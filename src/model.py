import re
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from my_utils.model_utils import make_1step_sched, my_lora_fwd
from noise_pred import HookBank, FuseHead, TimestepHead


VAE_LORA_TARGET_MODULES = (
    r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
)
# Only VAE encoder LoRA is trained/modulated; the decoder stays fixed as the
# SD-Turbo image renderer.
UNET_LORA_TARGET_MODULES = [
    "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
    "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj",
]


def get_layer_number(module_name):
    base_layers = {
        'down_blocks': 0,
        'mid_block': 4,
        'up_blocks': 5
    }

    if module_name == 'conv_out':
        return 9

    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break

    if base_layer is None:
        return None

    additional_layers = int(re.findall(r'\.(\d+)', module_name)[0])
    final_layer = base_layer + additional_layers
    return final_layer


def make_frontloaded_indices(num_train_timesteps=1000, n=100, gamma=2.5, final_step=1):
    """
    Generate a descending list of exactly n timestep indices:
    - Larger gaps at the beginning, smaller gaps near the end (gamma > 1)
    - Strictly decreasing
    - The last element is fixed to final_step (usually 1 instead of 0)
    """
    hi = num_train_timesteps - 1
    u = np.linspace(0.0, 1.0, n, endpoint=True)
    g = (1.0 - u) ** gamma
    tgt = np.rint(hi * g).astype(np.int64)

    out = np.empty(n, dtype=np.int64)
    out[-1] = final_step
    for i in range(n - 2, -1, -1):
        want = int(tgt[i])
        out[i] = max(min(want, hi), out[i + 1] + 1)

    assert out[-1] == final_step
    assert np.all(out[:-1] > out[1:])
    assert out[0] <= hi
    return out.tolist()


class InfraredIR(torch.nn.Module):
    def __init__(self, sd_path=None, pretrained_path=None, opt=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(self.device)
        self.noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        self.sched = make_1step_sched(sd_path, device=self.device)

        self.opt = opt
        self.stage = opt.get('stage', 'Stage1')
        self.N_timesteps = self.opt["N_timesteps"]

        num_train_ts = getattr(getattr(self.sched, "config", None), "num_train_timesteps", 1000)
        self.register_buffer(
            "candidate_ts",
            torch.tensor(
                make_frontloaded_indices(num_train_timesteps=num_train_ts, n=self.N_timesteps, gamma=2.5, final_step=1),
                dtype=torch.long,
                device=self.device,
            ),
            persistent=True,
        )

        # Candidate timestep lookup used by the dynamic timestep head.
        self._build_noise_tables()

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        unet.to(self.device)
        vae.to(self.device)
        self.unet, self.vae = unet, vae
        self.text_encoder.requires_grad_(False)

        # Encoder hooks collect degradation-aware features for timestep prediction.
        self.fuser = FuseHead(
            in_chs=[128, 256, 512, 512],
            out_ch_latent=128
        ).to(self.device)
        self.t_head = TimestepHead(c_lat=128, txt_dim=0).to(self.device)
        self.hookbank = HookBank()
        self._register_vae_hooks()

        # Task prompts drive group-wise LoRA modulation for VAE encoder and UNet.
        self.current_task = self.opt['current_task']
        self.task_prompts = nn.ParameterDict({
            task: nn.Parameter(torch.ones(self.opt['prompt_len'], 512))
            for task in self.opt['task_list']
        })
        self.lora_rank_unet = self.opt['lora_rank_unet']
        self.lora_rank_vae = self.opt['lora_rank_vae']
        self.target_modules_vae = VAE_LORA_TARGET_MODULES
        self.target_modules_unet = UNET_LORA_TARGET_MODULES
        checkpoint = None
        if pretrained_path is not None:
            # Load pretrained checkpoint. Obsolete tensors or differently shaped
            # prompt heads are skipped for backward compatibility.
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(
                r=self.lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=self.target_modules_vae,
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            self._load_matching_state_dict(vae, checkpoint["state_dict_vae"])

            unet_lora_config = LoraConfig(
                r=self.lora_rank_unet,
                init_lora_weights="gaussian",
                target_modules=self.target_modules_unet,
            )
            unet.add_adapter(unet_lora_config)
            self._load_matching_state_dict(unet, checkpoint["state_dict_unet"])

            self._load_matching_state_dict(self.fuser, checkpoint["state_dict_fuser"])
            self._load_matching_state_dict(self.t_head, checkpoint["state_dict_t_head"])

            # Restore task prompt parameters when they exist in the checkpoint.
            for task_name, tensor in checkpoint["task_prompts"].items():
                if task_name in self.task_prompts:
                    self.task_prompts[task_name].data.copy_(
                        tensor.to(self.task_prompts[task_name].device, dtype=self.task_prompts[task_name].dtype)
                    )
            print(f"[task_prompts] loaded: {len(checkpoint['task_prompts'])} entries")
        else:
            # Create fresh LoRA adapters when training from the base SD-Turbo model.
            print("Initializing model with random weights")
            vae_lora_config = LoraConfig(r=self.lora_rank_vae, init_lora_weights="gaussian",
                target_modules=self.target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            unet_lora_config = LoraConfig(r=self.lora_rank_unet, init_lora_weights="gaussian",
                target_modules=self.target_modules_unet
            )
            unet.add_adapter(unet_lora_config)

            # The timestep predictor is project-specific, so it starts from scratch.
            self.fuser.apply(self._init_weights)
            self.t_head.apply(self._init_weights)

        # Prompt modulation is block-wise: all LoRA modules in the same encoder or
        # UNet block share one rank-by-rank modulation matrix.
        self.vae_encoder_lora_layers = self._collect_lora_layers(vae, name_filter="encoder")
        self.unet_lora_layers = self._collect_lora_layers(unet)
        self.vae_encoder_lora_groups = self._group_lora_layers(
            self.vae_encoder_lora_layers,
            self._vae_encoder_lora_group_key,
        )
        self.unet_lora_groups = self._group_lora_layers(
            self.unet_lora_layers,
            self._unet_lora_group_key,
        )
        self.vae_encoder_lora_group_names = list(self.vae_encoder_lora_groups.keys())
        self.unet_lora_group_names = list(self.unet_lora_groups.keys())

        # PEFT LoRA modules are patched so `de_mod` can be assigned before every
        # forward pass according to the current task prompt.
        self._patch_prompt_modulated_lora(vae, self.vae_encoder_lora_layers)
        self._patch_prompt_modulated_lora(unet, self.unet_lora_layers)

        self.prompt_mlp = self._build_prompt_mlp(self._prompt_modulation_dim()).to(self.device)
        self.prompt_mlp.apply(self._init_weights)
        if checkpoint is not None:
            self._load_matching_state_dict(self.prompt_mlp, checkpoint["state_dict_prompt_mlp"])


    @staticmethod
    def _init_weights(m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    @staticmethod
    def _build_prompt_mlp(output_dim: int):
        if output_dim <= 0:
            raise ValueError("Prompt MLP output_dim must be positive. No prompt-modulated LoRA layers were found.")
        return nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, output_dim),
        )

    @staticmethod
    def _load_matching_state_dict(module: nn.Module, source_state_dict):
        """Load checkpoint tensors that still match after architecture changes."""
        target_state_dict = module.state_dict()
        for key, value in source_state_dict.items():
            if key in target_state_dict and target_state_dict[key].shape == value.shape:
                target_state_dict[key] = value
        module.load_state_dict(target_state_dict)

    def set_current_task(self, task_name: str):
        """Select which task prompt controls the next forward pass."""
        if task_name not in self.task_prompts:
            raise ValueError(f"Unknown task '{task_name}'. Available tasks: {list(self.task_prompts.keys())}")
        self.current_task = task_name
        if hasattr(self.opt, "__setitem__"):
            self.opt["current_task"] = task_name

    def prompt_parameters(self):
        """Return trainable prompt parameters for the current training stage."""
        if self.stage == 'Stage1':
            return [self.task_prompts[self.current_task]]
        elif self.stage == 'Stage2':
            return list(self.task_prompts.parameters())
        raise ValueError(f"Unsupported stage: {self.stage}")


    def _register_vae_hooks(self):
        enc = self.vae.encoder
        name2mod = dict(enc.named_modules())

        wanted = [
            "conv_in",
            "down_blocks.1.resnets.1",
            "down_blocks.2.resnets.1",
            "mid_block.resnets.1",
        ]

        for w in wanted:
            if w in name2mod:
                self.hookbank.add(name2mod[w], f"enc.{w}")
            else:
                print(f"[hook] warn: {w} not found in VAE encoder")

        attn_candidate = None
        for n, m in enc.named_modules():
            if "mid_block" in n and ("attentions" in n or "transformer_blocks" in n):
                attn_candidate = (n, m); break
        if attn_candidate is not None:
            self.hookbank.add(attn_candidate[1], f"enc.{attn_candidate[0]}")


    def set_eval(self):
        super().eval()
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.fuser.eval()
        self.t_head.eval()
        self.prompt_mlp.eval()
        self.fuser.requires_grad_(False)
        self.t_head.requires_grad_(False)
        self.prompt_mlp.requires_grad_(False)

        for _, p in self.task_prompts.items():
            p.requires_grad_(False)

    def set_train(self):
        super().train()

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.fuser.requires_grad_(False)
        self.t_head.requires_grad_(False)
        self.prompt_mlp.requires_grad_(False)

        for _, p in self.task_prompts.items():
            p.requires_grad_(False)

        if self.stage == 'Stage1':
            # Stage1 learns shared restoration modules plus the prompt of the sampled task.
            self.unet.train()
            self.vae.train()
            self.fuser.train()
            self.t_head.train()
            self.prompt_mlp.train()

            self.fuser.requires_grad_(True)
            self.t_head.requires_grad_(True)
            self.prompt_mlp.requires_grad_(True)

            for t, p in self.task_prompts.items():
                p.requires_grad_(t == self.current_task)

            # UNet LoRA and conv_in remain trainable in Stage1.
            for n, _p in self.unet.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True

            self.unet.conv_in.requires_grad_(True)

            # VAE training is restricted to encoder LoRA; decoder parameters stay fixed.
            for n, _p in self.vae.named_parameters():
                if "lora" in n and "encoder" in n:
                    _p.requires_grad = True

        elif self.stage == 'Stage2':
            # Stage2 freezes restoration modules and adapts task prompts only.
            self.unet.eval()
            self.fuser.eval()
            self.t_head.eval()
            self.prompt_mlp.eval()
            
            self.vae.eval()

            for t, p in self.task_prompts.items():
                p.requires_grad_(True)


    def _build_noise_tables(self):
        """
        Build diffusion-noise lookup tables aligned with `candidate_ts`.
        """
        with torch.no_grad():
            dev = self.candidate_ts.device
            alphas_bar = self.sched.alphas_cumprod.to(dtype=torch.float32, device=dev)
            a_bar = alphas_bar[self.candidate_ts]
            noise_std = torch.sqrt((1.0 - a_bar).clamp_min(0.0))
            self.register_buffer("alpha_bar_table", a_bar, persistent=True)
            self.register_buffer("noise_std_table", noise_std, persistent=True)

    @torch.no_grad()
    def sigma_true_from_pair(self, lq: torch.Tensor, hq: torch.Tensor, in_latent: bool = True) -> torch.Tensor:
        """
        Estimate degradation strength from normalized LQ-HQ residual energy.
        Returns a tensor of shape [B] in [0, 1].
        """
        if in_latent:
            def to_lat(x):
                d = self.vae.encode(x).latent_dist
                return d.mode() * self.vae.config.scaling_factor
            xl, xh = to_lat(lq), to_lat(hq)
        else:
            xl, xh = lq, hq

        mse = (xl - xh).pow(2).flatten(1).mean(1)  # [B]
        xh_mean = xh.flatten(1).mean(1, keepdim=True)
        var = (xh - xh_mean.view(-1, 1, 1, 1)).pow(2).flatten(1).mean(1) + 1e-8  # [B]
        sigma_est = torch.sqrt((mse / var).clamp_min(0.0))
        sigma_est = (sigma_est.clamp(0.0, 2.0) / 2.0).clamp(0.0, 1.0)
        return sigma_est  # [B]

    @staticmethod
    def _collect_lora_layers(root_module: nn.Module, name_filter: Optional[str] = None):
        """Collect PEFT LoRA wrapper names without the `.base_layer` suffix."""
        lora_layers = []
        suffix = ".base_layer"
        for name, _ in root_module.named_modules():
            if not name.endswith(suffix):
                continue
            layer_name = name[:-len(suffix)]
            if name_filter is None or name_filter in layer_name:
                lora_layers.append(layer_name)
        return lora_layers

    @staticmethod
    def _vae_encoder_lora_group_key(layer_name: str):
        if layer_name.startswith("encoder.down_blocks."):
            match = re.match(r"encoder\.down_blocks\.\d+", layer_name)
            if match:
                return match.group(0)
        if layer_name.startswith("encoder.mid_block"):
            return "encoder.mid_block"
        if layer_name.startswith("encoder.conv_in"):
            return "encoder.conv_in"
        if layer_name.startswith("encoder.conv_out"):
            return "encoder.conv_out"
        return layer_name

    @staticmethod
    def _unet_lora_group_key(layer_name: str):
        for block_name in ("down_blocks", "up_blocks"):
            if layer_name.startswith(f"{block_name}."):
                match = re.match(rf"{block_name}\.\d+", layer_name)
                if match:
                    return match.group(0)
        if layer_name.startswith("mid_block"):
            return "mid_block"
        if layer_name.startswith("conv_in"):
            return "conv_in"
        if layer_name.startswith("conv_out"):
            return "conv_out"
        return layer_name

    @staticmethod
    def _group_lora_layers(layer_names, group_key_fn):
        """Group LoRA modules so one prompt matrix is shared per block."""
        groups = {}
        for layer_name in layer_names:
            group_name = group_key_fn(layer_name)
            groups.setdefault(group_name, []).append(layer_name)
        return groups

    @staticmethod
    def _patch_prompt_modulated_lora(root_module: nn.Module, layer_names):
        layer_name_set = set(layer_names)
        for name, module in root_module.named_modules():
            if name in layer_name_set:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

    @staticmethod
    def _set_groupwise_lora_modulation(root_module: nn.Module, lora_groups, group_names, modulation: torch.Tensor):
        """Assign one modulation matrix to every LoRA module in each group."""
        layer_to_modulation = {}
        for group_idx, group_name in enumerate(group_names):
            for layer_name in lora_groups[group_name]:
                layer_to_modulation[layer_name] = modulation[group_idx]

        for name, module in root_module.named_modules():
            if name in layer_to_modulation:
                module.de_mod = layer_to_modulation[name]

    def _prompt_modulation_dim(self):
        """Total prompt-MLP output size for all VAE encoder and UNet LoRA groups."""
        num_vae_params = len(self.vae_encoder_lora_group_names) * self.lora_rank_vae ** 2
        num_unet_params = len(self.unet_lora_group_names) * self.lora_rank_unet ** 2
        return num_vae_params + num_unet_params

    def _apply_prompt_lora_modulation(self, batch_size: int):
        # One task prompt produces one modulation matrix per VAE encoder/UNet group.
        task_embed = self.task_prompts[self.current_task].mean(dim=0, keepdim=True)
        modulation = self.prompt_mlp(task_embed)

        num_vae_groups = len(self.vae_encoder_lora_group_names)
        num_unet_groups = len(self.unet_lora_group_names)
        vae_dim = num_vae_groups * self.lora_rank_vae ** 2
        vae_modulation, unet_modulation = torch.split(
            modulation,
            [vae_dim, num_unet_groups * self.lora_rank_unet ** 2],
            dim=-1,
        )
        vae_modulation = vae_modulation.reshape(
            num_vae_groups,
            1,
            self.lora_rank_vae,
            self.lora_rank_vae,
        )
        unet_modulation = unet_modulation.reshape(
            num_unet_groups,
            1,
            self.lora_rank_unet,
            self.lora_rank_unet,
        )

        vae_modulation = vae_modulation.expand(-1, batch_size, -1, -1).contiguous()
        unet_modulation = unet_modulation.expand(-1, batch_size, -1, -1).contiguous()

        self._set_groupwise_lora_modulation(
            self.vae,
            self.vae_encoder_lora_groups,
            self.vae_encoder_lora_group_names,
            vae_modulation,
        )
        self._set_groupwise_lora_modulation(
            self.unet,
            self.unet_lora_groups,
            self.unet_lora_group_names,
            unet_modulation,
        )

    def _encode_prompt(self, prompt: Optional[list], batch_size: int, device: torch.device):
        if prompt is None:
            prompt = ["" for _ in range(batch_size)]
        caption_tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        return self.text_encoder(caption_tokens)[0]


    def forward(
        self,
        c_t: torch.Tensor,
        prompt: Optional[list] = None,
        *,
        hq_for_snr: Optional[torch.Tensor] = None,
        in_latent: bool = False,
    ):
    
        caption_enc = self._encode_prompt(prompt, batch_size=c_t.shape[0], device=c_t.device)

        self._apply_prompt_lora_modulation(batch_size=c_t.shape[0])
        self.hookbank.clear()
        dist = self.vae.encode(c_t).latent_dist
        encoded_control = dist.mode() * self.vae.config.scaling_factor

        # Multi-scale encoder features determine which diffusion timestep to use.
        feats = []
        for k, v in self.hookbank.buffers.items():
            feats.append(v)
        
        B, C, Ht, Wt = encoded_control.shape
        fused = self.fuser(feats, (Ht, Wt))
        sigma_pred = torch.sigmoid(self.t_head(fused)).flatten()  # [B]

        loss_sigma = None
        training_with_sigma = self.training and (hq_for_snr is not None)
        if training_with_sigma:
            with torch.no_grad():
                sigma_true = self.sigma_true_from_pair(c_t.detach(), hq_for_snr.detach(), in_latent=in_latent)
                
            loss_sigma = F.smooth_l1_loss(sigma_pred, sigma_true)


        # Map continuous degradation strength to the nearest candidate timestep.
        with torch.no_grad():
            diffs_pred = (self.noise_std_table[None, :] - sigma_pred[:, None]).abs()
            idx_pred  = diffs_pred.argmin(dim=1)
            t_used    = self.candidate_ts[idx_pred]

        if not self.training:
            print(t_used) 
 
        noisy_latent = encoded_control
        model_pred = self.unet(noisy_latent, t_used, encoder_hidden_states=caption_enc,).sample
        
        # Scheduler step is applied by timestep group because step() expects scalar t.
        x_denoised = torch.empty_like(encoded_control)
        unique_ts = torch.unique(t_used)
        for t in unique_ts:
            idxs = (t_used == t).nonzero(as_tuple=False).squeeze(1)
            t_scalar = int(t.item())

            sub_pred = model_pred[idxs]
            sub_noisy = noisy_latent[idxs]


            self.sched.timesteps = torch.tensor([t, 0], device=sub_pred.device, dtype=torch.long)


            sub_out   = self.sched.step(sub_pred, t_scalar, sub_noisy, return_dict=True).prev_sample
            x_denoised[idxs] = sub_out

        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        
        if training_with_sigma:
            return output_image, loss_sigma
        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip_conv" in k}
        sd["state_dict_fuser"] = {k: v for k, v in self.fuser.state_dict().items()}
        sd["state_dict_t_head"] = {k: v for k, v in self.t_head.state_dict().items()}
        sd["state_dict_prompt_mlp"] = {k: v for k, v in self.prompt_mlp.state_dict().items()}
        sd["task_prompts"] = {k: v for k, v in self.task_prompts.items()}
        torch.save(sd, outf)
