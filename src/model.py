import os
import re
import sys
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
from typing import Optional

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

    additional_layers = int(re.findall(r'\.(\d+)', module_name)[0]) #sum(int(num) for num in re.findall(r'\d+', module_name))
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
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        self.sched = make_1step_sched(sd_path)

        self.opt = opt
        self.stage = opt.get('stage', 'Stage1')
        # Candidate timesteps
        self.N_timesteps = self.opt["N_timesteps"]

        num_train_ts = getattr(getattr(self.sched, "config", None), "num_train_timesteps", 1000)
        self.register_buffer(
            "candidate_ts",
            torch.tensor(
                make_frontloaded_indices(num_train_timesteps=num_train_ts, n=self.N_timesteps, gamma=2.5, final_step=1),
                dtype=torch.long,
                device="cuda",
            ),
            persistent=True,
        )

        # Build log-SNR lookup table aligned with candidate_ts
        self._build_noise_tables()

        # VAE / UNet LoRA settings
        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        target_modules_vae = r"^(encoder|decoder)\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]


        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.text_encoder.requires_grad_(False)

       # Fusion module and timestep prediction head
        self.fuser = FuseHead(
            in_chs=[128, 256, 512, 512],    # Replace with actual hooked channel sizes
            out_ch_latent=128
        ).cuda()
        self.t_head = TimestepHead(c_lat=128, txt_dim=0).cuda()
        self.hookbank = HookBank()
        self._register_vae_hooks()

        # ===== Stage 2 preparation =====
        # Task Prompts, single token
        self.current_task = self.opt['current_task']  # Default task
        self.task_prompts = nn.ParameterDict({
            task: nn.Parameter(torch.ones(self.opt['prompt_len'], 512))
            for task in self.opt['task_list']
        })
        self.lora_rank_unet = self.opt['lora_rank_unet']
        self.lora_rank_vae = self.opt['lora_rank_vae']
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        # self.vae_de_mlp = nn.Sequential(
        #     nn.Linear(num_embeddings * 4, 256),
        #     nn.ReLU(True),
        # )

        self.prompt_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, self.lora_rank_vae ** 2 * 8)
        )
       # ===== End of Stage 2 preparation =====


        if pretrained_path is not None:
            # Load pretrained checkpoin
            sd = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)

            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

            _sd_fuser = self.fuser.state_dict()
            for k in sd["state_dict_fuser"]:
                _sd_fuser[k] = sd["state_dict_fuser"][k]
            self.fuser.load_state_dict(_sd_fuser)

            _sd_t_head = self.t_head.state_dict()
            for k in sd["state_dict_t_head"]:
                _sd_t_head[k] = sd["state_dict_t_head"][k]
            self.t_head.load_state_dict(_sd_t_head)

            _sd_prompt_mlp = self.prompt_mlp.state_dict()
            for k in sd["state_dict_prompt_mlp"]:
                _sd_prompt_mlp[k] = sd["state_dict_prompt_mlp"][k]
            self.prompt_mlp.load_state_dict(_sd_prompt_mlp)

            # Load task-specific prompt parameters
            for task_name, tensor in sd["task_prompts"].items():
                if task_name in self.task_prompts:
                    self.task_prompts[task_name].data.copy_(
                        tensor.to(self.task_prompts[task_name].device, dtype=self.task_prompts[task_name].dtype)
                    )
            print(f"[task_prompts] loaded: {len(sd['task_prompts'])} entries")
            # else:
            #     print("[task_prompts] not found in checkpoint (skip)")


        else:
            # Initialize LoRA adapters from scratch
            print("Initializing model with random weights")
            vae_lora_config = LoraConfig(r=self.lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            unet_lora_config = LoraConfig(r=self.lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)

            # Randomly initialize fuser and timestep head
            def init_weights(m):
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                    torch.nn.init.ones_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            self.fuser.apply(init_weights)
            self.t_head.apply(init_weights)
            self.prompt_mlp.apply(init_weights)
            # default_init_weights([self.fuser, self.t_head], 1e-5)

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name and "decoder" in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])

        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)


        # print(self.vae_lora_layers)          

        # self.unet_lora_layers = []
        # for name, module in unet.named_modules():
        #     if 'base_layer' in name:
        #         self.unet_lora_layers.append(name[:-len(".base_layer")])


    def prompt_parameters(self):
        # Return only the prompt of the current task
        # (and any other parameters you want to train together)
        # params = [self.task_prompts[self.current_task]]
        # Append shared module parameters if needed
        # params += list(self.shared_module.parameters())
        # return params
        if self.stage == 'Stage1':
            return [self.task_prompts[self.current_task]]
        elif self.stage == 'Stage2':
            return list(self.task_prompts.parameters())


    def _register_vae_hooks(self):
        enc = self.vae.encoder
        name2mod = dict(enc.named_modules())

        wanted = [
            "conv_in",                     # High-resolution input stage
            "down_blocks.1.resnets.1",     # Middle-level feature
            "down_blocks.2.resnets.1",     # Near-latent feature
            "mid_block.resnets.1",         # Latent-scale feature
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
        super().eval()  # sets self.training = False
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
            # Stage1: fuser + t_head + prompt_mlp + task_prompts
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

            # Stage1: train UNet LoRA
            for n, _p in self.unet.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True

            # Stage1: train UNet conv_in
            self.unet.conv_in.requires_grad_(True)

            # Stage1: train VAE encoder LoRA
            for n, _p in self.vae.named_parameters():
                if "lora" in n and "encoder" in n:
                    _p.requires_grad = True

        elif self.stage == 'Stage2':
            # Stage2: VAE decoder LoRA + task prompts
            self.unet.eval()
            self.fuser.eval()
            self.t_head.eval()
            self.prompt_mlp.eval()
            
            self.vae.train()

            # for n, _p in self.vae.named_parameters():
            #     if "lora" in n and "decoder" in n:
            #         _p.requires_grad = True

            for t, p in self.task_prompts.items():
                p.requires_grad_(True)


    # -------- tables --------
    def _build_noise_tables(self):
        """
        Build tables aligned with candidate_ts:
        - alpha_bar_table: cumulative alpha_bar[t]
        - noise_std_table: sigma_t = sqrt(1 - alpha_bar[t])
        """
        with torch.no_grad():
            dev = self.candidate_ts.device
            alphas_bar = self.sched.alphas_cumprod.to(dtype=torch.float32, device=dev)  # [T]
            a_bar = alphas_bar[self.candidate_ts]                                       # [K]
            noise_std = torch.sqrt((1.0 - a_bar).clamp_min(0.0))                        # [K]
            self.register_buffer("alpha_bar_table", a_bar, persistent=True)
            self.register_buffer("noise_std_table", noise_std, persistent=True)

    @torch.no_grad()
    def sigma_true_from_pair(self, lq: torch.Tensor, hq: torch.Tensor, in_latent: bool = True) -> torch.Tensor:
        """
        Estimate degradation noise strength sigma_true ∈ [0, +∞) using a robust approximation:
        sigma_true ≈ sqrt( MSE(lq,hq) / Var(hq) ), then clamp to [0, 1]
        Return: sigma_true of shape [B] in [0, 1]
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
        sigma_est = torch.sqrt((mse / var).clamp_min(0.0))  # Approximate sqrt(1 - alpha_bar)
        # print("min-max",sigma_est.min().item(), sigma_est.max().item())
        # Clamp possible values > 1 and map them into [0, 1]
        # 2.0 is used to reduce the effect of extreme values during training
        sigma_est = (sigma_est.clamp(0.0, 2.0) / 2.0).clamp(0.0, 1.0)
        return sigma_est  # [B]


    # --------------------------- forward ---------------------------
    def forward(
        self,
        c_t: torch.Tensor,
        prompt: Optional[list] = None,
        *,
        hq_for_snr: Optional[torch.Tensor] = None,
        in_latent: bool = False,  # "latent" or "pixel"
    ):
    
        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]     # if needed

        self.hookbank.clear()
        dist = self.vae.encode(c_t).latent_dist
        encoded_control = dist.mode() * self.vae.config.scaling_factor  # or dist.mean

        # Collect multi-scale features from hooks
        feats = []
        for k, v in self.hookbank.buffers.items():
            feats.append(v)
        
        B, C, Ht, Wt = encoded_control.shape
        fused = self.fuser(feats, (Ht, Wt))
        sigma_pred = torch.sigmoid(self.t_head(fused)).flatten()  # [B]


        # Supervise sigma during training
        loss_sigma = None
        training_with_sigma = self.training and (hq_for_snr is not None)
        if training_with_sigma:
            with torch.no_grad():
                sigma_true = self.sigma_true_from_pair(c_t.detach(), hq_for_snr.detach(), in_latent=in_latent)  # [B]
                
            # print("sigma_true:", sigma_true, "sigma_pred:", sigma_pred)
            loss_sigma = F.smooth_l1_loss(sigma_pred, sigma_true)


        # Map continuous sigma to the nearest discrete timestep
        # Candidate table: sigma_t = sqrt(1 - alpha_bar[t])
        with torch.no_grad():
            diffs_pred = (self.noise_std_table[None, :] - sigma_pred[:, None]).abs()  # [B,K]
            idx_pred  = diffs_pred.argmin(dim=1)                                      # [B]
            t_used    = self.candidate_ts[idx_pred]
            # print(t_used)

        if not self.training:
            print(t_used) 
 
        noisy_latent = encoded_control
        model_pred = self.unet(noisy_latent, t_used, encoder_hidden_states=caption_enc,).sample
        
        # Apply scheduler step group by group, since step() expects a scalar t
        x_denoised = torch.empty_like(encoded_control)  # [B,C,H,W]
        unique_ts = torch.unique(t_used)
        for t in unique_ts:
            idxs = (t_used == t).nonzero(as_tuple=False).squeeze(1)
            t_scalar = int(t.item())

            sub_pred  = model_pred[idxs]   # UNet output for this timestep group
            sub_noisy = noisy_latent[idxs] # Corresponding latent inputs


            self.sched.timesteps = torch.tensor([t, 0], device=sub_pred.device, dtype=torch.long)


            sub_out   = self.sched.step(sub_pred, t_scalar, sub_noisy, return_dict=True).prev_sample
            x_denoised[idxs] = sub_out
        
        ########### task embedding to VAE decoder LoRA ############
        vae_embeds = self.prompt_mlp(self.task_prompts[self.current_task])  # [num_tasks, lora_rank_vae**2 * 8]

        vae_embeds = vae_embeds.reshape(-1, 8, self.lora_rank_vae, self.lora_rank_vae)  # [num_tasks, 8, r, r]
        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                split_name = layer_name.split(".")
                if split_name[1] == 'up_blocks':
                    block_id = int(split_name[2])
                    vae_embed = vae_embeds[:, block_id]
                elif split_name[1] == 'mid_block':
                    if split_name[2] == 'attentions' or split_name[2] == 'transformer_blocks':
                        vae_embed = vae_embeds[:, 4]
                    elif split_name[2] == 'resnets':
                        vae_embed = vae_embeds[:, 5]
                elif split_name[1] == 'conv_in':
                    vae_embed = vae_embeds[:, 6]
                else:
                    vae_embed = vae_embeds[:, -1]
                module.de_mod = vae_embed.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)


        # x_denoised = self.sched.step(model_pred, t_used, noisy_latent, return_dict=True).prev_sample
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
