import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import gc
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import diffusers

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from model import InfraredIR
from my_utils.training_utils import parse_args_paired_training, PairedDataset
from my_utils.dwt_loss import DWTLoss
from my_utils.task_router import (
    get_task_collate_fn,
    build_task_components,
    move_task_components_to_device,
    compute_task_loss,
    save_task_visualization,
)


def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


def set_model_task(accelerator, model, task_type, train_mode):
    """Set the task prompt state on the unwrapped model before a task batch."""
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.set_current_task(task_type)
    if train_mode:
        unwrapped_model.set_train()
    else:
        unwrapped_model.set_eval()
    return unwrapped_model


def get_task_prompt(task_prompts, task_type, default_prompt):
    """Return the configured task prompt, falling back to the CLI prompt."""
    if task_prompts is None:
        return default_prompt
    return task_prompts.get(task_type, default_prompt)


def main(args):
    config = OmegaConf.load(args.base_config)
    hyper_parameters = config.get("hyper_parameters", {})
    STAGE = hyper_parameters.get("stage", "Stage1")
    lambda_snr = float(hyper_parameters.get("lambda_snr", 10.0))

    # Each optimization step samples one task, then draws a full batch from that task.
    task_list = list(hyper_parameters.get("task_list", []))
    if len(task_list) == 0:
        task_list = list(config.train.keys())

    task_weights_cfg = config.get("task_weights", {})
    task_weights = []
    for task_name in task_list:
        if task_name not in task_weights_cfg:
            raise ValueError(
                f"Missing sampling weight for task '{task_name}'. "
                f"Please set task_weights.{task_name} in the yaml config."
            )
        task_weights.append(float(task_weights_cfg[task_name]))

    task_weights = np.array(task_weights, dtype=np.float64)
    if np.any(task_weights < 0) or task_weights.sum() <= 0:
        raise ValueError(f"Invalid task_weights: {task_weights_cfg}")
    task_probs = task_weights / task_weights.sum()

    num_val_visualizations_per_task = int(config.get("num_val_visualizations_per_task", 10))
    task_prompts = config.get("task_prompts", {})

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        for task_name in task_list:
            os.makedirs(os.path.join(args.output_dir, "eval", task_name), exist_ok=True)

    net_iir = InfraredIR(opt=hyper_parameters, sd_path=sd_path, pretrained_path=args.pretrained_path)
    net_iir.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_iir.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available, please install it by running `pip install xformers`"
            )

    if args.gradient_checkpointing:
        net_iir.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net="vgg")
    net_lpips.requires_grad_(False)
    dwt_loss = DWTLoss()

    # Frozen downstream task models provide task-aware losses for non-enhancement tasks.
    task_ctx = {}
    for task_name in task_list:
        task_ctx[task_name] = build_task_components(task_name, args)

    layers_to_opt = []
    prompt_params = (
        list(net_iir.task_prompts.parameters())
        if hasattr(net_iir, "task_prompts")
        else net_iir.prompt_parameters()
    )
    if STAGE == 'Stage1':
        # Stage1 trains shared restoration adapters plus task prompts.
        layers_to_opt = layers_to_opt + list(net_iir.fuser.parameters()) + list(net_iir.t_head.parameters()) + \
            list(net_iir.prompt_mlp.parameters()) + prompt_params

        for n, _p in net_iir.unet.named_parameters():
            if "lora" in n:
                assert _p.requires_grad
                layers_to_opt.append(_p)
        layers_to_opt += list(net_iir.unet.conv_in.parameters())

        for n, _p in net_iir.vae.named_parameters():
            if "lora" in n and "encoder" in n:
                assert _p.requires_grad
                layers_to_opt.append(_p)

    elif STAGE == 'Stage2':
        # Stage2 keeps the restoration backbone frozen and adapts prompts only.
        layers_to_opt = layers_to_opt + net_iir.prompt_parameters()
    else:
        raise ValueError(f"Unsupported stage: {STAGE}")

    dataset_train = {}
    dataset_val = {}
    dl_train = {}
    dl_val = {}

    for task_name in task_list:
        if task_name not in config.train:
            raise ValueError(f"Missing config.train.{task_name}")
        if task_name not in config.validation:
            raise ValueError(f"Missing config.validation.{task_name}")

        train_cfg = config.train[task_name]
        val_cfg = config.validation[task_name]

        if "task_type" not in train_cfg:
            train_cfg.task_type = task_name
        if "task_type" not in val_cfg:
            val_cfg.task_type = task_name

        collate_fn = get_task_collate_fn(task_name)

        dataset_train[task_name] = PairedDataset(train_cfg)
        dl_train[task_name] = torch.utils.data.DataLoader(
            dataset_train[task_name],
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            collate_fn=collate_fn,
        )

        dataset_val[task_name] = PairedDataset(val_cfg)
        dl_val[task_name] = torch.utils.data.DataLoader(
            dataset_val[task_name],
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare the restoration model, optimizer, schedulers, and task dataloaders together.
    prepare_objects = [net_iir, optimizer, lr_scheduler]
    small_targets_model_index = None
    if "small_targets" in task_ctx and task_ctx["small_targets"].get("task_model", None) is not None:
        small_targets_model_index = len(prepare_objects)
        prepare_objects.append(task_ctx["small_targets"]["task_model"])

    train_dl_start_index = len(prepare_objects)
    for task_name in task_list:
        prepare_objects.append(dl_train[task_name])
    val_dl_start_index = len(prepare_objects)
    for task_name in task_list:
        prepare_objects.append(dl_val[task_name])

    prepared = accelerator.prepare(*prepare_objects)

    net_iir = prepared[0]
    optimizer = prepared[1]
    lr_scheduler = prepared[2]

    if small_targets_model_index is not None:
        task_ctx["small_targets"]["task_model"] = prepared[small_targets_model_index]

    for idx, task_name in enumerate(task_list):
        dl_train[task_name] = prepared[train_dl_start_index + idx]
    for idx, task_name in enumerate(task_list):
        dl_val[task_name] = prepared[val_dl_start_index + idx]

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_iir.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    dwt_loss.to(accelerator.device, dtype=weight_dtype)

    for task_name in task_list:
        task_ctx[task_name] = move_task_components_to_device(
            task_type=task_name,
            task_ctx=task_ctx[task_name],
            device=accelerator.device,
            dtype=weight_dtype,
        )

    if accelerator.is_main_process:
        print("Multi-task training tasks:", task_list)
        print("Task sampling probabilities:", {t: float(p) for t, p in zip(task_list, task_probs)})

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    train_iter = {task_name: iter(dl_train[task_name]) for task_name in task_list}

    for epoch in range(args.num_training_epochs):
        while global_step < args.max_train_steps:
            # A batch is task-homogeneous; multi-task learning happens across steps.
            task_type = str(np.random.choice(task_list, p=task_probs))

            try:
                batch = next(train_iter[task_type])
            except StopIteration:
                train_iter[task_type] = iter(dl_train[task_type])
                batch = next(train_iter[task_type])

            with accelerator.accumulate(net_iir):
                set_model_task(accelerator, net_iir, task_type, train_mode=True)

                x_src = batch["lq"]
                x_tgt = batch["gt"]

                B = x_src.shape[0]
                task_prompt = get_task_prompt(task_prompts, task_type, args.pos_prompt)
                pos_tag_prompt = [task_prompt for _ in range(B)]

                x_tgt_pred, loss_snr = net_iir(
                    x_src.detach(),
                    pos_tag_prompt,
                    hq_for_snr=x_tgt.detach(),
                    in_latent=False,
                )

                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean",) * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float(),).mean() * args.lambda_lpips
                loss_dwt = dwt_loss(x_tgt_pred.float(), x_tgt.detach().float())
                if global_step > 100:
                    lambda_snr = 0
                loss_snr = loss_snr * lambda_snr

                if task_type == "enhancement":
                    loss_task = torch.zeros((), device=x_tgt_pred.device, dtype=x_tgt_pred.dtype)
                    task_outputs = {}
                else:
                    # Task loss backpropagates through the restored image into InfraredIR.
                    loss_task, task_outputs = compute_task_loss(
                        task_type=task_type,
                        task_ctx=task_ctx[task_type],
                        x_tgt_pred=x_tgt_pred,
                        batch=batch,
                    )

                loss = loss_l2 + loss_lpips + loss_dwt + loss_snr + loss_task

                accelerator.backward(loss, retain_graph=False)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "task_id": task_list.index(task_type),
                        f"train/{task_type}/loss_l2": loss_l2.detach().item(),
                        f"train/{task_type}/loss_lpips": loss_lpips.detach().item(),
                        f"train/{task_type}/loss_dwt": loss_dwt.detach().item(),
                        f"train/{task_type}/loss_snr": loss_snr.detach().item(),
                        f"train/{task_type}/loss_task": loss_task.detach().item(),
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "loss_dwt": loss_dwt.detach().item(),
                        "loss_snr": loss_snr.detach().item(),
                        "loss_task": loss_task.detach().item(),
                    }
                    progress_bar.set_postfix(
                        task=task_type,
                        loss_l2=logs["loss_l2"],
                        loss_snr=logs["loss_snr"],
                        loss_task=logs["loss_task"],
                    )

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(
                            args.output_dir,
                            "checkpoints",
                            f"model_{global_step}.pkl",
                        )
                        accelerator.unwrap_model(net_iir).save_model(outf)

                    if global_step % args.eval_freq == 1:
                        for val_task_type in task_list:
                            l_l2, l_lpips, l_dwt, l_task = [], [], [], []
                            val_count = 0

                            for val_step, batch_val in enumerate(dl_val[val_task_type]):
                                if val_step >= num_val_visualizations_per_task:
                                    break

                                x_src = batch_val["lq"]
                                x_tgt = batch_val["gt"]

                                B = x_src.shape[0]
                                assert B == 1, "Use batch size 1 for eval."

                                with torch.no_grad():
                                    # Validation uses the prompt and train/eval state of the current task.
                                    unwrapped_net = set_model_task(
                                        accelerator,
                                        net_iir,
                                        val_task_type,
                                        train_mode=False,
                                    )

                                    task_prompt = get_task_prompt(task_prompts, val_task_type, args.pos_prompt)
                                    pos_tag_prompt = [task_prompt for _ in range(B)]
                                    x_tgt_pred = unwrapped_net(x_src.detach(), pos_tag_prompt)

                                    loss_l2_val = F.mse_loss(
                                        x_tgt_pred.float(),
                                        x_tgt.detach().float(),
                                        reduction="mean",
                                    )
                                    loss_lpips_val = net_lpips(
                                        x_tgt_pred.float(),
                                        x_tgt.detach().float(),
                                    ).mean()
                                    loss_dwt_val = dwt_loss(x_tgt_pred.float(), x_tgt.detach().float())

                                    if val_task_type == "enhancement":
                                        loss_task_val = torch.zeros((), device=x_tgt_pred.device, dtype=x_tgt_pred.dtype)
                                        task_outputs = {}
                                    else:
                                        loss_task_val, task_outputs = compute_task_loss(
                                            task_type=val_task_type,
                                            task_ctx=task_ctx[val_task_type],
                                            x_tgt_pred=x_tgt_pred,
                                            batch=batch_val,
                                        )

                                l_l2.append(loss_l2_val.item())
                                l_lpips.append(loss_lpips_val.item())
                                l_dwt.append(loss_dwt_val.item())
                                l_task.append(loss_task_val.item())

                                if args.save_val and val_count < num_val_visualizations_per_task:
                                    outf = os.path.join(
                                        args.output_dir,
                                        "eval",
                                        val_task_type,
                                        f"val_{global_step}_{val_count}.png",
                                    )
                                    save_task_visualization(
                                        task_type=val_task_type,
                                        task_ctx=task_ctx[val_task_type],
                                        batch=batch_val,
                                        x_src=x_src,
                                        x_tgt=x_tgt,
                                        x_tgt_pred=x_tgt_pred,
                                        task_outputs=task_outputs,
                                        save_path=outf,
                                    )
                                    val_count += 1

                            logs[f"val/{val_task_type}/l2"] = safe_mean(l_l2)
                            logs[f"val/{val_task_type}/lpips"] = safe_mean(l_lpips)
                            logs[f"val/{val_task_type}/dwt"] = safe_mean(l_dwt)
                            logs[f"val/{val_task_type}/task"] = safe_mean(l_task)

                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
