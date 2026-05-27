import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gc
import tqdm
import math
import lpips
# import pyiqa
import clip
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

import diffusers
# import utils.misc as misc

from diffusers.utils.import_utils import is_xformers_available

from model import InfraredIR

from my_utils.testing_utils import parse_args_paired_testing, SingleDataset


def main(args):
    config = OmegaConf.load(args.base_config)

    hyper_parameters = config.get("hyper_parameters", {})
    STAGE = hyper_parameters.get("stage", "Stage1")

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

    # initialize net_iir
    net_iir = InfraredIR(opt=hyper_parameters, sd_path=sd_path, pretrained_path=args.pretrained_path)
    net_iir.set_eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_iir.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_iir.unet.enable_gradient_checkpointing()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_val = SingleDataset(config.validation)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare everything with our `accelerator`.
    net_iir = accelerator.prepare(net_iir)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_iir.to(accelerator.device, dtype=weight_dtype)
      
    for step, batch_val in enumerate(dl_val):
        lq_path = batch_val['lq_path'][0]
        (path, name) = os.path.split(lq_path)

        im_lq = batch_val['lq'].cuda()
        im_lq = im_lq.to(memory_format=torch.contiguous_format).float()    

        im_lq_resize = im_lq.contiguous() 
        im_lq_resize_norm = im_lq_resize * 2 - 1.0
        im_lq_resize_norm = torch.clamp(im_lq_resize_norm, -1.0, 1.0)
        resize_h, resize_w = im_lq_resize_norm.shape[2:]

        pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
        pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
        im_lq_resize_norm = F.pad(im_lq_resize_norm, pad=(0, pad_w, 0, pad_h), mode='reflect')
        
        B = im_lq.size(0)
        with torch.no_grad():
            pos_tag_prompt = [args.pos_prompt for _ in range(B)]
            x_tgt_pred = accelerator.unwrap_model(net_iir)(im_lq_resize_norm, pos_tag_prompt)
            x_tgt_pred = x_tgt_pred[:, :, :resize_h, :resize_w]
            out_img = (x_tgt_pred * 0.5 + 0.5).cpu().detach()

        out_single = out_img[0, 0:1, :, :] 
        output_pil = transforms.ToPILImage()(out_single)

        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.output_dir, fname+'.png')
        output_pil.save(outf)

    # print_results = evaluate(args.output_dir, args.ref_path, None)
    # out_t = os.path.join(args.output_dir, 'results.txt')
    # with open(out_t, 'w', encoding='utf-8') as f:
    #     for item in print_results:
    #         f.write(f"{item}\n")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args_paired_testing()
    main(args)
