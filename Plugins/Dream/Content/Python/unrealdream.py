#This is a version from stable diffusion optimized SD https://github.com/basujindal/stable-diffusion/tree/main/optimizedSD
#original repo https://github.com/basujindal/stable-diffusion
import unreal
unreal.log("Unreal Stable Diffusion - Let's dream!")

import argparse, os
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from einops import rearrange, repeat
from ldm.util import instantiate_from_config

def split_weighted_subprompts(text):
    """
    grabs all text up to the first occurrence of ':'
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        if ":" in text:
            idx = text.index(":") # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx+1:]
            # find value for weight
            if " " in text:
                idx = text.index(" ") # first occurence
            else: # no space, read to end
                idx = len(text)
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except: # couldn't treat as float
                    unreal.log_warning(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                    weight = 1.0
            else: # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx+1:]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(weight)
        else: # no : found
            if len(text) > 0: # there is still text though
                # take remainder as weight 1
                prompts.append(text)
                weights.append(1.0)
            remaining = 0
    return prompts, weights

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    unreal.log(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        unreal.log(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(path, h0, w0):

    image = Image.open(path).convert("RGB")
    w, h = image.size

    unreal.log(f"loaded input image of size ({w}, {h}) from {path}")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    unreal.log(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


config = f"{unreal.Paths.project_plugins_dir()}Dream/Content/Python/v1-inference.yaml"
ckpt = f"{unreal.Paths.project_plugins_dir()}Dream/Content/Python/model/model.ckpt"
outdir = unreal.Paths.screen_shot_dir()
init_img = f"{unreal.Paths.screen_shot_dir()}dream.png"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt", type=str, nargs="?", default="lions everywhere", help="the dream to render"
)

parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default=outdir)
parser.add_argument("--init-img", type=str, nargs="?", help="path to the input image", default=init_img)

parser.add_argument(
    "--skip_grid",
    action="store_true",
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action="store_true",
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--strength",
    type=float,
    default=0.5,
    help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=3,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--seed",
    type=int,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="CPU or GPU (cuda/cuda:0/cuda:1/...)",
)
parser.add_argument(
    "--unet_bs",
    type=int,
    default=1,
    help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
)
parser.add_argument(
    "--turbo",
    action="store_true",
    default=True,
    help="Reduces inference time on the expense of 1GB VRAM",
)
parser.add_argument(
    "--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast"
)
parser.add_argument(
    "--format",
    type=str,
    help="output image format",
    choices=["jpg", "png"],
    default="png",
)
parser.add_argument(
    "--sampler",
    type=str,
    help="sampler",
    choices=["ddim"],
    default="ddim",
)
opt = parser.parse_args()


tic = time.time()
os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir
grid_count = len(os.listdir(outpath)) - 1

if opt.seed == None:
    opt.seed = randint(0, 1000000)
seed_everything(opt.seed)
with unreal.ScopedSlowTask(4, "Loading weight") as slow_task3:
    slow_task3.make_dialog(True)               # Makes the dialog visible, if it isn't already
    sd = load_model_from_config(f"{ckpt}")
    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img, opt.H, opt.W).to(opt.device)

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.cdevice = opt.device
    model.unet_bs = opt.unet_bs
    model.turbo = opt.turbo
    slow_task3.enter_progress_frame(1)
    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = opt.device
    slow_task3.enter_progress_frame(2)
    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    slow_task3.enter_progress_frame(3)
    del sd
    if opt.device != "cpu" and opt.precision == "autocast":
        model.half()
        modelCS.half()
        modelFS.half()
        init_image = init_image.half()

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        assert opt.prompt is not None
        prompt = opt.prompt
        data = [batch_size * [prompt]]

    else:
        unreal.log(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = batch_size * list(data)
            data = list(chunk(sorted(data), batch_size))
    modelFS.to(opt.device)
    slow_task3.enter_progress_frame(4)

init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space

if opt.device != "cpu":
    mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
    modelFS.to("cpu")
    while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
        time.sleep(1)


assert 0.0 <= opt.strength <= 1.0, "can only work with strength in [0.0, 1.0]"
t_enc = int(opt.strength * opt.ddim_steps)
unreal.log(f"target t_enc is {t_enc} steps")


if opt.precision == "autocast" and opt.device != "cpu":
    precision_scope = autocast
else:
    precision_scope = nullcontext

seeds = ""
with torch.no_grad():
    all_samples = list()
    with unreal.ScopedSlowTask(opt.n_iter, "Unreal is dreaming!") as slow_task:
        slow_task.make_dialog(True)
        for n in trange(opt.n_iter, desc="Sampling"):
            if slow_task.should_cancel():         # True if the user has pressed Cancel in the UI
                break
            for prompts in tqdm(data, desc="data"):
                sample_path = outpath
                base_count = len(unreal.EditorAssetLibrary.list_assets(sample_path))

                with precision_scope("cuda"):
                    modelCS.to(opt.device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
                            time.sleep(1)

                    # encode (scaled latent)
                    z_enc = model.stochastic_encode(
                        init_latent,
                        torch.tensor([t_enc] * batch_size).to(opt.device),
                        opt.seed,
                        opt.ddim_eta,
                        opt.ddim_steps,
                    )
                    # decode it
                    samples_ddim = model.sample(
                        t_enc,
                        c,
                        z_enc,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        sampler = opt.sampler
                    )

                    modelFS.to(opt.device)
                    unreal.log("saving images")
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.{opt.format}")
                        )
                        seeds += str(opt.seed) + ","
                        opt.seed += 1
                        base_count += 1

                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim
                    unreal.log(f"memory_final = {torch.cuda.memory_allocated(device=opt.device) / 1e6}")
            slow_task.enter_progress_frame(n)


del modelFS
del modelCS
del model
torch.cuda.empty_cache()
toc = time.time()


toc = time.time()

time_taken = (toc - tic) / 60.0

unreal.log(f"Samples finished in {0:.2f} minutes and exported to {sample_path}\n Seeds used {seeds[:-1]}")
unreal.log(format(time_taken))
exit()
