import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from CT_dataloader import CTDataset, CTDataset_relative
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from torch.utils.data import Dataset
from torchvision import datasets
from glob import glob
import pandas as pd


from datetime import datetime
import matplotlib.pyplot as plt

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
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
        default=2,
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
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
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
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/nfs/turbo/coe-liyues/bowenbw/stable-diffusion/sd1.2/sd-v1-2.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    def sample_test(im_name_list, c, sc_ = None):
        start_code = None
        with torch.no_grad():
            uc = None
            if opt.scale != -1.0:
                uc = model.get_learned_conditioning(2 * [""])

                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=2,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 sc = sc_,
                                                 x_T=start_code)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                
                cur_bandwidth = sc_[0].item()

                plt.imsave(f"/scratch/liyues_root/liyues/shared_data/bowenbw/samples/{im_name_list[0]}_bandwidth{cur_bandwidth}.png", (x_samples_ddim[0]*255).astype('uint8'))
    
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

#     seed_everything(opt.seed) ####unseed this for training

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    model.train()
    model.learning_rate = 1e-5
    model.use_scheduler=False
    
    ds = CTDataset()
    batch_size = 2
    num_samples = 4
    params = {'batch_size': batch_size}
#     training_generator = torch.utils.data.DataLoader(ds, **params)

    ds2 = CTDataset_relative(num_samples = num_samples)
    print(f"number of data in the dataset is {num_samples}")
    training_generator = torch.utils.data.DataLoader(ds2, **params)
    prompt = ["Chest CT image"]
    iterations = 0
    opt_ = model.configure_optimizers()
#     model = load_model_from_config(config, "/nfs/turbo/coe-liyues/bowenbw/3DCT/checkpoints/SD_spatial_finetune_iter5000_2024031613.ckpt")
    sampler = DDIMSampler(model)
    now_ = datetime.now()
    ymdh = now_.strftime("%Y%m%d%H")
    iterations = 0
    img_paths = glob("/nfs/turbo/coe-liyues/bowenbw/photonic_devices/one_unit_cell_mat_64/*.npy")
    
    prompt = ["black and white shapes"]
    ############################################start fine tuning 3 slices##########################################
    sc_ = None
    
    labels = pd.read_csv("/nfs/turbo/coe-liyues/bowenbw/photonic_devices/label_all_THz.txt", sep = " ")
    
    
    
    for epoch in range(20000):
        sc_ =None
        c = model.get_learned_conditioning(batch_size * prompt)
        c = c.to(device)
        file_path = np.random.choice(img_paths)
        data = np.load(file_path)  # Assuming it has shape (64, 64)
        
        
        
        
        x = torch.rand(2, 3, 512, 512).to(model.first_stage_model.dtype).to(model.first_stage_model.device)
        sc_ = torch.rand(2).to(model.first_stage_model.dtype).to(model.first_stage_model.device)

        # Resize to (512, 512, 3)
        # First, convert the 2D data to 3D by duplicating the 64x64 grayscale to 3 channels
        
        for i in range(batch_size):
            file_path = np.random.choice(img_paths)
            data = np.load(file_path)  # Assuming it has shape (64, 64)
            
            img_ind = int((file_path.split("_")[-1]).split(".")[0])
            
            label = labels.loc[img_ind]['a7']
            data_3d = np.stack([data]*3, axis=-1)  # Shape becomes (64, 64, 3)

        # Resize using PIL
            image = Image.fromarray(np.uint8(data_3d))  # Convert to image format
            resized_image = image.resize((512, 512), Image.BICUBIC)

        # Convert the resized image back to a NumPy array
            resized_data = np.array(resized_image)  # Shape (512, 512, 3)

        # Transpose to (1, 3, 512, 512) for PyTorch
            x_ = torch.tensor(np.transpose(resized_data, (2, 0, 1)))
            x_= x_.to(model.first_stage_model.dtype).to(model.first_stage_model.device)
            x[i] = x_
            sc_[i] = label
            
        z = model.get_first_stage_encoding(model.encode_first_stage(x))
        loss, loss_dict = model.forward(z, c, sc = sc_)
        print(loss.item(), file = open("loss_1222024.txt", 'a'))
        loss.backward()
        opt_.step()
        opt_.zero_grad()
        
        if iterations % 100 == 0:
            print("cur iteration is: ", iterations)
            im_name_list = []
            for j in range(batch_size):
                im_name_list.append("photonic_iter" + str(iterations) + ymdh)
            sample_test(im_name_list, c, sc_ = sc_)
        if iterations % 2000 == 0:
            torch.save({'iterations':iterations,'state_dict': model.state_dict()}, "/scratch/liyues_root/liyues/shared_data/bowenbw/checkpoints/" + ymdh + "_photonics" + ".ckpt")
        iterations += 1

    
if __name__ == "__main__":
    main()
