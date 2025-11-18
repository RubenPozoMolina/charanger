import random

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapter

from utils.image_utils import ImageUtils


class IPAdapterUtils:

    model = None
    pipeline = None
    models_path = "models"


    def __init__(self, model="runwayml/stable-diffusion-v1-5"):
        self.model = model
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        self.download_models()
        vae_model_path = "stabilityai/sd-vae-ft-mse"
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(
            dtype=torch.float16)
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )

    def download_models(self):
        print("Downloading IP-Adapter models...")
        snapshot_download(
            repo_id="h94/IP-Adapter",
            local_dir=self.models_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("Download complete.")

    def generate_image_variations(
            self,
            input_image_path,
            output_image_path,
            num_variations=4,
            seed = random.randint(1, 100)
    ):
        image_utils = ImageUtils(input_image_path=input_image_path)
        image = image_utils.resize_image()
        image_encoder_path = self.models_path + "/models/image_encoder/"
        ip_ckpt = self.models_path + "/models/ip-adapter_sd15.bin"
        ip_model = IPAdapter(
            self.pipeline,
            image_encoder_path,
            ip_ckpt,
            "cuda"
        )
        images = ip_model.generate(
            pil_image=image,
            num_samples=num_variations,
            num_inference_steps=50,
            # height=image.height,
            # width=image.width,
            seed=seed
        )

        for i, image in enumerate(images):
            image.save(output_image_path + f"_{i}.png")
