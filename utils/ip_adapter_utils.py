import random

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, \
    ControlNetModel, StableDiffusionControlNetPipeline
from ip_adapter import IPAdapter

from utils.image_utils import ImageUtils


class IPAdapterUtils:
    model = None
    pipeline = None
    models_path = "models"
    noise_scheduler = None
    vae = None
    image_encoder_path = None
    ip_ckpt = None
    device = "cuda"
    ip_model = None

    def __init__(self, model="runwayml/stable-diffusion-v1-5"):
        self.download_models()
        self.model = model
        vae_model_path = "stabilityai/sd-vae-ft-mse"
        self.vae = AutoencoderKL.from_pretrained(vae_model_path).to(
            dtype=torch.float16
        )
        self.image_encoder_path = self.models_path + "/models/image_encoder/"
        self.ip_ckpt = self.models_path + "/models/ip-adapter_sd15.bin"
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float16,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            feature_extractor=None,
            safety_checker=None
        )
        self.ip_model = IPAdapter(
            self.pipeline,
            self.image_encoder_path,
            self.ip_ckpt,
            self.device
        )

    @staticmethod
    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

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
            seed=random.randint(1, 100)
    ):
        image_utils = ImageUtils(input_image_path=input_image_path)
        image = image_utils.resize_image()

        images = self.ip_model.generate(
            pil_image=image,
            num_samples=num_variations,
            num_inference_steps=50,
            # height=image.height,
            # width=image.width,
            seed=seed
        )

        for i, image in enumerate(images):
            image.save(output_image_path + f"_{i}.png")

    def generate_image_from_depth(
            self,
            input_image_path,
            depth_map_path,
            output_image_path,
    ):
        # load controlnet
        controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_path,
            torch_dtype=torch.float16
        )
        # load SD pipeline
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            feature_extractor=None,
            safety_checker=None
        )
        # read image prompt
        image_utils = ImageUtils(input_image_path=input_image_path)
        image = image_utils.resize_image(256, 256)


        depth_map = Image.open(depth_map_path)
        self.image_grid(
            [
                image.resize((256, 256)),
                depth_map.resize((256, 256))
            ], 1,
            2
        )
        ip_model = IPAdapter(
            self.pipeline,
            self.image_encoder_path,
            self.ip_ckpt,
            self.device
        )
        images = ip_model.generate(
            pil_image=image, image=depth_map,
            num_samples=1, num_inference_steps=50,
            seed=42
        )

        for i, image in enumerate(images):
            image.save(output_image_path + f"_{i}.png")
