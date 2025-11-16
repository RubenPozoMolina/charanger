from datetime import datetime

import torch
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

checkpoint = "lllyasviel/control_v11p_sd15_canny"

image = load_image(
    "assets/girl_dancing2.jpg"
)

image = np.array(image)

low_threshold = 50
high_threshold = 150
image_blur = cv2.GaussianBlur(image, (5, 5), 1.0)
image = cv2.Canny(image_blur, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)

control_image.save("output/control_dancing_girl.png")

controlnet = ControlNetModel.from_pretrained(
    checkpoint,
    torch_dtype=torch.float16
)

# model = "runwayml/stable-diffusion-v1-5"
model = "lykon/dreamshaper-8"
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# seed = random.randint(0, 100)
prompt = "A redhead girl with a blue dress dancing on the beach"
negative_prompt = """
deformed, ugly, mutilated, disfigured, bad anatomy, bad proportions,
extra limbs, cloned face, deformed face, malformed limbs, 
missing arms, missing legs, extra arms, extra legs, 
fused fingers, too many fingers, long neck, cross-eyed,
mutated hands, poorly drawn hands, poorly drawn face,
mutation, bad hands, extra fingers, text, watermark
"""

for seed in range(10):
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.0,
        controlnet_conditioning_scale=0.6,
        generator=generator,
        image=control_image,
        safety_checker=None,
        requires_safety_checker=False
    ).images[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image.save(f'output/{timestamp}_dancing_girl.png')
