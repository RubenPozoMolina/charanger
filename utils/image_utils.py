import datetime

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


class ImageUtils:
    image = None

    def __init__(self, input_image_path=None):
        self.image = input_image_path

    def resize_image(self, width=512, height=512):
        image = Image.open(self.image)

        # Calculate aspect ratios
        target_aspect = width / height
        source_width, source_height = image.size
        source_aspect = source_width / source_height

        # Determine crop dimensions to match target aspect ratio
        if source_aspect > target_aspect:
            # Image is wider than target, crop sides
            new_width = int(source_height * target_aspect)
            new_height = source_height
            left = (source_width - new_width) // 2
            top = 0
            right = left + new_width
            bottom = source_height
        else:
            # Image is taller than target, crop top/bottom
            new_width = source_width
            new_height = int(source_width / target_aspect)
            left = 0
            top = (source_height - new_height) // 2
            right = source_width
            bottom = top + new_height

        # Crop to target aspect ratio
        image = image.crop((left, top, right, bottom))

        # Resize to target dimensions
        image = image.resize((width, height), Image.LANCZOS)
        return image

    def resize_image_to_file(self, output_image_path, width=512, height=512):
        image = self.resize_image(width, height)
        image.save(output_image_path)

    @staticmethod
    def generate_image_from_text(
            prompt, negative_prompt,
            model, output_image_path, height=512, width=512,
    ):
        # Prepare the pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model, torch_dtype=torch.float16
        ).to("cuda")
        torch.cuda.empty_cache()
        pipe.enable_xformers_memory_efficient_attention()

        # Generate image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            height=height,
            width=width,
            strength=0.5
        ).images[0]

        # Save image
        output_file = output_image_path
        image.save(output_file)

    def pad_image(self, width=512, height=512, color=(0, 0, 0)):
        image = Image.open(self.image)

        # Calculate aspect ratios
        target_aspect = width / height
        source_width, source_height = image.size
        source_aspect = source_width / source_height

        if source_aspect > target_aspect:
            # Image is wider than target, fit width
            new_width = width
            new_height = int(width / source_aspect)
        else:
            # Image is taller than target, fit height
            new_height = height
            new_width = int(height * source_aspect)

        # Resize original image maintaining aspect ratio
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Create a new blank image with the target size and color
        new_image = Image.new("RGB", (width, height), color)

        # Paste the resized image onto the center of the new image
        paste_x = (width - new_width) // 2
        paste_y = (height - new_height) // 2
        new_image.paste(image, (paste_x, paste_y))

        return new_image

    def pad_image_to_file(self, output_image_path, width=512, height=512,
                          color=(0, 0, 0)):
        image = self.pad_image(width, height, color)
        image.save(output_image_path)
