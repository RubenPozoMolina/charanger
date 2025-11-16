import os
from datetime import datetime

from utils.image_utils import ImageUtils


class TestImageUtils:

    def test_resize_image(self):
        image_utils = ImageUtils("assets/girl_dancing2_old.jpg")
        image_utils.resize_image(
            "output/resized_image.jpg",
            width=720,
            height=1280
        )
        assert os.path.exists("output/resized_image.jpg")

    def test_generate_image_from_text(self):
        image_utils = ImageUtils()
        prompt = """
            A redhead girl with a blue dress dancing on the beach. Full body.
            Realistic. HD. Details
        """
        negative_prompt = """
        deformed, ugly, mutilated, disfigured, bad anatomy, bad proportions,
        extra limbs, cloned face, deformed face, malformed limbs, 
        missing arms, missing legs, extra arms, extra legs, 
        fused fingers, too many fingers, long neck, cross-eyed,
        mutated hands, poorly drawn hands, poorly drawn face,
        mutation, bad hands, extra fingers, text, watermark
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = f"output/{timestamp}_generated_image.jpg"
        image_utils.generate_image_from_text(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model="lykon/dreamshaper-8",
            output_image_path=output_file,
            width=720,
            height=1280
        )
        assert os.path.exists(output_file)