import logging

from diffusers.utils.testing_utils import load_image
from transformers import BlipProcessor, BlipForConditionalGeneration

class TextUtils:

    model = None
    processor = None

    def __init__(self, model="Salesforce/blip-image-captioning-large"):
        self.processor = BlipProcessor.from_pretrained(
            model,
            use_fast=True
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            model
        ).to("cuda")

    def get_text_from_image(
            self,
            input_image_path,
            input_text="a photography of"
    ):
        result = ""
        try:
            raw_image = load_image(input_image_path)
            inputs = self.processor(
                raw_image, input_text, return_tensors="pt"
            ).to("cuda")
            result = self.model.generate(**inputs)
            inputs = self.processor(raw_image, return_tensors="pt").to("cuda")
            result = self.model.generate(**inputs)
        except Exception as e:
            logging.error(f"Error generating text from image: {e}")
        return self.processor.decode(result[0], skip_special_tokens=True)