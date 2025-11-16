import cv2
import numpy as np
from PIL import Image
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
from transformers import pipeline


class ControlnetUtils:

    @staticmethod
    def image_to_control_canny(input_image_path, output_image_path):
        input_image = load_image(
            input_image_path
        )

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(input_image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)

        control_image.save(output_image_path)

    @staticmethod
    def image_to_control_pose(
            input_image_path,
            output_image_path,
            hand_and_face=True
    ):
        image = load_image(
            input_image_path
        )

        processor = OpenposeDetector.from_pretrained(
            'lllyasviel/ControlNet'
        )

        control_image = processor(
            image,
            hand_and_face=hand_and_face
        )
        control_image.save(output_image_path)

    @staticmethod
    def image_to_control_depth(input_image_path, output_image_path):
        image = load_image(input_image_path)
        depth_estimator = pipeline('depth-estimation')
        image = depth_estimator(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)

        control_image.save(output_image_path)