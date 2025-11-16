import os

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