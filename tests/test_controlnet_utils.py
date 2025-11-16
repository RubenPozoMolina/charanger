import os.path
from datetime import datetime

from utils.controlnet_utils import ControlnetUtils


class TestControlNetUtils:

    def test_image_to_control_canny(self):
        controlnet_utils = ControlnetUtils()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = f"output/{timestamp}_dancing_girl.png"
        controlnet_utils.image_to_control_canny(
            "assets/girl_dancing.jpg",
            output_file
        )
        os.path.isfile(output_file)

    def test_image_to_control_pose(self):
        controlnet_utils = ControlnetUtils()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = f"output/{timestamp}_dancing_girl.png"
        controlnet_utils.image_to_control_pose(
            "assets/girl_dancing.jpg",
            output_file
        )
        os.path.isfile(output_file)

    def test_image_to_control_depth(self):
        controlnet_utils = ControlnetUtils()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = f"output/{timestamp}_dancing_girl.png"
        controlnet_utils.image_to_control_depth(
            f"assets/girl_dancing.jpg",
            output_file
        )
        os.path.isfile(output_file)