import pytest

from utils.ip_adapter_utils import IPAdapterUtils

@pytest.fixture(scope="module")
def ip_adapter_utils():
    ip_adapter_utils = IPAdapterUtils("lykon/dreamshaper-8")
    return ip_adapter_utils

class TestIPAdapterUtils:

    def test_generate_image_variations(self, ip_adapter_utils: IPAdapterUtils):
        ip_adapter_utils.generate_image_variations(
            "assets/girl_dancing1.jpg",
            "output/girl_dancing1_variations"
        )

    def test_generate_image_from_depth(self, ip_adapter_utils: IPAdapterUtils):
        ip_adapter_utils.generate_image_from_depth(
            "assets/girl_dancing1.jpg",
            "assets/depth_map.png",
            "output/depth"
        )

    def test_generate_image_from_pose(self, ip_adapter_utils: IPAdapterUtils):
        ip_adapter_utils.generate_image_from_pose(
            "assets/girl_dancing1.jpg",
            "assets/pose_map.png",
            "output/pose"
        )