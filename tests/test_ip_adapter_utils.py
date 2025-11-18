from utils.ip_adapter_utils import IPAdapterUtils


class TestIPAdapterUtils:

    def test_generate_image_variations(self):
        ip_adapter_utils = IPAdapterUtils()
        ip_adapter_utils.generate_image_variations(
            "assets/girl_dancing1.jpg",
            "output/girl_dancing1_variations.png"
        )