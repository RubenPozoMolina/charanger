from utils.text_utils import TextUtils


class TestTextUtils:

    def test_image_to_text(self):
        text_utils = TextUtils()
        result = text_utils.get_text_from_image(
            "assets/girl_dancing1.jpg"
        )
        print(result)
        assert "beach" in result
        assert "blue" in result
