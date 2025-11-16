import os
from huggingface_hub import hf_hub_download


class ModelUtils:

    models_folder = None

    def __init__(self, models_folder="models"):
        self.models_folder = models_folder
        os.makedirs(self.models_folder, exist_ok=True)

    def download_model(self, repository_id, filename=None, version=None):
        try:
            hf_hub_download(
                repo_id=repository_id,
                filename=filename,
                revision=version,
                cache_dir=self.models_folder,
            )
        except Exception as e:
            raise Exception(
                f"Failed to download model from Hugging Face Hub: {e}"
            )