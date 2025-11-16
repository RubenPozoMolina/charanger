import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from io import BytesIO

import pytest

from utils.model_utils import ModelUtils


class TestModelUtils:

    def test_init_sets_attributes(self, tmp_path: Path):
        models_folder = str(tmp_path / "models")
        mu = ModelUtils(models_folder=models_folder)
        assert mu.models_folder == models_folder
        assert os.path.exists(models_folder)

    @patch('utils.model_utils.hf_hub_download')
    def test_download_model_success_with_all_parameters(self, mock_hf_download, tmp_path: Path):
        """Test successful model download with repository_id, filename, and version."""
        models_folder = str(tmp_path / "models")
        mu = ModelUtils(models_folder=models_folder)

        # Configure mock to return a file path
        mock_hf_download.return_value = str(tmp_path / "models" / "model.bin")

        # Call download_model
        mu.download_model(
            repository_id="test-org/test-model",
            filename="model.bin",
            version="v1.0"
        )

        # Verify hf_hub_download was called with correct parameters
        mock_hf_download.assert_called_once_with(
            repo_id="test-org/test-model",
            filename="model.bin",
            revision="v1.0",
            cache_dir=models_folder
        )

    @patch('utils.model_utils.hf_hub_download')
    def test_download_model_success_with_none_version(self, mock_hf_download, tmp_path: Path):
        """Test successful model download with None version parameter."""
        models_folder = str(tmp_path / "models")
        mu = ModelUtils(models_folder=models_folder)

        mock_hf_download.return_value = str(tmp_path / "models" / "pytorch_model.bin")

        # Call download_model without version
        mu.download_model(
            repository_id="test-org/test-model",
            filename="pytorch_model.bin",
            version=None
        )

        # Verify hf_hub_download was called with revision=None
        mock_hf_download.assert_called_once_with(
            repo_id="test-org/test-model",
            filename="pytorch_model.bin",
            revision=None,
            cache_dir=models_folder
        )

    @patch('utils.model_utils.hf_hub_download')
    def test_download_model_success_with_none_filename(self, mock_hf_download, tmp_path: Path):
        """Test successful model download with None filename parameter."""
        models_folder = str(tmp_path / "models")
        mu = ModelUtils(models_folder=models_folder)

        mock_hf_download.return_value = str(tmp_path / "models" / "default.bin")

        # Call download_model without filename
        mu.download_model(
            repository_id="test-org/test-model",
            filename=None,
            version="main"
        )

        # Verify hf_hub_download was called with filename=None
        mock_hf_download.assert_called_once_with(
            repo_id="test-org/test-model",
            filename=None,
            revision="main",
            cache_dir=models_folder
        )

    @patch('utils.model_utils.hf_hub_download')
    def test_download_model_raises_exception_on_failure(self, mock_hf_download, tmp_path: Path):
        """Test that download_model raises an exception when hf_hub_download fails."""
        models_folder = str(tmp_path / "models")
        mu = ModelUtils(models_folder=models_folder)

        # Configure mock to raise an exception
        mock_hf_download.side_effect = Exception("Network error")

        # Verify that download_model raises an exception with proper message
        with pytest.raises(Exception) as exc_info:
            mu.download_model(
                repository_id="test-org/test-model",
                filename="model.bin",
                version="v1.0"
            )

        assert "Failed to download model from Hugging Face Hub" in str(exc_info.value)
        assert "Network error" in str(exc_info.value)

    @patch('utils.model_utils.hf_hub_download')
    def test_download_model_uses_correct_cache_dir(self, mock_hf_download, tmp_path: Path):
        """Test that download_model uses the correct cache directory."""
        custom_folder = str(tmp_path / "custom_models")
        mu = ModelUtils(models_folder=custom_folder)

        mock_hf_download.return_value = str(tmp_path / "custom_models" / "model.bin")

        mu.download_model(
            repository_id="test-org/test-model",
            filename="model.bin"
        )

        # Verify cache_dir parameter is set correctly
        call_kwargs = mock_hf_download.call_args.kwargs
        assert call_kwargs['cache_dir'] == custom_folder

    @patch('utils.model_utils.hf_hub_download')
    def test_download_model_with_minimal_parameters(self, mock_hf_download, tmp_path: Path):
        """Test download_model with only repository_id (filename and version as None)."""
        models_folder = str(tmp_path / "models")
        mu = ModelUtils(models_folder=models_folder)

        mock_hf_download.return_value = str(tmp_path / "models" / "file.bin")

        # Call with only repository_id
        mu.download_model(repository_id="test-org/minimal-model")

        # Verify default values for filename and revision
        mock_hf_download.assert_called_once_with(
            repo_id="test-org/minimal-model",
            filename=None,
            revision=None,
            cache_dir=models_folder
        )
