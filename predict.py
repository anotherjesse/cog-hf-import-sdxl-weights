import io
import os
import shutil
import subprocess
import urllib
import zipfile
import tarfile
import time
import torch

from cog import BasePredictor, Input, Path

# from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline

from download import download_repo, tar_dir


class Predictor(BasePredictor):
    def predict(
        self,
        repo_id: str = Input(description="HF repo id: username/template", default=None),
        revision: str = Input(description="HF repo revision", default="main"),
        safetensors_url: str = Input(
            description="Single file safetensors url for weights", default=None
        ),
    ) -> Path:
        weights_dir = "weights"
        if os.path.exists(weights_dir):
            shutil.rmtree(weights_dir)

        if safetensors_url:
            download_repo(weights_dir, safetensors_url=safetensors_url)
        else:
            download_repo(weights_dir, repo_id=repo_id, revision=revision)

        out_file = "weights.tar"
        if os.path.exists(out_file):
            os.remove(out_file)

        tar_dir(weights_dir, out_file)

        return Path(out_file)
