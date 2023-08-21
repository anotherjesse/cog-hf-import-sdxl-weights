import tarfile
import time
import pathlib
import torch

from diffusers import StableDiffusionXLPipeline


def download_repo(
    dest,
    repo_id=None,
    safetensors_url=None,
    revision="main",
    cache_dir="diffusers-cache",
):
    """Download the model weights from the given URL"""
    print("Downloading weights...")
    start = time.time()
    if safetensors_url:
        pipe = StableDiffusionXLPipeline.from_single_file(
            safetensors_url,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
    pipe.save_pretrained(dest)
    print("Downloaded in {:0.2f}s".format(time.time() - start))


def tar_dir(weights_dir, out_file):
    start = time.time()
    directory = pathlib.Path(weights_dir)
    with tarfile.open(out_file, "w") as tar:
        for file_path in directory.rglob("*"):
            print(file_path)
            arcname = file_path.relative_to(directory)
            tar.add(file_path, arcname=arcname)

    print("compressed in {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    # download_repo("foobar", repo_id="stabilityai/stable-diffusion-xl-base-1.0")
    download_repo("foobar", safetensors_url="https://huggingface.co/bluepen5805/blue_pencil-XL/blob/main/blue_pencil-XL-v0.1.0.safetensors")
    tar_dir("foobar", "foobar.tar")
