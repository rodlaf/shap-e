from beam import App, Runtime, Image

import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

app = App(
    name="shap-e",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
        gpu="T4",
        image=Image(
            python_version="python3.9",
            python_packages=[
                "filelock",
                "Pillow",
                "torch",
                "fire",
                "humanize",
                "requests",
                "tqdm",
                "matplotlib",
                "scikit-image",
                "scipy",
                "numpy",
                "blobfile",
                "clip@git+https://github.com/openai/CLIP.git",
            ],
        ),
    ),
)


def load_device_and_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    return device, xm, model, diffusion


@app.rest_api(loader=load_device_and_model)
def generate(**inputs):
    device, xm, model, diffusion = inputs["context"]

    prompt = inputs["prompt"]
