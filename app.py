from beam import App, Runtime, Image, Output, Volume

import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

CACHE_PATH = "./cached_models"

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
    # Storage Volume for model weights
    volumes=[Volume(name="cached_models", path=CACHE_PATH)],
)


def preload_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    # TODO: save these models to cache

    return xm, model, diffusion


@app.rest_api(loader=preload_model, outputs=[Output("output.obj")])
def generate(**inputs):
    xm, model, diffusion = inputs["context"]
    prompt = inputs["prompt"]

    batch_size = 1
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    latent = latents[0]

    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f"output.obj", "w") as f:
        t.write_obj(f)

    # TODO: Can save directly to pocketbase here