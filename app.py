from beam import App, Runtime, Image, Output, Volume
import torch

import os
import dill as pickle

from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.sample import sample_latents
from shap_e.util.notebooks import decode_latent_mesh

import requests
import json

from dotenv import load_dotenv

load_dotenv()

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
                "dill",
                "python-dotenv",
                "clip@git+https://github.com/openai/CLIP.git",
            ],
        ),
    ),
    volumes=[Volume(name="cached_models", path=CACHE_PATH)]
)


def preload_model():
    print('pre-loading model...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    # with open(os.path.join(CACHE_PATH, 'transmitter.pickle'), 'rb') as handle:
    #     xm = pickle.load(handle)
    # with open(os.path.join(CACHE_PATH, 'text300M.pickle'), 'rb') as handle:
    #     model = pickle.load(handle)
    # with open(os.path.join(CACHE_PATH, 'diffusion.pickle'), 'rb') as handle:
    #     diffusion = pickle.load(handle)

    print('done pre-loading model')

    return xm, model, diffusion


@app.rest_api(loader=preload_model)
def generate(**inputs):
    xm, model, diffusion = inputs["context"]

    prompt = inputs["prompt"]
    submission_id = inputs["submission_id"]

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

    # save output
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'output.obj', 'w') as f:
        t.write_obj(f)

    # save to pocketbase
    auth_token = get_auth_token()
    upload_output(
        auth_token=auth_token, 
        submission_id=submission_id,
        file_path='output.obj',
    )


def get_auth_token() -> str:
    identity = os.getenv('POCKETBASE_IDENTITY')
    password = os.getenv('POCKETBASE_PASSWORD')
    base_url = os.getenv('POCKETBASE_URL')

    url = base_url + '/api/admins/auth-with-password'
    body = {
        'identity': identity,
        'password': password
    }

    response = requests.post(url, json=body)
    response_object = json.loads(response.content)

    return response_object['token']


def upload_output(
        auth_token: str, 
        file_path: str,
        submission_id: str
    ) -> None:
    base_url = os.getenv('POCKETBASE_URL')

    url = base_url + '/api/collections/submissions/records/' + submission_id
    files = {
        'output': open(file_path, 'rb')
    }
    headers = {
        'Authorization': auth_token
    }

    response = requests.patch(url, headers=headers, files=files)

    # print(response.content)
