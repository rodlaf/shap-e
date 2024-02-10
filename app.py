from beam import App, Runtime, Image, Volume
import subprocess

import os
import dill as pickle

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
        cpu=2,
        memory="8Gi",
        gpu="A10G",
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
                "pyyaml",
                "ipywidgets",
                "trimesh",
                "pygltflib",
                "clip@git+https://github.com/openai/CLIP.git",
            ],
        ),
    ),
    volumes=[Volume(name="cached_models", path=CACHE_PATH)],
)


def preload_model():
    print('pre-loading model...')

    with open(os.path.join(CACHE_PATH, 'transmitter.pickle'), 'rb') as handle:
        xm = pickle.load(handle)
    with open(os.path.join(CACHE_PATH, 'text300M.pickle'), 'rb') as handle:
        model = pickle.load(handle)
    with open(os.path.join(CACHE_PATH, 'diffusion.pickle'), 'rb') as handle:
        diffusion = pickle.load(handle)

    print('done pre-loading model')

    return xm, model, diffusion


@app.rest_api(loader=preload_model)
def generate(**inputs):
    auth_token = get_auth_token()

    xm, model, diffusion = inputs["context"]

    prompt = inputs["prompt"]
    submission_id = inputs["submission_id"]

    batch_size = 1
    guidance_scale = 15.0

    set_status(auth_token=auth_token, submission_id=submission_id, status='Generating 3D model...')
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
    
    set_status(auth_token=auth_token, submission_id=submission_id, status='Saving output...')

    # save output
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'output.obj', 'w') as f:
        t.write_obj(f)

    # save to pocketbase
    upload_output(
        auth_token=auth_token, 
        submission_id=submission_id,
        file_path='output.obj',
    )
    set_status(auth_token=auth_token, submission_id=submission_id, status='Converting obj to gltf...')
    # convert output to gltf
    result = subprocess.run(['python', 'OBJwVS_to_glTF.py', 'output.obj', 'output.gltf'], capture_output=True, text=True)
    print('stdout: ', result.stdout)
    print('stderr: ', result.stderr)
    set_status(auth_token=auth_token, submission_id=submission_id, status='Uploading gltf...')
    # save to pocketbase
    upload_gltf(
        auth_token=auth_token, 
        submission_id=submission_id,
        file_path='output.gltf',
    )
    set_status(auth_token=auth_token, submission_id=submission_id, status='Done!!!')



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


def upload_gltf(
        auth_token: str, 
        file_path: str,
        submission_id: str
    ) -> None:
    base_url = os.getenv('POCKETBASE_URL')

    url = base_url + '/api/collections/submissions/records/' + submission_id
    files = {
        'output_gltf': open(file_path, 'rb')
    }
    headers = {
        'Authorization': auth_token
    }

    response = requests.patch(url, headers=headers, files=files)


def set_status(
        auth_token: str, 
        submission_id: str,
        status: str,
) -> None:
    base_url = os.getenv('POCKETBASE_URL')
    url = base_url + '/api/collections/submissions/records/' + submission_id
    headers = {
        'Authorization': auth_token
    }
    body = {
        "status": status
    }
    response = requests.patch(url, headers=headers, json=body)
    print(response)
