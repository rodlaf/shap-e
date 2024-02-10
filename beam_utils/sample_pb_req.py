import requests
import json 
import os 


def get_auth_token():
    identity = 'rodneyslafuente@gmail.com'
    password = 'EnY5H,+LYy'

    url = 'https://text-to-3d.fly.dev/api/admins/auth-with-password'
    body = {
        'identity': identity,
        'password': password
    }

    r = requests.post(url, json=body)
    a = json.loads(r.content)

    return a['token']


def upload_output(auth_token):
    url = 'https://text-to-3d.fly.dev/api/collections/submissions/records'
    file = '/Users/rodney/workspace/shap-e/example_mesh_0.obj'
    body = {
        'prompt': 'some prompt', 
        'author': 'm1jfsuhyf2ls4g0', 
    }
    files = {
        'output': open(file, 'rb')
    }
    headers = {
        'Authorization': auth_token
    }
    r = requests.post(url, headers=headers, data=body, files=files)

    print(r.content)


auth_token = get_auth_token()
upload_output(auth_token=auth_token)