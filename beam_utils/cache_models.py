import torch
import os
import pickle

from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config


CACHE_PATH = "./cached_models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
xm = load_model("transmitter", device=device)
model = load_model("text300M", device=device)
diffusion = diffusion_from_config(load_config("diffusion"))

with open(os.path.join(CACHE_PATH, 'transmitter.pickle'), 'wb') as handle:
    pickle.dump(xm, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(CACHE_PATH, 'text300M.pickle'), 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(CACHE_PATH, 'diffusion.pickle'), 'wb') as handle:
    pickle.dump(diffusion, handle, protocol=pickle.HIGHEST_PROTOCOL)


