import os

import torch

from cvae.model.architecture import load_model_from_folder
from python_kmc.kmc_src.d1_1 import plot_functions_1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "../cvae/trained_models/isothermal_model")

cvae_model = load_model_from_folder(model_path)

start_lattice = torch.zeros(size=(1, 256))

for i in range(10):
    # Only pass a temperature if the model is temperature dependent
    ml_output = cvae_model.sample(
        n_samples=1,
        input_heights_int=start_lattice,
        device=device,
        temperature_kelvin=None,
    )[0]
    start_lattice = ml_output.squeeze(0)
    plot_functions_1d.plot_1d1_lattice(
        start_lattice.cpu().numpy(),
        title=f"Step {i+1}",
        min_height=64,
    )
