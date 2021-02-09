# %%
import os
from steer_cnp.datasets import MultiMNIST, MNISTDataset
from PIL import Image
import numpy as np

# %%
os.getcwd()
os.chdir("/data/ziz/not-backed-up/mhutchin/EquivCNP/")
# %%
data = MultiMNIST(
    "data",
    0.1,
    0.5,
    1.0,
    train=True,
    translate=False,
    rotate=False,
    n_digits=1,
    canvas_multiplier=1,
    include_blanks=True,
    seed=0,
)

# %%
Image.fromarray(
    (data.data[np.random.randint(len(data)), 0] * 255).numpy().astype("uint8")
)
# %%
m = MNISTDataset("data/mnist", 7, 390, 784, augment=True, download=True)

# %%
