# %%
import os
from equiv_cnp.datasets import MultiMNIST, MNISTDataset
from PIL import Image
import numpy as np

# %%
os.getcwd()
os.chdir("/data/ziz/not-backed-up/mhutchin/EquivCNP/")
# %%
data = MultiMNIST("data", 25, (3 * 28) ** 2, (3 * 28) ** 2, n_digits=2)

# %%
Image.fromarray(
    (data.data[np.random.randint(len(data)), 0] * 255).numpy().astype("uint8")
)
# %%
m = MNISTDataset("data/mnist", 7, 390, 784, augment=True, download=True)

# %%
