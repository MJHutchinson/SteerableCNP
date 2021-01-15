# %%
import numpy as np

import torch
import pytorch_lightning as pl
from torchvision.datasets import MNIST

import PIL
from PIL import Image

# %%

dataset = MNIST("data/mnist")

# %%


def make_multi_mnist(
    dataset, seed, rotate, translate, n_digits, base_size, canvas_size
):
    pl.seed_everything(seed)

    n = dataset.data.shape[0]

    images = np.zeros((n, canvas_size, canvas_size)).astype("uint16")

    max_shift = canvas_size - base_size
    max_rot = 360

    indices = np.stack([np.random.permutation(n) for i in range(n_digits)], axis=1)
    # indices = np.stack([np.arange(n) for i in range(n_digits)], axis=1)
    shifts = (
        np.random.randint(0, max_shift, size=(n, n_digits, 2))
        if translate
        else int((canvas_size - base_size) / 2) * np.ones((n, n_digits, 2), dtype=int)
    )
    rots = np.random.uniform(size=(n, n_digits)) * max_rot

    for i in range(n):
        for j in range(n_digits):
            img = Image.fromarray(dataset[indices[i, j]])
            if rotate:
                img = img.rotate(rots[i, j], resample=Image.BICUBIC)

            images[
                i,
                shifts[i, j, 0] : shifts[i, j, 0] + base_size,
                shifts[i, j, 1] : shifts[i, j, 1] + base_size,
            ] = (
                np.asarray(img)
                + images[
                    i,
                    shifts[i, j, 0] : shifts[i, j, 0] + base_size,
                    shifts[i, j, 1] : shifts[i, j, 1] + base_size,
                ]
            )

    images[images >= 255] = 255

    return images.astype("uint8")


# %%

multiimg = make_multi_mnist(
    dataset.data.numpy()[:10].astype("uint8"), 0, True, False, 2, 28, 28 * 3
)
# %%
