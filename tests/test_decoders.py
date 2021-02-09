# %%
import torch

from einops import rearrange, 

from steer_cnp.utils import get_e2_decoder, grid_2d, plot_vector_field

# %%

model = get_e2_decoder(
    -1,
    False,
    "irrep_little",
    [1],
    [1],
)

# %%

out = model(torch.randn((10, 3, 50, 50)))

# %%

model = get_e2_decoder(4, False, "regular_small", [1], [1], activation="normrelu")

# %%

n = 20
X = grid_2d(-4, 4, n)
inpt = torch.randn((1, 3, n, n))

out = model(inpt)

Y_in = rearrange(inpt[:, 1:3, :, :], "n c h w -> n (h w) c").squeeze()
Y_out = rearrange(out[:, 0:2, :, :], "n c h w -> n (h w) c").detach().squeeze()


# %%

plot_vector_field(X, Y_in, scale=100)
plot_vector_field(X, Y_out, scale=100)

# %%

field = model[1].in_type

for i, g in enumerate(field.testing_elements):
    Y_in_r = field.transform(inpt, g)
    Y_in_r = rearrange(Y_in_r, "b d w h -> b (w h) d")
    plot_vector_field(X, Y_in_r[0,..., 1:], scale=100)
# %%
