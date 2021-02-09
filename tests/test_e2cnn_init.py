# %%
import torch
from steer_cnp.utils import get_e2_decoder

# %%
m_vec = get_e2_decoder(4, False, "regular_huge", [[1]], [[1]])
# %%
i_vec = torch.randn(100, 3, 100, 100)
o_vec = m_vec(i_vec)
# %%
print(o_vec[:, 0, ...].mean())
print(o_vec[:, 0, ...].std())
print(o_vec[:, 1:, ...].mean())
print(o_vec[:, 1:, ...].std())
# %%
i_scl = torch.randn(100, 2, 40, 40)
# %%
torch.manual_seed(2)
m_scl = get_e2_decoder(4, False, "regular_huge", [[0]], [[0]])
o_scl = m_scl(i_scl)
print(o_scl[:, 0, ...].mean())
print(o_scl[:, 0, ...].std())
print(o_scl[:, 1, ...].mean())
print(o_scl[:, 1, ...].std())

print(torch.nn.functional.softplus(o_scl[:, 1, ...]).mean())
print(torch.nn.functional.softplus(o_scl[:, 1, ...]).std())
# %%
for p in m_scl.parameters():
    p.data[:] = p.data / 1.2
# %%
o_scl = m_scl(i_scl)
print(o_scl[:, 0, ...].mean())
print(o_scl[:, 0, ...].std())
print(o_scl[:, 1, ...].mean())
print(o_scl[:, 1, ...].std())

print(torch.nn.functional.softplus(o_scl[:, 1, ...]).mean())
print(torch.nn.functional.softplus(o_scl[:, 1, ...]).std())

# %%
