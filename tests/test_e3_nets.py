# %%
from e3nn.networks import ImageGatedConvNetwork

# %%
# little 1 000 params - might be impossible

hidden_reps_ids = [(1, 1, 0)]
size = 7
layers = 2

littlenet = ImageGatedConvNetwork(
    [(1, 1, 0)], hidden_reps_ids, [(3, 1, 0)], lmax=3, size=size, layers=layers
)

print(sum(p.numel() for p in littlenet.parameters()))

for name, p in littlenet.named_parameters():
    print(name, p.numel())
# %%
# small 20 000 params

hidden_reps_ids = [(1, 1, 0), (1, 1, 0)]
size = 7
layers = 3

smallnet = ImageGatedConvNetwork(
    [(1, 1, 0)], hidden_reps_ids, [(3, 1, 0)], lmax=3, size=size, layers=layers
)

print(sum(p.numel() for p in smallnet.parameters()))

for name, p in smallnet.named_parameters():
    print(name, p.numel())
# %%
# middle 100 000 params

hidden_reps_ids = [(6, 1, 0), (4, 1, 0), (2, 2, 0)]
size = 7
layers = 4

middlenet = ImageGatedConvNetwork(
    [(1, 1, 0)], hidden_reps_ids, [(3, 1, 0)], lmax=3, size=size, layers=layers
)

print(sum(p.numel() for p in middlenet.parameters()))

for name, p in middlenet.named_parameters():
    print(name, p.numel())
# %%
# big 100 000 params

hidden_reps_ids = [(15, 1, 0), (10, 1, 0), (5, 2, 0)]
size = 7
layers = 6

bignet = ImageGatedConvNetwork(
    [(1, 1, 0)], hidden_reps_ids, [(3, 1, 0)], lmax=3, size=size, layers=layers
)

print(sum(p.numel() for p in bignet.parameters()))

for name, p in bignet.named_parameters():
    print(name, p.numel())
# %%
# huge 2 000 000 params

hidden_reps_ids = [(23, 1, 0), (12, 1, 0), (5, 2, 0)]
size = 7
layers = 10

hugenet = ImageGatedConvNetwork(
    [(1, 1, 0)], hidden_reps_ids, [(3, 1, 0)], lmax=1, size=size, layers=layers
)

print(sum(p.numel() for p in hugenet.parameters()))

for name, p in hugenet.named_parameters():
    print(name, p.numel())

# %%
