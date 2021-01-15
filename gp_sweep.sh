# python train_gp_data.py -m \
#     hydra/launcher=submitit_slurm launcher=slurm \
#     model=T2,C4,C16,D4,D8 \
#     dataset=rbf,divfree,curlfree \
#     experiment_name=sweep_longer_lengthscales \

python train_gp_data.py -m \
    hydra/launcher=submitit_slurm launcher=slurm \
    model=C4 \
    dataset=divfree \
    experiment_name=match_peter_2 \
    model.embedding_kernel_length_scale=7.0 \
    model.embedding_kernel_learnable=False \
    model.output_kernel_length_scale=5.0 \
    model.context_in_target=True,False \
    model.lr=1e-4,3e-4,7e-4,1e-3 \