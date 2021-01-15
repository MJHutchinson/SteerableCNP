python train_gp_data.py \
    hydra/launcher=submitit_slurm launcher=slurm \
    model=C4 \
    dataset=rbf \
    experiment_name=rbf_test \
    model.embedding_kernel_length_scale=7.0 \
    model.embedding_kernel_learnable=False \
    model.output_kernel_length_scale=5.0 \
    model.context_in_target=False \
    model.lr=3e-4 \