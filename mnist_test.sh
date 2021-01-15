python train_image_data.py \
    hydra/launcher=submitit_slurm launcher=slurm \
    model=C4 \
    dataset=multimnist \
    experiment_name=multimnist_test \
    # model.embedding_kernel_length_scale=0.5 \
    # model.embedding_kernel_learnable=False \
    # model.output_kernel_length_scale=\${model.embedding_kernel_length_scale} \
    # model.output_kernel_learnable=False \
    # model.covariance_activation_parameters.min_var=0.001,0.01,0.1