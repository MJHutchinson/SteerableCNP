python train_image_data.py -m  \
    hydra/launcher=submitit_slurm launcher=slurm \
    model=T2,C4,D4,C16 \
    dataset=multimnist \
    experiment_name=multimnist_finetune \
    # finetune_epochs=0 \
    # debug=True \



    # model.covariance_activation_parameters.min_var=0.001,0.01,0.1 \
    # model.embedding_kernel_length_scale=0.2 \
    # model.output_kernel_length_scale=\${model.embedding_kernel_length_scale} \



    # model.embedding_kernel_learnable=False \
    # model.output_kernel_learnable=\${model.embedding_kernel_learnable} \

# python train_mnist_data.py -m  \
#     hydra/launcher=submitit_slurm launcher=slurm \
#     model=C4 \
#     dataset=mnist \
#     experiment_name=test_save \
#     epochs=1 \