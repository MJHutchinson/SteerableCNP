python train_image_data.py -m  \
    hydra/launcher=submitit_slurm launcher=slurm \
    model=T2,C4 \
    dataset=mnist \
    experiment_name=parameter_sweep \
    model.embedding_kernel_learnable=False,True \
    model.output_kernel_learnable=False,True \
    seed=1,2,3


# python train_image_data.py -m  \
#     hydra/launcher=submitit_slurm launcher=slurm \
#     model=T2,C4,D4,C16 \
#     dataset=mnist \
#     experiment_name=parameter_sweep \
#     model.min_cov=1e-2,1e-4,1e-6 \
#     model.covariance_activation=diagonal_softplus,diagonal_softplus_quadratic \

#     seed=1,2,3 

    # model.embedding_kernel_length_scale=0.1,0.5,1.0 \
    # model.output_kernel_length_scale=\${model.embedding_kernel_length_scale} \
    
    # finetune_epochs=0 \
    # debug=True \



    # model.covariance_activation_parameters.min_var=0.001,0.01,0.1 \
# python train_mnist_data.py -m  \
#     hydra/launcher=submitit_slurm launcher=slurm \
#     model=C4 \
#     dataset=mnist \
#     experiment_name=test_save \
#     epochs=1 \