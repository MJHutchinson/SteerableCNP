python mnist_test_models.py -m  \
    # hydra/launcher=submitit_slurm launcher=slurm \
    # hydra.launcher.array_parallelism=1 \
    model=GPrbf \
    dataset=mnist \
    experiment_name=mnist_experiments_more_chol \
    model.kernel_length_scale=0.1,0.2,0.5,1.0,2.0,3.0,4.0,5.0 \
    model.kernel_sigma_var=0.025,0.05,0.1,0.2,0.5,1.0,2.0,5.0 \
    epochs=0 \
    seed=1,9,17 \
    batch_size=300 \
    model.chol_noise=1e-6 \
