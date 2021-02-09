python mnist_train_models.py -m  \
    hydra/launcher=submitit_slurm launcher=slurm \
    model=T2,C4,D4,C8,D8,C16 \
    dataset=mnist \
    experiment_name=mnist_experiments_blanks \
    model.embedding_kernel_learnable=True \
    model.output_kernel_learnable=True \
    dataset.train_args.rotate=True,False \
    dataset.test_args.rotate=\${dataset.train_args.rotate} \
    dataset.train_args.include_blanks=True \
    min_context_fraction=0.01 \
    model.padding_mode=circular \
    seed=1,9,17  \

python mnist_train_models.py -m  \
    hydra/launcher=submitit_slurm launcher=slurm \
    model=CNP \
    dataset=mnist \
    experiment_name=mnist_experiments_blanks \
    dataset.train_args.rotate=True,False \
    dataset.test_args.rotate=\${dataset.train_args.rotate} \
    dataset.train_args.include_blanks=True \
    min_context_fraction=0.01 \
    epochs=30 \
    seed=1,9,17  \