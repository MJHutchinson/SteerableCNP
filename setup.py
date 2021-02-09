from distutils.core import setup

setup(
    name="SteerCNP",
    version="0.0.1",
    description="Steerable Conditional Neural Processes",
    author="Peter Holderrieth, Michael Hutchinson",
    packages=["steer_cnp"],
    install_requires=[
        "torch>=1.6.0",
        "torchvision>=0.7.0",
        "einops",
        "e2cnn",
    ],
)
