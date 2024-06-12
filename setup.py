from setuptools import find_packages, setup

setup(
    # this name must match the package name
    name="RicoNeuralNetPrototype",
    version="0.1.0",
    packages=find_packages(
        include=["RicoNeuralNetPrototype", "RicoNeuralNetPrototype.*"]
    ),
    install_requires=[
        # "posix_ipc",
    ],
)
