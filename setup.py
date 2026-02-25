from setuptools import setup, find_packages

setup(
    name="ising-cnn-phase-transitions",
    version="0.1.0",
    description="Phase transitions in the 2D Ising Model via CNN and Transfer Learning",
    author="YOUR NAME",
    author_email="your@email.com",
    url="https://github.com/YOUR_USERNAME/ising-cnn-phase-transitions",
    license="MIT",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "scipy>=1.12",
        "numba>=0.59",
        "torch>=2.2",
        "torchvision>=0.17",
        "h5py>=3.10",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "pyyaml>=6.0",
        "tqdm>=4.66",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
        "viz": [
            "grad-cam>=1.5",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
