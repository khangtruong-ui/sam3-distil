from setuptools import setup, find_packages

setup(
    name="sam3-distil",
    version="0.1.0",
    description="EfficientSAM3: Distilled, Lightweight SAM3 for Real-Time Promptable Concept Segmentation",
    author="EfficientSAM3 & SAM3-Distil Contributors",
    packages=find_packages(include=["sam3*", "stage1*"]),
    package_data={"sam3": ["assets/*", "assets/**/*", "**/*.yaml", "**/*.json"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=1.0.17",
        "numpy>=1.26.0",
        "tqdm",
        "ftfy>=6.1.1",
        "regex",
        "iopath>=0.1.10",
        "typing_extensions",
        "huggingface_hub",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "omegaconf>=2.3.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "pytest-cov>=4.0.0"],
        "lora": ["peft>=0.7.0", "bitsandbytes>=0.41.0", "accelerate>=0.25.0", "transformers>=4.36.0"],
    },
)
