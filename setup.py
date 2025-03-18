from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies needed for the library to function
install_requires = [
    "numpy>=2.0.0",
    "torch>=2.5.1",
    "torchvision>=0.20.0",
    "torchaudio>=2.5.0",
    "pandas>=2.2.0",
    "pydantic>=2.10.0",
    "ray>=2.40.0",
    "optuna>=4.0.0",
    "gym>=0.26.0",
    "networkx>=3.2.0",
    "scipy>=1.14.0",
    "dvc>=3.0.0",
    "aiohttp>=3.8.0",
    "pymongo>=4.0.0",
    "redis>=5.0.0",
    "SQLAlchemy>=2.0.0",
    "python-dotenv>=1.0.0",
    "tqdm>=4.60.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=8.0.0",
        "pylint>=3.0.0",
        "ruff>=0.0.0",
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
}

setup(
    name="metaq-star",
    version="0.2.5.1",
    author="saviornt",
    description="A Novel Machine Learning Framework Combining Meta-Learning and Q-Learning with Pathfinding Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saviornt/MetaQ-Star",  # Replace with actual URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
) 