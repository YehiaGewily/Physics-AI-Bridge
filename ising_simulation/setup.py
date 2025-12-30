from setuptools import setup, find_packages

setup(
    name="ising_simulation",
    version="0.1.0",
    description="Professional 2D Ising Model Simulation Package",
    author="Physics AI Bridge",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "scipy",
        "seaborn",
        "tqdm",
        "streamlit"
    ],
    python_requires='>=3.8',
)
