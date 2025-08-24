"""
Setup script for MARL Autonomous Vehicle package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="marl-autonomous-vehicle",
    version="1.0.0",
    author="MARL Team",
    author_email="marl@example.com",
    description="A modular Multi-Agent Reinforcement Learning system for autonomous vehicle coordination",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kmock930/MARL-autonomous-vehicle",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "ml": [
            "tensorflow>=2.8.0",
            "gym>=0.21.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "marl-train=marl_autonomous_vehicle.training.mappo_trainer:main",
        ],
    },
)