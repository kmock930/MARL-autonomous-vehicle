"""
Setup script for MARL Autonomous Vehicle package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "MARL Autonomous Vehicle - A Multi-Agent Reinforcement Learning system for autonomous vehicle coordination."

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            # Filter out comments and empty lines, and handle encoding issues
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle potential encoding issues in requirements.txt
                    try:
                        # Try to decode and clean the line
                        clean_line = ''.join(char for char in line if ord(char) < 128)
                        if clean_line and '==' in clean_line:
                            requirements.append(clean_line)
                    except:
                        continue
            return requirements
    except FileNotFoundError:
        return [
            'numpy>=1.19.0',
            'pandas>=1.3.0',
            'matplotlib>=3.3.0'
        ]

setup(
    name="marl-autonomous-vehicle",
    version="1.0.0",
    author="MARL Autonomous Vehicle Team",
    author_email="kmock930@example.com",
    description="A Multi-Agent Reinforcement Learning system for autonomous vehicle coordination",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/kmock930/MARL-autonomous-vehicle",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies that are essential
    ],
    extras_require={
        "ml": [
            "tensorflow>=2.8.0",
            "numpy>=1.19.0",
        ],
        "data": [
            "pandas>=1.3.0",
            "matplotlib>=3.3.0",
        ],
        "gym": [
            "gymnasium>=1.0.0",
            "gym>=0.26.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "coverage>=5.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "flake8>=3.8.0",
            "mypy>=0.900",
        ],
        "all": [
            "tensorflow>=2.8.0",
            "numpy>=1.19.0",
            "pandas>=1.3.0",
            "matplotlib>=3.3.0",
            "gymnasium>=1.0.0",
            "gym>=0.26.0",
        ]
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "marl-train=marl_autonomous_vehicle.training.mappo_trainer:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/kmock930/MARL-autonomous-vehicle/issues",
        "Source": "https://github.com/kmock930/MARL-autonomous-vehicle",
        "Documentation": "https://github.com/kmock930/MARL-autonomous-vehicle/docs",
    },
)