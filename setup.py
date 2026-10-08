from setuptools import setup

setup(
    name="crisp-imputer",
    version="1.2.0",
    description="CRISP: Compositional Ratio-based Imputation with Simplex Projection",
    author="wenyu2026",
    url="https://github.com/wenyu2026/crisp-imputer",
    py_modules=["crisp"],
    # The library itself is pure NumPy: no pandas, no scikit-learn.
    # (The benchmark scripts in the repository do need them — see requirements.txt.)
    install_requires=[
        "numpy>=1.20.0",
    ],
    python_requires=">=3.8",
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
    keywords="imputation missing-values compositional-data simplex materials-science",
)