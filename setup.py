import os
import setuptools

long_description = """Code to benchmark QAOA in the independent set problem."""

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

VERSION_PATH = os.path.join(os.path.dirname(__file__), "independent_set_benchmarking", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

setuptools.setup(
    name="independent_set_benchmarking",
    version=VERSION,
    description="independent set benchmarking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eggerdj/independent_set_benchmarking",
    author="Daniel Egger",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qaoa",
    packages=setuptools.find_packages(
        include=["independent_set_benchmarking", "independent_set_benchmarking.*"]
    ),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.8",
    zip_safe=False,
)
