import pathlib

import pkg_resources
from setuptools import find_packages, setup


with pathlib.Path("requirements.txt").open() as f:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

setup(
    name="ipad",
    version="0.0.2",
    license="MIT",
    license_files=("LICENSE",),
    description="iteratively pruning and distillation",
    URL="https://github.com/alipay/PainlessInferenceAcceleration",
    packages=find_packages(include=("ipad",)),
    install_requires=[],
    python_requires=">=3.8",
)