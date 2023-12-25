import pathlib

import pkg_resources
from setuptools import find_packages, setup


with pathlib.Path("requirements.txt").open() as f:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]


setup(
    name="pia",
    version="0.0.2",
    license="CC-BY-4.0",
    license_files=("LICENSE",),
    description="Inference framework of ANT GROUP",
    URL="https://github.com/alipay/PainlessInferenceAcceleration",
    packages=find_packages(),
    # install_requires=install_requires,
    python_requires=">=3.8",
)