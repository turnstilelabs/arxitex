from setuptools import find_packages, setup


def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name="arxitex",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
)
