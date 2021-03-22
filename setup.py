from setuptools import find_packages, setup

setup(
    name="python-37-template-repository",
    version="0.0.1",
    description="python template repository",
    install_requires=[],
    url="https://github.com/scatterlab/python-37-template-repository.git",
    author="ScatterLab",
    author_email="developers@scatterlab.co.kr",
    packages=find_packages(exclude=["tests"]),
)
