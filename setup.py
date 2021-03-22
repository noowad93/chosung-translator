from setuptools import find_packages, setup

setup(
    name="chosung-translator",
    version="0.0.1",
    description="초성해석기",
    install_requires=[],
    url="https://github.com/noowad93/chosung-translator",
    author="Dawoon Jung",
    author_email="dawoon@scatterlab.co.kr",
    packages=find_packages(exclude=["tests"]),
)
