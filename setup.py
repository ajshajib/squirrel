#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

# with open("HISTORY.rst") as history_file:
#     history = history_file.read()

requirements = []

test_requirements = []

setup(
    author="Anowar Shajib",
    author_email="ajshajib@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    description="Software to extract kinematics from IFU spectroscopy by deblending the components in a lensing system",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="squirrel",
    name="squirrel",
    packages=find_packages(include=["squirrel", "squirrel.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/ajshajib/squirrel",
    version="0.1.0",
    zip_safe=False,
)
