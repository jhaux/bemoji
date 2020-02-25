from setuptools import setup, find_packages


# allows to get version via python setup.py --version
__version__ = "0.1.0"


setup(
    name="bemoji",
    version=__version__,
    description="Behavior Analysis Dataset with Emojis",
    url="https://github.com/jhaux/bemoji",
    author="Johannes Haux",
    author_email="johannes.haux@iwr.uni-heidelberg.de",
    license="MIT",
    packages=find_packages(),
    package_data={"": ["*.yaml"]},
    install_requires=[
        "edflow",
        "tqdm",
        "Pillow",
        "numpy",
    ],
    zip_safe=False,
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
