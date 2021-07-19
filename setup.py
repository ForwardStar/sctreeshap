import glob
import os
import setuptools

import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
  long_description = fh.read()

setuptools.setup(
  name="sctreeshap",
  version="0.1.1",
  author="Haoxuan Xie",
  author_email="haoxuanxie@link.cuhk.edu.cn",
  description="sctreeshap: a cluster tree data structure, and for shap analysis",
  long_description=long_description,
  long_description_content_type="text/markdown",
  license="LICENSE",
  url="https://github.com/ForwardStar/sctreeshap",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3.5",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ]
)
