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
  ],
  install_requires=['shap>=0.37.0',
                    'matplotlib>=3.3.2',
                    'anndata>=0.7.6',
                    'numpy>=1.19.2',
                    'pandas>=0.25.2',
                    'scikit-learn>=0.23.1',
                    'xgboost>=1.3.3']
)
