import glob
import os
import setuptools

import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
  long_description = fh.read()

setuptools.setup(
  name="sctreeshap",
  version="0.1.0",
  author="Haoxuan Xie",
  author_email="haoxuanxie@link.cuhk.edu.cn",
  description="sctreeshap: a cluster tree data structure, and for shap analysis",
  long_description=long_description,
  long_description_content_type="text/markdown",
  license="LICENSE",
  url="https://github.com/pypa/sampleproject",
  packages=setuptools.find_packages(where='sctreeshap'),
  package_dir={'': 'sctreeshap'},
  py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob('sctreeshap/*.py')],
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
