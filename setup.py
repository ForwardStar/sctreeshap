import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
  long_description = fh.read()

setuptools.setup(
  name="sctreeshap",
  version="0.1.6",
  author="Haoxuan Xie",
  author_email="haoxuanxie@link.cuhk.edu.cn",
  url="https://github.com/ForwardStar/sctreeshap",
  py_modules=["sctreeshap"],
  description="sctreeshap: a cluster tree data structure, and for shap analysis",
  long_description=long_description,
  long_description_content_type="text/markdown",
  license="LICENSE",
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
                    'imblearn>=0.0',
                    'xgboost>=1.3.3']
)
