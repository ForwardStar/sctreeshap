import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
  long_description = fh.read()

setuptools.setup(
  name="sctreeshap",
  version="0.3.0",
  author="Haoxuan Xie",
  author_email="haoxuanxie@link.cuhk.edu.cn",
  url="https://github.com/ForwardStar/sctreeshap",
  py_modules=["sctreeshap"],
  description="sctreeshap: a cluster tree data structure, and for shap analysis",
  long_description=long_description,
  long_description_content_type="text/markdown",
  license="LICENSE",
  classifiers=[
  "Programming Language :: Python :: 3.8",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
  install_requires=['shap>=0.39.0',
                    'matplotlib>=3.4.2',
                    'anndata>=0.7.6',
                    'numpy>=1.19.5',
                    'pandas>=1.2.4',
                    'sklearn>=0.0',
                    'scikit-learn>=0.24.2',
                    'imblearn>=0.0',
                    'imbalanced-learn>=0.8.0',
                    'xgboost>=1.4.2'],
  python_requires='>=3.8'
)
