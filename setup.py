from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.3.4'
DESCRIPTION = 'Data Science Library'
LONG_DESCRIPTION = 'A package for data science'

setup(
    name="wolta",
    version=VERSION,
    author="iamalreadynoob",
    author_email="<sadikefe69@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['scikit-learn', 'pandas', 'numpy', 'hyperopt', 'catboost', 'imblearn', 'lightgbm', 'matplotlib', 'opencv-python'],
    keywords=['python', 'machine', 'learning', 'machine learning', 'data science', 'data'],
    py_modules=['data_tools', 'model_tools', 'progressive_tools', 'feature_tools', 'visual_tools'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
