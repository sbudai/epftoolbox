from setuptools import setup, find_packages
from os import path

here = path.abspath(path=path.dirname(p=__file__))

# Get the long description from the README file
with open(file=path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='epftoolbox',
    version='1.0',
    description='An open-access benchmark and toolbox for electricity price forecasting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jeslago/epftoolbox',
    author='Jesus Lago',
    author_email='jesuslagogarcia@gmail.com',
    license='GNU AGPLv3',
    python_requires='>=3.9, <=3.11',
    install_requires=['hyperopt>=0.2', 'tensorflow>=2.2', 'scikit-learn>=0.22',
                      'pandas>=1', 'numpy>=1', 'statsmodels>=0.11',
                      'matplotlib>=3', 'scipy>=1.4', 'setuptools>=69.0', 'requests>=2.31', 'cmap'],
    packages=find_packages(include=['epftoolbox', 'epftoolbox.*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python :: 3'
    ]
    )
