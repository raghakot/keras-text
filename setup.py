from setuptools import setup
from setuptools import find_packages


version = '0.1'

setup(name='keras-text',
      version=version,
      description='Text classification library for Keras',
      author='Raghavendra Kotikalapudi',
      author_email='ragha@outlook.com',
      url='https://github.com/raghakot/keras-text',
      download_url='https://github.com/raghakot/keras-text/tarball/{}'.format(version),
      license='MIT',
      install_requires=['keras>=2.1.2', 'six', 'spacy>=2.0.3', 'scikit-learn'],
      extras_require={
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      include_package_data=True,
      packages=find_packages())
