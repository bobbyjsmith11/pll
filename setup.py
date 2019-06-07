from setuptools import find_packages, setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pll',
      version="0.1",
      description="Design and simulation of RF phase-locked loops",
      long_description=readme(),
      classifiers=[
          'Development Status :: Alpha ::',
          'License :: OSI Approved :: MIT',
          'Programming Language :: Python 3.5',
      ],
      keywords='phase locked loop RF simulation',
      url="http://github.com/bobbyjsmith11/pll",
      author="Bobby Smith",
      author_email="bobbyjsmith11@gmail.com",
      license="MIT",
      packages=["pll"],
      install_requires=[
          'numpy',
      ],
      test_suite='nose.collector',
      tests_require=['nose',
                     'matplotlib'],
      zip_safe=False
      )
