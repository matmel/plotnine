"""
plotnine is an implementation of a *grammar of graphics* in Python,
it is based on ggplot2. The grammar allows users to compose plots
by explicitly mapping data to the visual objects that make up the
plot.

Plotting with a grammar is powerful, it makes custom (and otherwise
complex) plots are easy to think about and then create, while the
simple plots remain simple.

To find out about all building blocks that you can use to create a
plot, check out the documentation_. Since plotnine has an API
similar to ggplot2, where we lack in coverage the
`ggplot2 documentation`_ may be of some help.
"""
import os
from setuptools import find_packages, setup
import versioneer

__author__ = 'Hassan Kibirige'
__email__ = 'has2k1@gmail.com'
__description__ = "A grammar of graphics for python"
__license__ = 'GPL-2'
__url__ = 'https://github.com/has2k1/plotnine'
__classifiers__ = [
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3'
]


def check_dependencies():
    """
    Check for system level dependencies
    """
    pass


def get_required_packages():
    """
    Return required packages

    Plus any version tests and warnings
    """
    install_requires = ['six',
                        'mizani >= 0.4.1',
                        'matplotlib >= 2.1.0',
                        'numpy',
                        'scipy >= 1.0.0',
                        'patsy >= 0.4.1',
                        'statsmodels >= 0.8.0',
                        'pandas >= 0.21.0',
                        'geopandas >= 0.3.0',
                        'descartes >= 1.1.0'
                        ]
    return install_requires


def get_extra_packages():
    """
    Return extra packages

    Plus any version tests and warnings
    """
    extras_require = {
        'all':  ['scikit-learn', 'scikit-misc']
    }
    return extras_require


def get_package_data():
    """
    Return package data

    For example:

        {'': ['*.txt', '*.rst'],
         'hello': ['*.msg']}

    means:
        - If any package contains *.txt or *.rst files,
          include them
        - And include any *.msg files found in
          the 'hello' package, too:
    """
    baseline_images = [
        'tests/baseline_images/%s/*' % x
        for x in os.listdir('plotnine/tests/baseline_images')]
    csv_data = ['data/*.csv']
    package_data = {'plotnine': baseline_images + csv_data}
    return package_data


if __name__ == '__main__':
    check_dependencies()

    setup(name='plotnine',
          maintainer=__author__,
          maintainer_email=__email__,
          description=__description__,
          long_description=__doc__,
          license=__license__,
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          url=__url__,
          install_requires=get_required_packages(),
          extras_require=get_extra_packages(),
          packages=find_packages(),
          package_data=get_package_data(),
          classifiers=__classifiers__,
          zip_safe=False)
