from setuptools import setup

setup(
    name='demo_multilabel_classification',
    version='1.0',
    packages=['src'],
    url='',
    license='bsd',
    author='aurelien.massiot & lea.naccache',
    author_email='amassiot@octo.com, lnaccache@octo.com',
    description='Demo for multilabel classification article',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'plotly.express',
        'streamlit'
    ],
    extras_require={
        'test': ['pytest',
                 'mock']
    }
)
