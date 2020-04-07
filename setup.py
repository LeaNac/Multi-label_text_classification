from setuptools import setup

with open("requirements.txt", "r") as requirements_file:
    requirements = requirements_file.read().split()

setup(
    name='Multi-label_text_classification',
    version='0.0.1',
    packages=['src'],
    package_dir={"text_classif":"src"},
    url='',
    license='bsd',
    author='aurelien.massiot & lea.naccache',
    author_email='',
    description='',
    install_requires=requirements
)
