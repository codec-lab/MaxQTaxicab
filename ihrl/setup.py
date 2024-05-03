from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='ihrl',
    version='0.1',
    description='Inverse Hierarchical Reinforcement Learning',
    keywords = [],
    packages=find_packages(),
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False,
)

