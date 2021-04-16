from setuptools import setup

setup(
    name='pre-commit-hooks',
    description='A pre-commit hook to check that a token string does not appear in text files',
    url='https://github.com/abaisero/pre-commit-hooks',
    version='0.1.0',
    packages=['pre_commit_hooks'],
    scripts=['scripts/no-commit.py'],
)
