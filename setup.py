from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ir_utils',
    version='0.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'ir_utils-count_tokens = ir_utils.scripts.count_tokens:main',
        ]
    },
    install_requires=required
)
