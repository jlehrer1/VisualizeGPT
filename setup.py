from setuptools import setup, find_packages

setup(
    name='visualizegpt',
    version='0.1.0',
    description='A package that creates matplotlib and seaborn visualizations from pandas dataframes using OpenAI\'s LLMs',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/visualizegpt',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
        'seaborn',
        'openai',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.6, <4',
)