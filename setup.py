from setuptools import setup

setup(
    name='rnntools',
    version='0.0.1',
    description='RNN layers for nntools',
    author='Colin Raffel',
    author_email='craffel@gmail.com',
    url='https://github.com/craffel/rnntools',
    packages=['rnntools'],
    long_description="""RNN layers for nntools""",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python", "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    keywords='rnn',
    license='GPL',
    install_requires=[
        'nntools',
        'theano'
    ],
)
