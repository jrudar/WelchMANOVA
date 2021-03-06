import setuptools

setuptools.setup(name="WelchMANOVA",
                 version="1.0.0.dev",
                 author="Josip Rudar",
                 author_email="rudarj@uoguelph.ca",
                 description="Python Implementation of the Distance-Based Welch MANOVA",
                 url="https://github.com/jrudar/Distance-Based-Welch-MANOVA",
                 license = "MIT",
                 keywords = "ecology multivariate statistics",
                 packages=["WelchMANOVA"],
                 python_requires = ">=3.6",
                 install_requires = ["scikit-learn >= 0.23",
                                     "numpy >= 1.18",
                                     "scikit-bio >= 0.5.6",
                                     "deicode >= 0.2.4",
                                     "pandas >= 1.0.3",
                                     "statsmodels >= 0.10.0"],
                 classifiers=["Programming Language :: Python :: 3.6+",
                              "License :: MIT License",
                              "Operating System :: OS Independent",
                              "Topic :: Ecology :: Multivariate Statistics"])