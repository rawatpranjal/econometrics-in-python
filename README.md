# Econometrics in Python

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

A comprehensive collection of Python packages for econometrics, causal inference, and quantitative economics. This repository serves as a reference guide for researchers, data scientists, economists, and practitioners working with economic data.

## Contents

- [Linear Models](#linear-models)
- [Causal Inference](#causal-inference)
- [Double/Debiased Machine Learning](#doubledebiased-machine-learning)
- [Discrete Choice Models](#discrete-choice-models)
- [Panel Data Analysis](#panel-data-analysis)
- [Time Series](#time-series)
- [Difference-in-Differences & Synthetic Control](#difference-in-differences--synthetic-control)
- [High-Dimensional Methods](#high-dimensional-methods)
- [Bayesian Econometrics](#bayesian-econometrics)
- [Natural Language Processing for Economics](#natural-language-processing-for-economics)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Optimization & Simulation](#optimization--simulation)
- [Utilities & Development Tools](#utilities--development-tools)
- [Learning Resources](#learning-resources)
- [Contributing](#contributing)

---

## Linear Models

Core packages for regression analysis and traditional econometric modeling.

| Package | Description | Links |
|---------|-------------|-------|
| **Statsmodels** | Comprehensive library for estimating statistical models, conducting statistical tests, and exploring data | [Docs](https://www.statsmodels.org/) • [GitHub](https://github.com/statsmodels/statsmodels) |
| **Scikit-learn** | Machine learning library with various regression models, model selection, and evaluation metrics | [Docs](https://scikit-learn.org/) • [GitHub](https://github.com/scikit-learn/scikit-learn) |
| **Formulaic** | Implementation of Wilkinson formulas for statistical model specification | [Docs](https://formulaic.readthedocs.io/) • [GitHub](https://github.com/matthewwardrop/formulaic) |

---

## Causal Inference

Packages specifically designed for causal inference and treatment effect estimation.

| Package | Description | Links |
|---------|-------------|-------|
| **DoWhy** | End-to-end framework for causal inference with a four-step methodology for modeling assumptions, identification, estimation, and validation | [Docs](https://www.pywhy.org/dowhy/) • [GitHub](https://github.com/py-why/dowhy) |
| **CausalInference** | Implementation of statistical methods for causal inference, including propensity score matching and subclassification | [Docs](https://causalinferenceinpython.org) • [GitHub](https://github.com/laurencium/causalinference) |
| **CausalML** | Uplift modeling and causal inference with machine learning algorithms | [Docs](https://causalml.readthedocs.io/) • [GitHub](https://github.com/uber/causalml) |
| **EconML** | Methods for estimating heterogeneous treatment effects from observational data | [Docs](https://econml.azurewebsites.net/) • [GitHub](https://github.com/py-why/EconML) |

---

## Double/Debiased Machine Learning

Frameworks implementing double/debiased machine learning for causal parameter estimation.

| Package | Description | Links |
|---------|-------------|-------|
| **DoubleML** | Implementation of double/debiased machine learning for causal inference with many ML methods | [Docs](https://docs.doubleml.org/) • [GitHub](https://github.com/DoubleML/doubleml-for-py) |
| **EconML** | Implementation of double ML estimators for heterogeneous treatment effects | [Docs](https://econml.azurewebsites.net/) • [GitHub](https://github.com/py-why/EconML) |

---

## Discrete Choice Models

Packages for analyzing choice behavior and demand modeling.

| Package | Description | Links |
|---------|-------------|-------|
| **PyBLP** | Tools for Berry-Levinsohn-Pakes demand estimation for differentiated products | [Docs](https://pyblp.readthedocs.io/) • [GitHub](https://github.com/jeffgortmaker/pyblp) |
| **XLogit** | Discrete choice models, including multinomial and mixed logit with performance optimizations | [Docs](https://xlogit.readthedocs.io/) • [GitHub](https://github.com/arteagac/xlogit) |
| **PyLogit** | Flexible multinomial/conditional logit models with Pandas integration | [GitHub](https://github.com/timothyb0912/pylogit) |

---

## Panel Data Analysis

Tools for working with panel/longitudinal data.

| Package | Description | Links |
|---------|-------------|-------|
| **Linearmodels** | Advanced panel data regression and instrumental variable estimation | [Docs](https://bashtage.github.io/linearmodels/) • [GitHub](https://github.com/bashtage/linearmodels) |
| **PyFixest** | Fast fixed-effects estimation with multi-way clustering and methods for high-dimensional fixed effects | [GitHub](https://github.com/py-econometrics/pyfixest) |
| **rdrobust** | Tools for regression discontinuity designs | [GitHub](https://github.com/rdpackages/rdrobust) |

---

## Time Series

Packages specialized in time series forecasting and analysis.

| Package | Description | Links |
|---------|-------------|-------|
| **Prophet** | Procedure for forecasting time series data with additive models and trend+seasonal components | [Docs](https://facebook.github.io/prophet/) • [GitHub](https://github.com/facebook/prophet) |
| **StatsForecast** | Statistical time series forecasting models with speed optimizations | [Docs](https://nixtla.github.io/statsforecast/) • [GitHub](https://github.com/Nixtla/statsforecast) |
| **MLForecast** | Machine learning for time series forecasting | [GitHub](https://github.com/Nixtla/mlforecast) |
| **NeuralForecast** | Deep learning models for time series forecasting | [Docs](https://nixtla.github.io/neuralforecast/) • [GitHub](https://github.com/Nixtla/neuralforecast) |
| **pmdarima** | ARIMA equivalent of scikit-learn with auto-parameter selection | [Docs](https://alkaline-ml.com/pmdarima/) • [GitHub](https://github.com/alkaline-ml/pmdarima) |

---

## Difference-in-Differences & Synthetic Control

Packages for quasi-experimental methods.

| Package | Description | Links |
|---------|-------------|-------|
| **Differences** | Implementation of nonparametric difference-in-differences methods for staggered adoption designs | [Docs](https://bernardodionisi.github.io/differences/) • [GitHub](https://github.com/bernardodionisi/differences) |
| **SyntheticControlMethods** | Implementation of synthetic control methods for comparative case studies | [GitHub](https://github.com/OscarEngelbrektson/SyntheticControlMethods) |

---

## High-Dimensional Methods

Tools for high-dimensional econometrics and machine learning.

| Package | Description | Links |
|---------|-------------|-------|
| **HDM** | High-dimensional metrics for econometric applications | [GitHub](https://github.com/chuanli11/hdm) |
| **SparseTSCGM** | Sparse time series models using conditional graphical models | [GitHub](https://github.com/mlondschien/sparsetscgm) |

---

## Bayesian Econometrics

Packages for Bayesian analysis and probabilistic modeling.

| Package | Description | Links |
|---------|-------------|-------|
| **PyMC** | Probabilistic programming library for Bayesian analysis | [Docs](https://www.pymc.io/) • [GitHub](https://github.com/pymc-devs/pymc) |
| **Bambi** | High-level Bayesian model building interface | [Docs](https://bambinos.github.io/bambi/) • [GitHub](https://github.com/bambinos/bambi) |
| **LightweightMMM** | Bayesian modeling for marketing mix modeling | [GitHub](https://github.com/google/lightweight_mmm) |
| **NumPyro** | Probabilistic programming with NumPy and JAX backend for scalable inference | [Docs](https://num.pyro.ai/) • [GitHub](https://github.com/pyro-ppl/numpyro) |

---

## Natural Language Processing for Economics

Libraries for working with text data in economic contexts.

| Package | Description | Links |
|---------|-------------|-------|
| **Hugging Face Transformers** | State-of-the-art NLP for text classification, embeddings, and generation | [Docs](https://huggingface.co/transformers/) • [GitHub](https://github.com/huggingface/transformers) |
| **GenSim** | Topic modeling and document similarity analysis | [Docs](https://radimrehurek.com/gensim/) • [GitHub](https://github.com/RaRe-Technologies/gensim) |
| **spaCy** | Industrial-strength NLP with built-in entity recognition and parsing | [Docs](https://spacy.io/) • [GitHub](https://github.com/explosion/spaCy) |

---

## Synthetic Data Generation

Tools for generating synthetic economic data.

| Package | Description | Links |
|---------|-------------|-------|
| **SDV (Synthetic Data Vault)** | Comprehensive library for generating synthetic data | [Docs](https://sdv.dev/) • [GitHub](https://github.com/sdv-dev/SDV) |
| **Synthpop** | Synthetic population generation based on survey or aggregate data | [GitHub](https://github.com/alan-turing-institute/synthpop) |

---

## Optimization & Simulation

Libraries supporting automatic differentiation, optimization, and simulations.

| Package | Description | Links |
|---------|-------------|-------|
| **PyTorch** | Deep learning framework with automatic differentiation | [Docs](https://pytorch.org/) • [GitHub](https://github.com/pytorch/pytorch) |
| **JAX** | High-performance numerical computing with automatic differentiation | [Docs](https://jax.readthedocs.io/) • [GitHub](https://github.com/google/jax) |
| **QuantEcon** | Tools for computational economics, including dynamic programming and Monte Carlo simulation | [Docs](https://quantecon.org/) • [GitHub](https://github.com/QuantEcon/quantecon-py) |

---

## Utilities & Development Tools

Packages for improving code quality, efficiency, and maintainability.

| Package | Description | Links |
|---------|-------------|-------|
| **Pylint** | Code analysis tool for identifying errors and enforcing coding standards | [Docs](https://pylint.pycqa.org/) • [GitHub](https://github.com/PyCQA/pylint) |
| **TQDM** | Fast, extensible progress bar for Python | [Docs](https://tqdm.github.io/) • [GitHub](https://github.com/tqdm/tqdm) |
| **appelpy** | Simplified syntax for econometric models with Pandas integration | [Docs](https://appelpy.readthedocs.io/) • [GitHub](https://github.com/mfarragher/appelpy) |

---

## Learning Resources

Curated resources for learning econometrics with Python.

| Resource | Description | Link |
|----------|-------------|------|
| **QuantEcon Lectures** | Lecture series on quantitative economic modeling | [Website](https://quantecon.org/lectures/) |
| **Python for Econometrics** | Comprehensive textbook by Kevin Sheppard | [PDF](https://www.kevinsheppard.com/files/teaching/python/notes/python_introduction_2020.pdf) |
| **Causal Inference for The Brave and True** | Modern introduction to causal inference using Python | [Website](https://matheusfacure.github.io/python-causality-handbook/) |

---

## Contributing

Contributions are welcome! Here's how you can help improve this repository:

1. **Add new packages**: Submit a pull request with packages relevant to econometrics and quantitative economics
2. **Update documentation**: Help improve descriptions, add examples, or correct errors
3. **Report issues**: Let us know about broken links or outdated information

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Last updated: April 2025*
