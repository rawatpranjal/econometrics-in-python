# Econometrics in Python

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) <!-- Note: Link points to LICENSE file in the repository root -->
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md) <!-- Note: Link points to CONTRIBUTING.md file in the repository root -->

A comprehensive collection of Python packages for econometrics, causal inference, quantitative economics, and data analysis. This repository serves as a reference guide for researchers, data scientists, economists, and practitioners working with economic data. Most packages can be installed via `pip`.

## Contents

- [Core Libraries & Linear Models](#core-libraries--linear-models)
- [Panel Data & Fixed Effects](#panel-data--fixed-effects)
- [Instrumental Variables (IV) & GMM](#instrumental-variables-iv--gmm)
- [Causal Inference & Matching](#causal-inference--matching)
- [Double/Debiased Machine Learning (DML)](#doubledebiased-machine-learning-dml)
- [Program Evaluation Methods (DiD, SC, RDD)](#program-evaluation-methods-did-sc-rdd)
- [Time Series Analysis](#time-series-analysis)
- [High-Dimensional Methods](#high-dimensional-methods)
- [Discrete Choice Models](#discrete-choice-models)
- [Quantile Regression & Distributional Methods](#quantile-regression--distributional-methods)
- [Bayesian Econometrics](#bayesian-econometrics)
- [Marketing Mix Models (MMM)](#marketing-mix-models-mmm)
- [Spatial Econometrics](#spatial-econometrics)
- [Natural Language Processing for Economics](#natural-language-processing-for-economics)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Numerical Optimization & Computational Tools](#numerical-optimization--computational-tools)
- [Utilities & Econometric Infrastructure](#utilities--econometric-infrastructure)
- [Learning Resources](#learning-resources)
- [Contributing](#contributing)

---

## Core Libraries & Linear Models

Fundamental packages for statistical modeling, regression, and core econometric tasks.

| Package         | Description                                                                                                                  | Links                                                            | Installation             |
|-----------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|--------------------------|
| **Statsmodels** | Comprehensive library for estimating statistical models (OLS, GLM, etc.), conducting tests, and data exploration. Core tool. | [Docs](https://www.statsmodels.org/) • [GitHub](https://github.com/statsmodels/statsmodels) | `pip install statsmodels`  |
| **Scikit-learn**| Foundational ML library with regression models (incl. regularized), model selection, cross-validation, evaluation metrics. | [Docs](https://scikit-learn.org/) • [GitHub](https://github.com/scikit-learn/scikit-learn) | `pip install scikit-learn` |

---

## Panel Data & Fixed Effects

Tools for modeling data with both cross-sectional and time dimensions, including high-dimensional fixed effects.

| Package                  | Description                                                                                                                                          | Links                                                                        | Installation                         |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|--------------------------------------|
| **Linearmodels**         | Estimation of fixed, random, pooled OLS models for panel data. Also Fama-MacBeth and between/first-difference estimators.                             | [Docs](https://bashtage.github.io/linearmodels/) • [GitHub](https://github.com/bashtage/linearmodels) | `pip install linearmodels`           |
| **PyFixest**             | Fast estimation of linear models with multiple high-dimensional fixed effects (like R's `fixest`). Supports OLS, IV, Poisson, robust/cluster SEs.      | [Docs & GitHub](https://github.com/py-econometrics/pyfixest)                  | `pip install pyfixest`               |
| **pydynpd**              | Estimation of dynamic panel data models using Arellano-Bond (Difference GMM) and Blundell-Bond (System GMM). Includes Windmeijer correction & tests.  | [Docs (JOSS)](https://doi.org/10.21105/joss.04416) • [GitHub](https://github.com/dazhwu/pydynpd)        | `pip install pydynpd`                |
| **FixedEffectModelPyHDFE** | Solves linear models with high-dimensional fixed effects, supporting robust variance calculation and IV.                                             | [PyPI](https://pypi.org/project/FixedEffectModelPyHDFE/)                        | `pip install FixedEffectModelPyHDFE` |
| **duckreg**              | Out-of-core regression (OLS/IV) for very large datasets using DuckDB aggregation. Handles data that doesn't fit in memory.                           | [Docs & GitHub](https://github.com/py-econometrics/duckreg)                   | `pip install duckreg`                |

---

## Instrumental Variables (IV) & GMM

Packages for handling endogeneity using instrumental variables and generalized method of moments.

| Package                   | Description                                                                                                  | Links                                                                                           | Installation             |
|---------------------------|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------|
| **Linearmodels**          | Implements single-equation IV estimators (2SLS, LIML, GMM-IV) and system GMM (SUR, 3SLS).                     | [Docs](https://bashtage.github.io/linearmodels/) • [GitHub](https://github.com/bashtage/linearmodels) | `pip install linearmodels` |
| **py-econometrics `gmm`** | Lightweight package for setting up and estimating custom GMM models based on user-defined moment conditions. | [Docs & GitHub](https://github.com/py-econometrics/gmm)                                           | `pip install gmm`        |
| **Statsmodels**           | Includes basic IV/2SLS and GMM implementations within its broader framework.                                 | [Docs](https://www.statsmodels.org/) • [GitHub](https://github.com/statsmodels/statsmodels)          | `pip install statsmodels`  |

---

## Causal Inference & Matching

Tools for estimating causal effects using observational data, focusing on matching and structural approaches.

| Package            | Description                                                                                                                          | Links                                                                                 | Installation              |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|---------------------------|
| **DoWhy**          | End-to-end framework for causal inference based on causal graphs (DAGs) and potential outcomes. Covers identification, estimation, refutation. | [Docs](https://www.pywhy.org/dowhy/) • [GitHub](https://github.com/py-why/dowhy)             | `pip install dowhy`       |
| **CausalInference**| Implements classical causal inference methods like propensity score matching, inverse probability weighting, stratification.         | [Docs](https://causalinferenceinpython.org) • [GitHub](https://github.com/laurencium/causalinference) | `pip install CausalInference` |
| **CausalML**       | Focuses on uplift modeling and heterogeneous treatment effect estimation using machine learning techniques.                        | [Docs](https://causalml.readthedocs.io/) • [GitHub](https://github.com/uber/causalml)     | `pip install causalml`    |
| **CausalMatch**    | Implements Propensity Score Matching (PSM) and Coarsened Exact Matching (CEM) with ML flexibility for propensity score estimation. | [Docs & GitHub](https://github.com/bytedance/CausalMatch)                             | `pip install causalmatch` |
| **fastmatch**      | Fast k-nearest-neighbor matching for large datasets using Facebook's FAISS library.                                                  | [Docs & GitHub](https://github.com/py-econometrics/fastmatch)                         | `pip install fastmatch`   |

---

## Double/Debiased Machine Learning (DML)

Methods combining machine learning and econometrics for robust causal inference in high-dimensional settings.

| Package      | Description                                                                                                                          | Links                                                                                   | Installation        |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|---------------------|
| **DoubleML** | Implements the double/debiased ML framework (Chernozhukov et al.) for estimating causal parameters (ATE, LATE, POM) with ML nuisances. | [Docs](https://docs.doubleml.org/) • [GitHub](https://github.com/DoubleML/doubleml-for-py) | `pip install DoubleML`|
| **EconML**   | Microsoft toolkit for estimating heterogeneous treatment effects using DML, causal forests, meta-learners, and orthogonal ML methods.  | [Docs](https://econml.azurewebsites.net/) • [GitHub](https://github.com/py-why/EconML)     | `pip install econml`  |

---

## Program Evaluation Methods (DiD, SC, RDD)

Quasi-experimental methods for estimating causal effects from policy changes or natural experiments.

| Package                     | Description                                                                                                                | Links                                                                                   | Installation                      |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-----------------------------------|
| **csdid**                   | Python adaptation of the R `did` package. Implements multi-period DiD with staggered treatment timing (Callaway & Sant’Anna). | [GitHub](https://github.com/d2cml-ai/csdid)                                               | `pip install csdid`               |
| **Differences**             | Implements modern difference-in-differences methods for staggered adoption designs (e.g., Callaway & Sant'Anna).           | [Docs](https://bernardodionisi.github.io/differences/) • [GitHub](https://github.com/bernardodionisi/differences) | `pip install differences`         |
| **SyntheticControlMethods** | Implementation of synthetic control methods for comparative case studies when panel data is available.                     | [GitHub](https://github.com/OscarEngelbrektson/SyntheticControlMethods)                   | `pip install SyntheticControlMethods` |
| **rdrobust**                | Comprehensive tools for Regression Discontinuity Designs (RDD), including optimal bandwidth selection, estimation, inference. | [GitHub](https://github.com/rdpackages/rdrobust) • [PyPI](https://pypi.org/project/rdrobust/) | `pip install rdrobust`            |
| **rdd**                     | Toolkit for sharp RDD analysis, including bandwidth calculation and estimation, integrating with pandas.                 | [GitHub](https://github.com/evan-magnusson/rdd)                                         | `pip install rdd`                 |
| **mlsynth**              | Implements advanced synthetic control methods: forward DiD, cluster SC, factor models, and proximal SC. Designed for single-treated-unit settings. | [Docs](https://mlsynth.readthedocs.io/en/latest/) • [GitHub](https://github.com/jaredjgreathouse/mlsynth) | `pip install mlsynth` |

---

## Time Series Analysis

Packages specialized in analyzing time-ordered data, including forecasting and volatility modeling.

| Package          | Description                                                                                                             | Links                                                                             | Installation           |
|------------------|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|------------------------|
| **Statsmodels**  | Core library for classical time series models (ARIMA, VAR, VARMA, State Space, Exponential Smoothing).                  | [Docs](https://www.statsmodels.org/stable/tsa.html) • [GitHub](https://github.com/statsmodels/statsmodels) | `pip install statsmodels`|
| **ARCH**         | Specialized library for volatility modeling (ARCH, GARCH, etc.), unit root tests, cointegration, financial econometrics. | [Docs](https://arch.readthedocs.io/) • [GitHub](https://github.com/bashtage/arch)      | `pip install arch`     |
| **pmdarima**     | ARIMA modeling with automatic parameter selection (auto-ARIMA), similar to R's `forecast::auto.arima`.                  | [Docs](https://alkaline-ml.com/pmdarima/) • [GitHub](https://github.com/alkaline-ml/pmdarima) | `pip install pmdarima` |
| **Prophet**      | Forecasting procedure for time series with strong seasonality and trend components, developed by Facebook.                | [Docs](https://facebook.github.io/prophet/) • [GitHub](https://github.com/facebook/prophet) | `pip install prophet`  |
| **StatsForecast**| Fast implementation of popular statistical forecasting models (ETS, ARIMA, Theta) optimized for performance.            | [Docs](https://nixtla.github.io/statsforecast/) • [GitHub](https://github.com/Nixtla/statsforecast) | `pip install statsforecast`|
| **MLForecast**   | Uses machine learning models (e.g., LightGBM, XGBoost) for scalable time series forecasting tasks.                        | [Docs](https://nixtla.github.io/mlforecast/) • [GitHub](https://github.com/Nixtla/mlforecast) | `pip install mlforecast` |
| **NeuralForecast**| Deep learning models (N-BEATS, N-HiTS, RNNs) for time series forecasting, built on PyTorch Lightning.                  | [Docs](https://nixtla.github.io/neuralforecast/) • [GitHub](https://github.com/Nixtla/neuralforecast) | `pip install neuralforecast`|

---

## High-Dimensional Methods

Techniques for estimation and inference when the number of parameters is large relative to the sample size.

| Package                  | Description                                                                                                    | Links                                                                                   | Installation                         |
|--------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|--------------------------------------|
| **Scikit-learn**         | Implements regularized regression methods like Lasso, Ridge, and ElasticNet for sparse modeling/feature selection. | [Docs](https://scikit-learn.org/stable/modules/linear_model.html) • [GitHub](https://github.com/scikit-learn/scikit-learn) | `pip install scikit-learn`           |
| **PyFixest**             | Efficiently handles high-dimensional fixed effects in linear models (see Panel Data section).                | [Docs & GitHub](https://github.com/py-econometrics/pyfixest)                            | `pip install pyfixest`               |
| **FixedEffectModelPyHDFE** | Solves linear models with high-dimensional fixed effects (see Panel Data section).                           | [PyPI](https://pypi.org/project/FixedEffectModelPyHDFE/)                                | `pip install FixedEffectModelPyHDFE` |

---

## Discrete Choice Models

Packages for analyzing choice behavior, demand estimation, and modeling qualitative dependent variables.

| Package         | Description                                                                                                       | Links                                                                                           | Installation             |
|-----------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------|
| **Statsmodels** | Includes Logit, Probit, Multinomial Logit (MNLogit), and Conditional Logit models.                                | [Docs](https://www.statsmodels.org/stable/discretemod.html) • [GitHub](https://github.com/statsmodels/statsmodels) | `pip install statsmodels`  |
| **XLogit**      | Fast estimation of Multinomial Logit and Mixed Logit models, optimized for performance.                           | [Docs](https://xlogit.readthedocs.io/) • [GitHub](https://github.com/arteagac/xlogit)           | `pip install xlogit`     |
| **PyLogit**     | Flexible implementation of conditional/multinomial logit models with utilities for data preparation.              | [GitHub](https://github.com/timothyb0912/pylogit)                                               | `pip install pylogit`    |
| **PyBLP**       | Tools for estimating demand for differentiated products using the Berry-Levinsohn-Pakes (BLP) method.             | [Docs](https://pyblp.readthedocs.io/) • [GitHub](https://github.com/jeffgortmaker/pyblp)         | `pip install pyblp`      |
| **torch-choice**| PyTorch framework for flexible estimation of complex discrete choice models, leveraging GPU acceleration.           | [Docs](https://gsbdbi.github.io/torch-choice/) • [GitHub](https://github.com/gsbDBI/torch-choice) | `pip install torch-choice`|

---

## Quantile Regression & Distributional Methods

Methods for modeling the conditional quantiles or the entire conditional distribution of an outcome variable.

| Package            | Description                                                                                                 | Links                                                                                              | Installation             |
|--------------------|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|--------------------------|
| **Statsmodels**    | Provides an implementation of quantile regression.                                                        | [Docs](https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html) • [GitHub](https://github.com/statsmodels/statsmodels) | `pip install statsmodels`  |
| **pyqreg**         | Fast quantile regression solver using interior point methods, supporting robust and clustered standard errors. | [Docs & GitHub](https://github.com/mozjay0619/pyqreg)                                              | `pip install pyqreg`     |
| **quantile-forest**| Scikit-learn compatible implementation of Quantile Regression Forests for non-parametric estimation.         | [Docs](https://zillow.github.io/quantile-forest/) • [GitHub](https://github.com/zillow/quantile-forest) | `pip install quantile-forest`|

---

## Bayesian Econometrics

Packages for performing Bayesian inference and probabilistic modeling.

| Package         | Description                                                                                                          | Links                                                                                 | Installation         |
|-----------------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|----------------------|
| **PyMC**        | Flexible probabilistic programming library for Bayesian modeling and inference using MCMC algorithms (NUTS).           | [Docs](https://www.pymc.io/) • [GitHub](https://github.com/pymc-devs/pymc)                 | `pip install pymc`   |
| **Bambi**       | High-level interface for building Bayesian GLMMs, built on top of PyMC. Uses formula syntax similar to R's `lme4`. | [Docs](https://bambinos.github.io/bambi/) • [GitHub](https://github.com/bambinos/bambi)     | `pip install bambi`  |
| **NumPyro**     | Probabilistic programming library built on JAX for scalable Bayesian inference, often faster than PyMC.              | [Docs](https://num.pyro.ai/) • [GitHub](https://github.com/pyro-ppl/numpyro)             | `pip install numpyro`|
| **LightweightMMM**| Bayesian Marketing Mix Modeling (see Marketing Mix Models section).                                                | [GitHub](https://github.com/google/lightweight_mmm)                                   | `pip install lightweight_mmm`|

---

## Marketing Mix Models (MMM)

Specialized packages for attributing marketing impact and optimizing spend using statistical models.

| Package         | Description                                                                                                | Links                                                                                                   | Installation / Access             |
|-----------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-----------------------------------|
| **LightweightMMM**| Google's Bayesian approach to Marketing Mix Modeling focusing on channel attribution and budget optimization. | [Docs](https://lightweight-mmm.readthedocs.io) • [GitHub](https://github.com/google/lightweight_mmm)       | `pip install lightweight_mmm`     |
| **PyMC Marketing**| Collection of Bayesian marketing models built with PyMC, including MMM, CLV, and attribution.              | [Docs](https://www.pymc-marketing.io/) • [GitHub](https://github.com/pymc-labs/pymc-marketing)           | `pip install pymc-marketing`      |
| **MaMiMo**      | Lightweight Python library focused specifically on Marketing Mix Modeling implementation.                  | [GitHub](https://github.com/Garve/mamimo)                                                               | `pip install mamimo`              |
| **mmm_stan**    | Python/STAN implementation of Bayesian Marketing Mix Models.                                               | [GitHub](https://github.com/sibylhe/mmm_stan)                                                           | GitHub Repository                 |

---

## Spatial Econometrics

Tools for analyzing data with spatial dependencies or geographic structure.

| Package         | Description                                                                                                      | Links                                                                                    | Installation                |
|-----------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|-----------------------------|
| **PySAL (spreg)**| The spatial regression `spreg` module of PySAL. Implements spatial lag, error, IV models, and diagnostics.         | [Docs](https://pysal.org/spreg/) • [GitHub](https://github.com/pysal/spreg)                 | `pip install spreg`         |
| *(PySAL Core)*  | The broader PySAL ecosystem contains many tools for spatial data handling, weights, visualization, and analysis. | [Docs](https://pysal.org/) • [GitHub](https://github.com/pysal/pysal)                   | `pip install pysal`         |

---

## Natural Language Processing for Economics

Libraries for processing and analyzing textual data within economic contexts.

| Package                    | Description                                                                                                        | Links                                                                                       | Installation                |
|----------------------------|--------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------|
| **Hugging Face Transformers**| Access to thousands of pre-trained models for NLP tasks like text classification, summarization, embeddings, etc. | [Docs](https://huggingface.co/transformers/) • [GitHub](https://github.com/huggingface/transformers) | `pip install transformers`  |
| **Gensim**                 | Library focused on topic modeling (LDA, LSI) and document similarity analysis.                                     | [Docs](https://radimrehurek.com/gensim/) • [GitHub](https://github.com/RaRe-Technologies/gensim) | `pip install gensim`        |
| **spaCy**                  | Industrial-strength NLP library for efficient text processing pipelines (NER, POS tagging, etc.).                  | [Docs](https://spacy.io/) • [GitHub](https://github.com/explosion/spaCy)                       | `pip install spacy`         |

---

## Synthetic Data Generation

Tools for creating artificial datasets that mimic the statistical properties of real-world data.

| Package                      | Description                                                                                                   | Links                                                                      | Installation             |
|------------------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|--------------------------|
| **SDV (Synthetic Data Vault)** | Comprehensive library for generating synthetic tabular, relational, and time series data using various models. | [Docs](https://sdv.dev/) • [GitHub](https://github.com/sdv-dev/SDV)         | `pip install sdv`        |
| **Synthpop**                 | Port of the R package for generating synthetic populations based on sample survey data.                       | [GitHub](https://github.com/alan-turing-institute/synthpop)                | `pip install synthpop`   |

---

## Numerical Optimization & Computational Tools

Foundational libraries for numerical computation, automatic differentiation, and optimization.

| Package   | Description                                                                                   | Links                                                                     | Installation         |
|-----------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------|
| **JAX**   | High-performance numerical computing with autograd and XLA compilation on CPU/GPU/TPU.        | [Docs](https://jax.readthedocs.io/) • [GitHub](https://github.com/google/jax)   | `pip install jax`    |
| **PyTorch**| Popular deep learning framework with flexible automatic differentiation.                    | [Docs](https://pytorch.org/) • [GitHub](https://github.com/pytorch/pytorch)   | (See PyTorch website)|

---

## Utilities & Econometric Infrastructure

Tools for improving code quality, performance, reproducibility, and implementing specific econometric procedures.

| Package        | Description                                                                                                | Links                                                                                           | Installation              |
|----------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------|
| **wildboottest**| Fast implementation of various wild cluster bootstrap algorithms (WCR, WCU) for robust inference.          | [Docs](https://py-econometrics.github.io/wildboottest/) • [GitHub](https://github.com/py-econometrics/wildboottest) | `pip install wildboottest`|
| **appelpy**     | Applied econometrics library aiming for Stata-like syntax simplicity with Python's object-oriented power. | [GitHub](https://github.com/mfarragher/appelpy)                                                 | `pip install appelpy`     |
| **TQDM**        | Fast, extensible progress bar for loops and long-running operations.                                     | [Docs](https://tqdm.github.io/) • [GitHub](https://github.com/tqdm/tqdm)                           | `pip install tqdm`        |
| **Pylint**      | Code analysis tool for identifying errors, enforcing coding standards, and improving code quality.       | [Docs](https://pylint.pycqa.org/) • [GitHub](https://github.com/PyCQA/pylint)                     | `pip install pylint`      |

---

## Learning Resources

Curated resources for learning econometrics and quantitative economics with Python.

| Resource                     | Description                                                                                                     | Link                                                                                      |
|------------------------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **QuantEcon Lectures**       | High-quality lecture series on quantitative economic modeling, computational tools, and economics using Python/Julia. | [Website](https://quantecon.org/lectures/)                                                |
| **Python for Econometrics**  | Comprehensive intro notes by Kevin Sheppard covering Python basics, core libraries, and econometrics applications. | [PDF](https://www.kevinsheppard.com/files/teaching/python/notes/python_introduction_2023.pdf) |
| **Python Causality Handbook**| Modern introduction to causal inference methods (DiD, IV, RDD, Synth, ML-based) with Python code examples.       | [Website](https://matheusfacure.github.io/python-causality-handbook/)                     |
| **Coding for Economists**    | Practical guide by A. Turrell on using Python for modern econometric research, data analysis, and workflows.      | [Website](https://aeturrell.github.io/coding-for-economists/)                             |
| **The Missing Semester of Your CS Education (MIT)** | Teaches essential developer tools often skipped in formal education—command line, Git, Vim, scripting, debugging, etc. | [Website](https://missing.csail.mit.edu/) |
| **Machine Learning Specialization (Coursera)** | Beginner-friendly 3-course series by Andrew Ng covering core ML methods (regression, classification, clustering, trees, NN) with hands-on projects. | [Course](https://www.coursera.org/specializations/machine-learning-introduction/) |
| **Deep Learning Specialization (Coursera)** | Intermediate 5-course series by Andrew Ng covering deep neural networks, CNNs, RNNs, transformers, and real-world DL applications using TensorFlow. | [Course](https://www.coursera.org/specializations/deep-learning) |
| **Beyond Jupyter (TransferLab)** | Teaches software design principles for ML—modularity, abstraction, and reproducibility—going beyond ad hoc Jupyter workflows. Focus on maintainable, production-quality ML code. | [Website](https://transferlab.ai/trainings/beyond-jupyter/) |

---

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` file (linked via the badge at the top) in the repository root for details on how to:

1.  **Add new packages**: Submit a pull request with relevant, well-maintained packages.
2.  **Improve documentation**: Help enhance descriptions, add examples, or correct errors.
3.  **Report issues**: Notify us about broken links, outdated information, or suggest new categories.

*Last updated: April 2025*
