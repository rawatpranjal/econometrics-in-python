# Econometrics in Python

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) <!-- Note: Link points to LICENSE file in the repository root -->
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md) <!-- Note: Link points to CONTRIBUTING.md file in the repository root -->
 
A comprehensive collection of Python packages for econometrics, causal inference, quantitative economics, and data analysis. This repository serves as a reference guide for researchers, data scientists, economists, and practitioners working with economic data. Most packages can be installed via `pip`.

## Contents

- [Core Libraries & Linear Models](#core-libraries--linear-models)
- [Statistical Inference & Hypothesis Testing](#statistical-inference--hypothesis-testing)
- [Power Simulation & Design of Experiments](#power-simulation--design-of-experiments) 
- [Panel Data & Fixed Effects](#panel-data--fixed-effects)
- [Instrumental Variables (IV) & GMM](#instrumental-variables-iv--gmm)
- [Causal Inference & Matching](#causal-inference--matching)
- [Causal Discovery & Graphical Models](#causal-discovery--graphical-models)
- [Double/Debiased Machine Learning (DML)](#doubledebiased-machine-learning-dml)
- [Program Evaluation Methods (DiD, SC, RDD)](#program-evaluation-methods-did-sc-rdd)
- [Adaptive Experimentation & Bandits](#adaptive-experimentation--bandits)
- [Tree & Ensemble Methods for Prediction](#tree--ensemble-methods-for-prediction)
- [Time Series Forecasting](#time-series-forecasting)
- [Time Series Econometrics](#time-series-econometrics)
- [State Space & Volatility Models](#state-space--volatility-models)
- [Discrete Choice Models](#discrete-choice-models)
- [Structural Econometrics & Estimation](#structural-econometrics--estimation)
- [Quantile Regression & Distributional Methods](#quantile-regression--distributional-methods)
- [Bayesian Econometrics](#bayesian-econometrics)
- [Marketing Mix Models (MMM) & Business Analytics](#marketing-mix-models-mmm--business-analytics)
- [Spatial Econometrics](#spatial-econometrics)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Natural Language Processing Tools](#natural-language-processing-tools)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Numerical Optimization & Computational Tools](#numerical-optimization--computational-tools)
- [Standard Errors, Bootstrapping & Reporting](#standard-errors-bootstrapping--reporting)
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

## Statistical Inference & Hypothesis Testing

Packages providing functions for classical hypothesis testing, group comparisons, survival/duration analysis, and related statistical inference.

| Package         | Description                                                                                                                                       | Links                                                                                                | Installation             |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------|
| **Scipy.stats** | Foundational module within SciPy for a wide range of statistical functions, distributions, and hypothesis tests (t-tests, ANOVA, chi², KS, etc.).     | [Docs](https://docs.scipy.org/doc/scipy/reference/stats.html) • [GitHub](https://github.com/scipy/scipy) | `pip install scipy`      |
| **Statsmodels** | Includes dedicated modules for statistical tests (`stats`), ANOVA (`anova`), nonparametric methods, multiple testing corrections, contingency tables. | [Docs (stats)](https://www.statsmodels.org/stable/stats.html) • [GitHub](https://github.com/statsmodels/statsmodels) | `pip install statsmodels`  |
| **Pingouin**    | User-friendly interface for common statistical tests (ANOVA, ANCOVA, t-tests, correlations, chi², reliability) built on pandas & scipy.              | [Docs](https://pingouin-stats.org/) • [GitHub](https://github.com/raphaelvallat/pingouin)             | `pip install pingouin`   |
| **hypothetical**| Library focused on hypothesis testing: ANOVA/MANOVA, t-tests, chi-square, Fisher's exact, nonparametric tests (Mann-Whitney, Kruskal-Wallis, etc.). | [GitHub](https://github.com/aschleg/hypothetical)                                                     | `pip install hypothetical` |
| **lifelines**   | Comprehensive library for survival analysis: Kaplan-Meier, Nelson-Aalen, Cox regression, AFT models, handling censored data.                        | [Docs](https://lifelines.readthedocs.io/en/latest/) • [GitHub](https://github.com/CamDavidsonPilon/lifelines) | `pip install lifelines`  |
| **PyWhy-Stats**    | Part of the PyWhy ecosystem providing statistical methods specifically for causal applications, including various independence tests and power-divergence methods. | [Docs](https://pywhy-stats.readthedocs.io/) • [GitHub](https://github.com/py-why/pywhy-stats) | `pip install pywhy-stats` |

---

## Power Simulation & Design of Experiments

Tools for calculating statistical power, determining sample sizes, generating experimental designs (DoE), and implementing adaptive experimentation methods.

| Package             | Description (Focus)                                                                                                          | Links                                                                                             | Installation           |
|---------------------|------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|------------------------|
| **Statsmodels**     | Includes `stats.power` module for power/sample size calculations for t-tests, F-tests, Z-tests, Chi-squared tests.           | [Docs (Power)](https://www.statsmodels.org/stable/stats.html#power-and-sample-size-calculations) • [GitHub](https://github.com/statsmodels/statsmodels) | `pip install statsmodels`|
| **pyDOE2**          | Implements classical Design of Experiments: factorial (full/fractional), response surface (Box-Behnken, CCD), Latin Hypercube. | [Docs](https://pythonhosted.org/pyDOE2/) • [GitHub](https://github.com/clicumu/pyDOE2)                 | `pip install pyDOE2`   |
| **DoEgen**          | Automates generation and optimization of designs, especially for mixed factor-level experiments; computes efficiency metrics. | [GitHub](https://github.com/sebhaan/DoEgen)                                                        | `pip install DoEgen`   |
| **ADOpy**           | Bayesian Adaptive Design Optimization (ADO) for tuning experiments in real-time, with models for psychometric tasks.           | [Docs](https://adopy.readthedocs.io/en/latest/) • [GitHub](https://github.com/adopy/adopy)             | `pip install adopy`    |
| **Adaptive**        | Parallel active learning library for adaptive function sampling/evaluation, with live plotting for monitoring.               | [Docs](https://adaptive.readthedocs.io/en/latest/) • [GitHub](https://github.com/python-adaptive/adaptive) | `pip install adaptive` |

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
| **scikit-uplift**| Focuses on uplift modeling and estimating heterogeneous treatment effects using various ML-based methods. | [Docs](https://scikit-uplift.readthedocs.io/en/latest/) • [GitHub](https://github.com/maks-sh/scikit-uplift) | `pip install scikit-uplift`|
| **CausalPy**       | Developed by PyMC Labs, focuses specifically on causal inference in quasi-experimental settings. Specializes in scenarios where randomization is impossible or expensive. | [Docs](https://www.pymc-marketing.io/) • [GitHub](https://github.com/pymc-labs/pymc-marketing) | `pip install CausalPy`    |
| **CausalLib**      | IBM-developed package that provides a scikit-learn-inspired API for causal inference with meta-algorithms supporting arbitrary machine learning models. | [Docs](https://causallib.readthedocs.io/) • [GitHub](https://github.com/IBM/causallib) | `pip install causallib`   |
| **CausalPlayground** | Python library for causal research that addresses the scarcity of real-world datasets with known causal relations. Provides fine-grained control over structural causal models. | [Docs](https://causal-playground.readthedocs.io/) • [GitHub](https://github.com/causal-playground/causal-playground) | `pip install causal-playground` |

---

## Causal Discovery & Graphical Models

Libraries focused on learning causal structures (DAGs, Bayesian Networks) from data and performing inference using graphical models.

| Package                             | Description (Focus)                                                                                     | Links                                                                                             | Installation              |
|-------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|---------------------------|
| **Ananke**                          | Causal inference using graphical models (DAGs), including identification theory and effect estimation.    | [Docs](https://ananke.readthedocs.io/) • [GitHub]([https://github.com/py-why/Ananke](https://github.com/ghosthamlet/ananke))                   | `pip install ananke-causal` |
| **CausalNex**                       | Uses Bayesian Networks for causal reasoning, combining ML with expert knowledge to model relationships. | [GitHub](https://github.com/microsoft/causalnex)                                                  | `pip install causalnex`     |
| **Causal Discovery Toolbox (CDT)**  | Implements algorithms for causal discovery (recovering causal graph structure) from observational data.   | [Docs](https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html) • [GitHub](https://github.com/FenTechSolutions/CausalDiscoveryToolbox) | `pip install cdt`           |
| **DoWhy**                           | (See Causal Inference) Includes functionality for modeling assumptions with causal graphs (DAGs).         | [Docs](https://www.pywhy.org/dowhy/) • [GitHub](https://github.com/py-why/dowhy)                | `pip install dowhy`       |
| **Tigramite**      | Specialized package for causal inference in time series data implementing PCMCI, PCMCIplus, LPCMCI algorithms with conditional independence tests. | [Docs](https://tigramite.readthedocs.io/) • [GitHub](https://github.com/jakobrunge/tigramite) | `pip install tigramite`   |
| **gCastle**        | Huawei Noah's Ark Lab end-to-end causal structure learning toolchain emphasizing gradient-based methods with GPU acceleration (NOTEARS, GOLEM). | [Docs](https://gcastle.readthedocs.io/) • [GitHub](https://github.com/huawei-noah/trustworthyAI) | `pip install gcastle`     |
| **causal-learn**   | Comprehensive Python package serving as Python translation and extension of Java-based Tetrad toolkit for causal discovery algorithms. | [Docs](https://causal-learn.readthedocs.io/) • [GitHub](https://github.com/py-why/causal-learn) | `pip install causal-learn` |
| **LiNGAM**         | Specialized package for learning non-Gaussian linear causal models, implementing various versions of the LiNGAM algorithm including ICA-based methods. | [Docs](https://lingam.readthedocs.io/) • [GitHub](https://github.com/cdt15/lingam) | `pip install lingam`      |
| **py-tetrad**      | Python interface to Tetrad Java library using JPype, providing direct access to Tetrad's causal discovery algorithms with efficient data translation. | [GitHub](https://github.com/py-why/py-tetrad) | Available on GitHub (installation via git clone) |

---

## Double/Debiased Machine Learning (DML)

Methods combining machine learning and econometrics for robust causal inference in high-dimensional settings.

| Package      | Description                                                                                                                          | Links                                                                                   | Installation        |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|---------------------|
| **DoubleML** | Implements the double/debiased ML framework (Chernozhukov et al.) for estimating causal parameters (ATE, LATE, POM) with ML nuisances. | [Docs](https://docs.doubleml.org/) • [GitHub](https://github.com/DoubleML/doubleml-for-py) | `pip install DoubleML`|
| **EconML**   | Microsoft toolkit for estimating heterogeneous treatment effects using DML, causal forests, meta-learners, and orthogonal ML methods.  | [Docs](https://econml.azurewebsites.net/) • [GitHub](https://github.com/py-why/EconML)     | `pip install econml`  |
| **pydoublelasso** | Double‑post Lasso estimator for high‑dimensional treatment effects (Belloni‑Chernozhukov‑Hansen 2014). | [PyPI](https://pypi.org/project/pydoublelasso/) | `pip install pydoublelasso` |
| **pyhtelasso**    | Debiased‑Lasso detector of heterogeneous treatment effects in randomized experiments. | [PyPI](https://pypi.org/project/pyhtelasso/) | `pip install pyhtelasso` |

---

## Program Evaluation Methods (DiD, SC, RDD)

Quasi-experimental methods for estimating causal effects from policy changes or natural experiments.

| Package                     | Description                                                                                                                | Links                                                                                   | Installation                      |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-----------------------------------|
| **csdid**                   | Python adaptation of the R `did` package. Implements multi-period DiD with staggered treatment timing (Callaway & Sant’Anna). | [GitHub](https://github.com/d2cml-ai/csdid)                                               | `pip install csdid`               |
| **Differences**             | Implements modern difference-in-differences methods for staggered adoption designs (e.g., Callaway & Sant'Anna).           | [Docs](https://bernardodionisi.github.io/differences/) • [GitHub](https://github.com/bernardodionisi/differences) | `pip install differences`         |
| **SyntheticControlMethods** | Implementation of synthetic control methods for comparative case studies when panel data is available.                     | [GitHub](https://github.com/OscarEngelbrektson/SyntheticControlMethods)                   | `pip install SyntheticControlMethods` |
| **mlsynth**              | Implements advanced synthetic control methods: forward DiD, cluster SC, factor models, and proximal SC. Designed for single-treated-unit settings. | [Docs](https://mlsynth.readthedocs.io/en/latest/) • [GitHub](https://github.com/jaredjgreathouse/mlsynth) | `pip install mlsynth` |
| **rdrobust**                | Comprehensive tools for Regression Discontinuity Designs (RDD), including optimal bandwidth selection, estimation, inference. | [GitHub](https://github.com/rdpackages/rdrobust) • [PyPI](https://pypi.org/project/rdrobust/) | `pip install rdrobust`            |
| **rdd**                     | Toolkit for sharp RDD analysis, including bandwidth calculation and estimation, integrating with pandas.                 | [GitHub](https://github.com/evan-magnusson/rdd)                                         | `pip install rdd`                 |
| **CausalImpact** | Python port of Google's R package for estimating causal effects of interventions on time series using Bayesian structural time-series models. | [Docs](https://google.github.io/CausalImpact/CausalImpact/CausalImpact.html) (R) • [GitHub (Py)](https://github.com/tcassou/causal_impact) | `pip install causalimpact` |
| **pycinc**      | Changes‑in‑Changes (CiC) estimator for distributional treatment effects (Athey & Imbens 2006). | [PyPI](https://pypi.org/project/pycinc/) | `pip install pycinc` |
| **pyleebounds** | Lee (2009) sample‑selection bounds for treatment effects; trims treated distribution to match selection rates. | [PyPI](https://pypi.org/project/pyleebounds/) | `pip install pyleebounds` |

---

## Adaptive Experimentation & Bandits

Libraries for designing and evaluating adaptive experiments using multi-armed bandit (MAB) algorithms, covering stochastic, contextual, and more complex bandit settings.

| Package                     | Description (Focus)                                                                                                     | Links                                                                                                      | Installation              |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|---------------------------|
| **SMPyBandits**             | Comprehensive research framework for single/multi-player MAB algorithms (stochastic, adversarial, contextual).            | [Docs](https://smpybandits.github.io/) • [GitHub](https://github.com/SMPyBandits/SMPyBandits)                  | `pip install SMPyBandits` |
| **MABWiser**                | Production-ready, scikit-learn style library for contextual & stochastic bandits with parallelism and simulation tools.   | [Docs](https://fidelity.github.io/mabwiser/) • [GitHub](https://github.com/fidelity/mabwiser)              | `pip install mabwiser`    |
| **ContextualBandits**       | Implements a wide range of contextual bandit algorithms (linear, tree-based, neural) and off-policy evaluation methods. | [Docs](https://contextual-bandits.readthedocs.io/) • [GitHub](https://github.com/david-cortes/contextualbandits) | `pip install contextualbandits` |
| **BayesianBandits**         | Lightweight microframework for Bayesian bandits (Thompson Sampling) with support for contextual/restless/delayed rewards. | [Docs](https://rukulkarni.com/projects/bayesianbandits/) • [GitHub](https://github.com/IntelyCare/bayesianbandits) | `pip install bayesianbandits` |
| **Open Bandit Pipeline (OBP)**| Framework for **offline evaluation (OPE)** of bandit policies using logged data. Implements IPS, DR, DM estimators.      | [Docs](https://zr-obp.readthedocs.io/en/latest/) • [GitHub](https://github.com/st-tech/zr-obp)           | `pip install obp`         |
| **PyXAB**                   | Library for advanced bandit problems: X-armed bandits (continuous/structured action spaces) and online optimization.    | [Docs](https://pyxab.readthedocs.io/en/latest/) • [GitHub](https://github.com/huanzhang12/pyxab)             | `pip install pyxab`       |

---

## Tree & Ensemble Methods for Prediction

Powerful machine learning techniques like Random Forests and Gradient Boosting, often used for prediction, feature importance, and handling complex non-linear relationships. This section covers leading packages for both types of ensemble methods.

| Package               | Description                                                                                                                              | Links                                                                                                | Installation                             |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------|
| **Scikit-learn Ens.** | (`RandomForestClassifier`/`Regressor`) Widely-used, versatile implementation of Random Forests. Easy API and parallel processing support. | [Docs](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) • [GitHub](https://github.com/scikit-learn/scikit-learn) | `pip install scikit-learn`               |
| **cuML (RAPIDS)**     | GPU-accelerated implementation of Random Forests for significant speedups on large datasets. Scikit-learn compatible API.                 | [Docs](https://docs.rapids.ai/api/cuml/stable/) • [GitHub](https://github.com/rapidsai/cuml)             | `conda install ...` (See RAPIDS docs) |
| **XGBoost**           | High-performance, optimized gradient boosting library (also supports RF). Known for speed, efficiency, and winning competitions.            | [Docs](https://xgboost.readthedocs.io/) • [GitHub](https://github.com/dmlc/xgboost)                    | `pip install xgboost`                    |
| **LightGBM**          | Fast, distributed gradient boosting (also supports RF). Known for speed, low memory usage, and handling large datasets.                   | [Docs](https://lightgbm.readthedocs.io/) • [GitHub](https://github.com/microsoft/LightGBM)           | `pip install lightgbm`                   |
| **CatBoost**          | Gradient boosting library excelling with categorical features (minimal preprocessing needed). Robust against overfitting.                  | [Docs](https://catboost.ai/docs/) • [GitHub](https://github.com/catboost/catboost)                     | `pip install catboost`                   |
| **NGBoost**           | Extends gradient boosting to probabilistic prediction, providing uncertainty estimates alongside point predictions. Built on scikit-learn.   | [Docs](https://stanfordmlgroup.github.io/ngboost/) • [GitHub](https://github.com/stanfordmlgroup/ngboost) | `pip install ngboost`                    |

*(Note: Scikit-learn also provides GradientBoostingClassifier/Regressor. XGBoost and LightGBM are primarily Gradient Boosting but offer Random Forest modes.)*

---

## Time Series Forecasting

Packages focused on predicting future values of time series, including classical models, machine learning approaches, and specialized forecasting frameworks.

| Package          | Description                                                                                                             | Links                                                                             | Installation           |
|------------------|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|------------------------|
| **Statsmodels**  | Core implementations of classical forecasting models like ARIMA, SARIMAX, Exponential Smoothing (ETS), Unobserved Components (UCM). | [Docs (TSA)](https://www.statsmodels.org/stable/tsa.html) • [GitHub](https://github.com/statsmodels/statsmodels) | `pip install statsmodels`|
| **pmdarima**     | ARIMA modeling with automatic parameter selection (auto-ARIMA), similar to R's `forecast::auto.arima`.                  | [Docs](https://alkaline-ml.com/pmdarima/) • [GitHub](https://github.com/alkaline-ml/pmdarima) | `pip install pmdarima` |
| **Prophet**      | Forecasting procedure for time series with strong seasonality and trend components, developed by Facebook.                | [Docs](https://facebook.github.io/prophet/) • [GitHub](https://github.com/facebook/prophet) | `pip install prophet`  |
| **StatsForecast**| Fast, scalable implementations of popular statistical forecasting models (ETS, ARIMA, Theta, etc.) optimized for performance. | [Docs](https://nixtla.github.io/statsforecast/) • [GitHub](https://github.com/Nixtla/statsforecast) | `pip install statsforecast`|
| **MLForecast**   | Scalable time series forecasting using machine learning models (e.g., LightGBM, XGBoost) as regressors.                  | [Docs](https://nixtla.github.io/mlforecast/) • [GitHub](https://github.com/Nixtla/mlforecast) | `pip install mlforecast` |
| **NeuralForecast**| Deep learning models (N-BEATS, N-HiTS, Transformers, RNNs) for time series forecasting, built on PyTorch Lightning.     | [Docs](https://nixtla.github.io/neuralforecast/) • [GitHub](https://github.com/Nixtla/neuralforecast) | `pip install neuralforecast`|
| **sktime**       | Unified framework for various time series tasks, including forecasting with classical, ML, and deep learning models.       | [Docs](https://www.sktime.net/en/latest/) • [GitHub](https://github.com/sktime/sktime) | `pip install sktime`     |

---

## Time Series Econometrics

Libraries focused on modeling dynamic relationships, causality, conditional volatility, and structural properties in time series data using methods like VAR, VECM, GARCH, and impulse response analysis.

| Package             | Description (Focus)                                                                                                                   | Links                                                                                                                       | Installation             |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------|
| **Statsmodels**     | Implements Vector Autoregression (VAR), SVAR, VECM, impulse response functions, Granger causality tests, unit root/cointegration tests. | [Docs (VAR)](https://www.statsmodels.org/stable/vector_ar.html) • [Docs (Tests)](https://www.statsmodels.org/stable/tsa.html#statistical-tests) • [GitHub](https://github.com/statsmodels/statsmodels) | `pip install statsmodels`  |
| **ARCH**            | Specialized library for modeling and forecasting conditional volatility using ARCH, GARCH, EGARCH, and related models.                | [Docs](https://arch.readthedocs.io/) • [GitHub](https://github.com/bashtage/arch)                                          | `pip install arch`         |
| **LocalProjections**| Community implementations of Jordà (2005) Local Projections for estimating impulse responses without VAR assumptions.                 | [Example GitHub](https://github.com/elenev/localprojections)                                                                 | Install from source      |
| **Kats**            | Broad toolkit for time series analysis, including multivariate analysis, detection (outliers, change points, trends), feature extraction. | [Docs](https://facebookresearch.github.io/Kats/) • [GitHub](https://github.com/facebookresearch/Kats)                      | `pip install kats`       |

---

## State Space & Volatility Models

Libraries for representing and estimating models in state-space form (using Kalman filtering/smoothing) and for modeling stochastic volatility.

| Package             | Description (Focus)                                                                                                                | Links                                                                                                                                  | Installation               |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| **Statsmodels**     | Comprehensive state-space modeling framework (`tsa.statespace`) supporting ARIMA, VARMAX, structural time series (UCM), DFM, custom models. | [Docs (StateSpace)](https://www.statsmodels.org/stable/statespace.html) • [GitHub](https://github.com/statsmodels/statsmodels)                 | `pip install statsmodels`    |
| **stochvol**        | Efficient Bayesian estimation of stochastic volatility (SV) models using MCMC.                                                     | [Docs](https://stochvol.readthedocs.io/en/latest/) • [GitHub](https://github.com/rektory/stochvol)                                        | `pip install stochvol`     |
| **Metran**          | Specialized package for estimating Dynamic Factor Models (DFM) using state-space methods and Kalman filtering.                       | [GitHub](https://github.com/pastas/metran)                                                                                              | `pip install metran`       |
| **FilterPy**        | Focuses on Kalman filters (standard, EKF, UKF) and smoothers with a clear, pedagogical implementation style.                       | [Docs](https://filterpy.readthedocs.io/en/latest/) • [GitHub](https://github.com/rlabbe/filterpy)                                       | `pip install filterpy`     |
| **PyKalman**        | Implements Kalman filter, smoother, and EM algorithm for parameter estimation, including support for missing values and UKF.         | [PyPI](https://pypi.org/project/pykalman/) • [GitHub](https://github.com/pykalman/pykalman)                                               | `pip install pykalman`     |
| **PyMC Statespace** | (See Bayesian) Bayesian state-space modeling using PyMC, integrating Kalman filtering within MCMC for parameter estimation.          | [Docs](https://pymc-statespace.readthedocs.io/en/latest/) • [GitHub](https://github.com/pymc-devs/pymc-statespace) (Note: Check repo status) | `pip install pymc-statespace`|

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
| **Biogeme**         | Maximum likelihood estimation of parametric models, with strong support for complex discrete choice models.     | [Docs](https://biogeme.epfl.ch/index.html) • [GitHub](https://github.com/michelbierlaire/biogeme) | `pip install biogeme`   |

---

## Structural Econometrics & Estimation

Frameworks for specifying, simulating, and estimating structural economic models, often involving dynamic programming or complex likelihoods.

| Package             | Description (Focus)                                                                                             | Links                                                                                  | Installation          |
|---------------------|-----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|-----------------------|
| **respy**           | Simulation and estimation of finite-horizon dynamic discrete choice (DDC) models (e.g., labor/education choice).  | [Docs](https://respy.readthedocs.io/en/latest/) • [GitHub](https://github.com/OpenSourceEconomics/respy) | `pip install respy`     |
| **HARK**            | Toolkit for solving, simulating, and estimating models with heterogeneous agents (e.g., consumption-saving).    | [Docs](https://hark.readthedocs.io/en/latest/) • [GitHub](https://github.com/econ-ark/HARK)      | `pip install econ-ark`  |
| **Dolo**            | Framework for describing and solving economic models (DSGE, OLG, etc.) using a declarative YAML-based format.     | [Docs](https://dolo.readthedocs.io/en/latest/) • [GitHub](https://github.com/EconForge/dolo)       | `pip install dolo`      |
| **PyBLP**           | (See Discrete Choice) Estimation of demand using Berry-Levinsohn-Pakes (BLP) structural models.               | [Docs](https://pyblp.readthedocs.io/) • [GitHub](https://github.com/jeffgortmaker/pyblp)    | `pip install pyblp`     |
| **QuantEcon.py**    | Core library for quantitative economics: dynamic programming, Markov chains, game theory, numerical methods.    | [Docs](https://quantecon.org/python-lectures/) • [GitHub](https://github.com/QuantEcon/QuantEcon.py) | `pip install quantecon` |

---

## Quantile Regression & Distributional Methods

Methods for modeling the conditional quantiles or the entire conditional distribution of an outcome variable.

| Package            | Description                                                                                                 | Links                                                                                              | Installation             |
|--------------------|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|--------------------------|
| **Statsmodels**    | Provides an implementation of quantile regression.                                                        | [Docs](https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html) • [GitHub](https://github.com/statsmodels/statsmodels) | `pip install statsmodels`  |
| **pyqreg**         | Fast quantile regression solver using interior point methods, supporting robust and clustered standard errors. | [Docs & GitHub](https://github.com/mozjay0619/pyqreg)                                              | `pip install pyqreg`     |
| **quantile-forest**| Scikit-learn compatible implementation of Quantile Regression Forests for non-parametric estimation.         | [Docs](https://zillow.github.io/quantile-forest/) • [GitHub](https://github.com/zillow/quantile-forest) | `pip install quantile-forest`|
| **pyrifreg** | Recentered Influence‑Function (RIF) regression for unconditional quantile & distributional effects (Firpo et al., 2008). | [Docs & GitHub](https://github.com/vyasenov/pyrifreg) | `pip install pyrifreg` |

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

## Marketing Mix Models (MMM) & Business Analytics

Specialized packages for attributing marketing impact, customer analytics (CLV), and optimizing spend.

| Package          | Description                                                                                                | Links                                                                                                   | Installation / Access        |
|------------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|------------------------------|
| **LightweightMMM** | Google's Bayesian approach to Marketing Mix Modeling focusing on channel attribution and budget optimization. | [Docs](https://lightweight-mmm.readthedocs.io) • [GitHub](https://github.com/google/lightweight_mmm)       | `pip install lightweight_mmm`  |
| **PyMC Marketing**| Collection of Bayesian marketing models built with PyMC, including MMM, CLV, and attribution.              | [Docs](https://www.pymc-marketing.io/) • [GitHub](https://github.com/pymc-labs/pymc-marketing)           | `pip install pymc-marketing`      |
| **MaMiMo**      | Lightweight Python library focused specifically on Marketing Mix Modeling implementation.                  | [GitHub](https://github.com/Garve/mamimo)                                                               | `pip install mamimo`              |
| **mmm_stan**    | Python/STAN implementation of Bayesian Marketing Mix Models.                                               | [GitHub](https://github.com/sibylhe/mmm_stan)                                                           | GitHub Repository                 |
| **Lifetimes**    | Analyze customer lifetime value (CLV) using probabilistic models (BG/NBD, Pareto/NBD) to predict purchases. | [Docs](https://lifetimes.readthedocs.io/en/latest/) • [GitHub](https://github.com/CamDavidsonPilon/lifetimes) | `pip install lifetimes`      |

---

## Spatial Econometrics

Tools for analyzing data with spatial dependencies or geographic structure.

| Package         | Description                                                                                                      | Links                                                                                    | Installation                |
|-----------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|-----------------------------|
| **PySAL (spreg)**| The spatial regression `spreg` module of PySAL. Implements spatial lag, error, IV models, and diagnostics.         | [Docs](https://pysal.org/spreg/) • [GitHub](https://github.com/pysal/spreg)                 | `pip install spreg`         |
| *(PySAL Core)*  | The broader PySAL ecosystem contains many tools for spatial data handling, weights, visualization, and analysis. | [Docs](https://pysal.org/) • [GitHub](https://github.com/pysal/pysal)                   | `pip install pysal`         |

---

## Dimensionality Reduction

Libraries for reducing the number of variables in a dataset while preserving important information, including linear methods (PCA, Factor Analysis) and non-linear manifold learning techniques (t-SNE, UMAP).

| Package             | Description (Focus)                                                                                                        | Links                                                                                                         | Installation             |
|---------------------|----------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|--------------------------|
| **Scikit-learn**    | Foundational ML library with various dimensionality reduction methods: PCA, Factor Analysis, Isomap, t-SNE, etc.             | [Docs](https://scikit-learn.org/stable/modules/unsupervised_reduction.html) • [GitHub](https://github.com/scikit-learn/scikit-learn) | `pip install scikit-learn` |
| **FactorAnalyzer**  | Specialized library for Exploratory (EFA) and Confirmatory (CFA) Factor Analysis with rotation options for interpretability. | [Docs](https://factor-analyzer.readthedocs.io/en/latest/) • [GitHub](https://github.com/EducationalTestingService/factor_analyzer) | `pip install factor_analyzer`|
| **umap-learn**      | Fast and scalable implementation of Uniform Manifold Approximation and Projection (UMAP) for non-linear reduction.         | [Docs](https://umap-learn.readthedocs.io/en/latest/) • [GitHub](https://github.com/lmcinnes/umap)              | `pip install umap-learn`   |
| **openTSNE**        | Optimized, parallel implementation of t-distributed Stochastic Neighbor Embedding (t-SNE) for large datasets.             | [Docs](https://opentsne.readthedocs.io/en/stable/) • [GitHub](https://github.com/pavlin-policar/openTSNE)      | `pip install opentsne`   |

---

## Natural Language Processing for Economics

Libraries for processing and analyzing textual data within economic contexts.

| Package                    | Description                                                                                                        | Links                                                                                       | Installation                |
|----------------------------|--------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------|
| **Transformers**| Access to thousands of pre-trained models for NLP tasks like text classification, summarization, embeddings, etc. | [Docs](https://huggingface.co/transformers/) • [GitHub](https://github.com/huggingface/transformers) | `pip install transformers`  |
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

## Standard Errors, Bootstrapping & Reporting

Tools for robust statistical inference (corrected standard errors, bootstrapping) and creating publication-quality regression tables.

| Package                 | Description (Focus)                                                                                                                        | Links                                                                                                                       | Installation                 |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------|
| **Statsmodels**         | Core library for estimating robust standard errors (HC0-HC3, HAC, Clustered) via `get_robustcov_results`. Provides building blocks for manual bootstrap. | [Docs (Robust SE)](https://www.statsmodels.org/stable/regression.html#robust-standard-errors) • [GitHub](https://github.com/statsmodels/statsmodels) | `pip install statsmodels`      |
| **wildboottest**        | Fast implementation of various wild cluster bootstrap algorithms (WCR, WCU) for robust inference, especially with few clusters.                | [Docs](https://py-econometrics.github.io/wildboottest/) • [GitHub](https://github.com/py-econometrics/wildboottest)             | `pip install wildboottest`     |
| **SciPy Bootstrap**     | (`scipy.stats.bootstrap`) Computes bootstrap confidence intervals for various statistics using percentile, BCa methods.                      | [Docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) • [GitHub (SciPy)](https://github.com/scipy/scipy) | `pip install scipy`          |
| **Stargazer**           | Python port of R's stargazer for creating publication-quality regression tables (HTML, LaTeX) from `statsmodels` & `linearmodels` results.   | [GitHub](https://github.com/StatsReporting/stargazer)                                                                        | `pip install stargazer`      |

---

## Learning Resources

Curated resources for learning econometrics and quantitative economics with Python.

| Resource                     | Description                                                                                                     | Link                                                                                      |
|------------------------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **QuantEcon Lectures**       | High-quality lecture series on quantitative economic modeling, computational tools, and economics using Python/Julia. | [Website](https://quantecon.org/lectures/)                                                |
| **Python for Econometrics**  | Comprehensive intro notes by Kevin Sheppard covering Python basics, core libraries, and econometrics applications. | [PDF](https://www.kevinsheppard.com/files/teaching/python/notes/python_introduction_2023.pdf) |
| **Causal Inference for the Brave and True**| Modern introduction to causal inference methods (DiD, IV, RDD, Synth, ML-based) with Python code examples.       | [Website](https://matheusfacure.github.io/python-causality-handbook/)                     |
| **Coding for Economists**    | Practical guide by A. Turrell on using Python for modern econometric research, data analysis, and workflows.      | [Website](https://aeturrell.github.io/coding-for-economists/)                             |
| **The Missing Semester of Your CS Education (MIT)** | Teaches essential developer tools often skipped in formal education—command line, Git, Vim, scripting, debugging, etc. | [Website](https://missing.csail.mit.edu/) |
| **Machine Learning Specialization (Coursera)** | Beginner-friendly 3-course series by Andrew Ng covering core ML methods (regression, classification, clustering, trees, NN) with hands-on projects. | [Course](https://www.coursera.org/specializations/machine-learning-introduction/) |
| **Deep Learning Specialization (Coursera)** | Intermediate 5-course series by Andrew Ng covering deep neural networks, CNNs, RNNs, transformers, and real-world DL applications using TensorFlow. | [Course](https://www.coursera.org/specializations/deep-learning) |
| **Beyond Jupyter (TransferLab)** | Teaches software design principles for ML—modularity, abstraction, and reproducibility—going beyond ad hoc Jupyter workflows. Focus on maintainable, production-quality ML code. | [Website](https://transferlab.ai/trainings/beyond-jupyter/) |
| **Awesome Quant**    | Curated list of quantitative finance libraries and resources (many statistical/TS tools overlap with econometrics). | [Website](https://wilsonfreitas.github.io/awesome-quant/)                                          |

---

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` file (linked via the badge at the top) in the repository root for details on how to:

1.  **Add new packages**: Submit a pull request with relevant, well-maintained packages.
2.  **Improve documentation**: Help enhance descriptions, add examples, or correct errors.
3.  **Report issues**: Notify us about broken links, outdated information, or suggest new categories.

*Last updated: June 2025*
