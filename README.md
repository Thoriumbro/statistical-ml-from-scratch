# Statistical ML Toolkit: Classification & Ensemble Methods from Scratch

Implementation of core statistical machine learning algorithms from first
principles — generative classifiers, dimensionality reduction, linear
models, tree-based ensembles, and boosting — built primarily with NumPy
(scikit-learn used only where explicitly required, e.g. Lasso) and
evaluated on MNIST and FashionMNIST.

## Setup

```bash
git clone https://github.com/<your-username>/statistical-ml-from-scratch.git
cd statistical-ml-from-scratch
pip install -r requirements.txt
python download_data.py
```

## Structure

```
.
├── 01-mle-discriminant-analysis/
├── 02-pca-fda-classification/
├── 03-linear-models-and-trees/
├── 04-boosting-and-perceptron/
├── download_data.py
├── requirements.txt
└── README.md
```

## 1. MLE & Discriminant Analysis
`01-mle-discriminant-analysis/`

Estimated per-class mean and covariance matrices via Maximum Likelihood
Estimation (Gaussian assumption) on MNIST digits 0, 1, 2, then classified
using LDA (shared covariance) and QDA (per-class covariance). Class
separability was visualized with t-SNE prior to classification.

| Model | Test Accuracy |
|---|---|
| LDA | 81.0% |
| QDA | 99.3% |

QDA's per-class covariance modeling substantially outperformed LDA's shared-covariance assumption on this data.

## 2. PCA, FDA & Classification
`02-pca-fda-classification/`

Implemented PCA from scratch (covariance eigendecomposition) and Fisher's
Discriminant Analysis (explicit between-class/within-class scatter matrices,
generalized eigenvalue problem) for supervised dimensionality reduction,
then classified with LDA/QDA.

| Method | Train Acc | Test Acc |
|---|---|---|
| PCA (75% var) + LDA | 97.0% | 97.0% |
| PCA (90% var) + LDA | 98.0% | 97.0% |
| PCA (2 components) + LDA | 93.0% | 91.7% |
| PCA (75% var) + QDA | 98.7% | 95.7% |
| PCA (90% var) + QDA | 99.3% | 90.7% |
| PCA (2 components) + QDA | 94.0% | 93.0% |
| FDA + LDA | 100% | 88.0% |
| FDA + QDA | 100% | 88.0% |

PCA reconstruction achieved an average MSE of **0.062** across sampled test
images. FDA produced perfect training separation but generalized worse than
PCA+LDA — a supervised, low-dimensional projection can overfit small sample
sizes more easily than an unsupervised one.

## 3. Linear Models, Trees, Bagging & Random Forest
`03-linear-models-and-trees/`

Ridge and Lasso regression for multi-class classification (one-hot
targets), evaluated across λ with regularization-path and sparsity
analysis. Decision trees (Gini-based, 3 terminal nodes), Bagging, and
Random Forest built from scratch and compared on classification accuracy
and Out-of-Bag (OOB) error. A regression decision stump + bagged stumps
were also evaluated on FashionMNIST.

| Model | Test Accuracy | OOB Error |
|---|---|---|
| Ridge (best λ = 10) | 96.1% | — |
| Lasso (best λ = 0.0001) | 96.4% | — |
| Single Decision Tree | 78.2% | — |
| Bagging (5 trees) | 77.7% | 21.6% |
| Random Forest (k=3) | 76.0% | 26.2% |

Regression stump comparison (FashionMNIST): single stump MSE = **0.363**,
bagged stumps MSE = **0.359**, average OOB error = **0.353** — bagging
smooths piecewise-constant predictions and reduces variance.

Random Forest underperformed plain bagging here, likely due to shallow
trees (3 terminal nodes) and a small ensemble (5 trees) limiting the
benefit of feature randomization.

## 4. Boosting & Perceptron
`04-boosting-and-perceptron/`

AdaBoost and Gradient Boosting implemented from scratch with 300
sequential decision stumps on binary MNIST classification (digits 4 vs 9).
The Rosenblatt Perceptron was implemented from scratch and tested on two
synthetic 2D Gaussian datasets — one well-separated, one overlapping.

**AdaBoost** — best at 25 trees: train accuracy 80.5%, validation accuracy
80.6%, test accuracy **81.4%**. Accuracy rises sharply in early iterations
and plateaus, showing diminishing returns from additional stumps.

**Gradient Boosting** — swept learning rates η ∈ {0.001, 0.01, 0.1, 0.2,
0.5, 1}. Lower rates converge slowly but stably; higher rates converge fast
but risk overfitting. η ≈ 0.1–0.2 gave the best convergence/generalization
tradeoff.

**Perceptron:**

| Dataset | Covariance | Accuracy | Converged? |
|---|---|---|---|
| A (well-separated) | I | 100% | Yes |
| B (overlapping) | 3I | 99.2% | No |

Dataset A converged in a small number of epochs since the classes are
linearly separable; Dataset B never fully converged due to class overlap,
though it still reached high accuracy.

## Tech Stack
Python · NumPy · Pandas · Matplotlib · Scikit-learn (Lasso only)