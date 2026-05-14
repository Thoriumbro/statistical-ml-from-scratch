import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# ---------------- Load Dataset ----------------

dataset = np.load("mnist.npz")

xTrain = dataset["x_train"]
yTrain = dataset["y_train"]

xTest = dataset["x_test"]
yTest = dataset["y_test"]

# ---------------- Filter digits 0,1,2 ----------------

filterTrain = (yTrain <= 2)
xTrain = xTrain[filterTrain]
yTrain = yTrain[filterTrain]

filterTest = (yTest <= 2)
xTest = xTest[filterTest]
yTest = yTest[filterTest]

# ---------------- Flatten images ----------------

xTrain = xTrain.reshape(len(xTrain), -1)
xTest = xTest.reshape(len(xTest), -1)

# normalize
xTrain = xTrain / 255.0
xTest = xTest / 255.0

xTrain_original = xTrain.copy()
xTest_original = xTest.copy()

# ---------------- PCA ----------------

def PCAmean(X):
    rows = X.shape[0]
    cols = X.shape[1]

    meanVector = np.zeros(rows)

    for i in range(rows):
        total = 0
        for j in range(cols):
            total += X[i][j]

        meanVector[i] = total / cols

    return meanVector


def applyPCA(X, eigenval, eigenvec, m=None, var_retain=None):
    if m is None:
        total_var = np.sum(eigenval)
        target_var = var_retain * total_var

        sum_var = 0
        m = 0

        for i in range(len(eigenval)):
            sum_var += eigenval[i]
            m += 1

            if sum_var >= target_var:
                break

    Up = eigenvec[:, :m]

    return Up, m


def PCA_pipeline(xTrain, m=None, var_retain=None):
    xPCA = xTrain.T
    N = xPCA.shape[1]

    pmean = PCAmean(xPCA)
    xPCA = xPCA - pmean.reshape(xPCA.shape[0], 1)

    S = np.matmul(xPCA, xPCA.T) / (N - 1)

    eigenval, eigenvec = np.linalg.eigh(S)

    idx = np.argsort(eigenval)[::-1]
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    Up, m = applyPCA(xPCA, eigenval, eigenvec, m, var_retain)

    return Up, pmean, m


Up, pmean, m = PCA_pipeline(xTrain, m=10)

# training projection
xTrain = xTrain.T
xTrain = xTrain - pmean.reshape(pmean.shape[0], 1)
xTrain = np.matmul(Up.T, xTrain).T

# test projection
xTest = xTest.T
xTest = xTest - pmean.reshape(pmean.shape[0], 1)
xTest = np.matmul(Up.T, xTest).T

# ---------------- Ridge Regression ----------------

def addBias(xDataset):
    ones_train = np.ones((xDataset.shape[0], 1))
    xDataset = np.hstack((ones_train, xDataset))
    return xDataset

def RidgeRegression(xDataset, yDataset, lamdaVal):
    xDataset = addBias(xDataset)

    identityMat = np.eye(xDataset.shape[1])

    first = np.matmul(xDataset.T, xDataset) + lamdaVal * identityMat
    second = np.matmul(xDataset.T, yDataset)

    w = np.linalg.solve(first, second)

    return w

def oneHot(y):
    Y = np.zeros((y.shape[0], 3))
    for i in range(y.shape[0]):
        Y[i][y[i]] = 1
    return Y


def computeMSE(Ytrue, Ypred):
    return np.mean((Ytrue - Ypred) ** 2)

Ytrain = oneHot(yTrain)
Ytest = oneHot(yTest)

lamda = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10, 100]

train_mse = []
test_mse = []
ridge_weights = []

Xtrain_bias = addBias(xTrain)
Xtest_bias = addBias(xTest)

# ---------------- Ridge Training ----------------
for lam in lamda:
    w = RidgeRegression(xTrain, Ytrain, lam)

    ridge_weights.append(w[1:, :])   

    Ytrain_pred = np.matmul(Xtrain_bias, w)
    Ytest_pred = np.matmul(Xtest_bias, w)

    train_mse.append(computeMSE(Ytrain, Ytrain_pred))
    test_mse.append(computeMSE(Ytest, Ytest_pred))

# ---------------- Plot MSE ----------------

plt.plot(lamda, train_mse, label="Training MSE")
plt.plot(lamda, test_mse, label="Test MSE")

plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.title("Ridge Regression: MSE vs Lambda")
plt.legend()
plt.show()

# ---------------- Ridge Regularization Path ----------------

ridge_weights = np.array(ridge_weights)

plt.figure()

for i in range(ridge_weights.shape[1]):  
    plt.plot(lamda, ridge_weights[:, i, 1])

plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Coefficient Value")
plt.title("Ridge Regularization Path (Class 1)")
plt.show()


# ---------------- Lasso Regression ----------------

nonzero_counts = []
lasso_weights = []
train_mse_lasso = []
test_mse_lasso = []

for lam in lamda:
    W = []

    for k in range(3):
        yk = Ytrain[:, k]

        model = Lasso(alpha=lam, max_iter=10000)
        model.fit(xTrain, yk)

        W.append(model.coef_)

    W = np.array(W).T  

    lasso_weights.append(W)   

    nonzero_counts.append(np.sum(W != 0))

    Ytrain_pred = np.matmul(xTrain, W)
    Ytest_pred = np.matmul(xTest, W)

    train_mse_lasso.append(computeMSE(Ytrain, Ytrain_pred))
    test_mse_lasso.append(computeMSE(Ytest, Ytest_pred))

plt.figure()

plt.plot(lamda, train_mse_lasso, label="Training MSE (Lasso)")
plt.plot(lamda, test_mse_lasso, label="Test MSE (Lasso)")

plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.title("Lasso Regression: MSE vs Lambda")

plt.legend()
plt.show()

# ---------------- Lasso Sparsity Plot ----------------

plt.plot(lamda, nonzero_counts)

plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Number of Non-Zero Coefficients")
plt.title("Lasso Sparsity vs Lambda")
plt.show()

# ---------------- Lasso Regularization Path ----------------

lasso_weights = np.array(lasso_weights)

plt.figure()

for i in range(lasso_weights.shape[1]):
    plt.plot(lamda, lasso_weights[:, i, 1])

plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Coefficient Value")
plt.title("Lasso Regularization Path (Class 1)")
plt.show()


# ---------------- Find best lambda ----------------

best_lambda = lamda[np.argmin(test_mse)]
best_lambda_lasso = lamda[np.argmin(test_mse_lasso)]

# ---------------- PCA Dimensions ----------------
PCAval = [2, 5, 10, 20, 30]

train_errors = []
val_errors = []

PCA_info = []

for p in PCAval:
    Up, pmean, m = PCA_pipeline(xTrain_original, m=p)
    PCA_info.append([Up, pmean, m])

for Up, pmean, m in PCA_info:
    Xtrain = xTrain_original.T
    Xtrain = Xtrain - pmean.reshape(pmean.shape[0], 1)
    Xtrain = np.matmul(Up.T, Xtrain).T

    Xtest = xTest_original.T
    Xtest = Xtest - pmean.reshape(pmean.shape[0], 1)
    Xtest = np.matmul(Up.T, Xtest).T

    w = RidgeRegression(Xtrain, Ytrain, best_lambda)

    Xtrain_bias = addBias(Xtrain)
    Xtest_bias = addBias(Xtest)

    Ytrain_pred = np.matmul(Xtrain_bias, w)
    Ytest_pred = np.matmul(Xtest_bias, w)

    train_errors.append(computeMSE(Ytrain, Ytrain_pred))
    val_errors.append(computeMSE(Ytest, Ytest_pred))

# ---------------- Plot Model Complexity ----------------

plt.figure()

plt.plot(PCAval, train_errors, marker='o', label="Training MSE")
plt.plot(PCAval, val_errors, marker='o', label="Validation MSE")

plt.xlabel("Model Complexity (PCA Dimensions p)")
plt.ylabel("MSE")
plt.title("Training and Validation Error vs Model Complexity")

plt.legend()
plt.show()

# ---------------- Best PCA ----------------

best_p_index = np.argmin(val_errors)
best_p = PCAval[best_p_index]

print("Best PCA dimension:", best_p)
print("Best validation MSE:", val_errors[best_p_index])
print("Best ridge lambda:", best_lambda)
print("Best lasso lambda:", best_lambda_lasso)

# ---------------- Recompute PCA ----------------

Up, pmean, m = PCA_pipeline(xTrain_original, m=best_p)

Xtrain = xTrain_original.T
Xtrain = Xtrain - pmean.reshape(pmean.shape[0], 1)
Xtrain = np.matmul(Up.T, Xtrain).T

Xtest = xTest_original.T
Xtest = Xtest - pmean.reshape(pmean.shape[0], 1)
Xtest = np.matmul(Up.T, Xtest).T

# ---------------- Ridge Accuracy ----------------
w = RidgeRegression(Xtrain, Ytrain, best_lambda)

Xtest_bias = addBias(Xtest)
Ytest_pred = np.matmul(Xtest_bias, w)

y_pred_ridge = np.argmax(Ytest_pred, axis=1)
ridge_accuracy = np.mean(y_pred_ridge == yTest)

print("Ridge Test Classification Accuracy:", ridge_accuracy)


# ---------------- Lasso Accuracy ----------------

W = []

for k in range(3):
    yk = Ytrain[:, k]

    model = Lasso(alpha=best_lambda_lasso, max_iter=10000)
    model.fit(Xtrain, yk)

    W.append(model.coef_)

W = np.array(W).T

Ytest_pred_lasso = np.matmul(Xtest, W)
y_pred_lasso = np.argmax(Ytest_pred_lasso, axis=1)

lasso_accuracy = np.mean(y_pred_lasso == yTest)

print("Lasso Test Classification Accuracy:", lasso_accuracy)