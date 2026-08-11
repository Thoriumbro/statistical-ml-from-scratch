import numpy as np
import matplotlib.pyplot as plt

# Load Dataset

dataset = np.load("mnist.npz")

xTrain = dataset["x_train"]
yTrain = dataset["y_train"]

xTest = dataset["x_test"]
yTest = dataset["y_test"]

# Filter digits 4,9 

filterTrain = (yTrain == 4) | (yTrain == 9)
xTrain = xTrain[filterTrain]
yTrain = yTrain[filterTrain]

filterTest = (yTest == 4) | (yTest == 9)
xTest = xTest[filterTest]
yTest = yTest[filterTest]

yTrain = np.where(yTrain == 4, -1, 1)
yTest = np.where(yTest == 4, -1, 1)

# Flatten images 

xTrain = xTrain.reshape(len(xTrain), -1)
xTest = xTest.reshape(len(xTest), -1)

# normalize
xTrain = xTrain / 255.0
xTest = xTest / 255.0

idx_class1 = []
idx_class2 = []

for i in range(len(yTrain)):
    if (yTrain[i] == -1):
        idx_class1.append(i)
    else:
        idx_class2.append(i)

val_class1 = np.random.choice(idx_class1, 1000, replace=False)
val_class2 = np.random.choice(idx_class2, 1000, replace=False)

val_idx = list(val_class1) + list(val_class2)

train_idx = []
for i in range(len(yTrain)):
    if (i not in val_idx):
        train_idx.append(i)

val_idx = np.array(val_idx)
train_idx = np.array(train_idx)

xVal = xTrain[val_idx]
yVal = yTrain[val_idx]

xTrain_new = xTrain[train_idx]
yTrain_new = yTrain[train_idx]

# PCA 

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


Up, pmean, m = PCA_pipeline(xTrain_new, m=5)

# training projection
xTrain_new = xTrain_new.T
xTrain_new = xTrain_new - pmean.reshape(pmean.shape[0], 1)
xTrain_new = np.matmul(Up.T, xTrain_new).T

# test projection
xTest = xTest.T
xTest = xTest - pmean.reshape(pmean.shape[0], 1)
xTest = np.matmul(Up.T, xTest).T

xVal = xVal.T
xVal = xVal - pmean.reshape(pmean.shape[0], 1)
xVal = np.matmul(Up.T, xVal).T

# DECISION STUMP 

def trainTree(xTrain, yTrain):
    best_ssr = float('inf')

    for j in range(xTrain.shape[1]):
        xTrain_1d = xTrain[:, j]

        order = np.argsort(xTrain_1d)
        x_sorted = xTrain_1d[order]
        y_sorted = yTrain[order].astype(float)

        n = len(y_sorted)

        prefix_sum = [0] * n
        prefix_sq = [0] * n

        prefix_sum[0] = y_sorted[0]
        prefix_sq[0] = y_sorted[0] ** 2

        for i in range(1, n):
            prefix_sum[i] = prefix_sum[i-1] + y_sorted[i]
            prefix_sq[i] = prefix_sq[i-1] + y_sorted[i] ** 2

        total_sum = prefix_sum[n-1]
        total_sq = prefix_sq[n-1]

        for k in range(n - 1):
            threshold = (x_sorted[k] + x_sorted[k + 1]) / 2

            left_size = k + 1
            right_size = n - left_size

            left_sum = prefix_sum[k]
            left_sqsum = prefix_sq[k]

            right_sum = total_sum - left_sum
            right_sqsum = total_sq - left_sqsum

            ssr_l = left_sqsum - (left_sum * left_sum) / left_size
            ssr_r = right_sqsum - (right_sum * right_sum) / right_size

            ssr = ssr_l + ssr_r

            if (ssr < best_ssr):
                best_ssr = ssr
                best_feature = j
                best_threshold = threshold
                best_left_mean = left_sum / left_size
                best_right_mean = right_sum / right_size

    return best_feature, best_threshold, best_left_mean, best_right_mean


def predictTree(xDataset, feature, threshold, left_mean, right_mean):
    return np.where(xDataset[:, feature] <= threshold, left_mean, right_mean)

# GRADIENT BOOSTING 

learnRates = [0.001, 0.01, 0.1, 0.2, 0.5, 1]

for learnRate in learnRates:
    train_preds = np.zeros(len(xTrain_new))
    val_preds = np.zeros(len(xVal))
    test_preds = np.zeros(len(xTest))

    val_mse_list = []

    features = []
    thresholds = []
    left_means = []
    right_means = []

    print("\nLearning rate =", learnRate)

    for t in range(300):
        residuals = yTrain_new - train_preds

        feature, threshold, left_mean, right_mean = trainTree(xTrain_new, residuals)

        features.append(feature)
        thresholds.append(threshold)
        left_means.append(left_mean)
        right_means.append(right_mean)

        pred_train = predictTree(xTrain_new, feature, threshold, left_mean, right_mean)
        pred_val = predictTree(xVal, feature, threshold, left_mean, right_mean)
        pred_test = predictTree(xTest, feature, threshold, left_mean, right_mean)

        train_preds += learnRate * pred_train
        val_preds += learnRate * pred_val
        test_preds += learnRate * pred_test

        val_mse = np.mean((yVal - val_preds) ** 2)
        val_mse_list.append(val_mse)

    best_iter = np.argmin(val_mse_list)

    best_val_mse = val_mse_list[best_iter]
    test_preds_best = np.zeros(len(xTest))

    for t in range(best_iter + 1):
        preds = predictTree(xTest, features[t], thresholds[t], left_means[t], right_means[t])
        test_preds_best += learnRate * preds

    test_mse = np.mean((yTest - test_preds_best) ** 2)

    print("Best TREE:", best_iter + 1)
    print("Validation MSE:", best_val_mse)
    print("Test MSE:", test_mse)

    # plot

    plt.plot(range(1, len(val_mse_list)+1), val_mse_list, color = "black")
    plt.xlabel("Number of Trees")
    plt.ylabel("Validation MSE")
    plt.title(f"Validation MSE vs Trees (learning Rate={learnRate})")
    plt.show()
