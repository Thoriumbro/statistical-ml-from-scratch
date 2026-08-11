import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

# ---------------- Load Dataset ----------------

xTrain = idx2numpy.convert_from_file("train-images-idx3-ubyte")
yTrain = idx2numpy.convert_from_file("train-labels-idx1-ubyte")

xTest = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
yTest = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")

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

# ---------------- DECISION STUMP ----------------

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


def testTree(xDataset, yDataset, feature, threshold, left_mean, right_mean):
    preds = []

    for x in xDataset:
        if (x[feature] <= threshold):
            preds.append(left_mean)
        else:
            preds.append(right_mean)

    preds = np.array(preds)
    return np.mean((yDataset - preds) ** 2)


# ---------------- SINGLE STUMP ----------------

f_single, t_single, mL_single, mR_single = trainTree(xTrain, yTrain)

single_mse = testTree(xTest, yTest, f_single, t_single, mL_single, mR_single)

print("Single Stump MSE:", single_mse)


def randomDataset(xTrain, yTrain):
    n = xTrain.shape[0]

    indices = np.random.choice(n, size=n, replace=True)

    x_boot = xTrain[indices]
    y_boot = yTrain[indices]

    return x_boot, y_boot, indices

# ---------------- BAGGING ----------------

models = []

for i in range(5):
    x_boot, y_boot, _ = randomDataset(xTrain, yTrain)

    f, t, mL, mR = trainTree(x_boot, y_boot)

    models.append((f, t, mL, mR))


bag_preds = []

for x in xTest:
    total = 0

    for (f, t, mL, mR) in models:
        if (x[f] <= t):
            total += mL
        else:
            total += mR

    bag_preds.append(total / 5)

bag_preds = np.array(bag_preds)

bagging_mse = np.mean((yTest - bag_preds) ** 2)

print("Bagging Test MSE:", bagging_mse)


# ---------------- OOB ERROR ----------------

oob_errors = []
n = xTrain.shape[0]

for i in range(5):
    x_boot, y_boot, indices = randomDataset(xTrain, yTrain)

    f, t, mL, mR = trainTree(x_boot, y_boot)

    boot_idx = set(indices)
    oob_idx = [i for i in range(n) if (i not in boot_idx)]

    if (len(oob_idx) > 0):
        x_oob = xTrain[oob_idx]
        y_oob = yTrain[oob_idx]

        preds = []

        for x in x_oob:
            if (x[f] <= t):
                preds.append(mL)
            else:
                preds.append(mR)

        preds = np.array(preds)

        mse = np.mean((y_oob - preds) ** 2)
        oob_errors.append(mse)

print("Average OOB Error:", np.mean(oob_errors))

x_axis = xTest[:, 0]

y_true = yTest
y_stump = []
y_bag = bag_preds

for x in xTest:
    if (x[f_single] <= t_single):
        y_stump.append(mL_single)
    else:
        y_stump.append(mR_single)

plt.scatter(x_axis, y_true, label="True", alpha=0.4)
plt.scatter(x_axis, y_stump, label="Stump", alpha=0.4)
plt.scatter(x_axis, y_bag, label="Bagging", alpha=0.4)

plt.legend()
plt.title("Decision Stump vs Bagging")
plt.xlabel("First PCA Feature")
plt.ylabel("Output")
plt.show()