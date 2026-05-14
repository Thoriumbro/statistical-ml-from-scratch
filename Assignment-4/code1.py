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

#  Flatten images 

xTrain = xTrain.reshape(len(xTrain), -1)
xTest = xTest.reshape(len(xTest), -1)

# normalize
xTrain = xTrain / 255.0
xTest = xTest / 255.0

class1 = []
class2 = []

for i in range(len(yTrain)):
    if (yTrain[i] == -1):
        class1.append(i)
    else:
        class2.append(i)

val_class1 = np.random.choice(class1, 1000, replace=False)
val_class2 = np.random.choice(class2, 1000, replace=False)

val_idx = list(val_class1) + list(val_class2)

train_idx = []
for i in range(len(yTrain)):
    if (i not in val_idx):
        train_idx.append(i)

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)

xVal = xTrain[val_idx]
yVal = yTrain[val_idx]

xTrain_new = xTrain[train_idx]
yTrain_new = yTrain[train_idx]

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
    if (m is None):
        total_var = np.sum(eigenval)
        target_var = var_retain * total_var

        sum_var = 0
        m = 0

        for i in range(len(eigenval)):
            sum_var += eigenval[i]
            m += 1

            if (sum_var >= target_var):
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

N = len(xTrain_new)
w = np.ones(N) / N

def trainTree(X, y, w):
    n, d = X.shape
    best_error = float('inf')

    for j in range(d):
        x_col = X[:, j]
        idx = np.argsort(x_col)

        x_sorted = x_col[idx]
        y_sorted = y[idx]
        w_sorted = w[idx]

        w_class2 = np.where(y_sorted == 1, w_sorted, 0)
        w_class1 = np.where(y_sorted == -1, w_sorted, 0)

        cum_class2 = np.cumsum(w_class2)
        cum_class1 = np.cumsum(w_class1)

        total_class2 = cum_class2[-1]
        total_class1 = cum_class1[-1]

        err_left_class1 = cum_class2 + (total_class1 - cum_class1)
        err_left_class2 = cum_class1 + (total_class2 - cum_class2)

        for errors, left_pred in [(err_left_class1, -1), (err_left_class2, 1)]:
            i = np.argmin(errors)

            if (errors[i] < best_error):
                best_error = errors[i]
                best_feature = j
                best_threshold = x_sorted[i]
                best_left_pred = left_pred
                best_right_pred = -left_pred

    return best_feature, best_threshold, best_left_pred, best_right_pred


def testTree(X, y, feature, threshold, left_pred, right_pred, return_preds=False):
    n = len(X)
    preds = np.zeros(n)

    for i in range(n):
        if (X[i][feature] <= threshold):
            preds[i] = left_pred
        else:
            preds[i] = right_pred

    if (return_preds):
        return preds

    error = 0
    for i in range(n):
        if (preds[i] != y[i]):
            error += 1

    return error / n

features = []
thresholds = []
left_preds = []
right_preds = []
confidences = []
val_accuracies = []
val_final = np.zeros(len(xVal))

for t in range(300):
    feature, threshold, left_pred, right_pred = trainTree(xTrain_new, yTrain_new, w)


    preds = np.zeros(len(xTrain_new))
    for i in range(len(xTrain_new)):
        if (xTrain_new[i][feature] <= threshold):
            preds[i] = left_pred
        else:
            preds[i] = right_pred

    error = 0
    for i in range(len(w)):
        if (preds[i] != yTrain_new[i]):
            error += w[i]

    confidence = 0.5 * np.log((1 - error) / error)

    features.append(feature)
    thresholds.append(threshold)
    left_preds.append(left_pred)
    right_preds.append(right_pred)
    confidences.append(confidence)

    for i in range(len(w)):
        w[i] = w[i] * np.exp(-confidence * yTrain_new[i] * preds[i])

    w = w / np.sum(w)

    preds_val = np.zeros(len(xVal))
    for i in range(len(xVal)):
        if (xVal[i][feature] <= threshold):
            preds_val[i] = left_pred
        else:
            preds_val[i] = right_pred

    for i in range(len(val_final)):
        val_final[i] += confidence * preds_val[i]

    val_preds = np.sign(val_final)

    correct = 0
    for i in range(len(yVal)):
        if (val_preds[i] == yVal[i]):
            correct += 1

    val_acc = correct / len(yVal)
    val_accuracies.append(val_acc)

def predictBoost(X, upto=None):
    if (upto is None):
        upto = len(confidences)

    n = len(X)
    final = np.zeros(n)

    for t in range(upto):
        for i in range(n):
            if (X[i][features[t]] <= thresholds[t]):
                pred = left_preds[t]
            else:
                pred = right_preds[t]

            final[i] += confidences[t] * pred

    for i in range(n):
        if (final[i] > 0):
            final[i] = 1
        else:
            final[i] = -1

    return final

best_iter = np.argmax(val_accuracies)

train_preds = predictBoost(xTrain_new, best_iter + 1)
val_preds = predictBoost(xVal, best_iter + 1)
test_preds = predictBoost(xTest, best_iter + 1)

train_error = np.mean(train_preds != yTrain_new)
val_error = np.mean(val_preds != yVal)
test_error = np.mean(test_preds != yTest)

test_acc = np.mean(test_preds == yTest)

rand_trees = np.random.choice(range(300), 5, replace=False)

for t in rand_trees:
    train_preds = predictBoost(xTrain_new, t+1)
    val_preds = predictBoost(xVal, t+1)
    test_preds = predictBoost(xTest, t+1)

    train_acc = np.mean(train_preds == yTrain_new)
    val_acc = np.mean(val_preds == yVal)
    test_acc = np.mean(test_preds == yTest)

    print("\nTree:", t+1)
    print("Train Acc:", train_acc)
    print("Val Acc:", val_acc)
    print("Test Acc:", test_acc)

print("\nBEST TREE:", best_iter + 1)

print("Feature:", features[best_iter])
print("Threshold:", thresholds[best_iter])
print("Confidence:", confidences[best_iter])

train_preds = predictBoost(xTrain_new, best_iter + 1)
val_preds = predictBoost(xVal, best_iter + 1)
test_preds = predictBoost(xTest, best_iter + 1)

train_acc = np.mean(train_preds == yTrain_new)
val_acc = np.mean(val_preds == yVal)
test_acc = np.mean(test_preds == yTest)

print("Train Accuracy:", train_acc)
print("Validation Accuracy:", val_acc)
print("Test Accuracy:", test_acc)

# plot 

plt.plot(range(1, 300 + 1), val_accuracies, color='black')
plt.xlabel("Number of Trees")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs Trees")
plt.show()