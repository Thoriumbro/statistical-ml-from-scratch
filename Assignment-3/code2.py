import numpy as np
import math
import matplotlib.pyplot as plt

# Load the datasetset
dataset = np.load("mnist.npz")

# Extract arrays
xTrain = dataset["x_train"]  
yTrain = dataset["y_train"]

xTest = dataset["x_test"]    
yTest = dataset["y_test"]

# filter training data
filterTrain = (yTrain <= 2)

xTrain = xTrain[filterTrain]
yTrain = yTrain[filterTrain]

# filter test data
filterTest = (yTest <= 2)

xTest = xTest[filterTest]
yTest = yTest[filterTest]

xTrain = xTrain.transpose(0, 2, 1).reshape(len(xTrain), -1)
xTest  = xTest.transpose(0, 2, 1).reshape(len(xTest), -1)

xTrain = xTrain / 255.0
xTest  = xTest / 255.0

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

# ---- training projection ----
xTrain = xTrain.T
xTrain = xTrain - pmean.reshape(pmean.shape[0], 1)

xTrain_pca = np.matmul(Up.T, xTrain)

# ---- test projection ----
xTest = xTest.T
xTest = xTest - pmean.reshape(pmean.shape[0], 1)

xTest_pca  = np.matmul(Up.T, xTest)

xTrain = xTrain_pca.T
xTest  = xTest_pca.T

# ---------------- Gini ----------------
def gini(y):
    n = len(y)
    g = 1
    
    for val in set(y):
        count = 0
        for x in y:
            if x == val:
                count += 1
        
        p = count / n
        g = g - p * p
    
    return g

def weighted_gini(y1, y2):
    n = len(y1) + len(y2)

    w1 = len(y1) / n
    w2 = len(y2) / n

    g1 = gini(y1)
    g2 = gini(y2)

    return w1 * g1 + w2 * g2

# ---------------- Best Split ----------------

def best_split(X, y, feature_subset=None):
    best_feature = None
    best_thresh = None
    best_score = float('inf')

    if(feature_subset is None):
        features = range(len(X[0]))
    else:
        features = feature_subset

    for j in features:
        vals = []
        for i in range(len(X)):
            vals.append(X[i][j])

        thresh = np.median(vals)

        left = []
        right = []

        for i in range(len(vals)):
            if(vals[i] <= thresh):
                left.append(y[i])
            else:
                right.append(y[i])

        if(len(left) == 0 or len(right) == 0):
            continue

        score = weighted_gini(left, right)

        if score < best_score:
            best_score = score
            best_feature = j
            best_thresh = thresh

    return best_feature, best_thresh

# ---------------- Train Tree ----------------

def majority(arr):
    freq = {}
    for val in arr:
        if val in freq:
            freq[val] += 1
        else:
            freq[val] = 1

    max_count = -1
    ans = None
    for key in freq:
        if(freq[key] > max_count):
            max_count = freq[key]
            ans = key

    return ans

def train_tree(X, y, k=None):
    p = len(X[0])

    f1, t1 = best_split(X, y)

    X_left, y_left = [], []
    X_right, y_right = [], []

    for i in range(len(X)):
        row = X[i]
        label = y[i]

        if(row[f1] <= t1):
            X_left.append(row)
            y_left.append(label)
        else:
            X_right.append(row)
            y_right.append(label)

    if(gini(y_left) > gini(y_right)):
        split_side = "left"
        X_split, y_split = X_left[:], y_left[:]
    else:
        split_side = "right"
        X_split, y_split = X_right[:], y_right[:]


    if(k is not None):
        features = np.random.choice(p, k, replace=False)
    else:
        features = None

    f2, t2 = best_split(X_split, y_split, features)

    if(split_side == "left"):
        y_l1, y_l2 = [], []

        for i in range(len(X_left)):
            if(X_left[i][f2] <= t2):
                y_l1.append(y_left[i])
            else:
                y_l2.append(y_left[i])

        leaf1 = majority(y_l1)
        leaf2 = majority(y_l2)
        leaf3 = majority(y_right[:])

    else:
        y_r1, y_r2 = [], []

        for i in range(len(X_right)):
            if(X_right[i][f2] <= t2):
                y_r1.append(y_right[i])
            else:
                y_r2.append(y_right[i])

        leaf1 = majority(y_left[:])
        leaf2 = majority(y_r1)
        leaf3 = majority(y_r2)

    return {
        "f1": f1, "t1": t1,
        "f2": f2, "t2": t2,
        "split_side": split_side,
        "leaf1": leaf1,
        "leaf2": leaf2,
        "leaf3": leaf3
    }

# ---------------- Prediction ----------------

def predict_tree(model, x):
    f1, t1 = model["f1"], model["t1"]
    f2, t2 = model["f2"], model["t2"]

    if(x[f1] <= t1):
        if(model["split_side"] != "left"):
            return model["leaf1"]
        if(x[f2] <= t2):
            return model["leaf1"]
        return model["leaf2"]

    else:
        if(model["split_side"] != "right"):
            return model["leaf3"]
        if(x[f2] <= t2):
            return model["leaf2"]
        return model["leaf3"]

# ---------------- SINGLE TREE ----------------

tree = train_tree(xTrain, yTrain)

y_pred = []
for i in range(len(xTest)):
    pred = predict_tree(tree, xTest[i])
    y_pred.append(pred)

correct = 0
for i in range(len(yTest)):
    if y_pred[i] == yTest[i]:
        correct += 1

accuracy = correct / len(yTest)

print("\n--- Single Tree ---")
print("Accuracy:", accuracy)

labels = [0, 1, 2]

for c in labels:
    correct = 0
    total = 0

    for i in range(len(yTest)):
        if(yTest[i] == c):
            total += 1
            if(y_pred[i] == yTest[i]):
                correct += 1

    if(total > 0):
        print("Class", c, "Accuracy:", correct / total)
    else:
        print("Class", c, "Accuracy: 0")

# ---------------- BAGGING ----------------

def bootstrap(X, y):
    n = len(X)

    idx = np.random.choice(n, n, replace=True)

    Xb = []
    yb = []
    for i in range(len(idx)):
        Xb.append(X[idx[i]])
        yb.append(y[idx[i]])

    oob = []
    for i in range(n):
        if i not in idx:
            oob.append(i)

    return Xb, yb, oob


trees = []
oob_errors = []

for _ in range(5):
    Xb, yb, oob_idx = bootstrap(xTrain, yTrain)

    model = train_tree(Xb, yb)
    trees.append(model)

    if(len(oob_idx) > 0):
        preds = []

        for i in range(len(oob_idx)):
            idx = oob_idx[i]
            pred = predict_tree(model, xTrain[idx])
            preds.append(pred)

        wrong = 0
        for i in range(len(oob_idx)):
            idx = oob_idx[i]
            if(preds[i] != yTrain[idx]):
                wrong += 1

        err = wrong / len(oob_idx)
        oob_errors.append(err)

print("\n--- Bagging ---")

total = 0
for e in oob_errors:
    total += e

if(len(oob_errors) > 0):
    avg_error = total / len(oob_errors)
else:
    avg_error = 0

print("Average OOB Error:", avg_error)

def bagging_predict(x):
    preds = []

    for i in range(len(trees)):
        pred = predict_tree(trees[i], x)
        preds.append(pred)

    freq = {}
    for val in preds:
        if(val in freq):
            freq[val] += 1
        else:
            freq[val] = 1

    max_count = -1
    ans = None
    for key in freq:
        if(freq[key] > max_count):
            max_count = freq[key]
            ans = key

    return ans

y_pred = []
for i in range(len(xTest)):
    pred = bagging_predict(xTest[i])
    y_pred.append(pred)

correct = 0
for i in range(len(yTest)):
    if y_pred[i] == yTest[i]:
        correct += 1

accuracy = correct / len(yTest)
print("Bagging Accuracy:", accuracy)


for c in labels:
    correct = 0
    total = 0

    for i in range(len(yTest)):
        if yTest[i] == c:
            total += 1
            if y_pred[i] == yTest[i]:
                correct += 1

    if total > 0:
        print("Class", c, "Accuracy:", correct / total)
    else:
        print("Class", c, "Accuracy: 0")

# ---------------- RANDOM FOREST ----------------

rf_trees = []
rf_oob_errors = []
k = 3

for _ in range(5):
    Xb, yb, oob_idx = bootstrap(xTrain, yTrain)

    model = train_tree(Xb, yb, k)
    rf_trees.append(model)

    if len(oob_idx) > 0:
        wrong = 0

        for i in range(len(oob_idx)):
            idx = oob_idx[i]
            pred = predict_tree(model, xTrain[idx])
            if pred != yTrain[idx]:
                wrong += 1

        err = wrong / len(oob_idx)
        rf_oob_errors.append(err)

total = 0
for e in rf_oob_errors:
    total += e

print("\n--- Random Forest ---")
print("OOB Error:", total / len(rf_oob_errors))


def rf_predict(x):
    preds = []
    for t in rf_trees:
        preds.append(predict_tree(t, x))
    return majority(preds)
