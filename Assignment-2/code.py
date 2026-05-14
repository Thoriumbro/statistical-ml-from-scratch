import numpy as np
import sklearn
import math
import matplotlib.pyplot as plt
import idx2numpy

# -----------Loading MNIST dataset-------

# load training data
xTrain = idx2numpy.convert_from_file("train-images.idx3-ubyte")
yTrain = idx2numpy.convert_from_file("train-labels.idx1-ubyte")

# load test data
xTest = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
yTest = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")


# -----------Filtering Only Digti Classes (0, 1, 2)-------

# filter training data
filterTrain = (yTrain <= 2)

xTrain = xTrain[filterTrain]
yTrain = yTrain[filterTrain]

# filter test data
filterTest = (yTest <= 2)

xTest = xTest[filterTest]
yTest = yTest[filterTest]


# -----------Converting Images into Feature Vectors-------

xTrain = xTrain.transpose(0, 2, 1).reshape(len(xTrain), -1)
xTest  = xTest.transpose(0, 2, 1).reshape(len(xTest), -1)

xTrain = xTrain / 255.0
xTest  = xTest / 255.0


# -----------Picking Random 100 Samples from each Class-------

digits = [0, 1, 2]

def randomSamples(X, y):
    xNew = []
    yNew = []

    for digit in digits:
        idx = []
        for i in range(len(y)):
            if(y[i] == digit):
                idx.append(i)

        random = np.random.choice(idx, 100, replace=False)

        for i in random:
            xNew.append(X[i])
            yNew.append(y[i])

    return np.array(xNew), np.array(yNew)

xTrain, yTrain = randomSamples(xTrain, yTrain)
xTest,  yTest  = randomSamples(xTest, yTest)

# # -----------MLE Estimates-------

# mean
def fmean(X):
    N = X.shape[0]
    dim = X.shape[1]

    meanVector = np.zeros(dim)

    for x in X:
        meanVector += x

    meanVector = meanVector / N

    return meanVector

# covariance matrix
def fcovariance(X):
    N = X.shape[0]
    d = X.shape[1]

    covMatrix = np.zeros((d, d))
    meanVector = fmean(X)

    for x in X:
        diff = (x - meanVector).reshape(d, 1)
        covMatrix += np.matmul(diff, diff.T)

    covMatrix = covMatrix / N

    return covMatrix

# -----------Applying LDA to the Test Data-------

def avgCovariance(LDAinfo):
    d = LDAinfo[0][2].shape[0]
    newMatrix = np.zeros((d, d))

    for _, _, cov in LDAinfo:
        newMatrix += cov

    newMatrix /= len(LDAinfo)

    np.fill_diagonal(newMatrix, newMatrix.diagonal() + 1e-5)

    return newMatrix

# LDA equation = (μᵀ Σ⁻¹) x - 1/2 (μᵀ Σ⁻¹ μ)
def LDA(x, LDAinfo, avgCov):
    trueClass = None
    invCov = np.linalg.inv(avgCov)
    maxLDAscore = -np.inf

    for digit, mean, _ in LDAinfo:
        # first term: μᵀ Σ⁻¹ x
        term1 = np.matmul(np.matmul(mean, invCov), x)

        # second term: μᵀ Σ⁻¹ μ
        term2 = -0.5 * np.matmul(np.matmul(mean, invCov), mean)

        LDAscore = term1 + term2

        if(LDAscore > maxLDAscore):
            maxLDAscore = LDAscore
            trueClass = digit
    

    return trueClass

# -----------Applying QDA to the Test Data-------

# QDA equation = 1/2 log |Σ| - 1/2 (x-μ)ᵀ Σ⁻¹ (x-μ)
def precompute_QDA(LDAinfo):
    QDAinfo = []

    for digit, mean, cov in LDAinfo:
        d = cov.shape[0]

        np.fill_diagonal(cov, cov.diagonal() + 1e-5)

        invCov = np.linalg.inv(cov)
        detCov = np.linalg.det(cov)

        detCov = max(detCov, 1e-12)
        detCov = math.log(detCov)

        QDAinfo.append((digit, mean, invCov, detCov))

    return QDAinfo

def QDA(x, QDAinfo, show=False):
    trueClass = None
    maxScore = -np.inf

    for digit, mean, invCov, logDet in QDAinfo:
        diff = x - mean

        # first term: -1/2 log |Σ|
        term1 = -0.5 * logDet

        # second term: -1/2 (x-μ)ᵀ Σ⁻¹ (x-μ)
        term2 = -0.5 * np.matmul(np.matmul(diff, invCov), diff)

        QDAscore = term1 + term2

        if(QDAscore > maxScore):
            maxScore = QDAscore
            trueClass = digit


    return trueClass

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

    Y = np.matmul(Up.T, X)

    return Y, Up, m

def PCA_pipeline(xTrain, m=None, var_retain=None):
    xPCA = xTrain.T

    pmean = PCAmean(xPCA)
    xPCA = xPCA - pmean.reshape(xPCA.shape[0], 1)

    S = np.matmul(xPCA, xPCA.T) / 299

    eigenval, eigenvec = np.linalg.eigh(S)

    idx = np.argsort(eigenval)[::-1]
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    Y, Up, m = applyPCA(xPCA, eigenval, eigenvec, m, var_retain)

    return Y, Up, pmean, m


# ---------------- FDA ----------------

def FDAmean(X):
    N = X.shape[0]
    dim = X.shape[1]

    meanVector = np.zeros(dim)

    for x in X:
        meanVector += x

    meanVector = meanVector / N

    return meanVector

def FDA_pipeline(xTrain, yTrain, digits):
    d = xTrain.shape[1]

    # Between-class scatter Sb
    Sb = np.zeros((d, d))
    ovr_mean = FDAmean(xTrain)

    for digit in digits:
        Xc = xTrain[yTrain == digit]
        N = Xc.shape[0]

        mean = FDAmean(Xc)

        diff = (mean - ovr_mean).reshape(d, 1)
        Sb += N * np.matmul(diff, diff.T)

    # Within-class scatter Sw
    Sw = np.zeros((d, d))

    for digit in digits:
        Xc = xTrain[yTrain == digit]
        mean = FDAmean(Xc)

        for x in Xc:
            diff = (x - mean).reshape(d, 1)
            Sw += np.matmul(diff, diff.T)

    np.fill_diagonal(Sw, Sw.diagonal() + 1e-5)

    # Generalized eigenvalue problem
    W = np.matmul(np.linalg.inv(Sw), Sb)
    eigval, eigvec = np.linalg.eig(W)

    idx = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, idx].real
    
    w = eigvec[:, :2]

    Y = np.matmul(xTrain, w)

    return Y, w

# ---------------- PCA + LDA CLASSIFICATION (75% variance retained) ----------------

print("\n========== PCA + LDA ==========")

yPCA_train, Up, pmean, m = PCA_pipeline(xTrain, var_retain=0.75)
xTrain_PCA = yPCA_train.T

# Train LDA on PCA 
LDAinfo = []

for digit in digits:
    vX = xTrain_PCA[yTrain == digit]

    mean = fmean(vX)
    cov  = fcovariance(vX)

    LDAinfo.append((digit, mean, cov))

avgCov = avgCovariance(LDAinfo)

# Train Accuracy 
correct = 0
for i in range(len(xTrain)):
    xtrain = xTrain[i]

    xtrain_centered = xtrain - pmean
    ytrain = np.matmul(Up.T, xtrain_centered)

    pred = LDA(ytrain, LDAinfo, avgCov)

    if (pred == yTrain[i]):
        correct += 1


lda_train_accuracy = correct / len(yTrain)
print("LDA Train Accuracy (75%) :", lda_train_accuracy)

# Test Accuracy
correct = 0
for i in range(len(xTest)):
    xtest = xTest[i]

    xtest_centered = xtest - pmean
    ytest = np.matmul(Up.T, xtest_centered)

    pred = LDA(ytest, LDAinfo, avgCov)

    if(pred == yTest[i]):
        correct += 1

lda_accuracy = correct / len(yTest)
print("LDA Test Accuracy (75%) :", lda_accuracy)


# ---------------- PCA + LDA CLASSIFICATION (90% variance retained) ----------------

yPCA_train, Up, pmean, m = PCA_pipeline(xTrain, var_retain=0.90)
xTrain_PCA = yPCA_train.T

LDAinfo = []

for digit in digits:
    vX = xTrain_PCA[yTrain == digit]

    mean = fmean(vX)
    cov  = fcovariance(vX)

    LDAinfo.append((digit, mean, cov))

avgCov = avgCovariance(LDAinfo)

# Train Accuracy 
correct = 0
for i in range(len(xTrain)):
    xtrain = xTrain[i]

    xtrain_centered = xtrain - pmean
    ytrain = np.matmul(Up.T, xtrain_centered)

    pred = LDA(ytrain, LDAinfo, avgCov)

    if (pred == yTrain[i]):
        correct += 1


lda_train_accuracy = correct / len(yTrain)
print("LDA Train Accuracy (90%) :", lda_train_accuracy)

correct = 0
for i in range(len(xTest)):
    xtest = xTest[i]

    xtest_centered = xtest - pmean
    ytest = np.matmul(Up.T, xtest_centered)

    pred = LDA(ytest, LDAinfo, avgCov)

    if (pred == yTest[i]):
        correct += 1

lda_accuracy = correct / len(yTest)
print("LDA Test Accuracy (90%) :", lda_accuracy)

# ---------------- PCA + LDA CLASSIFICATION (2 Principle Components) ----------------

yPCA_train, Up, pmean, m = PCA_pipeline(xTrain, m=2)
xTrain_PCA = yPCA_train.T

LDAinfo = []

for digit in digits:
    vX = xTrain_PCA[yTrain == digit]

    mean = fmean(vX)
    cov  = fcovariance(vX)

    LDAinfo.append((digit, mean, cov))

avgCov = avgCovariance(LDAinfo)

# Train Accuracy
correct = 0
for i in range(len(xTrain)):
    xtrain = xTrain[i]

    xtrain_centered = xtrain - pmean
    ytrain = np.matmul(Up.T, xtrain_centered)

    pred = LDA(ytrain, LDAinfo, avgCov)

    if (pred == yTrain[i]):
        correct += 1


lda_train_accuracy = correct / len(yTrain)
print("LDA Train Accuracy (2 PCs) :", lda_train_accuracy)

correct = 0
for i in range(len(xTest)):
    xtest = xTest[i]

    xtest_centered = xtest - pmean
    ytest = np.matmul(Up.T, xtest_centered)

    pred = LDA(ytest, LDAinfo, avgCov)

    if (pred == yTest[i]):
        correct += 1

lda_accuracy = correct / len(yTest)
print("LDA Test Accuracy (2 PCs) :", lda_accuracy)

# ---------------- PCA + QDA CLASSIFICATION (75% variance retained) ----------------

print("\n========== PCA + QDA ==========")

yPCA_train, Up, pmean, m = PCA_pipeline(xTrain, var_retain=0.75)
xTrain_PCA = yPCA_train.T

# Train QDA
LDAinfo = []

for digit in digits:
    vX = xTrain_PCA[yTrain == digit]

    mean = fmean(vX)
    cov  = fcovariance(vX)

    LDAinfo.append((digit, mean, cov))

QDAinfo = precompute_QDA(LDAinfo)

# Train Accuracy
correct = 0
for i in range(len(xTrain)):
    xtrain = xTrain[i]

    xtrain_centered = xtrain - pmean
    ytrain = np.matmul(Up.T, xtrain_centered)

    pred = QDA(ytrain, QDAinfo)

    if (pred == yTrain[i]):
        correct += 1

qda_train_accuracy = correct / len(yTrain)
print("QDA Train Accuracy (75%) :", qda_train_accuracy)

# ---------- Test Accuracy ----------

correct = 0
for i in range(len(xTest)):
    xtest = xTest[i]

    xtest_centered = xtest - pmean
    ytest = np.matmul(Up.T, xtest_centered)

    pred = QDA(ytest, QDAinfo)

    if (pred == yTest[i]):
        correct += 1

qda_test_accuracy = correct / len(yTest)
print("QDA Test Accuracy (75%) :", qda_test_accuracy)

# ---------------- PCA + QDA CLASSIFICATION (90% variance retained) ----------------

yPCA_train, Up, pmean, m = PCA_pipeline(xTrain, var_retain=0.90)
xTrain_PCA = yPCA_train.T

LDAinfo = []

for digit in digits:
    vX = xTrain_PCA[yTrain == digit]

    mean = fmean(vX)
    cov  = fcovariance(vX)

    LDAinfo.append((digit, mean, cov))

QDAinfo = precompute_QDA(LDAinfo)

# Train Accuracy
correct = 0
for i in range(len(xTrain)):
    xtrain = xTrain[i]

    xtrain_centered = xtrain - pmean
    ytrain = np.matmul(Up.T, xtrain_centered)

    pred = QDA(ytrain, QDAinfo)

    if (pred == yTrain[i]):
        correct += 1

qda_train_accuracy = correct / len(yTrain)
print("QDA Train Accuracy (90%) :", qda_train_accuracy)

# Test Accuracy
correct = 0
for i in range(len(xTest)):
    xtest = xTest[i]
    
    xtest_centered = xtest - pmean
    ytest = np.matmul(Up.T, xtest_centered)

    pred = QDA(ytest, QDAinfo)

    if (pred == yTest[i]):
        correct += 1

qda_test_accuracy = correct / len(yTest)

print("QDA Test Accuracy (90%) :", qda_test_accuracy)

# ---------------- PCA + QDA CLASSIFICATION (2 Principal Components) ----------------

yPCA_train, Up, pmean, m = PCA_pipeline(xTrain, m=2)
xTrain_PCA = yPCA_train.T

LDAinfo = []

for digit in digits:
    vX = xTrain_PCA[yTrain == digit]

    mean = fmean(vX)
    cov  = fcovariance(vX)

    LDAinfo.append((digit, mean, cov))

QDAinfo = precompute_QDA(LDAinfo)

# Train Accuracy
correct = 0
for i in range(len(xTrain)):
    xtrain = xTrain[i]

    xtrain_centered = xtrain - pmean
    ytrain = np.matmul(Up.T, xtrain_centered)

    pred = QDA(ytrain, QDAinfo)

    if (pred == yTrain[i]):
        correct += 1

qda_train_accuracy = correct / len(yTrain)
print("QDA Train Accuracy (2 PCs) :", qda_train_accuracy)

# Test Accuracy
correct = 0
for i in range(len(xTest)):
    xtest = xTest[i]

    xtest_centered = xtest - pmean
    ytest = np.matmul(Up.T, xtest_centered)

    pred = QDA(ytest, QDAinfo)

    if (pred == yTest[i]):
        correct += 1

qda_test_accuracy = correct / len(yTest)

print("QDA Test Accuracy (2 PCs) :", qda_test_accuracy)

# ---------------- PCA Reconstruction ----------------

print("\n========== PCA Reconstruction ==========")

total = 0

for i in range(5):
    x = xTest[i]

    x_centered = x - pmean
    y = np.matmul(Up.T, x_centered)

    x_recon = np.matmul(Up, y) + pmean

    mse = np.mean((x - x_recon) ** 2)

    total += mse

    print("Sample", i + 1, "MSE:", mse)

print("Average MSE:", total / 5)

# ---------------- FDA + LDA CLASSIFICATION ----------------

print("\n========== FDA Results ==========")

# Apply FDA
Ytrain_FDA, w = FDA_pipeline(xTrain, yTrain, digits)
Ytest_FDA = np.matmul(xTest, w)

# Train LDA 
LDAinfo_FDA = []

for digit in digits:
    vX = Ytrain_FDA[yTrain == digit]

    mean = fmean(vX)
    cov  = fcovariance(vX)

    LDAinfo_FDA.append((digit, mean, cov))


avgCov_FDA = avgCovariance(LDAinfo_FDA)

# LDA Train Accuracy
correct = 0
for i in range(len(Ytrain_FDA)):
    pred = LDA(Ytrain_FDA[i], LDAinfo_FDA, avgCov_FDA)

    if (pred == yTrain[i]):
        correct += 1

lda_train_accuracy = correct / len(yTrain)

# LDA Test Accuracy 
correct = 0
for i in range(len(Ytest_FDA)):
    pred = LDA(Ytest_FDA[i], LDAinfo_FDA, avgCov_FDA)

    if (pred == yTest[i]):
        correct += 1

lda_test_accuracy = correct / len(yTest)

# ---------------- FDA + qDA CLASSIFICATION ----------------

# Train QDA
QDAinfo_FDA = precompute_QDA(LDAinfo_FDA)

# QDA Train Accuracy
correct = 0
for i in range(len(Ytrain_FDA)):
    pred = QDA(Ytrain_FDA[i], QDAinfo_FDA)

    if (pred == yTrain[i]):
        correct += 1

qda_train_accuracy = correct / len(yTrain)

# QDA Test Accuracy
correct = 0
for i in range(len(Ytest_FDA)):

    pred = QDA(Ytest_FDA[i], QDAinfo_FDA)

    if pred == yTest[i]:
        correct += 1

qda_test_accuracy = correct / len(yTest)

print("LDA Train Accuracy :", lda_train_accuracy)
print("LDA Test Accuracy  :", lda_test_accuracy)

print("QDA Train Accuracy :", qda_train_accuracy)
print("QDA Test Accuracy  :", qda_test_accuracy)

# ---------- PCA 2D ----------

yviz, upviz, meanviz, m = PCA_pipeline(xTrain, var_retain=0.75)
xviz = yviz.T

plt.figure()

for digit in digits:
    pts = xviz[yTrain == digit]

    plt.scatter(pts[:,0], pts[:,1], label="Digit " + str(digit))

plt.title("PCA (75% variance retain)")
plt.legend()
plt.show()

# ---------- FDA 2D ----------

Ytrain_FDA_vis, w_vis = FDA_pipeline(xTrain, yTrain, digits)

plt.figure()

for digit in digits:
    pts = Ytrain_FDA_vis[yTrain == digit]

    plt.scatter(pts[:,0], pts[:,1], label="Digit " + str(digit))

plt.title("FDA (Train Set)")
plt.legend()
plt.show()

Ytest_FDA = np.matmul(xTest, w)

plt.figure()

for digit in digits:

    pts = Ytest_FDA[yTest == digit]

    plt.scatter(pts[:,0], pts[:,1], label="Digit " + str(digit))

plt.title("FDA (Test Set)")
plt.legend()
plt.show()