import numpy as np
import sklearn
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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


# -----------t-sne plots of Train and Test Set-------

tsne = TSNE(n_components=2, random_state=42)

# train set
train_2d = tsne.fit_transform(xTrain)

plt.figure()
plt.scatter(train_2d[:, 0], train_2d[:, 1], c=yTrain, cmap="plasma")
plt.title("t-SNE — Train Set")
plt.show()

# test set
test_2d = tsne.fit_transform(xTest)

plt.figure()
plt.scatter(test_2d[:, 0], test_2d[:, 1], c=yTest, cmap="plasma")
plt.title("t-SNE — Test Set")
plt.show()


# -----------MLE Estimates-------

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

# For each Digit Class (0, 1, 2)
LDAinfo = []

for digit in digits:
    vX = xTrain[yTrain == digit]

    mean = fmean(vX)
    cov = fcovariance(vX)

    LDAinfo.append((digit, mean, cov))


# -----------Applying LDA to the Test Data-------

def avgCovariance(LDAinfo):
    d = LDAinfo[0][2].shape[0]
    newMatrix = np.zeros((d, d))

    for _, _, cov in LDAinfo:
        newMatrix += cov

    newMatrix /= len(LDAinfo)

    # To prevent singular matrix
    np.fill_diagonal(newMatrix, newMatrix.diagonal() + 1e-5)

    return newMatrix

# LDA equation = (μᵀ Σ⁻¹) x - 1/2 (μᵀ Σ⁻¹ μ)
def LDA(x, LDAinfo, avgCov, show=False):
    trueClass = None
    maxLDAscore = -np.inf

    for digit, mean, _ in LDAinfo:
        # first term: μᵀ Σ⁻¹ x
        term1 = np.matmul(np.matmul(mean, invCov), x)

        # second term: μᵀ Σ⁻¹ μ
        term2 = -0.5 * np.matmul(np.matmul(mean, invCov), mean)

        LDAscore = term1 + term2

        if show:
            print(f"digit {digit}: {round(LDAscore,2)}")

        if(LDAscore > maxLDAscore):
            maxLDAscore = LDAscore
            trueClass = digit
    
    if show:
        print("LDA value = ", round(maxLDAscore,2))

    return trueClass


avgCov = avgCovariance(LDAinfo)
invCov = np.linalg.inv(avgCov)
correct = 0

for i in range(len(xTest)):
    pred = LDA(xTest[i], LDAinfo, avgCov)

    if(pred == yTest[i]):
        correct += 1

lda_correct = correct
lda_accuracy = correct / len(yTest)


# -----------Applying QDA to the Test Data-------

# QDA equation = 1/2 log |Σ| - 1/2 (x-μ)ᵀ Σ⁻¹ (x-μ)
def precompute_QDA(LDAinfo):
    QDAinfo = []

    for digit, mean, cov in LDAinfo:
        d = cov.shape[0]

        # To prevent singular matrix
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

        if show:
            print(f"digit {digit}: {round(QDAscore,2)}")

        if(QDAscore > maxScore):
            maxScore = QDAscore
            trueClass = digit

    if show:
        print("QDA max score:", round(maxScore, 2))

    return trueClass

QDAinfo = precompute_QDA(LDAinfo)
correct = 0

for i in range(len(xTest)):
    pred = QDA(xTest[i], QDAinfo)

    if(pred == yTest[i]):
        correct += 1

qda_correct = correct
qda_accuracy = correct / len(yTest)

# -----------Final Results-------

print("\n========== Classification Summary ==========")

print(f"\nLDA -> Correct: {lda_correct}/300")
print("LDA accuracy:", round(lda_accuracy, 3), f"({round(lda_accuracy*100,2)}% Correct)")
print(f"\nQDA -> Correct: {qda_correct}/300")
print("QDA accuracy:", round(qda_accuracy,3), f"({round(qda_accuracy*100,2)}% Correct)")

idx = int(input("\nEnter any index for LDA & QDA values(0–299): "))

if idx < 0 or idx >= len(xTest):
    print("Invalid index!")
else:
    print("\nRandom index:", idx)
    print("True class:", yTest[idx])

    print("\n--- LDA scores ---")
    LDA(xTest[idx], LDAinfo, avgCov, show=True)

    print("\n--- QDA scores ---")
    QDA(xTest[idx], QDAinfo, show=True)
