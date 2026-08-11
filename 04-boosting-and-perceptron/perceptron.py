import numpy as np
import matplotlib.pyplot as plt

covA = np.identity(2)
meanA = [-3, -3]

covB = 3 * covA
meanB = [3, 3]

def split(x, y):
    idx = np.random.permutation(len(x))
    s = int(0.7*len(x))
    return x[idx[:s]], y[idx[:s]], x[idx[s:]], y[idx[s:]]

# -------- Perceptron --------

def train(x, y):
    w = np.zeros(2)
    b = 0
    err = []
    n = len(x)
    learnRate = 0.01

    for j in range(300):
        e = 0
        for i in range(n):
            percep_val = np.dot(w, x[i]) + b
            if ((y[i] * percep_val) <= 0):
                w += learnRate * y[i] *x[i]
                b += learnRate * y[i]
                e += 1

        err.append(e)
        if (e == 0): break

    return w, b, err

def acc(X, y, w, b):
    return np.mean(np.sign(np.matmul(X, w) + b) == y)

def generate_and_split(cov, check_non_sep=False):
    while True:
        X_neg = np.random.multivariate_normal(meanA, cov, 200)
        X_pos = np.random.multivariate_normal(meanB, cov, 200)

        X = np.vstack((X_neg, X_pos))
        y = np.concatenate((-np.ones(200), np.ones(200)))

        Xtr, ytr, Xte, yte = split(X, y)

        if (not check_non_sep):
            return Xtr, ytr, Xte, yte

        w, b, _ = train(Xtr, ytr)

        if (np.any(np.sign(np.matmul(Xtr, w) + b) != ytr)):
            return Xtr, ytr, Xte, yte

XA_train, yA_train, XA_test, yA_test = generate_and_split(covA)
XB_train, yB_train, XB_test, yB_test = generate_and_split(covB, True)

weightA, biasA, errorsA = train(XA_train, yA_train)
weightB, biasB, errorsB = train(XB_train, yB_train)

print("Dataset A Test Accuracy:", acc(XA_test, yA_test, weightA, biasA))
print("Dataset B Test Accuracy:", acc(XB_test, yB_test, weightB, biasB))


plt.figure()
plt.plot(errorsA, color = "black")
plt.title("Dataset A - Misclassifications per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Errors")
plt.show()

plt.figure()
plt.plot(errorsB, color = "black")
plt.title("Dataset B - Misclassifications per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Errors")
plt.show()


def plot_boundary(X_train, y_train, X_test, y_test, w, b, title):
    plt.figure()

    plt.scatter(X_train[y_train==-1][:,0], X_train[y_train==-1][:,1])
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1])

    plt.scatter(X_test[y_test==-1][:,0], X_test[y_test==-1][:,1], marker='x')
    plt.scatter(X_test[y_test==1][:,0], X_test[y_test==1][:,1], marker='x')

    x_vals = np.linspace(-10, 10, 100)

    if abs(w[1]) > 1e-6:
        y_vals = -(w[0]*x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, 'k--')

    plt.title(title)
    plt.show()

plot_boundary(XA_train, yA_train, XA_test, yA_test, weightA, biasA, "Dataset A Boundary")
plot_boundary(XB_train, yB_train, XB_test, yB_test, weightB, biasB, "Dataset B Boundary")