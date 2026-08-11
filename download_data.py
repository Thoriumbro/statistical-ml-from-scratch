"""
Downloads MNIST and FashionMNIST datasets used across this project.
Run this once before executing any script in the subfolders.
"""
import urllib.request
import os

os.makedirs("data", exist_ok=True)

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
mnist_path = "data/mnist.npz"

if not os.path.exists(mnist_path):
    print("Downloading MNIST...")
    urllib.request.urlretrieve(MNIST_URL, mnist_path)
    print("Saved to", mnist_path)
else:
    print("MNIST already present.")

print("\nFor FashionMNIST, download manually from:")
print("https://www.kaggle.com/datasets/zalando-research/fashionmnist/data")
print("and place it in the data/ folder.")

print("\nFor raw MNIST idx-ubyte files (used in folders 01/02), download from:")
print("https://www.kaggle.com/datasets/hojjatk/mnist-dataset")
print("and place them in the data/ folder.")