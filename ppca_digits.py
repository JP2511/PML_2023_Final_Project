import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from keras.datasets import mnist
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from torchmetrics import StructuralSimilarityIndexMeasure


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
train_y = mnist_trainset.targets
train_X = torch.reshape(mnist_trainset.data, (60000, 28*28))


print(train_y.shape)

pca = PCA(n_components=2)
trans = pca.fit_transform(train_X)  #transform data

# scatter plot from the resulting PCA values to compare it to the solution implementation
scatter_x = trans[:,0]
scatter_y = trans[:,1]

fig, axs = plt.subplots(10, 10, figsize=(10, 10), sharex = True, sharey = True)

print(train_y==0)

for i in range(10):
  mask = train_y==i
  poi = trans[mask]
  rev = pca.inverse_transform(poi[:10])
  for j in range(10):
    axs[i,j].imshow(rev[j, :].reshape((28,28)))

plt.show()
  