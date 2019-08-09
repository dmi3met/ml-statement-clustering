# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float
# from pylab import imshow


def psnr(a,b):  # <-
    pass


image = imread('parrots.jpg')
image_data = img_as_float(image)
x, y, z = image_data.shape
X = np.reshape(image_data, (x * y, z))

cls = KMeans(init='k-means++', random_state=241,n_clusters=2)
y_pred = cls.fit_predict(X)

X_median = X.copy()
print(cls.n_clusters)
for i in range(cls.n_clusters):
    X_median[y_pred==i] = np.median(X[y_pred==i], axis=0)

print(psnr(image_data, X_median))  # <-

X_mean = X.copy()
print(cls.n_clusters)
for i in range(cls.n_clusters):
    X_mean[y_pred==i] = np.mean(X[y_pred==i], axis=0)

#
# plt.imshow(np.reshape(X_median, (x, y, z)))
# plt.show()
# plt.imshow(np.reshape(X_mean, (x, y, z)))
# plt.show()