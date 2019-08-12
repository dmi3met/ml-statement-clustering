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


def psnr(a, b):  # <-
    maxi = 1
    sum = 0
    length = len(a)
    for i in range(length):
        for j in range(3):
            sum += (a[i][j] - b[i][j]) ** 2

    mse = sum / (length*3)
    result = 10 * np.log10(maxi ** 2 / mse)
    return result


image = imread('parrots.jpg')
image_data = img_as_float(image)
x, y, z = image_data.shape
X = np.reshape(image_data, (x * y, z))

for n_cl in range(2, 20):
    cls = KMeans(init='k-means++', random_state=241,n_clusters=n_cl)
    y_pred = cls.fit_predict(X)
    X_median = X.copy()
    X_mean = X.copy()

    for i in range(cls.n_clusters):
        X_median[y_pred==i] = np.median(X[y_pred==i], axis=0)
        X_mean[y_pred == i] = np.mean(X[y_pred == i], axis=0)

    print(n_cl)
    print(psnr(X, X_median))
    print(psnr(X, X_mean))

#
# plt.imshow(np.reshape(X_median, (x, y, z)))
# plt.show()
# plt.imshow(np.reshape(X_mean, (x, y, z)))
# plt.show()