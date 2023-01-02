import numpy as np
import pandas as pd
import cv2 as cv2
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
#from IPython.display import display

image_raw = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
print(image_raw.shape)

# Displaying the image
plt.figure(figsize=[8,4])
plt.imshow(image_raw, cmap=plt.cm.gray)
plt.show()
pca = PCA(n_components=2)
image_pca = pca.fit_transform(image_raw)
print(image_pca.shape)

# Displaying the image
plt.figure(figsize=[8,4])
plt.imshow(image_pca)
plt.show()
imagesDf = pd.DataFrame(data = image_pca, columns = ['principalcomponent1','principalcomponent2'])
#display(imagesDf)
imagesDf.plot.scatter(x="principalcomponent1", y="principalcomponent2")
plt.show()