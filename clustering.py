from google.colab import drive

drive.mount("/content/drive")

from tensorflow.python.client import device_lib

device_lib.list_local_devices()

from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import VGG16
from keras.models import Model

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from random import randint
import pandas as pd
import pickle

path = r"/content/drive/MyDrive/output/output"
os.chdir(path)

images = []
with os.scandir(path) as files:
    for file in files:
        if file.name.endswith(".png"):
            images.append(file.name)
len(images)

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)


def extract_features(file, model):
    img = load_img(file, target_size=(224, 224))
    img = np.array(img)
    reshaped_img = img.reshape(1, 224, 224, 3)
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx, use_multiprocessing=True)
    return features


data = {}
for image in images:
    feat = extract_features(image, model)
    data[image] = feat
filenames = np.array(list(data.keys()))

feat = np.array(list(data.values()))
feat.shape

feat = feat.reshape(-1, 4096)
x = feat
x.shape

from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(x)
tsne_df = pd.DataFrame(data=tsne_data, columns=("x", "y"))
tsne_df

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 30))

visualizer.fit(tsne_df)
visualizer.show()
x.shape

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer


model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(2, 50), metric="calinski_harabasz", timings=False
)

visualizer.fit(tsne_df)
visualizer.show()

wcss = []
for i in range(1, 50):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=22)
    kmeans.fit(tsne_df)
    wcss.append(kmeans.inertia_)
wcss

plt.plot(range(1, 50), wcss)

kmeans = KMeans(n_clusters=7, random_state=22)
kmeans.fit(tsne_df)

label = kmeans.fit_predict(tsne_df)
kmeans.labels_

from sklearn.metrics import silhouette_score

score = silhouette_score(tsne_df, kmeans.labels_)
print("Silhouetter Score: %.3f" % score)
tsne_df["label"] = label
tsne_df.to_csv("/content/drive/df.csv", index=False)

from matplotlib.pyplot import figure

u_labels = np.unique(label)
figure(figsize=(12, 8), dpi=80)
for i in u_labels:
    plt.scatter(tsne_df[label == i, 0], tsne_df[label == i, 1], label=i)
plt.legend()
plt.show()

u_labels = np.unique(label)


def graph(i):
    plt.scatter(x[label == i, 0], x[label == i, 1], label=i)
    plt.legend()
    plt.show()


graph(3)

u_labels = np.unique(label)

for i in u_labels:
    plt.scatter(x[label == i, 0], x[label == i, 1], label=i)
plt.legend()
plt.show()

x[label == 0, 1].shape
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
pca.fit(x)
reduced = pca.transform(x)
reduced

df1 = pd.DataFrame(reduced)
df1
df1["label"] = label
df1.shape

u_labels = np.unique(label)


def graph(i):
    figure(figsize=(12, 8), dpi=80)
    plt.scatter(reduced[label == i, 0], reduced[label == i, 1], label=i)
    plt.legend()
    plt.show()
    print(len(label == i))


df2 = pd.DataFrame(reduced[label])
df2["label"] = label
df2.to_csv("/content/drive/t3.csv", index=False)
pd.DataFrame(reduced[label == 3, 0])
graph(0)

from matplotlib.pyplot import figure

u_labels = np.unique(label)
figure(figsize=(12, 8), dpi=80)
for i in u_labels:
    plt.scatter(reduced[label == i, 0], reduced[label == i, 1], label=i)
plt.legend()
plt.show()

from sklearn.cluster import DBSCAN

db = DBSCAN(eps=50, min_samples=3)
db.fit(reduced)
label = db.labels_

df = pd.DataFrame(label)
df.value_counts()

y_pred = db.fit_predict(reduced)
plt.figure(figsize=(10, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=y_pred, cmap="Paired")
plt.title("Clusters determined by DBSCAN")
from matplotlib import pyplot as plt

plt.savefig("/content/drive/db.png")

u_labels = np.unique(label)


def graph(i):
    plt.scatter(reduced[label == i, 0], reduced[label == i, 1], label=i)
    plt.legend()
    plt.show()


list(label).count(-1)
graph(-1)

from sklearn.metrics import silhouette_score

score = silhouette_score(reduced, label)
print("Silhouetter Score: %.3f" % score)
