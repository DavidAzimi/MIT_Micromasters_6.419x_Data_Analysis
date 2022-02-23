import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

data_path = './data/p2_unsupervised/X.npy'

raw = np.load(data_path)
X = np.log2(raw+1)

########################################
### Variance Explained by # PCs plot ###
########################################
# N = 200
# PCs = np.array(range(1,N+1))
# pca1 = PCA(n_components=N)
# pca2 = PCA(n_components=N)
# Xfit=pca1.fit(X)
# logfit=pca2.fit(X_log)
# Xsum=np.cumsum(pca1.explained_variance_ratio_)
# logsum=np.cumsum(pca2.explained_variance_ratio_)
# 
# plt.plot(PCs, Xsum, color = 'blue', marker = "*")
# plt.plot(PCs, logsum, color = 'red', marker = "o")
# plt.title("np.cumsum by # of principal components")
# plt.xlabel("Principal Components")
# plt.ylabel("Explained Variance")
# plt.show()

#### Choose PCA N principal components ####
N = 12
pca = PCA(n_components=N)
pca.fit(X)
#print(pca.explained_variance_ratio_)
#print(pca.singular_values_)
X_pca = pca.transform(X)

#########################################
###  Mean values of Kmeans clusters   ###
#########################################
kmeans_pred = KMeans(n_clusters=7).fit_predict(X_pca)
X_lab=np.insert(X, 0, kmeans_pred, axis=1)
df = pd.DataFrame(X_lab)
means = np.array(df.groupby(0).mean())
figs, (ax1, ax2) = plt.subplots(1,2)
pca = PCA(n_components=2)
pca.fit(means)
means_pca = pca.transform(means)
ax1.scatter(means_pca[:,0], means_pca[:,1])
ax1.set_title('PCA of cluster means')
embedding = MDS(n_components=2)
means_mds = embedding.fit_transform(means)
ax2.scatter(means_mds[:,0], means_mds[:,1])
ax2.set_title('MDS of cluster means')
plt.show()

#########################################
### Dimension Reduction Visualization ###
#########################################
# ### Tune plot ###
# fig, ax = plt.subplots()
# X_tsne = TSNE(n_components=2, perplexity=100).fit_transform(X_pca)
# ax.scatter(X_tsne[:,0], X_tsne[:,1])#, c=kmeans_pred)
# ax.set_title('t-SNE two components, 12 PCA, 100 perplexity')
# plt.show()

# ### 3 Plots in one ###
# figs, axs = plt.subplots(1,3)
# kmeans_pred = KMeans(n_clusters=3).fit_predict(X_pca)
# # PCA first two principal components
# axs[0].scatter(X_pca[:,0], X_pca[:,1], c=kmeans_pred)
# axs[0].set_title('PCA first two PCs')
# # MDS in 2 dimensions
# embedding = MDS(n_components=2)
# X_mds = embedding.fit_transform(X)
# axs[1].scatter(X_mds[:,0], X_mds[:,1], c=kmeans_pred)
# axs[1].set_title('MDS two dimensions')
# # t-SNE on first N PCs with perplexity p
# p = 100
# X_tsne = TSNE(n_components=2, perplexity=p).fit_transform(X_pca)
# axs[2].scatter(X_tsne[:,0], X_tsne[:,1], c=kmeans_pred)
# axs[2].set_title('t-SNE two components, PCA N=%i, perplexity=%i'%(N, p))
# plt.show()

#########################################
###          Elbow Plot               ###
#########################################
# fig, (ax1,ax2) = plt.subplots(1,2)
# inertias, distortions = [], []
# map1, map2 = {}, {}
# K = range(1,10)
# for k in K:
#     # Building and fitting the model
#     kmeanModel = KMeans(n_clusters=k).fit(X_pca)
#     distortions.append(sum(np.min(cdist(X_pca, kmeanModel.cluster_centers_,
#                                         'euclidean'), axis=1)) / X_pca.shape[0])
#     inertias.append(kmeanModel.inertia_)
#     map1[k] = sum(np.min(cdist(X_pca, kmeanModel.cluster_centers_,
#                                    'euclidean'), axis=1)) / X_pca.shape[0]
#     map2[k] = kmeanModel.inertia_
# # Distortion
# print('Distortions')
# for key, val in map1.items():
#     print(f'{key} : {val}')
# ax1.plot(K, distortions, 'bx-')
# ax1.set_title('The Elbow Method using Distortion')
# # Inertia
# print('Inertias')
# for key, val in map2.items():
#     print(f'{key} : {val}')
# ax2.plot(K, inertias, 'bx-')
# ax2.set_title('The Elbow Method using Inertia')
# plt.show()

