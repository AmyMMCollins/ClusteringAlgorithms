from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
kmeans = KMeans(n_clusters=3, random_state=111)
kmeans.fit(iris.data)

pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)

plt.subplot(221)
#plt.figure('Figure 13-1')
for i in range(0, pca_2d.shape[0]):
    if iris.target[i] == 0:
        c1 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
    elif iris.target[i] == 1:
        c2 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
    elif iris.target[i] == 2:
        c3 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')
plt.legend([c1,c2,c3], ['Setosa', 'Versicolor', 'Virginica'])
plt.title('PCA')
                        
plt.subplot(222)
#plt.figure('Figure 13-2')
for i in range(0, pca_2d.shape[0]):
    if kmeans.labels_[i] == 1:
        c4 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
    elif kmeans.labels_[i] == 0:
        c5 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
    elif kmeans.labels_[i] == 2:
        c6 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')
plt.legend([c4,c5,c6], ['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.title('K-means 3')
                        

kmeans = KMeans(n_clusters=2, random_state=111)
kmeans.fit(iris.data)
plt.subplot(223)
#plt.figure('Figure 13-2')
for i in range(0, pca_2d.shape[0]):
    if kmeans.labels_[i] == 1:
        c4 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
    elif kmeans.labels_[i] == 0:
        c5 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
plt.legend([c4,c5], ['Cluster 1', 'Cluster 2'])
plt.title('K-means 2')

kmeans = KMeans(n_clusters=4, random_state=111)
kmeans.fit(iris.data)
plt.subplot(224)
#plt.figure('Figure 13-2')
for i in range(0, pca_2d.shape[0]):
    if kmeans.labels_[i] == 1:
        c1 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
    elif kmeans.labels_[i] == 0:
        c2 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
    elif kmeans.labels_[i] == 2:
        c3 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')
    elif kmeans.labels_[i] == 3:
        c4 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='y', marker='+')
plt.legend([c1,c2, c3, c4], ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])
plt.title('K-means 4')

plt.show()
plt.show()

