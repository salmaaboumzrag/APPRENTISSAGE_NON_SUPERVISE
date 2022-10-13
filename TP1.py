import numpy as np
import matplotlib.pyplot as plt
import time
import kmedoids
import scipy.cluster.hierarchy as shc

from sklearn import cluster
from sklearn import metrics
from scipy.io import arff


path='/home/aboumzrag/Bureau/artificial/'
databrut=arff.loadarff(open(path+"square2.arff",'r'))
#print (databrut[0])
data=[[x[0],x[1]] for x in databrut[0]]
labels_true=[x[2] for x in databrut[0]]

f0=[f[0] for f in data]
f1=[f[1] for f in data]
plt.scatter(f0,f1,s=8)
plt.title("Donnees initiales")
#plt.show()
#############################

# print("Appel Kmeans pour une valeur fixe de k")
# tps1=time.time()
# k=3
# model=cluster.KMeans(n_clusters=k , init='k-means++')
# model.fit(data)
# tps2=time.time()
# labels=model.labels_
# iteration=model.n_iter_

# plt.scatter(f0,f1, c=labels,s=8)
# plt.title("Donnees apres Kmeans")
# #plt.show()
# print('nb clusters=',k," ,nb iter=",iteration ,"..... runtime = ",round((tps2-tps1)*1000,2)," ms")


###################################"
#good sizes
#bad smiles



# tps1 = time.time()
# k=3
# distmatrix = metrics.pairwise.manhattan_distances(data)
# fp= kmedoids.fasterpam(distmatrix, k)
# tps2 = time.time()
# iter_kmed = fp.n_iter
# labels_kmed = fp.labels

# print("\n Loss with FasterPAM:",fp.loss)

# plt.scatter(f0,f1, c=labels_kmed,s=8)
# plt.title("Donnees apres KMedoid")
# plt.show()
# print('nb clusters=',k," ,nb iter=",iter_kmed ,"runtime = ",round((tps2-tps1)*1000,2)," ms")
# print("rand score: ", metrics.rand_score(labels_true,labels_kmed))
# print("mutual info score: ", metrics.mutual_info_score(labels_true, labels_kmed))


# distmatrix = metrics.pairwise.euclidean_distances(data)
# scores = []
# cluster = [k for k in range(2,11)]
# for k in range(2,11):
#     fp= kmedoids.fasterpam(distmatrix, k)
#     tps2 = time.time()
#     iter_kmed = fp.n_iter
#     labels_kmed = fp.labels
#     scores.append(metrics.silhouette_score(data, fp.labels, metric='euclidean'))
    
# plt.plot(cluster,scores)
# plt.show()

print("dendograme'single' Données initiales")

linked_mat = shc.linkage(data,'single')

plt.figure(figsize=(12,12))
shc.dendrogram(linked_mat,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=False)
plt.show()

#set distance threshold

tps1= time.time()
