import numpy as np
import cv2

import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import MiniBatchKMeans

# X为样本特征(D)，Y为样本簇类别， 共1000个样本(y,Q)，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=500, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2],
                  random_state =9)
y_pred = MiniBatchKMeans(n_clusters=4, batch_size=3000, random_state=9).fit(X)#K=4
y_pred2 = y_pred.cluster_centers_
test=np.array([[3,4],[5,21]]).astype(np.float32)
print(y_pred2[0])
print(X[0])
bf = cv2.BFMatcher()
matches = bf.knnMatch(X.astype(np.float32),y_pred2.astype(np.float32), k=1)
# for m in matches:
#     print(m[0].trainIdx)
c=np.array([1,2,3,4]
           )
list1=[[1,2],[21,2]]
list2=[[1,2],[21,22]]
list3=np.dot(list2,list1)

