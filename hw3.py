import numpy as np

### ANSWERS TO QUESTION 6 ###

A = np.matrix('0 14; 6 9')
print np.linalg.eig(A)

### ANSWERS TO QUESTION 7 ###
a = np.matrix('-0.75 -1.5 0.5; 3.25 2.5 0.5; 1.25 -2.5 -3.5;-3.75 1.5 2.5')

b = np.cov(a.T)

w,v=np.linalg.eig(b)
print a
print b
print w

import sklearn.decomposition as skd
# .fit computes the principal components (n_components of them)
# The columns of W are the eigenvectors of the covariance matrix of X
pca = skd.PCA(n_components = 3)
skd.PCA.fit(pca,a)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(a)

print Z

### ANSWERS TO QUESTION 5 ###

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70)
n_samples, h, w = lfw_people.images.shape
npix = h*w
fea = lfw_people.data
def plt_face(x):
	global h,w
	plt.imshow(x.reshape((h, w)), cmap=plt.cm.gray)
	plt.xticks([])
#plt.figure(figsize=(10,20))
#nplt = 4
#for i in range(nplt):
	#plt.subplot(1,nplt,i+1)
	#plt_face(fea[i])
#plt.show()

plt_face(fea[3])
#plt.show() #UNCOMMENT THIS LINE TO GET ANSWER FOR 5A

plt_face(np.mean(fea,axis=0))
#plt.show() #UNCOMMENT THIS LINE TO GET ANSWER FOR 5B

pca = skd.PCA(n_components = 5)
skd.PCA.fit(pca,fea)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(fea)
print '5C - Values of the associated 5 attributes of image 4'
print Z[3]

pca = skd.PCA(n_components = 5)
skd.PCA.fit(pca,fea)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(fea)
A=np.dot(W, Z[3])+np.mean(fea,axis=0)
plt_face(A)
print 'Some values for image A', A
#plt.show() #UNCOMMENT THIS LINE TO GET ANSWER FOR 5D

pca = skd.PCA(n_components = 50)
skd.PCA.fit(pca,fea)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(fea)
B=np.dot(W, Z[3])+np.mean(fea,axis=0)
plt_face(B)
print 'Some values for image B', B
#plt.show()  #UNCOMMENT THIS LINE TO GET ANSWER FOR 5D
