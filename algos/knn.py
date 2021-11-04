import numpy as np


class KNN():
	def __init__(self,k=5):
		self.X=None
		self.y=None
		self.k=k

	def fit(self,X,y):
		self.X=X
		self.y=y

	def calcualate_euclidean_distance(self,a,b):

		dist=np.array(a) - np.array(b)
		dist=np.square(dist)
		dist=np.sum(dist)
		return dist

	def predict(self,a):

		# calculate euclidean distance
		euclidean_distances=[]
		for row in self.X:
			euclidean_distance=self.calcualate_euclidean_distance(row,a)
			euclidean_distances.append(euclidean_distance)
		euclidean_distances=list(zip(euclidean_distances,self.y))
		euclidean_distances=sorted(euclidean_distances,key=lambda x:x[0])
		euclidean_distances=euclidean_distances[:self.k]
		predicted_classes=[x[1] for x in euclidean_distances]
		print(predicted_classes)
		predicted_classes=np.sum(np.array(predicted_classes),axis=0)
		print(predicted_classes)
		predicted_class=np.argmax(predicted_classes)
		print(predicted_classes)
		return predicted_class


knn=KNN()
X=np.random.random((100,2))
y=np.random.randint(0,2,[100,2])
knn.fit(X,y)
p=knn.predict(np.random.random((20,2)))
print(p)
