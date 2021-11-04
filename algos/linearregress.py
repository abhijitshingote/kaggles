import numpy as np
import pandas as pd


class LinearRegression():

	def __init__(self,lr=0.001):
		self.lr=lr
		self.w=None
		self.b=None

	def fit(self,X,y,epochs=1):
		X,y=np.array(X),np.array(y)
		# Weight Matrix
		self.w=np.zeros(X.shape[1])
		self.b=0
		

		for epoch in range(epochs):
			print(f'epoch:{epoch}')
			# Loss
			yhat=X.dot(self.w) + np.ones(X.shape[0])*self.b
			loss=np.square(y-yhat)
			print(f'loss:{np.sum(loss)}')
			# Gradients
			w_gradient=np.transpose(X).dot(loss)/len(y)
			b_gradient=sum(loss)/len(y)
			# Update Weights
			self.w=self.w-self.lr*w_gradient
			self.b=self.b-self.lr*b_gradient
			#Zero the gradients
			w_gradient,b_gradient=np.zeros(w_gradient.shape),0

	def predict(self,X):

		y=X.dot(self.w)+self.b 
		return y



lr=LinearRegression()
X=np.random.random([100,2])
y=np.append(np.zeros(50) , np.ones(50))
print(X.shape)
print(y.shape)
lr.fit(X,y,10)
print(lr.predict(np.random.random([20,2])))
# a=np.array([5,4,3,2,1])
# print(np.where(a==a))
# print(sum(a==np.array([5,3,4,2,1])))
