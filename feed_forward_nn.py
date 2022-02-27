import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera

class network:

	def __init__(self,training_data,labels,blank_spots,layers_dims,epochs,lr):
		self.training_data = training_data
		self.labels = labels
		self.blank_spots = blank_spots
		self.layers_dims = layers_dims
		self.epochs = epochs
		self.lr = lr
		self.params = {}
		self.fig = plt.figure()
		self.camera = Camera(self.fig)


	def init_params(self):

		for i in range(1,len(self.layers_dims)):
			self.params['W'+str(i)] = torch.rand(self.layers_dims[i],self.layers_dims[i-1],dtype=torch.float64,requires_grad=True)
			self.params['b'+str(i)] = torch.rand(self.layers_dims[i],1,dtype=torch.float64,requires_grad=True)

	def linear_forward(self,A_prev,W,b,activation):

		if activation == 'tanh':
			Z = torch.matmul(W,A_prev) + b
			A_next = torch.tanh(Z)

		if activation =='sigmoid':
			Z = torch.matmul(W,A_prev) + b
			A_next = torch.sigmoid(Z)

		return A_next

	def forward_pass(self,data):

		A_prev = data

		for i in range(1,len(self.layers_dims)-1):
			A_prev = self.linear_forward(A_prev,self.params['W'+str(i)],self.params['b'+str(i)],'tanh')	

		predict = self.linear_forward(A_prev,self.params['W'+str(len(self.layers_dims)-1)],self.params['b'+str(len(self.layers_dims)-1)],'sigmoid')

		return predict
	def grad_sum(self):
        
		grad_sum = 0

		for tensor in self.params.keys():
			grad_sum += sum(sum(abs(self.params[tensor].grad)))

		return grad_sum

	def graph_epoch_results(self,prediction,epoch):
		#b/c each dataset is partially randomly initialized we generate a median decision boundary
		#visualization is improved if median is generated every epoch. Terrible practice, would not replicate that method in any use case other than visualizations			
		median = torch.median(prediction)

		blue = torch.where(prediction < median)
		red = torch.where(prediction > median)
		plt.xlabel('x values')
		plt.ylabel('y values')
		plt.title(f'Neural Network Decision Boundary Visualization')

		#predicting whitespace
		plt.scatter(self.blank_spots[0][blue[1]],self.blank_spots[1][blue[1]],c='skyblue',label='blue prediction')
		plt.scatter(self.blank_spots[0][red[1]],self.blank_spots[1][red[1]],c='pink',label = 'red prediction')

		#re-drawing training data points
		plt.scatter(self.training_data[0][0:30].numpy(),self.training_data[1][0:30].numpy(),c='blue',label = 'blue point')
		plt.scatter(self.training_data[0][30:60].numpy(),self.training_data[1][30:60].numpy(),c='red',label = 'red point')

		self.camera.snap()

	def run(self):

		#tensorboard writer was used to narrow down on ideal lr parameters through random sampling
		# writer = SummaryWriter(f'/Users/timothygould/ml_practice/nn_decision_boundary_vis/tensorboard/sgd/lr_{self.lr}')

		self.init_params()


		loss = nn.BCELoss()	#binary cross entropy loss. Used for binary classification probs

		optimizer = torch.optim.SGD(self.params.values(),lr=self.lr,momentum=0) #basic stochastic gradient descent
		
		for i in range(0,self.epochs):
			predict= self.forward_pass(self.training_data)
			predict_blanks = self.forward_pass(self.blank_spots)
			#need to find a way to seperate out params w/ no gradients so prediction doesn't get tracked back in autograd
			self.graph_epoch_results(predict_blanks,i)

			cost = loss(predict,self.labels)
			cost.backward()
			grad_sum = self.grad_sum()
			# writer.add_scalar('cost',cost,i)
			# writer.add_scalar('grad_sum',grad_sum,i)

			optimizer.step()
			optimizer.zero_grad()
		# writer.flush()
		# writer.close()
		animation = self.camera.animate()
		animation.save('NN_Visualization.gif')
