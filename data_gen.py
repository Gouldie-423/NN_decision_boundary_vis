import matplotlib.pyplot as plt
import random as rand
import torch 
import feed_forward_nn as ff
import random
from itertools import permutations
#order of ops
#generate blank tensors
#populate with random data
#save copy in matplotlib readable format, only needs to be created once.
#feed data into nn
#predict every other blank point on graph to visualize decision boundary
#save those predictions in matplotlib readable format
#take matplotlib graphics and turn into gif

def data_gen():

	#low and high ranges for each group distribution
	blue_x_low,blue_x_high,blue_y_low,blue_y_high = 40,70,50,100
	red_x_low,red_x_high,red_y_low,red_y_high = 0,40,0,50
	
	#creating blank tensors to fill partially random data into.
	#hard stopping at 60 total samples. May decrease further
	training_data = torch.zeros(2,60,dtype=torch.float64,requires_grad=False)
	labels = torch.zeros(1,60,dtype=torch.float64,requires_grad=False)
	blank_spots = torch.zeros(2,10_100,dtype=torch.float64,requires_grad=False)

	#using two explicit for loops to ensure that partially random data has slight gap when plotted
	#blue
	for i in range(0,30):
		training_data[0][i] = rand.randint(blue_x_low,blue_x_high)
		training_data[1][i] = rand.randint(blue_y_low,blue_y_high)
	#red
	for i in range(30,60):
		training_data[0][i] = rand.randint(red_x_low,red_x_high)
		training_data[1][i] = rand.randint(red_y_low,red_y_high)
		labels[0][i] = 1

	#generating 'blank' spots to run through nn prediction
	plot_range = [i for i in range(0,101)] #graph is a 101x101 plot if you include 0 on each axis
	combs = list(permutations(plot_range,2)) #permutation of all 2 integer combinations of plot range
	for i in range(0,len(combs)):
		blank_spots[0][i] = combs[i][0]
		blank_spots[1][i] = combs[i][1]

	return training_data,labels,blank_spots

#test to make sure I can visualize how random distributions are looking
def initial_graph(training_data,blank_spots):

	plt.figure()
	plt.scatter(blank_spots[0][:].numpy(),blank_spots[1][:].numpy(),c='green')
	plt.scatter(training_data[0][0:30].numpy(),training_data[1][0:30].numpy(),c='blue')
	plt.scatter(training_data[0][30:60].numpy(),training_data[1][30:60].numpy(),c='red')
	plt.show()

#used to either randomly sample learning rates or find a custom range once some solid hp's are sampled for
def learning_rate_gen(num_samples,custom_range = None):
    learning_rates = []
    base = [10,100]
    
    if custom_range != None:
        for i in range(0,num_samples):
            test = random.uniform(custom_range[0],custom_range[1])
            learning_rates.append(test)
    
    else:
    
        for base in base:
            for i in range(0,num_samples):
                test = -5 *random.uniform(0,1)
                learning_rate = base**test
                learning_rates.append(learning_rate)

    return learning_rates

learning_rates = learning_rate_gen(1,[.39,.41]) #only generating a single lr within range b/c it's already been sampled for in backtesting
training_data,labels,blank_spots = data_gen()
# initial_graph(training_data,blank_spots)

training_data = torch.div(training_data,100) #plane is 100x100, dividing data by 100 to get numbers closer to zero for easier training
blank_spots = torch.div(blank_spots,100)

layers_dims = [2,3,2,1] #[features, nodes in layer1, nodes in layer2, nodes in layer3]...can add as many layers/nodes as you want. explicit for loop creates params

for rate in learning_rates:
	test = ff.network(training_data,labels,blank_spots,layers_dims,30,rate)
	test.run()
