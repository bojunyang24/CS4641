# hw1.py

import os.path
import numpy
import numpy as np

######################################
#
# FUNCTIONS YOU WILL NEED TO MODIFY:
#  - linreg_closed_form
#  - loss
#  - linreg_grad_desc
#  - random_fourier_features
#
######################################
def linreg_model_sample(Theta,model_X):
	if model_X.shape[1]==1:
		## get a bunch of evenly spaced X values in the same range as the passed in data
		sampled_X = numpy.linspace(model_X.min(axis=0),model_X.max(axis=0),100)
		## get the Y values for our sampled X values by taking the dot-product with the model
		## Note: we're appending a column of all ones so we can do this with a single matrix-vector multiply
		sampled_Y = numpy.hstack([numpy.ones((sampled_X.shape[0],1)),sampled_X]).dot(Theta)
		return sampled_X, sampled_Y
	elif model_X.shape[1]==2:
		## Unfortunately, plotting surfaces is a bit more complicated, first we need
		## a set of points that covers the area we want to plot. numpy.meshgrid is a helper function
		## that will create two NxN arrays that vary over both the X and Y range given.
		sampled_X, sampled_Y = numpy.meshgrid(model_X[:,0],model_X[:,1])
		## We can't just do a simple matrix multiply here, because plot_surface(...) is going to expect NxN arrays like
		## those generated by numpy.meshgrid(...). So here we're explicitly pulling out the components of Theta as
		## scalars and multiplying them across each element in the X and Y arrays to get the value for Z
		sampled_Z = sampled_X*Theta[1]+sampled_Y*Theta[2]+Theta[0]
		return sampled_X, sampled_Y, sampled_Z

def plot_helper(data_X, data_Y, model_X=None, model_Y=None, model_Z=None):
	import matplotlib.pyplot
	## 2D plotting
	## data_X.shape[1] is the number of columns in data_X, just as data_X.shape[0] is the number of rows
	if data_X.shape[1]==1: 
		fig1 = matplotlib.pyplot.figure() ## creates a new figure object that we can plot into
		fig1.gca().scatter(data_X,data_Y) ## creates a scatterplot with the given set of X and Y points
		## If we were given a model, we need to plot that
		if not(model_X is None) and not(model_Y is None):
			## Plot the data from the model
			## Note: we're using plot(...) instead of scatter(...) because we want a smooth curve
			fig1.gca().plot(model_X,model_Y,color='r')
		## The graph won't actually be displayed until we .show(...) it. You can swap this with savefig(...) if you
		## instead want to save an image of the graph instead of displaying it. You can also use the interface to save an
		## image after displaying it
		matplotlib.pyplot.show() #fig1.show()
	## 3D plotting
	elif data_X.shape[1]==2:
		## This import statement 'registers' the ability to do 3D projections/plotting with matplotlib
		from mpl_toolkits.mplot3d import Axes3D
		fig1 = matplotlib.pyplot.figure()
		## The format for 3D scatter is similar to 2D; just add the third dimension to the argument list
		fig1.gca(projection='3d').scatter(data_X[:,0],data_X[:,1],data_Y)
		if not(model_X is None) and not(model_Y is None) and not(model_Z is None):
			## Now, with our X, Y, and Z arrays (all NxN), we can use plot_surface(...) to create a nice 3D surface
			fig1.gca(projection='3d').plot_surface(model_X, model_Y, model_Z,linewidth=0.0,color=(1.0,0.2,0.2,0.75))
		matplotlib.pyplot.show() #fig1.show()
	else:
		## Matplotlib does not yet have the capability to plot in 4D
		print('Data is not in 2 or 3 dimensions, cowardly refusing to plot! (data_X.shape == {})'.format(data_X.shape))

## Data loading utility function
def load_data(fname,directory='data'):
	data = numpy.loadtxt(os.path.join(directory,fname),delimiter=',')
	rows,cols = data.shape
	X_dim = cols-1
	Y_dim = 1
	return data[:,:-1].reshape(-1,X_dim), data[:,-1].reshape(-1,Y_dim)

def vis_linreg_model(train_X, train_Y, Theta):
	sample_X, sample_Y = linreg_model_sample(Theta,train_X)
	#NOTE: this won't work directly with 3D data. Write your own function, or modify this one
	#to generate plots for 2D-noisy-lin.txt or other 3D data.
	plot_helper(train_X, train_Y, sample_X, sample_Y)

def vis_linreg_model_3d(train_X, train_Y, Theta):
	sample_X, sample_Y, sample_Z = linreg_model_sample(Theta,train_X)
	plot_helper(train_X, train_Y, sample_X, sample_Y, sample_Z)

###################
# YOUR CODE BELOW #
###################
def linreg_closed_form(train_X, train_Y):
	'''
	Computes the optimal parameters for the given training data in closed form


	Args:
		train_X (N-by-D numpy array): Training data features as a matrix of row vectors (train_X[i][j] is the jth component of the ith example)
		train_Y (length N numpy array): The training data target as a length N vector


	Returns:
		A length D+1 numpy array with the optimal parameters	
	'''
	X = train_X
	Theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, train_Y)) # TODO: compute the closed form solution here. Note: using numpy.linalg.lstsq(...) is *not* the correct answer
	return Theta

###################
# YOUR CODE BELOW #
###################
def loss(Theta, train_X, train_Y):
	'''
	Computes the squared loss for the given setting of the parameters given the training data


	Args:
		Theta (length D+1 numpy array): the parameters of the model
		train_X (N-by-D numpy array): Training data features as a matrix of row vectors (train_X[i][j] is the jth component of the ith example)
		train_Y (length N numpy array): The training data target as a length N vector


	Returns:
		The (scalar) loss for the given parameters and data.
	'''
	# ones = np.ones((train_X.shape[0], 1))
	# X = np.concatenate((ones, train_X), axis=1)
	X = train_X
	h = np.dot(X, Theta)
	rv = np.sum((h - train_Y) ** 2) * (1/(2 * train_Y.shape[0])) # TODO: compute the loss here.
	return rv

###################
# YOUR CODE BELOW #
###################
def linreg_grad_desc(initial_Theta, train_X, train_Y, alpha=0.05, num_iters=500, print_iters=True):
	'''
	Fits parameters using gradient descent


	Args:
		initial_Theta ((D+1)-by-1 numpy array): The initial value for the parameters we're optimizing over
		train_X (N-by-D numpy array): Training data features as a matrix of row vectors (train_X[i][j] is the jth component of the ith example)
		train_Y (N-by-1 numpy array): The training data target as a vector
		alpha (float): the learning rate/step size, defaults to 0.1
		num_iters (int): number of iterations to run gradient descent for, defaults to 500


	Returns:
		The history of theta's and their associated loss as a list of tuples [ (Theta1,loss1), (Theta2,loss2), ...]
	'''
	cur_Theta = initial_Theta
	step_history = list()
	for k in range(1,num_iters+1):
		cur_loss = loss(cur_Theta, train_X, train_Y)
		step_history.append((cur_Theta, cur_loss))
		if print_iters:
			print("Iteration: {} , Loss: {} , Theta: {}".format(k,cur_loss,cur_Theta))
		#TODO: Add update equation here
		X = train_X
		cur_Theta = cur_Theta - alpha * ((np.dot(np.dot(X.T, X), cur_Theta) - (np.dot(X.T, train_Y))) * (1/train_Y.shape[0]))
	return step_history

def apply_RFF_transform(X,Omega,B):
	'''
	Transforms features into a Fourier basis with given samples

		Given a set of random inner products and translations, transform X into the Fourier basis, Phi(X)
			phi_k(x) = cos(<x,omega_k> + b_k)                           #scalar form
			Phi(x) = sqrt(1/D)*[phi_1(x), phi_2(x), ..., phi_NFF(x)].T  #vector form
			Phi(X) = [Phi(x_1), Phi(x_2), ..., Phi(x_N)].T              #matrix form


	Args:
		X (N-by-D numpy array): matrix of row-vector features (may also be a single row-vector)
		Omega (D-by-NFF numpy array): matrix of row-vector inner products
		B (NFF length numpy array): vector of translations



	Returns:
		A N-by-NFF numpy array matrix of transformed points, Phi(X)
	'''
	# return numpy.sqrt(1.0/Omega.shape[1])*numpy.cos(X.dot(Omega)+B)
	Phi = numpy.sqrt(1.0/Omega.shape[1])*numpy.cos(X.dot(Omega)+B)
	ones = np.ones((X.shape[0], 1))
	Phi = np.concatenate((ones, Phi), axis=1)
	return Phi

##################
# YOUR CODE HERE #
##################
def random_fourier_features(train_X, train_Y, num_fourier_features=100, alpha=0.1, num_iters=500, print_iters=False):
	'''
	Creates a random set of Fourier basis functions and fits a linear model in this space.

		Randomly sample num_fourier_features's non-linear transformations of the form:

			phi_k(x) = cos(<x,omega_k> + b_k)
			Phi(x) = sqrt(1/D)*[phi_1(x), phi_2(x), ..., phi_NFF(x)]

		where omega_k and b_k are sampled according to (Rahimi and Recht, 20018). 


	Args:
		train_X (N-by-D numpy array): Training data features as a matrix of row vectors (train_X[i][j] is the jth component of the ith example)
		train_Y (length N numpy array): The training data target as a length N vector
		num_fourier_features (int): the number of random features to generate


	Returns:
		Theta (numpy array of length num_fourier_features+1): the weights for the *transformed* model
		Omega (D-by-num_fourier_features numpy array): the inner product term of the transformation
		B (numpy array of length num_fourier_features): the translation term of the transformation
	'''
	# You will find the following functions useful for sampling:
	# 	numpy.random.multivariate_normal() for normal random variables
	#	numpy.random.random() for Uniform random variables
	# D by NFF numpy array
	D = train_X.shape[1]
	mean = np.zeros(D)
	sigma = np.sqrt(2 * np.pi) ** (D-1)
	cov = np.zeros((D,D))
	np.fill_diagonal(cov, 1)
	Omega = np.random.multivariate_normal(mean, cov, (num_fourier_features)) 	# TODO: sample inner-products
	Omega = Omega.T
	# NFF length numpy array
	B = np.random.random(size=(num_fourier_features)) * (2 * np.pi)		# TODO: sample translations
	Phi = apply_RFF_transform(train_X,Omega,B)
	# here's an example of using numpy.random.random()
	# to generate a vector of length = (num_fourier_features), between -0.1 and 0.1
	initial_Theta = (numpy.random.random(size=(num_fourier_features+1,1))-0.5)*0.2
	step_history = linreg_grad_desc(initial_Theta,Phi,train_Y,alpha=alpha,num_iters=num_iters,print_iters=print_iters)
	print('loss: ' + str(step_history[-1][1]))
	return step_history[-1][0], Omega, B

def rff_model_sample(Theta,Omega,B,model_X):
	sampled_X = numpy.linspace(model_X.min(axis=0),model_X.max(axis=0),100)
	Phi = apply_RFF_transform(sampled_X,Omega,B)
	sampled_Y = Phi.dot(Theta)
	return sampled_X, sampled_Y

def vis_rff_model(train_X, train_Y, Theta, Omega, B):
	sample_X, sample_Y = rff_model_sample(Theta,Omega,B, train_X)
	plot_helper(train_X, train_Y, sample_X, sample_Y)

if __name__ == '__main__':
	### Import data
	data_X, data_Y = load_data('1D-no-noise-lin.txt')
	# data_X, data_Y = load_data('2D-noisy-lin.txt')
	# data_X, data_Y = load_data('1D-exp-samp.txt')
	# data_X, data_Y = load_data('1D-exp-uni.txt')
	# data_X, data_Y = load_data('1D-quad-uni.txt')
	# data_X, data_Y = load_data('1D-quad-uni-noise.txt')

	# X = np.concatenate((data_X[0,:].reshape((1,data_X.shape[1])), data_X), axis=0) # duplicate example
	# data_X[:,0] = data_X[:,1] # duplicate feature

	# append ones vector
	ones = np.ones((data_X.shape[0], 1))
	ones = data_X[:,0].reshape((data_X.shape[0], 1))
	X = np.concatenate((ones, data_X), axis=1)

	# data_Y = np.concatenate((data_Y[0,:].reshape((1,data_Y.shape[1])), data_Y), axis = 0) # duplicate example

	# # lin-reg optimal thetas and losses
	theoretical_theta, theoretical_loss, _, _ = np.linalg.lstsq(X, data_Y, rcond=1)
	theoretical_loss = theoretical_loss / (2 * data_Y.shape[0])

	hw=0 # lin reg
	# hw=1 # grad desc
	# hw=2 # rff
	if hw==0: # closed form
		Theta = linreg_closed_form(X, data_Y)
		Loss = loss(Theta, X, data_Y)

		print('\nClosed form theta: ' + str(np.around(Theta[0], 100)), str(np.around(Theta[1], 100))) # 1D
		# print('\nClosed form theta: ' + str(Theta[0]), str(Theta[1]), str(Theta[2])) # 2D
		print('Closed form loss: ' + str(np.around(Loss, 100)))

		print('\ntheoretical_theta: ' + str(np.around(theoretical_theta[0], 100)), str(np.around(theoretical_theta[1], 100))) # 1D
		# print('\nTheoretical_theta: ' + str(theoretical_theta[0]), str(theoretical_theta[1]), str(theoretical_theta[2])) # 2D
		print('Theoretical_loss: ' + str(theoretical_loss[0]))

		## visualize models
		# vis_linreg_model_3d(data_X, data_Y, Theta)
		vis_linreg_model(data_X, data_Y, Theta)
	elif hw==1:
		## Gradient Descent
		init_Theta = np.zeros((X.shape[1], 1))
		step_history = linreg_grad_desc(init_Theta, X, data_Y, alpha=1.3, num_iters=20)
		# step_history = linreg_grad_desc(init_Theta, X, data_Y)
		grad_desc_theta = step_history[-1][0]
		grad_desc_loss = step_history[-1][1]
		print('\ngrad_desc_theta: ' + str(np.around(grad_desc_theta[0], 100)), str(np.around(grad_desc_theta[1], 100))) # 1D
		# print('\ngrad_desc_theta: ' + str(grad_desc_theta[0]), str(grad_desc_theta[1]), str(grad_desc_theta[2])) # 2D
		print('grad_desc_loss: ' + str(np.around(grad_desc_loss, 100)))

		print('\nTheoretical_theta: ' + str(np.around(theoretical_theta[0], 100)), str(np.around(theoretical_theta[1], 100))) # 1D
		# print('\ntheoretical_theta: ' + str(theoretical_theta[0]), str(theoretical_theta[1]), str(theoretical_theta[2])) # 2D
		print('Theoretical_loss: ' + str(theoretical_loss[0]))
		## visualize models
		# vis_linreg_model_3d(data_X, data_Y, grad_desc_theta)
		# vis_linreg_model(data_X, data_Y, grad_desc_theta)
	elif hw==2:
		# rff_theta, rff_omega, rff_b = random_fourier_features(data_X, data_Y)
		rff_theta, rff_omega, rff_b = random_fourier_features(data_X, data_Y, num_fourier_features=800, alpha=0.1, num_iters=500)
		print('rff_theta: ' + str(rff_theta.shape))
		print('rff_omega: ' + str(rff_omega.shape))
		print('rff_b: ' + str(rff_b.shape))
		vis_rff_model(data_X, data_Y, rff_theta, rff_omega, rff_b)