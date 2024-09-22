import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Please read the free response questions before starting to code.

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None, 
	batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):
	"""
	This function either trains or evaluates a model. 

	training mode: the model is trained and evaluated on a validation set, if provided. 
				   If no validation set is provided, the training is performed for a fixed 
				   number of epochs. 
				   Otherwise, the model should be evaluted on the validation set 
				   at the end of each epoch and the training should be stopped based on one
				   of these two conditions (whichever happens first): 
				   1. The validation loss stops improving. 
				   2. The maximum number of epochs is reached.

    testing mode: the trained model is evaluated on the testing set

	training mode:
		for each epoch < #max_epoch_num:
			train
			if validation_set provided:
				validate
				calculate loss
				if loss - prev_loss < threshold:
					break

    Inputs: 

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset object generated using the class MyDataset 
    valid_set: the validation dataset object generated using the class MyDataset
    test_set: the testing dataset object generated using the class MyDataset
    batch_size: number of training samples fed to the model at each training step
	learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model 
    stop_thr: if the validation loss from one epoch to the next is less than this
              value, stop training
    shuffle: determines if the shuffle property of the DataLoader is on/off

    Outputs when running_mode == 'train':

    model: the trained model 
    loss: dictionary with keys 'train' and 'valid'
    	  The value of each key is a list of loss values. Each loss value is the average
    	  of training/validation loss over one epoch.
    	  If the validation set is not provided just return an empty list.
    acc: dictionary with keys 'train' and 'valid'
    	 The value of each key is a list of accuracies (percentage of correctly classified
    	 samples in the dataset). Each accuracy value is the average of training/validation 
    	 accuracies over one epoch. 
    	 If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss: the average loss value over the testing set. 
    accuracy: percentage of correctly classified samples in the testing set. 
	
	Summary of the operations this function should perform:
	1. Use the DataLoader class to generate trainin, validation, or test data loaders
	2. In the training mode:
	   - define an optimizer (we use SGD in this homework)
	   - call the train function (see below) for a number of epochs untill a stopping
	     criterion is met
	   - call the test function (see below) with the validation data loader at each epoch 
	     if the validation set is provided

    3. In the testing mode:
       - call the test function (see below) with the test data loader and return the results

	"""
	epoch_num = 0
	prev_loss = float('inf')

	if running_mode == "test":
		#test  using test set
		test_loader = DataLoader(test_set, batch_size = len(test_set), shuffle = shuffle)
		_est_loss, _est_acc = _test(model, test_loader)
		return _est_loss, _est_acc

	else: #" mode = train"
		loss_dict = {"train": [], "valid": []}
		acc_dict = {"train": [], "valid": []}
		while epoch_num < n_epochs:
			epoch_num += 1
			# train your model

			train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
			optimizer = optim.SGD(model.parameters(), lr = learning_rate)
			model, _est_loss, _est_acc = _train(model,train_loader,optimizer)
			loss_dict["train"].append(_est_loss)
			acc_dict["train"].append(_est_acc)

			#validate if validation set!= None
			if valid_set!= None:
				valid_loader = DataLoader(valid_set, batch_size = len(train_set), shuffle = shuffle)
				_est_loss, _est_acc = _test(model, valid_loader)

				loss_dict["valid"].append(_est_loss)
				acc_dict["valid"].append(_est_acc)

				#stop if prev_loss - loss < thresh
				print("prev loss: ", prev_loss, "loss: ", _est_loss)
				if abs(_est_loss - prev_loss) < stop_thr:
					print("break!")
					break
				else:
					prev_loss = _est_loss

		return model, loss_dict, acc_dict



def _train(model,data_loader,optimizer,device=torch.device('cpu')):

	"""
	This function implements ONE EPOCH of training a neural network on a given dataset.
	Example: training the Digit_Classifier on the MNIST dataset


	Inputs:
	model: the neural network to be trained:
		model.fc1, model.fc2, ... 4 layers.
	data_loader: for loading the netowrk input and targets from the training dataset
	optimizer: the optimiztion method, e.g., SGD . Assume model parameters have been plugged in here!
	device: we run everything on CPU in this homework

	Outputs:
	model: the trained model
	train_loss: average loss value on the entire training dataset
	train_accuracy: average accuracy on the entire training dataset = % * 100?
	"""
	total_loss = torch.from_numpy(np.array([0]))
	count = 0
	correct = 0
	for data in data_loader:
		X,y = data
		model.zero_grad()
		output = model(X.float())

		count += 1.0
		if torch.argmax(output) == y:
			correct+=1.0

		loss = F.cross_entropy(output, y.long(), reduction="sum")

		loss.backward()
		optimizer.step()
		total_loss = torch.add(total_loss, loss)


	accuracy = 100 * correct / count
	avg_loss = total_loss.item()/count

	return model, avg_loss, accuracy


def _test(model, data_loader, device=torch.device('cpu')):
	"""
	This function evaluates a trained neural network on a validation set
	or a testing set. 

	Inputs:
	model: trained neural network
	data_loader: for loading the netowrk input and targets from the validation or testing dataset
	device: we run everything on CPU in this homework

	Output:
	test_loss: average loss value on the entire validation or testing dataset 
	test_accuracy: percentage of correctly classified samples in the validation or testing dataset
	"""

	model.eval()		#?
	test_loss = 0
	correct = 0.0

	with torch.no_grad():
		for data, target in data_loader:
			data, target= data.to(device), target.to(device)
			output = model(data.float())
			test_loss += F.cross_entropy(output, target.long(), reduction="sum").item()
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(data_loader.dataset)

	accuracy = 100.0 * correct/len(data_loader.dataset)

	return test_loss, accuracy






