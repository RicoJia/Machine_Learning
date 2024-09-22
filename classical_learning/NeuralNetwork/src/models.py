import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.
	
	Network architecture:
	- Input layer
	- First hidden layer: fully connected layer of size 128 nodes
	- Second hidden layer: fully connected layer of size 64 nodes
	- Output layer: a linear layer with one node per class (in this case 10)

	Activation function: ReLU for both hidden layers

    """
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # final_result = F.log_softmax(x, dim=1)
        return x

class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.
    
    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self):
        super(Dog_Classifier_FC, self).__init__()
        self.fc1 = nn.Linear(12288,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # final_result = F.log_softmax(x, dim=1)
        return x


class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.
    
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers
    
    There should be a maxpool after each convolution. 
    
    The sequence of operations looks like this:
    
    1. Apply convolutional layer with stride and kernel size specified
    - note: uses hard-coded in_channels and out_channels
    - read the problems to figure out what these should be!
	2. Apply the activation function (ReLU)
	3. Apply 2D max pooling with a kernel size of 2

    Inputs: 
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()

                #        [(6, 3, 5, 5), (64, 16, 5, 5)]
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size= kernel_size[0][0], stride= stride[0][0])   #first two terms are: (input_channel, output channel)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(16,32, kernel_size= kernel_size[1][0], stride= stride[1][0])
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(5408,10)

    def forward(self, input):

        # print("input shape: ",input.shape)
        #change size from (3, 64,64) to (16, 64,64)

        x = F.relu(self.conv1(input.view(1,3,64, 64)))
        #(16, 32, 32)

        x = self.pool1(x)

        # ([1, 32, 52, 52])
        x = F.relu(self.conv2(x))

        # 32, 48, 48])
        x = self.pool2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        return x



class Synth_Classifier(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying 
    synthesized images.
    
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 2)

    Activation function: ReLU for both hidden layers
    
    There should be a maxpool after each convolution. 
    
    The sequence of operations looks like this:
    
    	1. Apply convolutional layer with stride and kernel size specified
		- note: uses hard-coded in_channels and out_channels
		- read the problems to figure out what these should be!
	2. Apply the activation function (ReLU)
	3. Apply 2D max pooling with a kernel size of 2

    Inputs: 
    kernel_size: list of length 3 containing kernel sizes for the three convolutional layers
                 e.g., kernel_size = [(5,5), (3,3),(3,3)]
    stride: list of length 3 containing strides for the three convolutional layers
            e.g., stride = [(1,1), (1,1),(1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Synth_Classifier, self).__init__()
        #        [(6, 3, 5, 5), (64, 16, 5, 5)]
        self.conv1 = torch.nn.Conv2d(1, 2, kernel_size= kernel_size[0][0], stride= stride[0][0])   #first two terms are: (input_channel, output channel)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(2,4, kernel_size= kernel_size[1][0], stride= stride[1][0])
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(4,8, kernel_size= kernel_size[1][0], stride= stride[1][0])
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        #here it really depends on your input size!!
        self.fc1 = torch.nn.Linear(8,2)
        
    def forward(self, input):

        x = F.relu(self.conv1(input.view(1,1,28, 28)))
        #(16, 32, 32)
        x = self.pool1(x)

        # ([1, 32, 52, 52])
        x = F.relu(self.conv2(x))
        # 32, 48, 48])
        x = self.pool2(x)

        # ([1, 32, 52, 52])
        x = F.relu(self.conv3(x))
        # 32, 48, 48])
        x = self.pool3(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        return x















