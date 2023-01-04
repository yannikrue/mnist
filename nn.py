import numpy
import scipy.special
import pickle

# Class for the neural netowrk
class NeuralNetwork:
    
    # Construction for the neural network
    def __init__(self, i, h, o, lr):

        # Set architecture for neural network
        self.inputnodes = i
        self.hiddennodes = h
        self.outputnodes = o
        self.learningrate = lr
        
        # Create weight matrices which define the connections between each layer
        self.weights_in_hi = numpy.random.normal(0.0, pow(self.inputnodes, -0.5), (self.hiddennodes, self.inputnodes))
        self.weights_hi_out = numpy.random.normal(0.0, pow(self.hiddennodes, -0.5), (self.outputnodes, self.hiddennodes))
    
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)        
        pass

    
    # Method to train the neural network
    def train(self, inputs_list, targets_list):

        # Prepare the data
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # Calculate inputs and values of hidden layer
        hidden_inputs = numpy.dot(self.weights_in_hi, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate inputs and value of output layer
        final_inputs = numpy.dot(self.weights_hi_out, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        # Calculate error of outputs
        output_errors = targets - final_outputs

        # Calculate error in hidden layser by spliting the output error by the weights
        hidden_errors = numpy.dot(self.weights_hi_out.T, output_errors) 
        
        # Correct weights of the layer based on the error
        self.weights_hi_out += self.learningrate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.weights_in_hi += self.learningrate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        # Checks for accuracy of this training run
        if numpy.argmax(targets) == numpy.argmax(final_outputs):
            return True
        else:
            return False
            
    # Method saves the weights into a pickle file
    def saveModel(self):
        with open('assets/model.pkl', 'wb') as f:
            pickle.dump(self.weights_in_hi, f)
            pickle.dump(self.weights_hi_out, f)
            pass
        pass

    # Method loads the weights from a pickle file
    def loadModel(self):
        with open('assets/model.pkl', 'rb') as f:
            self.weights_in_hi = pickle.load(f)
            self.weights_hi_out = pickle.load(f)
            pass
        pass
    
    # Method queries the neural network
    def query(self, inputs_list):
        # Prepare data
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # Calculate inputs and values of hidden layer
        hidden_inputs = numpy.dot(self.weights_in_hi, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate inputs and value of output layer
        final_inputs = numpy.dot(self.weights_hi_out, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs