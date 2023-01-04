import numpy
from PIL import Image
import tkinter as tk

from nn import NeuralNetwork
from webcam import webcam

# Class that manages backend like neural network training, query, webcam and data preparation
class Main:

    # Constructor that initialises the neural network
    def __init__(self, i, h, o, l):
        self.input_nodes = i
        self.hidden_nodes = h
        self.output_nodes = o
        self.learning_rate = l
        self.nn = NeuralNetwork(i, h, o, l)
        pass

    # Loads a presaved model from the neural network
    def loadModel(self):
        self.nn.loadModel()
        pass
    
    # Method loads training data and trains the neural network
    # updates text field in the gui with current training stats
    def trainModel(self, textField):

        # Load data from file
        training_data_file = open("/Users/ynk/Desktop/code/mnist/mnist_dataset/mnist_train.csv", 'r')
        training_data_list = training_data_file.readlines()
        training_data_file.close()

        # epochs is the number the data is fed trough the neural network
        epochs = 5
        length = len(training_data_list)

        for e in range(epochs):
            correct = 0
            total = 0
            acc = 0

            # Loop goes trough every datapoint in dataset
            for record in training_data_list:

                # Prepare input and target values
                all_values = record.split(',')
                inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                targets = numpy.zeros(self.output_nodes) + 0.01
                targets[int(all_values[0])] = 0.99

                # Pass trough neural network
                guess = self.nn.train(inputs, targets)
                total += 1

                # Counts correct guessed datapoints for accuracy
                if guess:
                    correct += 1
                    pass
                acc = round(correct * 100 / total)

                # Updates text field every 1000th datapoint
                if total%1000 == 0:
                    msg = "Epoch {}/{}\nCurrent accuracy {}%\nSample {}/{}".format(e+1, epochs, acc, total, length)
                    self.printText(textField, msg)
                    textField.update_idletasks()
                    pass
                pass
            pass
        pass

    # Method takes input from webcam and predicts number
    # displays it to the text field in the gui
    def openCamera(self, textField):
        
        # Prepare inputs
        values = webcam.video().split(',')
        inputs = (numpy.asfarray(values) / 255.0 * 0.99) + 0.01

        # Pass trough neural network and calculate predictions
        outputs = self.nn.query(inputs)
        prediction = numpy.argmax(outputs)
        probability = numpy.round(outputs[prediction][0], 2)

        # Display Prediction in text field
        msg = "Prediction: {} with a probability of {}%".format(prediction, probability)
        self.printText(textField, msg)
        pass

    # Method runs the drawing trough the neural network and
    # displays the prediction in text field of the gui
    def runDrawing(self, textField):

        # Prepare inputs from image
        im = Image.open("assets/image.png")
        gray_im = im.convert("L")
        pixels = list(gray_im.getdata())
        pixels_str = ",".join(str(p) for p in pixels)

        values = pixels_str.split(',')
        inputs = (numpy.asfarray(values) / 255.0 * 0.99) + 0.01
        
        # Pass trough neural network and calculate predictions
        outputs = self.nn.query(inputs)
        prediction = numpy.argmax(outputs)
        probability = numpy.round(outputs[prediction][0], 2)

        # Display Prediction in text field
        msg = "Prediction: {} with a probability of {}%".format(prediction, probability)
        self.printText(textField, msg)
        pass

    # Method test the neural network on a never seen test dataset
    def performance(self, textField):
        
        # Load data from file
        test_data_file = open("/Users/ynk/Desktop/code/mnist/mnist_dataset/mnist_test.csv", 'r')
        test_data_list = test_data_file.readlines()
        test_data_file.close()

        correct = 0
        total = 0
        acc = 0

        # Loop trough all recors in da test dataset
        for record in test_data_list:

            # Prepare inputs
            all_values = record.split(',')
            correct_label = int(all_values[0])
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # Pass trough neural network
            outputs = self.nn.query(inputs)
            label = numpy.argmax(outputs)
            total += 1

            if label == correct_label:
                correct += 1
                pass
            pass
        
        # Display test accuracy in text field
        acc = round(correct * 100 / total)
        msg = "Performance = {}%".format(acc)
        self.printText(textField, msg)
        pass

    # Method takes text field element and message and displays it to it
    def printText(self, textField, msg):
        textField.delete("1.0", tk.END)
        textField.insert(tk.END, msg)
        pass
    pass
