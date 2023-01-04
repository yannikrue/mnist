import numpy
import matplotlib.pyplot
from tqdm import tqdm
from PIL import Image
import tkinter as tk

from nn import neuralNetwork
from webcam import webcam

class Main:
    def __init__(self, i, h, o, l):
        self.input_nodes = i
        self.hidden_nodes = h
        self.output_nodes = o
        self.learning_rate = l
        self.nn = neuralNetwork(i, h, o, l)

    def loadModel(self):
        self.nn.loadModel()
    
    def trainModel(self, textField):
        # load the mnist training data CSV file into a list
        training_data_file = open("/Users/ynk/Desktop/code/mnist/mnist_dataset/mnist_train.csv", 'r')
        training_data_list = training_data_file.readlines()
        training_data_file.close()
        # train the neural network

        # epochs is the number of times the training data set is used for training
        epochs = 5
        length = len(training_data_list)

        textField.insert(tk.END, "Epochs:", epochs)
        for e in range(epochs):
            # go through all records in the training data set
            correct = 0
            total = 0
            acc = 0
            for record in training_data_list:
                # split the record by the ',' commas
                all_values = record.split(',')
                # scale and shift the inputs
                inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = numpy.zeros(self.output_nodes) + 0.01
                # all_values[0] is the target label for this record
                targets[int(all_values[0])] = 0.99
                guess = self.nn.train(inputs, targets)
                total += 1
                if guess:
                    correct += 1
                acc = round(correct * 100 / total)
                if total%1000 == 0:
                    msg = "Epoch {}/{}\nCurrent accuracy {}%\nSample {}/{}".format(e+1, epochs, acc, total, length)
                    self.printText(textField, msg)
                    textField.update_idletasks()

    def openCamera(self, textField):
        values = webcam.video().split(',')
        inputs = (numpy.asfarray(values) / 255.0 * 0.99) + 0.01
        outputs = self.nn.query(inputs)

        prediction = numpy.argmax(outputs)
        probability = numpy.round(outputs[prediction][0], 2)
        msg = "Prediction:", prediction, "with a probability of", probability
        self.printText(textField, msg)

    def runDrawing(self, textField):
        im = Image.open("assets/image.png")
        gray_im = im.convert("L")
        pixels = list(gray_im.getdata())
        pixels_str = ",".join(str(p) for p in pixels)

        values = pixels_str.split(',')
        inputs = (numpy.asfarray(values) / 255.0 * 0.99) + 0.01
        outputs = self.nn.query(inputs)

        prediction = numpy.argmax(outputs)
        probability = numpy.round(outputs[prediction][0], 2)
        msg = "Prediction:", prediction, "with a probability of", probability
        self.printText(textField, msg)

    def performance(self, textField):
        # load the mnist test data CSV file into a list
        test_data_file = open("/Users/ynk/Desktop/code/mnist/mnist_dataset/mnist_test.csv", 'r')
        test_data_list = test_data_file.readlines()
        test_data_file.close()

        # test the neural network

        # scorecard for how well the network performs, initially empty
        scorecard = []

        # go through all the records in the test data set
        for record in test_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # correct answer is first value
            correct_label = int(all_values[0])
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # query the network
            outputs = self.nn.query(inputs)
            # the index of the highest value corresponds to the label
            label = numpy.argmax(outputs)
            # append correct or incorrect to list
            if (label == correct_label):
                # network's answer matches correct answer, add 1 to scorecard
                scorecard.append(1)
            else:
                # network's answer doesn't match correct answer, add 0 to scorecard
                scorecard.append(0)
                pass
            pass

        # calculate the performance score, the fraction of correct answers
        scorecard_array = numpy.asarray(scorecard)
        msg = "Performance =", (scorecard_array.sum() * 100/ scorecard_array.size), "%"
        self.printText(textField, msg)

    def printText(self, textField, msg):
        textField.delete("1.0", tk.END)
        textField.insert(tk.END, msg)