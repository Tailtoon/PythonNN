import sys
import pygame
import struct
import numpy
import pickle
import time

trainCoef = 0.8

class NN:
    def __init__(self, amountL, *args):
        self.amountL = amountL
        self.layers = []
        self.length = len(args)
        for i in range(amountL):
            if (i == 0):
                self.layers.append([Neuron(args[i]) for j in range(args[i])])
            else:
                self.layers.append([Neuron(args[i - 1]) for j in range(args[i])])
        

    def train(self, x, y):
        self.answer(x)
        # delta calculation
        i = self.amountL - 1
        for layer in reversed(self.layers):
            j = 0
            for neuron in layer:
                if (i == self.amountL - 1):
                    neuron.lastDelta = y[j] - neuron.lastOUT
                else:
                    summ = 0
                    for prevNeuron in self.layers[i + 1]:
                        summ += prevNeuron.w[j] * prevNeuron.lastDelta
                    neuron.lastDelta = summ
                j += 1
            i -= 1
        # weight change
        i = 0
        for layer in self.layers:
            for neuron in layer:
                neuronPart = trainCoef * neuron.lastDelta * neuron.derivative()
                #print(neuronPart, end='$')
                for j in range(len(neuron.w)):
                    if (i == 0):
                        neuron.w[j] = neuron.w[j] + neuronPart * x[j]
                    else:
                        neuron.w[j] = neuron.w[j] + neuronPart * self.layers[i - 1][j].lastOUT
                    #print(neuronPart, neuronPart * x[j], end="%\n")
            i += 1


    def answer(self, x):
        curx = x
        for layer in self.layers:
            y = numpy.empty(len(layer))
            i = 0
            for neuron in layer:
                y[i] = neuron.out(curx)
                i += 1
            curx = y
        return curx

class Neuron:
    def __init__(self, n):
        self.n = n
        self.w = numpy.random.uniform(-1, 1, n)
        self.lastOUT = 0
        self.lastDelta = 0

    def activation(self, NET):
        return 1/(1+numpy.exp(-NET))

    def derivative(self):
        return (self.lastOUT * (1 - self.lastOUT))

    def out(self, x):
        NET = numpy.dot(x, self.w)
        self.lastOUT = self.activation(NET)
        return self.lastOUT

def maxInList(l):
    maxPos = 0
    maximum = 0
    i = 0
    for item in l:
        if (item > maximum):
            maximum = item
            maxPos = i
        i += 1
    return maxPos

def main():
    time0 = time.process_time()

    print(sys.byteorder)
    trainFile = open("train-images.idx3-ubyte", "rb")
    trainLabelsFile = open("train-labels.idx1-ubyte", "rb")
    testFile = open("t10k-images.idx3-ubyte", "rb")
    testLabelsFile = open("t10k-labels.idx1-ubyte", "rb")

    magic, amount, rows, columns = struct.unpack(">4i", trainFile.read(16))
    print(magic, amount, rows, columns)

    magic, amount = struct.unpack(">2i", trainLabelsFile.read(8))

    trainList = []
    trainLabelList = []
    for i in range(amount - 59000):
        trainX = numpy.empty(rows * columns)
        trainY = numpy.zeros(10)
        for j in range(rows * columns):
            trainX[j] = int(struct.unpack(">B", trainFile.read(1))[0]) / 255
        label = int(struct.unpack(">B", trainLabelsFile.read(1))[0])
        trainY[label] = 1.
        trainList.append(trainX)
        trainLabelList.append(trainY)
    print(len(trainList), len(trainLabelList))

    # try:
    #     inp = open("trained.nn", "rb")
    #     NeuronNetwork = pickle.load(inp)
    #     inp.close()
    #     print("NN found")
    # except:
    NeuronNetwork = NN(2, 28 * 28, 10)
    print("Train started")
    print()
    for i in range(amount - 59000):
        print('\r', i, end='')
        NeuronNetwork.train(trainList[i], trainLabelList[i])
    #Check one 
    tmp = NeuronNetwork.answer(trainList[0])
    print(tmp)
    print(maxInList(tmp), maxInList(trainLabelList[0]))

    print(time.process_time() - time0)

    # error = 0
    # good = 0
    # for i in range(amount - 59000):
    #     ans = NeuronNetwork.answer(trainList[i])
    #     error += numpy.linalg.norm(numpy.array(ans) - numpy.array(trainLabelList[i]))
    #     if (maxInList(ans) == maxInList(trainLabelList[i])):
    #         good += 1
    #     #print(maxInList(ans), maxInList(trainLabelList[i]))
    # print(good, amount - 59000 - good, error)

    # outp = open("trained.nn", "wb")
    # pickle.dump(NeuronNetwork, outp, pickle.HIGHEST_PROTOCOL)
    # outp.close()

    #print(NeuronNetwork.answer(trainX))

    trainFile.close()
    trainLabelsFile.close()
    testFile.close()
    testLabelsFile.close()
        
def pgcheckimages():
    HEIGHT = 450
    WIDTH = 400
    FPS = 60

    FramePerSec = pygame.time.Clock()

    surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Check hand. numbers")

    print(sys.byteorder)
    file = open("train-images.idx3-ubyte", "rb")
    magic, amount, rows, columns = struct.unpack(">4i", file.read(16))
    print(magic, amount, rows, columns)

    surface.fill((0, 0, 0))

    for am in range(4):
        for i in range(rows):
            for j in range(columns):
                colortmp = int(struct.unpack(">B", file.read(1))[0])
                pix_array = pygame.PixelArray(surface)
                pix_array[28 * am + j, 28 * am + i] = (colortmp, colortmp, colortmp)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        pygame.display.update()

if __name__ == "__main__":
    main()