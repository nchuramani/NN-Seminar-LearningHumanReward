# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from Utils import DataUtil, ImageProcessingUtil

from tochange import MCCNNExperiments_old
import os
import cv2
from Networks import MCCNN
import numpy
import matplotlib.pyplot as plt
import theano
from theano import tensor as T
import time
from _collections import defaultdict
from iCubPhoto import takeSnapshot
import shutil


def getPictures():
    images = []
    #os.makedirs("/tmp/gestures/")
    if os.path.exists("/tmp/gestures/"): 
        shutil.rmtree("/tmp/gestures")        
        
    os.makedirs("/tmp/gestures/")

    print "Taking snapshots"
    f = takeSnapshot("/tmp/gestures/", 10, 1)
                
    while (len(os.listdir("/tmp/gestures")) <10):
          time.sleep(1)
          
    imagesList = os.listdir("/tmp/gestures")
    print "Image list size:", imagesList
    print "Image list size:", len(imagesList)
    for img in imagesList:
        frame = cv2.imread("/tmp/gestures/"+img)
        images.append(frame)
    return images          
              
def initEmotion():
    
    print "Init Emotion for iCub : Neutral"
    emoString = "echo 'set all hap '  | yarp rpc /icub/face/emotions/in  " 
    f=os.popen(emoString)
    emoString = "echo 'set raw M07 '  | yarp rpc /icub/face/emotions/in  " 
    #print emoString
    f=os.popen(emoString)
          
def detectFace(img):
    img_copy = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(
        "/CNN/haarcascade_frontalface_alt.xml")
    rects = cascade.detectMultiScale(img, 1.2, 3, 1, (20, 20))

    if len(rects) == 0:

        #Return complete image if no face is detected
        x, y = numpy.asarray(img).shape
        return img_copy, (0, 0, x, y)
    rects[:, 2:] += rects[:, :2]

    return box(rects, img_copy)


def box(rects, img):
    for x1, y1, x2, y2 in rects:
        # cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
        img = img[y1:y2, x1:x2]
        return img, (x1, y1, x2, y2)

def readFile(fileDirectory):
    images = []
    targetFile = open(fileDirectory, 'r')
    for image in targetFile:
        image = image.split(",")
        image = [float(i) for i in image]
        images.append(image)
    targetFile.close()
    #print images
    return numpy.array(images,dtype=theano.config.floatX)

def writeFile(fileDirectory, inputVector):
    targetFile = open(fileDirectory, 'w')
    for number in inputVector:
        #for value in range(len(number)):
        targetFile.write(str(number))
        #if not number == len(number)-1:
        #    targetFile.write(",")
        targetFile.write("\n")
    targetFile.close()

def loadData(imageDirectory,network,trainingParameters,networkTopology):
    valueForRangeCK = os.listdir(imageDirectory)
    for f in valueForRangeCK:
        if f.startswith('.'):
            valueForRangeCK.remove(f)
    for c in valueForRangeCK:
        images = os.listdir(imageDirectory + "/" + c)
        for f in images:
            if f.startswith('.'):
                images.remove(f)
        images = sorted(images, key=lambda x: int(x.split(".")[0]))
        print "S:", images
        for imgs in images:
            print "Reading:", imageDirectory + "/" + c + "/" + imgs
            # change
            img = cv2.imread(imageDirectory + "/" + c + "/" + imgs)
            # change
            image, rects = ImageProcessingUtil.detectFace(img)
            print rects
            image = ImageProcessingUtil.resize(image, networkTopology[6][0][0][4])

            Output = MCCNN.classify(network[-1], [image], trainingParameters[4])[0]

            Output += 1


            inputVector.append(Output)

    return inputVector

# NeuralNetwork Class
class Neural_Network(object):
    def __init__(self):
        # Define HyperParameters
        self.T = [[1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, 4], [1, 5, 5], [1, 6, 6], [1, 7, 7],
                  [2, 1, 1], [2, 2, 2], [2, 3, 3], [2, 4, 4], [2, 5, 5], [2, 6, 6], [2, 7, 7],
                  [3, 1, 1], [3, 2, 2], [3, 3, 3], [3, 4, 4], [3, 5, 5], [3, 6, 6], [3, 7, 7],
                  [4, 1, 1], [4, 2, 2], [4, 3, 3], [4, 4, 4], [4, 5, 5], [4, 6, 6], [4, 7, 7],
                  [5, 1, 1], [5, 2, 2], [5, 3, 3], [5, 4, 4], [5, 5, 5], [5, 6, 6], [5, 7, 7],
                  [6, 1, 1], [6, 2, 2], [6, 3, 3], [6, 4, 4], [6, 5, 5], [6, 6, 6], [6, 7, 7],
                  [7, 1, 1], [7, 2, 2], [7, 3, 3], [7, 4, 4], [7, 5, 5], [7, 6, 6], [7, 7, 7]]

        self.inputLayerSize = 1
        self.outputLayerSize = 1
        self.hiddenLayer1Size = 3
        self.hiddenLayer2Size = 3

        self.actions=['NoAction','Anger', 'Disgust', 'Fear','Happy', 'Neutral', 'Sadness', 'Surprise']
        # Weights

        self.w1 = numpy.random.randn(self.inputLayerSize, self.hiddenLayer1Size)
        self.w2 = numpy.random.randn(self.hiddenLayer1Size, self.hiddenLayer2Size)
        self.w3 = numpy.random.randn(self.hiddenLayer2Size, self.outputLayerSize)

        # other vars
        self.z2 = numpy.zeros((1, self.hiddenLayer1Size))
        self.a2 = numpy.zeros((1, self.hiddenLayer1Size))
        self.z3 = numpy.zeros((1, self.hiddenLayer2Size))
        self.a3 = numpy.zeros((1, self.hiddenLayer2Size))
        self.z4 = numpy.zeros((1, self.outputLayerSize))

    def forward(self, x):
        # Propagate inputs through net
        self.z2 = numpy.dot(x, self.w1)

        self.a2 = self.sigmoid(self.z2)

        self.z3 = numpy.dot(self.a2, self.w2)


        self.a3 = self.sigmoid(self.z3)

        self.z4 = numpy.dot(self.a3, self.w3)


        y_pred = 2.0 * numpy.tanh(self.z4)


        return y_pred

    def sigmoid(self, z, deriv=False):
        # Apply Sigmoid activation function Or Derivative of Sigmoid
        if (deriv == True):
            return numpy.exp(-z) / ((1 + numpy.exp(-z)) ** 2)
        return 1 / (1 + numpy.exp(-z))
    def dtanh(self,z):
        # Apply Derivative of Tanh

        return 1 - numpy.tanh(z)**2

    def costFunction(self, x, training):

        y_pred, a_pred = self.chooseAction(x)
        if training :
            
            #y1 = int(input("Enter Reward: "))
            y= self.computeReward(x, a_pred)
            print "Computed Reward: ", y
            J = 0.5 * ((y - y_pred)**2)
            print "Cost: ", J
            return J, y, y_pred, a_pred
        else:
            return y_pred, a_pred

    def costFunctionPrime(self, x, y, y_pred, a_pred):
        # Compute Derivative w.r.t w1, w2

        s_t1 = self.getNewState(x,a_pred)
        s_diff = (s_t1 - int(x[0]))

        y_pred = self.forward(s_diff)

        delta4 = numpy.multiply(- (y - y_pred), 2.0 * self.dtanh(self.z4))


        djdw3 = numpy.dot(self.a3.T, delta4)

        delta3 = numpy.multiply(numpy.dot(delta4, self.w3.T), self.sigmoid(self.z3, deriv=True))

        djdw2 = numpy.dot(self.a2.T, delta3)

        delta2 = numpy.multiply(numpy.dot(delta3, self.w2.T), self.sigmoid(self.z2, deriv=True))

        djdw1 = numpy.dot(s_diff, delta2)

        return djdw1, djdw2, djdw3

    def getParams(self):
        # Get w1, w2 and w3 in a vector
        params = numpy.concatenate((self.w1.ravel(), self.w2.ravel(), self.w3.ravel()))
        return params

    def setParams(self, params):
        # Set w1, w2, w3 using single param method
        w1_start = 0
        w1_end = self.hiddenLayer1Size * self.inputLayerSize
        self.w1 = numpy.reshape(params[w1_start:w1_end], (self.inputLayerSize, self.hiddenLayer1Size))
        w2_end = w1_end + self.hiddenLayer2Size * self.hiddenLayer1Size
        self.w2 = numpy.reshape(params[w1_end:w2_end], (self.hiddenLayer1Size, self.hiddenLayer2Size))
        w3_end = w2_end + self.outputLayerSize * self.hiddenLayer2Size
        self.w3 = numpy.reshape(params[w2_end:w3_end], (self.hiddenLayer2Size, self.outputLayerSize))

    def update(self, x, y, y_pred, cost, a_pred):
        djdw1, djdw2, djdw3 = self.costFunctionPrime(x, y, y_pred, a_pred)
        learningrate = 0.1

        self.w1 = self.w1 - learningrate * djdw1
        self.w2 = self.w2 - learningrate * djdw2
        self.w3 = self.w3 - learningrate * djdw3


    def getNewState(self, x, action):
        newState = 0
        for i in range(len(self.T)):
            if self.T[i][0] == int(x[0]) and self.T[i][1] == action:
                newState = self.T[i][2]
                break
        return newState


    def getActions(self, x):
        getaction = []
        for i in range(len(self.T)):
            if self.T[i][0] == int(x[0]):
                getaction.append(self.T[i][1])
        return getaction

    def chooseAction(self, x):
        projectedReward = []
        print "Input: ", x[0]
        for a in self.getActions(x):
            # print "Action: ", a
            s_t1 = self.getNewState(x, action=a)
            s_diff = (s_t1 - x[0])
            # print "S_diff: ", s_diff
            reward = self.forward(s_diff)
            projectedReward.append(reward)
            # print "Reward: " + str(a) + " : ", reward

        # print "Max Reward:", numpy.amax(projectedReward)
        self.selectActionToDisplay(numpy.argmax(projectedReward) + 1)

        return numpy.amax(projectedReward), numpy.argmax(projectedReward) + 1

    def selectActionToDisplay(self, action_pred):


        if action_pred == 1:
            print "Action for Max Reward: ", action_pred, self.actions[action_pred]
            
            emoString = "echo 'set all ang '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)        
            emoString = "echo 'set raw M07 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)


        elif (action_pred == 2):

            print "Action for Max Reward: ", action_pred, self.actions[action_pred]
            
            emoString = "echo 'set raw R05 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)
            emoString = "echo 'set raw L05 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)
            emoString = "echo 'set raw M08 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)

            
            

        elif (action_pred == 3):

            print "Action for Max Reward: ", action_pred, self.actions[action_pred]
            
            emoString = "echo 'set raw R05 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)
            emoString = "echo 'set raw L05 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)
            emoString = "echo 'set raw M04 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)


        elif (action_pred == 4):

            print "Action for Max Reward: ", action_pred, self.actions[action_pred]
            
            emoString = "echo 'set all hap '  | yarp rpc /icub/face/emotions/in  " 

            #print emoString
            f=os.popen(emoString)

        elif (action_pred == 5):

            print "Action for Max Reward: ", action_pred, self.actions[action_pred]
            emoString = "echo 'set all hap '  | yarp rpc /icub/face/emotions/in  " 
            f=os.popen(emoString)
            emoString = "echo 'set raw M07 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)
            
        elif (action_pred == 6):

            print "Action for Max Reward: ", action_pred, self.actions[action_pred]
            
            emoString = "echo 'set raw R05 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)
            emoString = "echo 'set raw L05 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)
            emoString = "echo 'set raw M07 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)

        elif (action_pred == 7):

            print "Action for Max Reward: ", action_pred, self.actions[action_pred]
            
            emoString = "echo 'set raw R0C '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)
            emoString = "echo 'set raw L0C '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)
            emoString = "echo 'set raw M05 '  | yarp rpc /icub/face/emotions/in  " 
            #print emoString
            f=os.popen(emoString)



    def computeReward(self, x, a):
        #Actions=['Anger', 'Disgust', 'Fear','Happy', 'Neutral', 'Sadness', 'Surprise']
        
        positive = [4, 7]
        negative = [1, 2, 3, 6]
        '''
            Implicit Reward
        '''
        emotionClass = []
        frames = getPictures()

        if len(frames) > 1:
            for i in range(len(frames)):
                faceImg, rects = detectFace(frames[i])
                faceImg = ImageProcessingUtil.resize(faceImg, networkTopology[6][0][0][4])
                emotionClass.append(MCCNN.classify(network[-1], [faceImg], trainingParameters[4])[0] + 1)

            d = defaultdict(int)
            for i in emotionClass:
   				d[i] += 1
            result = max(d.iteritems(), key=lambda yy: yy[1])
            print "Result: ", result[0]

        if result[0] in positive:
            y = 2.0
        elif result[0] in negative:
            y = -2.0
        else:
            y = 0

        '''
            Explicit Reward
        '''
        # if(x == result[0]):
        #    y = 2.0
        #    #print "Yes: ", a
        # elif(x in positive and result[0] in positive ):
        #    y = 1.0
        # elif(x in negative and result[0] in negative):
        #    y = 1.0
        # elif(x in positive and result[0] in negative or (x in negative and result[0] in positive)):
        #    y = -2.0
        # elif(x == 5 or result[0]):
        #    y = -1.0
        # else:
        #    y = 0.0
        return y




# Trainer Class to train the network
class trainer(object):
    def __init__(self, N):
        self.N = N
        self.J = []
        self.J1=[]

    def getInputEmotion(self):

        emotionClass = []
        effectEmotion = []
        frames = getPictures()

        if len(frames) > 1:
            for i in range(len(frames)):
                faceImg, rects = detectFace(frames[i])
                faceImg = ImageProcessingUtil.resize(faceImg, networkTopology[6][0][0][4])
                emotionClass.append(MCCNN.classify(network[-1], [faceImg], trainingParameters[4])[0] + 1)
                print emotionClass
            d = defaultdict(int)
            for i in emotionClass:
                d[i] += 1
            result = max(d.iteritems(), key=lambda x: x[1])
            effectEmotion.append(result[0])
            
            if (len(effectEmotion)) > 0:
                ret = True
            else:
                ret = False

            #  Show Frame
            #x1, y1, x2, y2 = rects
            #if (effectEmotion[0] == 1):
            #    cv2.rectangle(frames[len(frames) - 1], (x1, y1), (x2, y2), (0, 0, 255), 2)
            #elif (effectEmotion[0] == 2):
            #    cv2.rectangle(frames[len(frames) - 1], (x1, y1), (x2, y2), (255, 255, 0), 2)
            #elif (effectEmotion[0] == 3):
            #    cv2.rectangle(frames[len(frames) - 1], (x1, y1), (x2, y2), (0, 255, 0), 2)
            #elif (effectEmotion[0] == 4):
            #    cv2.rectangle(frames[len(frames) - 1], (x1, y1), (x2, y2), (255, 0, 255), 2)
            #elif (effectEmotion[0] == 5):
            #    cv2.rectangle(frames[len(frames) - 1], (x1, y1), (x2, y2), (0, 255, 255), 2)
            #elif (effectEmotion[0] == 6):
            #    cv2.rectangle(frames[len(frames) - 1], (x1, y1), (x2, y2), (192, 192, 192), 2)
            #elif (effectEmotion[0] == 7):
            #    cv2.rectangle(frames[len(frames) - 1], (x1, y1), (x2, y2), (255, 0, 0), 2)
            #cv2.imshow('frame', frames[len(frames) - 1])
            #cv2.waitKey(20)
            

        else:
            print "Frame Drop"

        return effectEmotion, ret

    def getCamera(self):
        # When working with PC
        cap = cv2.VideoCapture(0)

        return cap

    def costFunctionWrapper(self, x):

        cost, y, y_pred, a_pred =  self.N.costFunction(x, training=True)
        self.J.append(cost)

        self.N.update(x, y, y_pred, cost, a_pred)

    def train(self, trainX, repetitions, live=False):

        if live == True:
            print "Running Training for ", str(repetitions), " epochs"
            for i in range(repetitions):
                print "Online learning with Human. Epoch: ", i+1

                #cap = self.getCamera()
                i = 1
                #Number of interactions per epoch
                while(i <=100):
                    i+=1
                    emoString = "echo 'set raw M00 '  | yarp rpc /icub/face/emotions/in  "
                    #print emoString
                    f=os.popen(emoString)
                    emoString = "echo 'set raw R00 '  | yarp rpc /icub/face/emotions/in  "
                    #print emoString
                    f=os.popen(emoString)
                    emoString = "echo 'set raw L00 '  | yarp rpc /icub/face/emotions/in  "
                    #print emoString
                    f=os.popen(emoString)
                    gett = int(input("Get Image "))
                    inputClass, ret = self.getInputEmotion()
                    if ret:
                        self.costFunctionWrapper(inputClass)

                if repetitions > 1:
                    cost = 0
                    for j in range(len(self.J)):
                        cost += self.J[j]
                    avgcost = cost / len(self.J)
                    print "AvgCost: ", avgcost
                    self.J1.append(avgcost)

                else:
                    self.J1 = self.J

        else:

            print "Running Training for ", str(repetitions), " epochs"
            for i in range(repetitions):
                print "Epoch: ", i

                for i in range(len(trainX)):
                    self.costFunctionWrapper(trainX[i])
                if repetitions > 1:
                    cost = 0
                    for j in range(len(self.J)):
                        cost += self.J[j]
                    avgcost = cost / len(self.J)
                    # self.J = []
                    print "AvgCost: ", avgcost
                    self.J1.append(avgcost)
                    # if avgcost == 0:
                    #     break
                else:
                    self.J1 = self.J


                #self.J = []

    def test(self,testX):

        for i in range(len(testX)):
            y_pred, a_pred= self.N.costFunction(testX[i], training=False)

            print "Input: ", self.N.actions[int(testX[i][0])], " Action: ", self.N.actions[a_pred]




#Vector for MLP input
inputVector = []

#Vector for MLP Output

outputVector=[]
# change
modelDirectory = "/CNN/model/repetition_13_BestTest_testCK9_.save"
# change
imageDirectory = "/CK"

imageTestDirectory = ""

#change
trainDataDirectory = "/MLP/Seminar_train.txt"
trainTestDataDirectory = "/MLP/Seminar_test.txt"
mlpweights = "/MLP/weights.txt"



if __name__ == "__main__":
    # change
    networkTopology, trainingParameters, experimentParameters, visualizationParameters, networkState = DataUtil.loadNetworkState(
         modelDirectory)
    #
    #experimentParameters[0] = ""
    # # change
    experimentParameters[0] = os.path.dirname(os.path.abspath(__file__))
    experimentParameters.append(False)
    # #change
    saveNetworkParameters = [False]
    # # change
    network = MCCNNExperiments_old.runExperiment(networkTopology, trainingParameters, experimentParameters,
                                                  visualizationParameters, saveNetworkParameters)
    #
    # #change
    # inputVector = loadData(imageDirectory,network,trainingParameters, networkTopology)
    # writeFile(trainDataDirectory, inputVector)
    # outputVector = loadData(imageTestDirectory,network,trainingParameters, networkTopology)
    # writeFile(trainTestDataDirectory, outputVector)

    live = True

    trainX = readFile(trainDataDirectory)
    #testX = readFile(trainTestDataDirectory)

    numpy.random.shuffle(trainX)

    users = 1
    repetitions = 10

    for i in range(users):

        NN = Neural_Network()

        TT = trainer(NN)

        if live:
            TT.train(None, repetitions, live=live)
        else:
            TT.train(trainX, repetitions, live=live)

        print "W1: ", TT.N.w1
        print "W2: ", TT.N.w2
        print "W3: ", TT.N.w3
        interation = "User" + str(i + 1)
        if live:
            plt.plot(TT.J, label=interation)
        else:
            plt.plot(TT.J1, label=interation)

    print "READY FOR TESTING"

    # NN = Neural_Network()
    # TT = trainer(NN)
    '''Example of Weights Learnt by the System
    # TT.N.w1 = [[-1.13748894, -1.22647179, 0.83771153]]
    # TT.N.w2 = [[ 0.16759007, -1.52348757, -1.05919119],
    #            [-0.93934464, -3.01719703,  1.21539892],
    #            [-0.94775398, -1.10942093,  0.33797288]]
    # TT.N.w3 = [[ 2.40257395],
    #            [-0.47988554],
    #            [ 3.45891562]]
    '''
    # TT.test(trainX)
    # if live:
    #     plt.plot(TT.J, label=interation)
    # else:
    #     plt.plot(TT.J1, label=interation)

    #
    if live:
        plt.ylabel('Cost')
        plt.xlabel('Iterations')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc="upper center",
                   ncol=5, mode="expand", borderaxespad=0.1)
        plt.savefig('Seminar_Learning_live.png')

    else:
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc="upper center",
                   ncol=2, mode="expand", borderaxespad=0.)
        if repetitions > 1:
            plt.ylabel('Avg. Cost')
            plt.xlabel('Epochs')
            plt.savefig('Seminar_Learning1.png')

        else:
            plt.ylabel('Cost')
            plt.xlabel('Images')
            plt.savefig('Seminar_Learning2.png')

# # -*- coding: utf-8 -*-

