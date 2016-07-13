# -*- coding: utf-8 -*-

import numpy
import cPickle
import os
import theano
import theano.tensor as T
import random
import pylab
import itertools
import cv2




import ImageProcessingUtil
#import AudioProcessingUtil


#INPUT_TYPE =  {"Common": "common", "Color":"color"}
LAYER_TYPE =  {"Common": "common", "SobelX":"sobelX", "SobelY":"sobelY", "Gabor":"gabor", "AudioFilters":"audioFilters"}

LOAD_IMAGES_STRATEGY = {"Batches":0, "LoadAll":1}

IMAGE_STRUCTURE = {"Sequence":"Sequence", "Static":"Static", "StaticInSequence":"StaticInSequence"}

FEATURE_TYPE = {"GrayScale":("grayScale",1), "RGB":("RGB",3), "Whiten":("whiten",1), 
                "MFCC":("MFCC",1), "MFCC_Delta":("MFCC_Delta",1),"MFCC_DeltaDelta":("MFCCMFCC_DeltaDelta",1), 
                "POW":("POW",1), "MAG":("MAG",1), "POW_MAG":("POW_MAG",2)}

ACTIVATION_FUNCTION = {"Tanh":"Tanh", "ReLU":"ReLU"}

DATA_MODALITY = {"Image":"Image", "Face":"Face","Audio":"Audio"}

HIDDEN_LAYER_TYPE = {"Common":"Common", "LSTM":"LSTM", "Splitted":"Splitted"}

OUTPUT_LAYER_TYPE = {"Classification":"Classification", "Localization":"Localization"}

GABOR_FILTERS_DIRECTORY = "/informatik/isr/wtm/home/barros/Desktop/old/ijcnnExperiments/jaffe_cae_GPU0/20/params"

 
#Method that reads the file containing the features.
# The feature file must be formated as follows:
    #<label>,<featuress eparated with a coma>
    #<label>,<featuress eparated with a coma>
    #<label>,<featuress eparated with a coma>
    #etc...
#This method separate the data in two lists:
#one with the label and one with the features.
#These two lists are passed to the separateSets method.    
  
def loadImage(imageSets,channel, dataStructure, batchIndex, batchSize, imageSize, loadImagesStrategy, dataModality, isUsingLSTM):
    
    if isUsingLSTM:
        dataStructure = IMAGE_STRUCTURE["Sequence"]    
        
    if dataStructure == IMAGE_STRUCTURE["Static"] or dataStructure == IMAGE_STRUCTURE["StaticInSequence"]:        
        if loadImagesStrategy == LOAD_IMAGES_STRATEGY["LoadAll"]:            
            images , labelSubSet =  imageSets[0][channel][batchIndex * batchSize: (batchIndex + 1) * batchSize], imageSets[1][channel][batchIndex * batchSize: (batchIndex + 1) * batchSize]                                                                         
        else:    
            imagesSubSet = imageSets[0][channel][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            labelSubSet = imageSets[1][channel][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            
            for image in imagesSubSet:    
                img = prepareData(image, dataModality, imageSize, featureType, sliceSize) 
                if not img == []:                                      
                    images.append(img)                
             
    elif dataStructure == IMAGE_STRUCTURE["Sequence"]:
        if loadImagesStrategy == LOAD_IMAGES_STRATEGY["LoadAll"]:            
            images , labelSubSet =  imageSets[0][channel][batchIndex * batchSize: (batchIndex + 1) * batchSize], imageSets[1][channel][batchIndex * batchSize: (batchIndex + 1) * batchSize]                                                                         
            
        else:
            imagesSubSet = imageSets[0][channel][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            labelSubSet = imageSets[1][channel][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            
            for sequences in imagesSubSet:                
                sequence = []
                addSequence = True                
                for image in sequences:                    
                    img = prepareData(image, dataModality, imageSize, featureType, sliceSize)    
                    if image == []:
                            addSequence = False
                            break                                   
                    sequence.append(img)
                if addSequence:
                    images.append(sequence)                    

    #return (images.eval(), labelSubSet.eval())
    #return (numpy.asarray(images,dtype=theano.config.floatX) , numpy.asarray(labelSubSet,dtype=theano.config.floatX) )
#    print "Shape image vector:", numpy.array(images).shape
#    print "Shape image vector[0]:", numpy.array(images[0]).shape
#    print "Shape image vector[0][0]:", numpy.array(images[0][0]).shape
#    print "Shape image Set vector_X:", numpy.array(imageSets[0]).shape
#    print "Shape image in Set vector_X:", numpy.array(imageSets[0][0][0]).shape
#    print "Shape image Set vector_Y:", numpy.array(imageSets[1]).shape
#    
#    print "Shape label Set vector_Y:", labelSubSet
    if isUsingLSTM:
        newLabelSubSet = []        
        for s in range(len(images[0])):            
                newLabelSubSet.append(labelSubSet[0])            
#        print "newLabelSubset Shape:", numpy.array(newLabelSubSet).shape
        labelSubSet = newLabelSubSet
        
        
        #raw_input("here")       
        
    return (images, labelSubSet)
 
def readFeatureFile(featuresDirectory, randomized,  percentTrain, percentValid, useColor):
        
    #print "USe color:", useColor    
    directory = featuresDirectory        
    f = open(directory, 'r')        
    inputs = []
    outputs = []            
    for line in f:
        li = line.split(",")            
        outputs.append(int(li[0]))
        li.remove(li[0])
        features = [] 
        colorNumber = 0
        color = []
        for i in li:   
            if not i == 0:
                try:
                    if useColor:
                        color.append(i)
                        colorNumber = colorNumber +1                        
                        if(colorNumber==3):                        
                            features.append(color)                            
                            colorNumber = 0                        
                            color = []
                    else:
                        features.append(i)
                except:
                     pass
        
        features = numpy.swapaxes(features, 0,1)        
        inputs.append(features)    
        
    f.close()    


    if randomized:
        return randomizeSet(inputs,outputs)
    else:
        return separateSets(inputs,outputs,percentTrain,percentValid)
    #return (numpy.array(inputs),numpy.array(outputs))



def readFeatureFileColor(featuresDirectory, randomized,  percentTrain, percentValid, color):
        
    directory = featuresDirectory        
    f = open(directory, 'r')        
    inputs = []
    outputs = []            
    for line in f:
        li = line.split(",")            
        outputs.append(int(li[0]))
        li.remove(li[0])
        features = []       
        color = []
        colorNumber = 0
        for i in li:   
            if not i == 0:
                try:          
                    color.append(i)
                    colorNumber = colorNumber +1
                    if(colorNumber==3):
                        features.append(colorNumber)
                        colorNumber = 0
                except:
                     pass
        inputs.append(features)    
        
    f.close()    
    
    
 
    
    if randomized:
        return randomizeSet(inputs,outputs)
    else:
        return separateSets(inputs,outputs,percentTrain,percentValid)
    #return (numpy.array(inputs),numpy.array(outputs))

def randomizeSet(inputs, outputs):
    
    
    positions = []
    for p in range(len(inputs)):
        positions.append(p)
        
    random.shuffle(positions)
        
    newInputs = []
    newOutputs = []
    for p in positions:
        newInputs.append(inputs[p])
        newOutputs.append(outputs[p])
        
    return (newInputs,newOutputs)


#This method separates the set in trhee sub-sets with 
#shufled and ramdonly chose values. Each subset has a 
# pre-defined amount of values, passed as a parameter.
# For each subset a list of positions is created
#and  then filled with sorted values from the original set.
# After the amount of values in each list is reached, the values
# of each position are copied to a final list.
def separateSets(inputSet,outputSet, pTrain, pValid, log, classesLabels):
             
    positionsSetTrain = []
    positionsSetValidate = []
    positionsSetTest = []
    
    patterns = []    
        
    for o in outputSet:
        if not o in patterns:
            patterns.append(o)
     
    for c in patterns:
       outputsInThisClass = []
       for i in range(len(outputSet)):           
           if( outputSet[i] == c):
               outputsInThisClass.append(i)
               
       positionsTrainSet = []
       positionsValidateSet = []
       positionsTestSet = []
       percentTest = int(len(outputsInThisClass)* ( 100-pTrain-pValid)/float(100))
       percentTrain = int(len(outputsInThisClass) * pTrain/float(100))       
       percentValid = int(len(outputsInThisClass) * pValid/float(100))               
       
       log.printMessage(("---Class: ", c, "-", classesLabels[str(c)]))
       log.printMessage(("----Total elements: ", len(outputsInThisClass)))
       
       while len(outputsInThisClass) >0:
                                                 
           if len(positionsTrainSet) <= percentTrain:    
               rnd = random.randint(0,len(outputsInThisClass)-1)
               positionsTrainSet.append(outputsInThisClass[rnd])
               outputsInThisClass.remove(outputsInThisClass[rnd])
           
           if len(positionsValidateSet) <= percentValid:      
               rnd = random.randint(0,len(outputsInThisClass)-1)
               positionsValidateSet.append(outputsInThisClass[rnd])
               outputsInThisClass.remove(outputsInThisClass[rnd])

           if len(positionsTestSet) <= percentTest:
               rnd = random.randint(0,len(outputsInThisClass)-1)
               positionsTestSet.append(outputsInThisClass[rnd])
               outputsInThisClass.remove(outputsInThisClass[rnd])                
       
       log.printMessage(("----Elements in train set: ", len(positionsTrainSet)))
       log.printMessage(("----Elements in test set: ", len(positionsTestSet)))
       log.printMessage(("----Elements in valid set: ", len(positionsValidateSet))) 
        
       for i in positionsTrainSet:
           positionsSetTrain.append(i)
                      
       for i in positionsValidateSet:
           positionsSetValidate.append(i)
           
       for i in positionsTestSet:
           positionsSetTest.append(i)
          
    inputSetTrain = []
    outputSetTrain = []

    inputSetValidate = []
    outputSetValidate = []

    inputSetTest = []
    outputSetTest = []   
    
    
    random.shuffle(positionsSetTrain)
    random.shuffle(positionsSetValidate)
    random.shuffle(positionsSetTest)
    
    for i in positionsSetTrain:
        inputSetTrain.append(inputSet[i])
        outputSetTrain.append(outputSet[i])
    for i in positionsSetValidate:
        inputSetValidate.append(inputSet[i])
        outputSetValidate.append(outputSet[i])    
    for i in positionsSetTest:
        inputSetTest.append(inputSet[i])
        outputSetTest.append(outputSet[i])  
                    
    return ( (numpy.array(inputSetTrain),numpy.array(outputSetTrain)), (numpy.array(inputSetValidate),numpy.array(outputSetValidate)), (numpy.array(inputSetTest),numpy.array(outputSetTest)))


def loadNetworkState (directory):
    
    f = file(directory, 'rb')    
    
    networkTopology = cPickle.load(f)
    
    trainingParameters = cPickle.load(f)
    
    experimentParameters = cPickle.load(f)
    
    visualizationParameters= cPickle.load(f)
    
    networkState = cPickle.load(f)        

    networkTopology[8] = networkState
    
    trainingParameters[0] = False
    trainingParameters[3] = 1
    
    experimentParameters[4] = False
    experimentParameters[3] = 1
    
    #print visualizationParameters
    #visualizationParameters[3] = (True, True, False, False)
            
    return (networkTopology, trainingParameters, experimentParameters, visualizationParameters,networkState)

   
def createNetworkState(parametersChannels, parametersHiddenLayer, parametersCrossChannels, parametersClassifier, channels, crossChannels):
    
    networkState = []      
    channelStates = []
    for channel in range(len(parametersChannels)):       
        channelState = []
        for layer in range(len(parametersChannels[channel])):   
                layerState = []    
                layerState.append(parametersChannels[channel][layer].params)                
                
                usingInhibition = channels[channel][1][layer][4]
                
                if usingInhibition != None:
                    layerState.append(parametersChannels[channel][layer].paramsInhibition)                                
                channelState.append(layerState)
        channelStates.append(channelState)
        
    networkState.append(channelStates)    
    
    crossChannelStates = []
    for channel in range(len(parametersCrossChannels)):       
        channelState = []
        for layer in range(len(parametersCrossChannels[channel])):   
                layerState = []    
                layerState.append(parametersCrossChannels[channel][layer].params)                
                
                usingInhibition = crossChannels[channel][1][layer][4]
                
                if usingInhibition != None:
                    layerState.append(parametersCrossChannels[channel][layer].paramsInhibition)                                
                crossChannelStates.append(layerState)
        crossChannelStates.append(channelState)        
        
    networkState.append(crossChannelStates)     
    
    hiddenLayersState = []     
    for hiddenLayer in range(len(parametersHiddenLayer)):             
             hiddenLayersState.append(parametersHiddenLayer[hiddenLayer].params)
    
    networkState.append(hiddenLayersState)
    
    parametersClassifierState = []    
    for classifierLayer in range(len(parametersClassifier)):             
             parametersClassifierState.append(parametersClassifier[classifierLayer].params)
             
    networkState.append(parametersClassifierState)     
    
    return networkState



def saveNetwork(networkTopology,trainingParameters, experimentParameters, visualizationParameters, networkState, directory):
    
    f = file(directory, 'wb')
    
    cPickle.dump(networkTopology, f, protocol=cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(trainingParameters, f, protocol=cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(experimentParameters, f, protocol=cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(visualizationParameters, f, protocol=cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(networkState, f, protocol=cPickle.HIGHEST_PROTOCOL)
        
        
    f.close()    
    
    
#Method that saves the state of a MCNN
def saveState(params,directory):    
        
    f = file(directory, 'wb')
    for obj in params:                    
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

#Method that load a state of a MCNN
def loadState(directory,parametersToLoad):
    
    f = file(directory, 'rb')
    loaded_objects = []
    for i in range(parametersToLoad):                
        loaded_objects.append(cPickle.load(f))
    f.close()

    return loaded_objects    

#Method that load a state of a MCNN
def loadStates(directory,layers):
    
    #print "Loading ", layers, " state from state in :", directory
    
    f = file(directory, 'rb')
    loaded_objects = []
    for i in range(layers):                
        loaded_objects.append(cPickle.load(f))
    f.close()

    return loaded_objects 


def createTrainingSequence(data,timeStep):
    sequencesX = []    
    for h in range(len(data[0])):
        sequenceX = []
        for i in range(timeStep):
            sequenceX.append(data[0][h])        
        sequencesX.append(sequenceX)        
            
      
    return (sequencesX,data[1])    
            

def createSequence(data, timeStep):
    classes = []
    sequencesX = []
    sequencesY = []
         
    for i in range(len(data[1])):
        if(not data[1][i] in classes):
            classes.append(data[1][i])
            classSequence = []                        
            for j in range(len(data[1])):            
                if(data[1][j] == data[1][i]):                      
                    classSequence.append(data[0][j])                        
            #print "Class: ", data[1][i], " - Elements:", len(classSequence)
            for subset in itertools.combinations(classSequence,timeStep):
                sequencesX.append(subset)                
                sequencesY.append(data[1][i])                
    
    #print "Classes: ", sequencesY
    positions = []
    for i in range(len(sequencesX)):              
        positions.append(i)
    

    random.shuffle(positions)
    
    sequencesXShuffled = []
    sequencesYShuffled = []
    for i in positions:
        sequencesXShuffled.append(sequencesX[i])
        sequencesYShuffled.append(sequencesY[i])
    
    return (sequencesXShuffled,sequencesYShuffled)


def readFeatureFileSequence(featuresDirectory, percentTrain, percentValid, timeStep):
        
    directory = featuresDirectory        
    f = open(directory, 'r')        
    inputs = []
    outputs = []    
    
    sequences = []        
    sequenceNumber = 0
    
    for line in f:
        li = line.split(",")
        output = int(li[0])
        li.remove(li[0])
        features = []
        for i in li:   
            if not i == 0:
                try:
                    features.append(float(i))
                except:
                     pass
        sequences.append(features)
        sequenceNumber = sequenceNumber+1
                            
        if (sequenceNumber % timeStep) == 0 and sequenceNumber != 0:            
            inputs.append(sequences)            
            outputs.append(output)        
            sequences = []
            
    return separateSets(inputs,outputs,percentTrain,percentValid)    

   
   
   

def loadDataCrossValidation(log, featuresDirectory, datasetDivision, dataType):
    
    log.startNewStep("Loading Data "+str(datasetDivision)+"-Cross Validation")

    # Load the dataset
    log.printMessage(("Data Type:", dataType))
    log.printMessage(("Loading from: ", featuresDirectory))
    inputs = []
    outputs = []
    
    if(dataType == "Static"):
        inputSet,outputSet = readFeatureFile(featuresDirectory,True,0,0)
        
        intensEachSet = int(len(inputSet)/datasetDivision)        
        
        
        for i in range(datasetDivision):
            
            
            if(i == datasetDivision-1):
                
                newSetInput = inputSet[i*intensEachSet:]
                newSetOutput = outputSet[i*intensEachSet:]
                
                inputs.append(newSetInput)
                outputs.append(newSetOutput)
            else:           
                
                newSetInput = inputSet[i*intensEachSet:i*intensEachSet+intensEachSet]
                newSetOutput = outputSet[i*intensEachSet:i*intensEachSet+intensEachSet]
                inputs.append(newSetInput)
                outputs.append(newSetOutput)
                
               
    #print "Input:", inputs[0]
    
    log.printMessage(("Numbers of sets: ", datasetDivision))        
    log.printMessage(("Elements in each input set: ", len(inputs[0])))        
    log.printMessage(("--- Each Element in each input set: ", len(inputs[0][0])))                     
    log.printMessage(("Elements in each output set: ", len(outputs[0])))        
    
            
            
    return (inputs,outputs)

#input is an numpy.ndarray of 2 dimensions (a matrix)
#witch row's correspond to an example. target is a
#numpy.ndarray of 1 dimensions (vector)) that have the same length as
#the number of rows in the input. It should give the target
#target to the example with the same index in the input.

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """    
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    
    return shared_x, T.cast(shared_y, 'int32')  

 
#Metod used for loading the features for the network. After obtain the
# three lists, training, validation and testing, it transform the lists 
#in theano shared variables.
def loadData(log, experimentDirectory, channels, loadImagesStrategy, nFoldPositionsSet, nFoldIndex, channelsRepositories, classesLabel, output):    
    log.startNewStep("Loading Data")
    
    if nFoldPositionsSet == []:    
        trainPositions, validationPositions,testPositions, channelsRepositories, classesLabel =  readPositions(experimentDirectory, channels, loadImagesStrategy, log)    
        nFoldPositionsSet = trainPositions
    if len (channels[0][0][3]) == 2:
        log.printMessage(("Reading NFold sets"))                
                
        setPositions = (nFoldPositionsSet[0][nFoldIndex], nFoldPositionsSet[1][nFoldIndex])
        testSet, audioSet = createSetFromPositions(setPositions, channels, channelsRepositories[0], loadImagesStrategy, output, experimentDirectory)
        validSet = testSet              
        
        trainSet = []            
        positionsInput = []
        positionsOutput = []
        for setIndex in range(len(nFoldPositionsSet[0])):
            if not setIndex == nFoldIndex:
                                 
                for p in range(len(nFoldPositionsSet[0][setIndex])):
                    positionsInput.append(nFoldPositionsSet[0][setIndex][p])
                    positionsOutput.append(nFoldPositionsSet[1][setIndex][p])
        
        setPositions = (positionsInput, positionsOutput)           
        trainSet, audioSets = createSetFromPositions(setPositions, channels, channelsRepositories[0], loadImagesStrategy, output, experimentDirectory) 
        
                                                
    else:                          
        log.printMessage(("Reading Train set"))     

        trainSet, audioSet = createSetFromPositions(trainPositions, channels, channelsRepositories[0], loadImagesStrategy, output, experimentDirectory)              

        log.printMessage(("Reading Valid set"))        
        validSet, audioSet = createSetFromPositions(validationPositions, channels, channelsRepositories[1], loadImagesStrategy, output, experimentDirectory)              
        log.printMessage(("Reading Test set"))        
        testSet, audioSet = createSetFromPositions(testPositions, channels, channelsRepositories[2], loadImagesStrategy, output, experimentDirectory)             
        nFoldPositionsSet = []
    
    
        
    for channel in range(len(channels)):
        inputStructure = channels[channel][0]
        dataAugmentation = inputStructure[5]
        
        log.printMessage(("Data modality: ", inputStructure[1]))
        log.printMessage(("---Input images: ", inputStructure[0]))        
        log.printMessage(("---Image size: ", inputStructure[4]))
        log.printMessage(("---Image structure: ", inputStructure[2]))
        log.printMessage(("---Data directory: ", inputStructure[3]))
        log.printMessage(("---Data augmentation: ", inputStructure[5]))
        log.printMessage(("---Image color Space: ", inputStructure[6]))
        if dataAugmentation:
                log.printMessage(("Augmenting training set"))
                trainSet[0][channel], trainSet[1][channel] = useDataAugmentation((trainSet[0][channel], trainSet[1][channel]), inputStructure[2])
                
        log.printMessage(("--- Elements in train Data: ", len(trainSet[0][channel])))
        log.printMessage(("--- Elements in validation Data: ", len(validSet[0][channel])))
        log.printMessage(("--- Elements in test Data: ", len(testSet[0][channel]))) 
    
    return (trainSet,validSet,testSet), classesLabel, nFoldPositionsSet, audioSet, channelsRepositories
   
def useDataAugmentation(dataset, imageStructure):
    inputs,outputs = dataset
    
    newInputs = []
    newOutputs = []                
        
    for img in range(len(inputs)):
        if imageStructure == IMAGE_STRUCTURE["Static"]:
            newImages = ImageProcessingUtil.dataAugmentation(inputs[img])
            newInputs.append(inputs[img])
            newOutputs.append(outputs[img])
            for image in newImages:
                newInputs.append(inputs[img])
                newOutputs.append(outputs[img])
        else:
            sequences = []
            newInputs.append(inputs[img])
            newOutputs.append(outputs[img])
            
            numberOfNewSequences = len(ImageProcessingUtil.dataAugmentation(inputs[img][0]))
            for i in range(numberOfNewSequences):
                newOutputs.append(outputs[img])
                sequences.append([])
                         
            for a in range(len(inputs[img])):         
                #print "a:", a
                #print "Images in this sequence:", len(inputs[img])                
                newImages = ImageProcessingUtil.dataAugmentation(inputs[img][a])
                for nImg in range(len(newImages)):
                    sequences[nImg].append(newImages[nImg])
                
            
            for sequence in sequences:
               newInputs.append(sequence)
               
    inputs = None 
    outputs = None 
    
    return randomizeSet(newInputs,newOutputs)
   
        
            
    

def createFolder(directory):
    if not os.path.exists(directory): os.makedirs(directory)


def _blob(x,y,area,colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = numpy.sqrt(area) / 2
    xcorners = numpy.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = numpy.array([y - hs, y - hs, y + hs, y + hs])
    pylab.fill(xcorners, ycorners, colour, edgecolor=colour)

def saveHintonDiagram(W, directory):
    maxWeight = None
    #print "Weight: ", W
    """
    Draws a Hinton diagram for visualizing a weight matrix. 
    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.
    """
    reenable = False
    if pylab.isinteractive():
        pylab.ioff()
    pylab.clf()
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2**numpy.ceil(numpy.log(numpy.max(numpy.abs(W)))/numpy.log(2))

    pylab.fill(numpy.array([0,width,width,0]),numpy.array([0,0,height,height]),'gray')
    pylab.axis('off')
    pylab.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,w/maxWeight),'white')
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,-w/maxWeight),'black')
    if reenable:
        pylab.ion()
    #pylab.show()
    pylab.savefig(directory)
    

def createFeatureVector(imagesList, imageSize):
    
     
    inputs = []
    outputs = []
    for i in imagesList:
        
        img = cv2.imread(i[0])
        img = numpy.array(img) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img - img.mean()
        img = img / numpy.std(img)
        

        #img = numpy.array(img)
        newx,newy = imageSize #new size (w,h)
        img = cv2.resize(img,(newx,newy))
        #cv2.imwrite("//informatik2/wtm/home/barros/Documents/Experiments/JaffeDataset/test.jpg",img)
         
        outputs.append(int(i[1])) 
        
        imageFeatures = []
        for x in img:
           for y in x:
                imageFeatures.append(y)                                
        
        inputs.append(imageFeatures)
    
    return (inputs,outputs)


def createFeatureFileJaffeExperiment(directory, log, imageSize):


    log.startNewStep("Loading Data Leave one out experiment ")

    # Load the dataset    
    log.printMessage(("Loading from: ", directory))
    
    testingSet = []
    trainingSet = []
    
    classes = os.listdir(directory)
    classIndex = 0
    for c in classes:
        persons = []
        images = os.listdir(directory+"/"+c+"/")
        for i in images:
            name = i[0:2]
            if not name in persons:
                persons.append(name)

        imagesPerPerson = []                
        for i in range(len(persons)):
            imagesPerPerson.append([])
                
        for i in images:
            personIndex = 0
            for p in persons:

                if p in i:                                        
                    imagesPerPerson[personIndex].append((directory+"/"+str(c)+"/"+i,classIndex))

                    break
                personIndex = personIndex+1        
        
        for iP in range(len(imagesPerPerson)):            
             random.shuffle(imagesPerPerson[iP])             
             testingSet.append(imagesPerPerson[iP][+len(imagesPerPerson[iP])-1])
             imagesPerPerson[iP].pop()
             trainingSet.extend(imagesPerPerson[iP])
        
        classIndex = classIndex+1
        
    random.shuffle(trainingSet)    
    random.shuffle(testingSet)
    
    train_set = createFeatureVector(trainingSet,imageSize)
    test_set = createFeatureVector(testingSet,imageSize)

    
    #print train_set[0][0]
    log.printMessage(("Elements in train Data: ", len(train_set[0])))        
    log.printMessage(("--- Each Element in train Data: ", len(train_set[0][0])))    
    log.printMessage(("Elements in test Data: ", len(test_set[0])))
    log.printMessage(("--- Each Element in test Data: ", len(test_set[0][0])))
        
    test_set_x, test_set_y = shared_dataset(test_set)    
    train_set_x, train_set_y = shared_dataset(train_set)

    return [(train_set_x, train_set_y), (train_set_x, train_set_y),
            (test_set_x, test_set_y)]        



def writeSingleFile(features, location, color):
       # print features        
        f = open(location,"w")
        for featureSet in features:         
           # print "Cada FeatureSet tem:", len(featureSet)
            featureNumber = 0
            
            for feature in featureSet:               
#                print "Cada feature tem:", len(feature)  
                if featureNumber == 0:  
                    f.write(str(feature))
                    featureNumber = featureNumber+1
                    f.write(",")
                    
                else:
                    if color:                    
                        for c in feature:
                            f.write(str(c))
                            
                            featureNumber = featureNumber+1
                            if(featureNumber < len(featureSet)*3):
                                f.write(",")   
                            
                    else:        
                        f.write(str(feature))
                        
                        featureNumber = featureNumber+1
                        if(featureNumber < len(featureSet)):
                            f.write(",")    
                 
            f.write("\n")
        f.close()
        


def prepareDataLive(image,dataModality, imageSize):
                    
        if dataModality == DATA_MODALITY["Image"]:            
            data = ImageProcessingUtil.resize(image,imageSize)                
        elif dataModality == DATA_MODALITY["Face"]:            
            image = ImageProcessingUtil.detectFace(image)    
            data = ImageProcessingUtil.resize(image,imageSize)
                                  
        return data, image 
        
        
def prepareData(data,dataModality, imageSize, featureType, sliceSize):
        

        if dataModality == DATA_MODALITY["Image"]:
             #data = AudioProcessingUtil.extractMFCC(data, "", 0, sliceSize)
             #AudioProcessingUtil.audioFeatureExtractor(data, sliceSize, featureType, imageSize)     
             
            #print "data:", data
            data = cv2.imread(data) 
            data = ImageProcessingUtil.resize(data,imageSize) 
            #print "SHape:", numpy.shape(data)
            #data2 = data.T
            #data2 = ImageProcessingUtil.resize(data,imageSize) 
            #print "SHape:", numpy.shape(data2)
            #cv2.imwrite("/data/gaussian/ExperimentsTwoFaces/LoadedImage.png",data)
            #cv2.imwrite("/data/gaussian/ExperimentsTwoFaces/LoadedImage.png",data2)            
            #raw_input("here")
            
            #cv2.imwrite("/informatik2/wtm/home/barros/Workspace/MCCNN2/temp3.png", data)            
            
        elif dataModality == DATA_MODALITY["Face"]:
            data = cv2.imread(data)            
            data, rects = ImageProcessingUtil.detectFace(data)
            data = ImageProcessingUtil.resize(data,imageSize)
        elif dataModality == DATA_MODALITY["Audio"]:        
            data = AudioProcessingUtil.audioFeatureExtractor(data, sliceSize, featureType, imageSize)
        return data            
        


def createTestInputAudio(data, channels):
    inputs = []

    for channel in range(len(channels)):                        
        inputStructure = channels[channel][0]
        dataModality = inputStructure[1]                 
        imageSize = inputStructure[4] 
        featureType = inputStructure[6]
        sliceSize = inputStructure[0]
        
        images = prepareData(data, dataModality, imageSize, featureType, sliceSize) 
        inputs.append(images)
        
    return inputs
            
            
            

def createSetFromPositions(positionsSet, channels, channelRepositories, loadImagesStrategy, output, basicDirectory):
        
    setAudioFiles = []
    setAudioFilesOutput = []
    
    outputType = output[5][3]
    outputFolder = output[5][4]
    
    
    setInputs =  []
    setOutputs = []
    for channel in channels:
        setInputs.append([])
        setOutputs.append([])
    
    #print "Ihereee",  channelRepositories[1][positionsSet[0][578]]
    for trainPos in range(len(positionsSet[0])):
        for channel in range(len(channels)):
            #print "Ihereee", channel, " --- ",   channelRepositories[channel][positionsSet[0][trainPos]]
            #print "TrainPos", trainPos
            data = channelRepositories[channel][positionsSet[0][trainPos]]
            label = positionsSet[1][trainPos]
            
            inputStructure = channels[channel][0]
            dataModality = inputStructure[1]
            imageStructure = inputStructure[2]            
            imageSize = inputStructure[4] 
            #print "Create Set From Position Image Size:", imageSize
            featureType = inputStructure[6]
            sliceSize = inputStructure[0]            
            
            if imageStructure==IMAGE_STRUCTURE["Sequence"]:                
                imageSequence = []         
                sortedSequenceList = os.listdir(data)
                sortedSequenceList = sorted(sortedSequenceList, key=lambda x: int(x.split(".")[0]))
                
                for image in sortedSequenceList:                    
                    if loadImagesStrategy == LOAD_IMAGES_STRATEGY["LoadAll"]:                             
                        image = prepareData(data+"/"+image, dataModality, imageSize, featureType, sliceSize)   
                        if dataModality == DATA_MODALITY["Audio"]:
                            
                            setAudioFiles.append(data+"/"+image)
                            setAudioFilesOutput.append(label)
                            
                            for i in image:                                
                                imageSequence.append(i) 
                                
                        else:
                            imageSequence.append(image) 
                    else:
                        imageSequence.append(data+"/"+image)   
                                  
                setInputs[channel].append(imageSequence)
                
                if outputType == OUTPUT_LAYER_TYPE["Localization"]:
                        thisOutputFolder = data.split("/")
                        thisOutputFolder = basicDirectory+"/"+outputFolder + "/" + thisOutputFolder[-2]+"/"+ thisOutputFolder[-1]+".txt.txt"
                        target = readTargetLocalizationFile(thisOutputFolder)
                                                
                        setOutputs[channel].append(target)     
                else:
                        setOutputs[channel].append(label)     
                
            elif imageStructure==IMAGE_STRUCTURE["StaticInSequence"]:   
                imageIndex = inputStructure[7] 
                #print "Data:", data
                #print "data:", os.listdir(data)
                #print "Index:", imageIndex
                data = data+"/"+os.listdir(data)[imageIndex]
                dataDirectory = data                
                print "DataDirectory:", data
                if loadImagesStrategy == LOAD_IMAGES_STRATEGY["LoadAll"]:     
                        data = prepareData(data, dataModality, imageSize, featureType, sliceSize) 
                
                if dataModality == DATA_MODALITY["Audio"]:
                    setAudioFiles.append(dataDirectory)
                    setAudioFilesOutput.append(label)
                    
                    for i in data:                                                        
                        setInputs[channel].append(i)
                        setOutputs[channel].append(label)
                else:
                    setInputs[channel].append(data)
                    
                    if outputType == OUTPUT_LAYER_TYPE["Localization"]:
                        dataDirectory = dataDirectory.replace("//","/")
                        thisOutputFolder = dataDirectory.split("/")
    

                        thisOutputFolder = basicDirectory+"/"+outputFolder + "/" + "Fabo"+"/"+thisOutputFolder[-2]+".txt.txt"                           
                        target = readTargetLocalizationFile(thisOutputFolder)
                        
                        setOutputs[channel].append(target)     
                    else:
                        setOutputs[channel].append(label)  
                                                       
                
            elif imageStructure==IMAGE_STRUCTURE["Static"]:  
                    dataDirectory = data        

                    if loadImagesStrategy == LOAD_IMAGES_STRATEGY["LoadAll"]:     
                        data = prepareData(data, dataModality, imageSize, featureType, sliceSize)    
#                        setAudioFiles.append(dataDirectory)
#                        setAudioFilesOutput.append(label)
#                        for i in data:                                                        
#                            setInputs[channel].append(i)
#                            setOutputs[channel].append(label)
                                                  
                    if dataModality == DATA_MODALITY["Audio"]:
                        setAudioFiles.append(dataDirectory)
                        setAudioFilesOutput.append(label)
                        for i in data:                                                        
                            setInputs[channel].append(i)
                            setOutputs[channel].append(label)
                    else:
                        setInputs[channel].append(data)                        
                        if outputType == OUTPUT_LAYER_TYPE["Localization"]:   

                            dataDirectory = dataDirectory.replace("//","/")
                            dataName = dataDirectory.split("/")
                                
#                            print "baseDirectory:",basicDirectory
#                            print "outputFolder:",outputFolder
#                            print "Label:",label
#                            print "dataDirectory:",dataDirectory
#                            print "dataName:",dataName                            
                            dataName = dataName[-1].split(".")[-2]
                            thisOutputFolder = basicDirectory+"/"+outputFolder+"/"+str(label)+"/"+dataName+".txt.txt"
#                            print "FileDirectory:",thisOutputFolder
                            
                            #thisOutputFolder =basicDirectory+"/"+outputFolder + "/" + thisOutputFolder[-2]+"/"+ thisOutputFolder[-1].split(".")[0]+"."+thisOutputFolder[-1].split(".")[1]+".txt"                            
                            target = readTargetLocalizationFile(thisOutputFolder)
                            #raw_input("here")
                            
                            setOutputs[channel].append(target)     
                        else:
                            setOutputs[channel].append(label)                                                                      
                                                  
    #setInputs,setOutputs = randomizeSet(setInputs,setOutputs)
    return (setInputs, setOutputs),  (setAudioFiles, setAudioFilesOutput)
   


    
def readPositions(baseDirectory, channels, loadImagesStrategy, log):    
    
    
    if len (channels[0][0][3]) == 2:
            
        classesLabels = {}
        
        numberOfSets = channels[0][0][3][1]
                     
        inputNumber = 0
        labelNumber = 0
        inputPositions = []    
        outputs = []    
        imageGuideDirectory = baseDirectory + "/" + channels[0][0][3][0]
        print "IMAGE" ,imageGuideDirectory
        sortedLabelsList = os.listdir(imageGuideDirectory+"/")
        #sortedLabelsList.remove(".DS_Store")
        for f in sortedLabelsList:
            if f.startswith('.'):
                sortedLabelsList.remove(f)
        sortedLabelsList.sort()
        for label in sortedLabelsList:    

            for position in range(len(os.listdir(imageGuideDirectory+"/"+label))):
                inputPositions.append(inputNumber)  
                outputs.append(labelNumber)
                classesLabels[str(labelNumber)] = label
                inputNumber = inputNumber+1    
            labelNumber = labelNumber+1
        
        
#        orderedClassesLabels = {}
#        for ocl in sorted(classesLabels):
#            orderedClassesLabels[str(ocl)] = classesLabels[ocl]
#            
#        print "ordered:",  orderedClassesLabels
#        raw_input("here")
        inputPositions,outputPositions = randomizeSet(inputPositions,outputs)    
        
        positionsInputSet = []
        positionsOutputSet = []
        numberOfSamples = len(outputPositions)
        
        elementsPerSet = numberOfSamples / numberOfSets
        for i in range(numberOfSets):
            positionFrom = i*elementsPerSet
            positionTo = (i+1)*elementsPerSet
            
            positionsInputSet.append( inputPositions[positionFrom:positionTo])
            positionsOutputSet.append( outputPositions[positionFrom:positionTo])
                        
        
        channelRepositories = []        
        for channel in channels:       
                
            dataDirectory = channel[0][3][0]
            
            channelRepositoryInput = []         
            
            sortedLabelsList = os.listdir(baseDirectory+"/"+dataDirectory+"/")
            for f in sortedLabelsList:
                if f.startswith('.'):
                    sortedLabelsList.remove(f)
            print "here", sortedLabelsList
            sortedLabelsList.sort()
            for label in sortedLabelsList:  
                sortedSampleList = os.listdir(baseDirectory+"/"+dataDirectory+"/"+label)
                for f in sortedSampleList:
                    if f.startswith('.'):
                        sortedSampleList.remove(f)
                sortedSampleList = sorted(sortedSampleList, key=lambda x: int(x.split(".")[0]))
                
                for example in sortedSampleList: 
                    channelRepositoryInput.append(baseDirectory+"/"+dataDirectory+"/"+label+"/"+example)  
                    
            channelRepositories.append(channelRepositoryInput)
            
        channelRepositories = [channelRepositories,channelRepositories,channelRepositories]  
        trainPositions = (positionsInputSet,positionsOutputSet)
        validationPositions = (positionsInputSet,positionsOutputSet) 
        testPositions = (positionsInputSet,positionsOutputSet)              
        
    elif len (channels[0][0][3]) == 4:
        
        
        classesLabels = {}
        
        percentTrain = channels[0][0][3][1]
        percentValid = channels[0][0][3][2]
             
        inputNumber = 0
        labelNumber = 0
        inputPositions = []    
        outputs = []    
        imageGuideDirectory = baseDirectory + "/" + channels[0][0][3][0]        
        sortedLabelsList = os.listdir(imageGuideDirectory+"/")
        for f in sortedLabelsList:
            if f.startswith('.'):
                sortedLabelsList.remove(f)
                
        sortedLabelsList.sort()
        """ SOLUTION TO THE PROBLEM"""
        ################################

        ################################
    
        for label in sortedLabelsList:
            valueForRange = os.listdir(imageGuideDirectory+"/"+label)
            for f in valueForRange:
                if f.startswith('.'):
                    valueForRange.remove(f)
        ################################

            for position in range(len(valueForRange)):
                
                inputPositions.append(inputNumber)  
                outputs.append(labelNumber)
                classesLabels[str(labelNumber)] = label
                inputNumber = inputNumber+1    
            labelNumber = labelNumber+1

        
        inputPositions,outputPositions = randomizeSet(inputPositions,outputs)    
        
        trainPositions, validationPositions, testPositions = separateSets(inputPositions,outputPositions, percentTrain, percentValid, log, classesLabels)        
        
        channelRepositories = []        
        for channel in channels:       
                
            dataDirectory = channel[0][3][0]
            
            channelRepositoryInput = []         
            
            sortedLabelsList = os.listdir(baseDirectory+"/"+dataDirectory+"/")
            for f in sortedLabelsList:
                if f.startswith('.'):
                    sortedLabelsList.remove(f)
                
            sortedLabelsList.sort()
            for label in sortedLabelsList:
                sortedSampleList = os.listdir(baseDirectory+"/"+dataDirectory+"/"+label)
                for f in sortedSampleList:
                    if f.startswith('.'):
                        sortedSampleList.remove(f)
                

                sortedSampleList = sorted(sortedSampleList, key=lambda x: int(x.split(".")[0]))
                
                for example in sortedSampleList: 
                    channelRepositoryInput.append(baseDirectory+"/"+dataDirectory+"/"+label+"/"+example)                                        
            channelRepositories.append(channelRepositoryInput)
        channelRepositories = [channelRepositories,channelRepositories,channelRepositories]     
    else:
        
        inputNumber = 0
        labelNumber = 0
        inputPositionsTraining = []    
        outputPositionsTraining = []        
        classesLabels = {}
        sortedLabelsList = os.listdir(baseDirectory + "/" + channels[0][0][3][0])
        for f in sortedLabelsList:
            if f.startswith('.'):
                sortedLabelsList.remove(f)
                
        sortedLabelsList.sort()
        for label in sortedLabelsList:      
            
            for position in range(len(os.listdir(baseDirectory + "/" + channels[0][0][3][0]+"/"+label))):
                inputPositionsTraining.append(inputNumber)  
                outputPositionsTraining.append(labelNumber)
                inputNumber = inputNumber+1    
                classesLabels[str(labelNumber)] = label
            labelNumber = labelNumber+1
        
                          
        inputNumber = 0
        labelNumber = 0
        inputPositionsValidation = []    
        outputPositionsValidation = []        
        sortedLabelsList = os.listdir(baseDirectory + "/" + channels[0][0][3][1])
        for f in sortedLabelsList:
            if f.startswith('.'):
                sortedLabelsList.remove(f)
                
        sortedLabelsList.sort()
        for label in sortedLabelsList:        
            for position in range(len(os.listdir(baseDirectory + "/" + channels[0][0][3][1]+"/"+label))):
                inputPositionsValidation.append(inputNumber)  
                outputPositionsValidation.append(labelNumber)
                inputNumber = inputNumber+1    
            labelNumber = labelNumber+1
    
        inputNumber = 0
        labelNumber = 0
        inputPositionsTesting = []    
        outputPositionsTesting = []   
        sortedLabelsList = os.listdir(baseDirectory + "/" + channels[0][0][3][2])
        for f in sortedLabelsList:
            if f.startswith('.'):
                sortedLabelsList.remove(f)
                
        sortedLabelsList.sort()
        for label in sortedLabelsList:        
            for position in range(len(os.listdir(baseDirectory + "/" + channels[0][0][3][2]+"/"+label))):
                inputPositionsTesting.append(inputNumber)  
                outputPositionsTesting.append(labelNumber)
                inputNumber = inputNumber+1    
            labelNumber = labelNumber+1        
            

        
        trainPositions = randomizeSet(inputPositionsTraining,outputPositionsTraining)
        
        validationPositions = randomizeSet(inputPositionsValidation,outputPositionsValidation)
        testPositions = randomizeSet(inputPositionsTesting,outputPositionsTesting)
        
        channelRepositoriesTraining = []        
        for channel in channels:       
                
            dataDirectory = channel[0][3][0]
            
            channelRepositoryInput = []         
                
                
            sortedLabelsList = os.listdir(baseDirectory+"/"+dataDirectory+"/")
            for f in sortedLabelsList:
                if f.startswith('.'):
                    sortedLabelsList.remove(f)
                
            sortedLabelsList.sort()
            for label in sortedLabelsList:  
                sortedSampleList = os.listdir(baseDirectory+"/"+dataDirectory+"/"+label)
                for f in sortedSampleList:
                    if f.startswith('.'):
                        sortedSampleList.remove(f)
                
                sortedSampleList = sorted(sortedSampleList, key=lambda x: int(x.split(".")[0]))
                for example in os.listdir(baseDirectory+"/"+dataDirectory+"/"+label): 
                    channelRepositoryInput.append(baseDirectory+"/"+dataDirectory+"/"+label+"/"+example)                                        
            channelRepositoriesTraining.append(channelRepositoryInput)
            
        channelRepositoriesValidation = []        
        for channel in channels:       
                
            dataDirectory = channel[0][3][1]
            
            channelRepositoryInput = []         
            
            sortedLabelsList = os.listdir(baseDirectory+"/"+dataDirectory+"/")
            for f in sortedLabelsList:
                if f.startswith('.'):
                    sortedLabelsList.remove(f)
                
            sortedLabelsList.sort()
            for label in sortedLabelsList:  
                sortedSampleList = os.listdir(baseDirectory+"/"+dataDirectory+"/"+label)
                for f in sortedSampleList:
                    if f.startswith('.'):
                        sortedSampleList.remove(f)
                
                #print "Directory:", baseDirectory+"/"+dataDirectory+"/"+label
                
                sortedSampleList = sorted(sortedSampleList, key=lambda x: int(x.split(".")[0]))
                for example in os.listdir(baseDirectory+"/"+dataDirectory+"/"+label): 
                    channelRepositoryInput.append(baseDirectory+"/"+dataDirectory+"/"+label+"/"+example)                                        
            channelRepositoriesValidation.append(channelRepositoryInput)    
        
        channelRepositoriesTesting = []          
        for channel in channels:       
                
            dataDirectory = channel[0][3][2]
            
            channelRepositoryInput = []         
            
            sortedLabelsList = os.listdir(baseDirectory+"/"+dataDirectory+"/")
            for f in sortedLabelsList:
                if f.startswith('.'):
                    sortedLabelsList.remove(f)
                
            sortedLabelsList.sort()
            for label in sortedLabelsList:  
                sortedSampleList = os.listdir(baseDirectory+"/"+dataDirectory+"/"+label)
                for f in sortedSampleList:
                    if f.startswith('.'):
                        sortedSampleList.remove(f)
                
                sortedSampleList = sorted(sortedSampleList, key=lambda x: int(x.split(".")[0]))
                for example in os.listdir(baseDirectory+"/"+dataDirectory+"/"+label): 
                    channelRepositoryInput.append(baseDirectory+"/"+dataDirectory+"/"+label+"/"+example)                                        
            channelRepositoriesTesting.append(channelRepositoryInput)
        
        channelRepositories = [channelRepositoriesTraining,channelRepositoriesValidation,channelRepositoriesTesting]         
     
    classesLabels = sorted(classesLabels.items())
    
    #print classesLabels
    
    
    
    return (trainPositions,validationPositions,testPositions, channelRepositories,classesLabels)
     
def readTargetLocalizationFile(directory):
    target = []
    targetFile = open(directory,"r")
    for line in targetFile:        
        target.append(float(line))
    
    return numpy.array(target,dtype=theano.config.floatX)
    