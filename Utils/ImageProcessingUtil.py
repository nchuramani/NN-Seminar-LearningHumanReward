# -*- coding: utf-8 -*-

import cv2
import numpy
import PIL
import PIL.Image
import os
import shutil
import random
import theano 
import theano.tensor as T
from scipy.ndimage.filters import gaussian_filter


from scipy import ndimage


import DataUtil


def resize(image, size): 
        
        image = numpy.array(cv2.resize(image,(size[0],size[1])))                   
         
        #image = numpy.array(cv2.resize(image,size))  
#        if numpy.shape(image)[0] == size[1]:        
#            #print "Image Data:", numpy.array(image).shape
#            image = image.swapaxes(0,1)
#        
#            #print "Image Size:", size
#            #print "Image Data2:", numpy.array(image).shape
#           # raw_input("here")
#        
        return image
        
def resizeInterpolation(image, size, interpolation_factor):            
        return numpy.array(cv2.resize(image,size, interpolation=interpolation_factor))                
        
def resizeFactor(image, factor):
    return cv2.resize(image,None,fx=factor, fy=factor, interpolation = cv2.INTER_AREA)
        


def dataAugmentation(image):

    #  Resize original Image
    newImages = []
    newImg = resizeFactor(image, .5)
    newImg = resizeFactor(newImg, 2)        
    newImages.append(newImg)
        
    rows = image.shape[0]
    cols = image.shape[1]
    
    for factor in [5, 7 , 9 , 11 , 12]:
        #Y+ Shifting
        newImg = image[0:rows-rows/factor, 0:cols]
        newImg = resize(newImg, (rows,cols))
        newImages.append(newImg)
#        newImg = resizeFactor(newImg, .5)
#        nweImg = resizeFactor(newImg, 2)
#        newImages.append(nweImg)

#        #Y- Shifting
        newImg = image[rows/factor:rows, 0:cols]
        newImg = resize(newImg, (rows,cols))
        newImages.append(newImg)        
#        newImg = resizeFactor(newImg, .5)
#        nweImg = resizeFactor(newImg, 2)
#        newImages.append(nweImg)        
#        
#        
#        #X+ Shifting
        newImg = image[0:rows, 0:cols-cols/factor]
        newImg = resize(newImg, (rows,cols))
        newImages.append(newImg)
#        newImg = resizeFactor(newImg, .5)
#        nweImg = resizeFactor(newImg, 2)
#        newImages.append(nweImg)        
#
#        #X- Shifting
        newImg = image[0:rows, cols/factor:cols]
        newImg = resize(newImg, (rows,cols))
        newImages.append(newImg)
#        newImg = resizeFactor(newImg, .5)
#        nweImg = resizeFactor(newImg, 2)
#        newImages.append(nweImg)                
    
    newImages.append(image)
    
    return newImages
        

        
def gaussian_kernel(size, sigma):
    one = numpy.zeros((size, size), dtype=theano.config.floatX)
    one[size//2, size//2] = 1
    
    return gaussian_filter(one, sigma)            

def whitenImage(x, imageStructure, numberOfImages):
    
    whiten_kernel_1 = theano.shared(gaussian_kernel(9, 0.2))
    whiten_kernel_2 = theano.shared(gaussian_kernel(9, 1.0))
    
    if imageStructure == DataUtil.IMAGE_STRUCTURE["Static"] or imageStructure == DataUtil.IMAGE_STRUCTURE["StaticInSequence"]:
        tmp1 = T.nnet.conv2d(x, whiten_kernel_1.dimshuffle('x', 'x', 0, 1), border_mode="full")
        tmp2 = T.nnet.conv2d(x, whiten_kernel_2.dimshuffle('x', 'x', 0, 1), border_mode="full")
        resultImage = ( tmp1 - tmp2)[:, :, 4:-4, 4:-4]
    elif imageStructure == DataUtil.IMAGE_STRUCTURE["Sequence"]:
        kernel1 = whiten_kernel_1.dimshuffle('x', 'x', 0, 1)
        kernel2 = whiten_kernel_2.dimshuffle('x', 'x',0, 1)
        
       # one = numpy.zeros((10,3,64,64))
        
        print "Shape1:", kernel1.eval().shape
        print "Shape2:", kernel2.eval().shape        
        resultImage = []
        for i in range(numberOfImages):            
           # two = one[:,i,:,:]
            xNow = x[:,i,:,:]
            xNow = xNow.dimshuffle(0,'x',1,2)
            
            #print "SHape:", two.shape            
            #raw_input("here")
            
            tmp1 = T.nnet.conv2d(xNow, whiten_kernel_1.dimshuffle('x', 'x', 0, 1), border_mode="full")
            tmp2 = T.nnet.conv2d(xNow, whiten_kernel_2.dimshuffle('x', 'x', 0, 1), border_mode="full")
            #tmp2 = tmp2[:, :, 4:-4, 4:-4]
            #tmp1 = tmp1[:, :, 4:-4, 4:-4]
            resultImage.append(( tmp1 - tmp2)[:, :, 4:-4, 4:-4])
           # resultImage1.append(tmp1)
           # resultImage2.append(tmp2)
            #tmp1.append(t1)
            #tmp2.append(t2)                    
        resultImage = theano.tensor.concatenate(resultImage,1)   
        #resultImage2 = theano.tensor.concatenate(resultImage2,1)    
        #resultImage = (resultImage1-resultImage2)
        #tmp2 = theano.tensor.concatenate(tmp2,1)   
        #tmp1 = T.nnet.conv2d(x, whiten_kernel_1.dimshuffle('x', 'x', 0, 1), border_mode="full")
        #tmp2 = T.nnet.conv2d(x, whiten_kernel_2.dimshuffle('x', 'x',0, 1), border_mode="full")
        #tmp1 = T.nnet.conv3d2d.conv3d(signals=x, filters=whiten_kernel_1.dimshuffle('x', 'x', 0, 1, 'x'),border_mode='valid')    
        #tmp2 = T.nnet.conv3d2d.conv3d(signals=x, filters=whiten_kernel_2.dimshuffle('x', 'x', 0, 1, 'x'),border_mode='valid')    
    
    return resultImage
        
    
    
def normalizeInputTheanoFunction(x):
    
    x = x - T.mean(x)
    return x / T.std(x)
    

def grayImageTheanoFunction(x):
                                
    return  T.dot(x, theano.shared(numpy.asarray([0.299, 0.587, 0.144],dtype=theano.config.floatX),borrow=True))


       
    
def concatenateImageSidebySide(img1, img2):
        #print img2.size
        h1, w1 = 10,10
        h2, w2 = 10,10
        im3 = PIL.Image.new("RGB", (max(h1, h2), w1+w2))
        im3.paste(img1)
        im3.paste(img2, (0, w1, h2, w2))
        return im3
        
def concatenateImageStacked(img1, img2):
        h1, w1 = 10,10
        h2, w2 = 10,10
        im3 = PIL.Image.new("RGB", (h1, h2, max(w1+w2)))
        im3.paste(img1)
        im3.paste(img2, (h1, 0, h2, w2))
        return im3        
        
def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )
       
def fig2data ( fig ):
    fig.canvas.draw()
    
    # Now we can save it to a numpy array.
    data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
    data = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
    return data
    
        
def imageConvolving():
    
    img1 = cv2.imread("/informatik2/wtm/home/barros/Pictures/Datasets/Dynamic Gestures Dataset/images/30.04.2014/Subject1/rS1Abort.ogvdir/0/out0027.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.medianBlur(img1,15)
#    img1 = img1*-1
    cv2.imshow('image1',img1)
    #ret2,img1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   
    
    img2 = cv2.imread("/informatik2/wtm/home/barros/Pictures/Datasets/Dynamic Gestures Dataset/images/30.04.2014/Subject1/rS1Abort.ogvdir/0/out0052.png")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)    
    img2 = cv2.medianBlur(img2,15)
    #ret2,img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   
    cv2.imshow('image2',img2)
    
    img3 = cv2.absdiff(img2, img1)  
    ret2,img3 = cv2.threshold(img3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #img3 = (img1+img2)*10
    img3 = cv2.medianBlur(img3,5)
#    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY) 
    
    
    
    """
    x,y = 3,3    
    W_bound = numpy.sqrt(x * y)
    rng = numpy.random.RandomState(23455)
    Ww = numpy.asarray(
             rng.uniform(low=-W_bound, high=W_bound, size=(1,x,y))) 
    for i in range(len(Ww)):
            for u in range(len(Ww[i])):
                 Ww[i][u] = [1,1,1]
                 Ww[i][u] = [1,1,1]
                 Ww[i][u] = [1,1,1]
             
    

    img3 = sg.convolve(img1, Ww[0], "valid") 
    """    
    cv2.imshow('image',img3)
    cv2.imwrite("/informatik2/wtm/home/barros/Pictures/Datasets/Dynamic Gestures Dataset/teste.png",img3)

    while True: 
        cv2.imshow('image',img3)
        cv2.waitKey(20)
    cv2.destroyAllWindows()

def detectSkin(image):       

        skin_min = numpy.array([0, 10, 100],numpy.uint8)
        skin_max = numpy.array([30, 200, 255],numpy.uint8)    
        
        blur_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        tre_green = cv2.inRange(blur_hsv, skin_min, skin_max)
        
        return tre_green


def doConvShadow(images):

        
        total = []
        
        binaries = []        
        
        for i in range(len(images)):                      
            img4 = numpy.asarray(images[i])            
           
            if i == 0:
                previous = img4
            else:
                
                newImage = cv2.absdiff(img4, previous)
                
                binaries.append(newImage)
          

                
        for i in range(len(binaries)):            
           
            weight = float(i)/float(len(binaries))
            
            
            img4 = numpy.asarray(binaries[i]) * weight
            #img4 = numpy.asarray(binaries[i])
             
            if(i==0):
                total = img4
                
            else:
                #total = cv2.absdiff(img4, total)
                total = total + (img4)
                
        return total
          

def doConv(images):

        total = []    
        previous = 0
        binaries = []
        #print "Size: ", len(images)
        for i in range(len(images)):       
            img4 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)    
            img4 = numpy.asarray(img4)
            if i == 0:
                previous = img4
                total = img4
            else:
                newImage = cv2.absdiff(img4, previous)
                binaries.append(newImage)              
            previous = img4
      
        for i in range(len(binaries)):            
                #img4 = sg.convolve(numpy.asarray(images[i]), Ww[0], "valid")       
    
                img4 = numpy.asarray(binaries[i])
                 
                if(i==0):
                    total = img4
                    #total = img4
                else:
                    #total = total + (img4*weights[i])
                    total = cv2.absdiff(img4 , total)
        
                 
        total = total*10
#        total = cv2.cvtColor(total, cv2.COLOR_BGR2GRAY)    
        ret2,total = cv2.threshold(total,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        total = cv2.medianBlur(total,15)        
        return total
#imageConvolving()

def detectFace(img):     
    
        img2 = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
        
        cascade = cv2.CascadeClassifier("/users/nikhilchuramani/Desktop/IAS/SoSe16/Independent Studies/Workbench/haarcascade_frontalface_alt.xml")
        rects = cascade.detectMultiScale(img, 1.3, 4, 1, (20,20))
    
        if len(rects) == 0:            
            return img2
        rects[:, 2:] += rects[:, :2]
        
        return box(rects,img2),rects

def box(rects, img):        
        for x1, y1, x2, y2 in rects:
            
            #cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
            img = img[y1:y2, x1:x2]
            
            #newx,newy = 28,28 #new size (w,h)
            #newimage = cv2.resize(img2,(newx,newy))
            #cv2.imwrite(path, img2);
            return img
  

def applySkinSegmentation(image,directory):
    
    im_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    skin_ycrcb_mint = numpy.array((0, 133, 77))
    skin_ycrcb_maxt = numpy.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)       
    
    #for x in range(len(skin_ycrcb)):
    #    for y in range(len(skin_ycrcb[x])):            
    #        if skin_ycrcb[x][y]==0:            
    #            skin_ycrcb[x][y] = 255
    #        else:
    #            skin_ycrcb[x][y] = 0    
                
    cv2.imwrite(directory, skin_ycrcb)
    return skin_ycrcb
 

def normalizeInputs(lenght,directory, saveDirectory):    
    classes = os.listdir(directory)
    for c in classes:
        sequences = os.listdir(directory+"/"+c)    
        for s in sequences:
            files = os.listdir(directory+"/"+c+"/"+s)
            files = lenghtNormalization(files,lenght)            
            for f in files:                
                if  f != 0 and not "backup" in f:
                    if not os.path.exists(saveDirectory+"/"+str(c)+"/"+str(s)+"__/"):            
                        os.makedirs(saveDirectory+"/"+str(c)+"/"+str(s)+"__/")                
                    shutil.copyfile(directory+"/"+str(c)+"/"+str(s)+"/"+str(f),saveDirectory+"/"+str(c)+"/"+str(s)+"__/"+str(f))     
 
        
        
            
            
def lenghtNormalization(features, lenght):
        
        normalizedFeatures = []
        if(len(features) < lenght):
            normalizedFeatures = features
            while(len(normalizedFeatures) < lenght):
                normalizedFeatures.append(0)
        
        else:
            runner = (len(features) / lenght) + 1
            i = 0
            positionsToBeInserted = []
            while(i < len(features)):
                positionsToBeInserted.append(i)
                i += runner
            
            while(len(positionsToBeInserted) != lenght):                
                i = random.randrange(len(features))
                haveItem = False
                for h in positionsToBeInserted:
                    if(h == i): 
                        haveItem = True
                        
                if(not haveItem):
                    positionsToBeInserted.append(i)
           
            positionsToBeInserted.sort()
            for x in positionsToBeInserted:
                normalizedFeatures.append(features[x])
               
        return normalizedFeatures       
              
def applyMaskCam3DCorpus():
    directory = "/informatik2/wtm/home/barros/Documents/Experiments/cam3D/imagesRaw/"
    saveDirectory = "/informatik2/wtm/home/barros/Documents/Experiments/cam3D/imagesSegmented/"
    classs = os.listdir(directory)  
    for c in classs:
        examples = os.listdir(directory+c+"/")
        for e in examples:
            #print "Reading Example: ", directory+c+"/"+e
            images = os.listdir(directory+c+"/"+e+"/images/")
            masks = os.listdir(directory+c+"/"+e+"/depth/")
            for i in range(len(images)):
                #print "Reading image: ", directory+c+"/"+e+"/images/"+images[i]
                img = cv2.imread(directory+c+"/"+e+"/images/"+images[i])
                mask = cv2.imread(directory+c+"/"+e+"/depth/"+masks[i],0)
                
                mask[mask != 2] = 0
                mask[mask == 2] = 255
                #mask[mask != 2 ] = 0
                #for x in range(len(mask)):
                #    for y in range(len(mask[x])):
                #        if mask[x][y]==2:            
                #            mask[x][y] = 255
                #        else:
                #            mask[x][y] = 0
                #cv2.imwrite(saveDirectory+c+"/"+e+"/"+str(i)+".png",mask)
                res = cv2.bitwise_and(img,img,mask = mask)   
                #print "Saving image: ", saveDirectory+c+"/"+e+"/"+str(i)+".png"
                if not os.path.exists(saveDirectory+c+"/"+e+"/"):            
                    os.makedirs(saveDirectory+c+"/"+e+"/")
                cv2.imwrite(saveDirectory+c+"/"+e+"/"+str(i)+".png",res)
    
   
    #idx = (mask!=0)
    #dst = img
    #dst[idx] = img[idx]
    #testFace = cv2.imread("/informatik2/wtm/home/barros/Documents/Experiments/cam3D/images/Happy/seg5/images/frame000001.png")
    
    #detectFace(testFace,"/informatik2/wtm/home/barros/Documents/Experiments/cam3D/images/Happy/seg1/face.png",testFace)
    #skin = applySkinSegmentation(img,"/informatik2/wtm/home/barros/Documents/Experiments/cam3D/images/Happy/seg1/skin.png") 
    #res = cv2.bitwise_and(res,res,mask = skin)
    
    #cv2.imwrite("/informatik2/wtm/home/barros/Documents/Experiments/cam3D/images/Happy/seg1/mask.png",mask)
    #cv2.imwrite("/informatik2/wtm/home/barros/Documents/Experiments/cam3D/images/Happy/seg1/test.png",res)



def extractFace(directoryFace, directorySave):
    classes = os.listdir(directoryFace)
    for c in classes:
        #sequences = os.listdir(directoryFace+"/"+c)    
        #for s in sequences:
            #files = os.listdir(directoryFace+"/"+c+"/"+s)            
            files = os.listdir(directoryFace+"/"+c+"/")            
            for f in files:
                #img = cv2.imread(directoryFace+c+"/"+s+"/"+f) 
                img = cv2.imread(directoryFace+c+"/"+f) 
                #if not os.path.exists(directorySave+c+"/"+s+"/"):            
                #    os.makedirs(directorySave+c+"/"+s+"/")
                if not os.path.exists(directorySave+c+"/"):            
                    os.makedirs(directorySave+c+"/")
                #print directoryFace+c+"/"+s+"/"+f
                #print directoryFace+c+"/"+f
                #detectFace(img,directorySave+c+"/"+s+"/"+f,img)
                detectFace(img,directorySave+c+"/"+f,img)
                
                
def createFeatureFileGrayScaleSequences(imageSize, imageDirectory, featuresDirectory, featuresName, log):

        directory = imageDirectory        
        
        classesPath = os.listdir(directory)                
        featuresSetGray =  []        
        classNumber = 0
        for classs in classesPath:   
            sequences = os.listdir(directory+"/"+classs)
            for s in sequences:
                files = os.listdir(directory+os.sep+classs+os.sep+s+os.sep)      
                for image in files:                    
                 if (not "txt" in image and not "db" in image):                    
                    log.printMessage(("Reading:", directory+os.sep+classs+os.sep+s+os.sep+image))
                    img = cv2.imread(directory+os.sep+classs+os.sep+s+os.sep+image)
                    featuresGray = grayImage(img,imageSize,False,"")

                    featuresGray = whiten(featuresGray) 
                    featuresGray = resize(featuresGray,imageSize) 

                                                                                                                                                        
                    
                    newFeatures = [] 
                    newFeatures.append(int(classNumber)) 
                    
                    for x in featuresGray:
                       for y in x:
                            newFeatures.append(y)                                
                    
                    featuresSetGray.append(newFeatures)                
                
            classNumber = classNumber+1                             
        DataUtil.writeSingleFile(featuresSetGray,featuresDirectory+os.sep+featuresName, False) 




                
    

def createFeatureFile(imageSize, imageDirectory, featuresDirectory, featuresName, log, inputType):

        print "Input type:", inputType, "-", DataUtil.INPUT_TYPE["Sequence"]
        
        if inputType == DataUtil.INPUT_TYPE["3D"] or inputType == DataUtil.INPUT_TYPE["Sequence"]:
            createFeatureFileGrayScaleSequences(imageSize, imageDirectory,featuresDirectory, featuresName,log)
        else:    
            directory = imageDirectory        
            
            classesPath = os.listdir(directory)                
            featuresSetGray =  []        
            classNumber = 0
            for classs in classesPath:        
                    files = os.listdir(directory+os.sep+classs+os.sep)                      
                    for image in files:                    
                     if ( not "db" in image and not ".directory" in image):                
                        log.printMessage(("Reading:", directory+os.sep+classs+os.sep+image))
                        img =  Image.open(directory+os.sep+classs+os.sep+image)
                        img = numpy.array(img)   
                        if len(img.shape) == 3:
                            b,g,r = cv2.split(img)
                            img = cv2.merge([r,g,b])
                        
                        
                        #cv2.imwrite("/informatik2/wtm/home/barros/demo_ws/src/dialog/scripts/networkPosture/test/test1.jpg",img)
                        if not inputType ==  DataUtil.INPUT_TYPE["Color"]:
                            img = grayImage(img,imageSize,False,"")
                            
                        img = whiten(img) 
                        #cv2.imwrite("/informatik2/wtm/home/barros/demo_ws/src/dialog/scripts/networkPosture/test/test2.jpg",img)
                        img = resize(img,imageSize)
                        #cv2.imwrite("/informatik2/wtm/home/barros/demo_ws/src/dialog/scripts/networkPosture/test/test3.jpg",img)
                        
                        newFeatures = [] 
                        newFeatures.append(int(classNumber)) 
                        
                        for x in img:
                           for y in x:
                               if inputType ==  DataUtil.INPUT_TYPE["Color"]:
                                   colors = []
                                   for c in y:
                                       colors.append(c)
                                   newFeatures.append(colors)  
                               else:
                                    newFeatures.append(y)                                
                        
                        featuresSetGray.append(newFeatures)                
                    
                    classNumber = classNumber+1          
                    
            #print "Total features:", len(featuresSetGray)
            DataUtil.writeSingleFile(featuresSetGray,featuresDirectory+os.sep+featuresName, inputType ==  DataUtil.INPUT_TYPE["Color"])
        

 
def convertFloatImage(image):
    scale = numpy.max(numpy.abs(image))
    if numpy.any(image < 0):
        result = 255. * ((0.5 * image / scale) + 0.5)
    else:
        result = 255. * image / scale
    return result.astype(numpy.uint8)
 
def whiten(image):
    
#    return image
#    
    tmp = image - image.mean()
    return tmp / numpy.std(tmp)
   
#    if image.ndim > 3:
#        raise TypeError('Not more than 3 dimensions supported')
#
#    if image.ndim == 3:
#        tmp = numpy.empty_like(image)
#        for c in range(image.shape[2]):
#            tmp[:, :, c] = whiten(image[:, :, c])
#
#        result = numpy.zeros_like(image)
#        for c1 in range(image.shape[2]):
#            for c2 in range(image.shape[2]):
#                if c1 == c2:
#                    result[:, :, c1] += tmp[:, :, c2]
#                else:
#                    result[:, :, c1] -= tmp[:, :, c2]
#
#        return result
#
#    sigma1 = 0.2
#    img1 = ndimage.gaussian_filter(image, sigma1)
#    sigma2 = 5 * sigma1
#    img2 = ndimage.gaussian_filter(image, sigma2)
#    result = img1 - img2
#    
#    tmp = result - result.mean()
#    
#    return tmp / numpy.std(tmp)
    
    #return result
    
    
#whitenAll("/informatik2/wtm/home/barros/Documents/Experiments/JaffeDataset/images/","/informatik2/wtm/home/barros/Documents/Experiments/JaffeDataset/JaffeWhiten/")        
#applyMaskCam3DCorpus()
#createFeatureFileGrayScaleSequences((28,28),"/informatik2/wtm/home/barros/Pictures/Datasets/CambridgeDataset/Set1_30Elements/","/informatik2/wtm/home/barros/Documents/Experiments/Cambridge/features/","28x28Sequences")
#normalizeInputs(20,"/informatik2/wtm/home/barros/Documents/Experiments/FABO/Face_Only_Expressions/","/informatik2/wtm/home/barros/Documents/Experiments/FABO/Face_Only_Expressions_20/")
#extractFace("/informatik/isr/wtm/home/barros/Documents/Experiments/FABO/images/","/informatik/isr/wtm/home/barros/Documents/Experiments/FABO/images_Body_faceOnly_expression/")
    

#applySkinSegmentation(cv2.imread("//export/gestures/Subject1/Abort/Sequence_0/test.png"),"/export/gestures/tests/test2.png" )