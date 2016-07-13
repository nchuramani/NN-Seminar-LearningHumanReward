# -*- coding: utf-8 -*-

import datetime
import os

class LogUtil:
    
    logFile = ""
    
    def createFolder(self,directory):
        if not os.path.exists(directory):            
            os.makedirs(directory)
    
    def startNewStep(self,stepName):
        
        print str(datetime.datetime.now()) + ": ------------------------------------------------------------------\n"
        print str(datetime.datetime.now()) + ":" + str(stepName) + "\n"
        print str(datetime.datetime.now()) + ": ------------------------------------------------------------------\n"
        
        write = str(datetime.datetime.now()) + ": ------------------------------------------------------------------\n"
        write += str(datetime.datetime.now()) + ":" + str(stepName) + "\n"
        write += str(datetime.datetime.now()) + ": ------------------------------------------------------------------\n"
        
        f = open(self.logFile,"a")        
        f.write(str(write)+"\n")        
        f.close() 

        
    def printMessage(self,message):
        
        f = open(self.logFile,"a") 
        
        for m in str(message).split("\n"):
            print str(datetime.datetime.now()) + ": - ", m
            write = str(datetime.datetime.now()) + ": - ", m
                           
        
            f.write(str(write)+"\n")        
        f.close() 
    
    def createLog(self,experimentName,directory):        
        self.createFolder(directory)        
        self.logFile = directory + str(datetime.datetime.now()).replace(" ", "")+"_"+experimentName + ".txt"
        


    
