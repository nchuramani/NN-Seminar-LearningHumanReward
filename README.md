# NN-Seminar-LearningHumanReward
Seminar Submission for Neural Networks SoSe2016

This project is the submission for NN-Seminar SoSe 2016 organized by WTM Group at the Department of Informatik, Universität Hamburg. 
The project consists of the following files:
- DataUtil.py: (refer barros@informatik.uni-hamburg.de)
  - DataUtil is used to keep the data consistent in a way that all images loaded are converted to grayscale before they can be used, storing and loading networks etc. It is used particularly here to load the pre-trained CNN (repetition_13_BestTest_testCK9_.save) parameters.
- ImageProcessingUtil.py: (refer barros@informatik.uni-hamburg.de)
  - ImageProcessingUtil is used to perform operation on images such as resizing the image, detecting faces in the image (using haarscale) etc. For the larger network, it used to perform smoothening operations, data-augmentation etc. using the images.  
- MCCNNExperiments_old.py: (refer barros@informatik.uni-hamburg.de)
  - MCCNNExperiments_old is used to load the network (CNN) from the parameters extracted by DataUtil.py. This network is then used to classify images into emotion categories.
- MCCNN.py: (refer barros@informatik.uni-hamburg.de)
  - MCCNN represents the CNN classifier returned by MCCNNExperiments_old. It is used to classify the emotions in categories. It is also used to get information about the CNN and its performance.
- haarcascade_frontalface_alt.xml:
  - It is the predefined xml file used to detect faces in images using CV2 functionality. It is a standard file which is freely available online.
- repetition_13_BestTest_testCK9_.save:
  - Pre-trained CNN on Cohn-Kanade dataset for seven emotions
- tamer_implementation_iCub.py:
  - The main implementation in the project. Consists of the code for creating and training the MLP and also loading the CNN. It uses some functionality from the iCub python wrapper used at WTM (more details can be obtained from barros@informatik.uni-hamburg.de) to use the on-board camera as well as the iCub robot head emotion module to represent different emotions. 
