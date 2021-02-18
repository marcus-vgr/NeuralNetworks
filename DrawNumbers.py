"""
In this file a simple app is constructed using PyQt5 to allow the user draw numbers with his mouse. 
The prediction of the drawing is then shown based on the results of the already trained Neural Network 
with the MNIST handwritten dataset  (see MNIST_handwritten jupyter notebook for details).
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from PyQt5 import QtGui, QtWidgets, QtCore
import cv2
import NeuralNetwork as NN
from scipy import ndimage

# Loading model
path_model = 'Digit_Recognizer.model'
model = NN.Model.load('Digit_Recognizer.model')


"""
In order to use our trained model, the images to be predicted have to must processed in the same way as the MNIST
dataset. Therefore, for each handwritten image we have to:

1) Resize to 28x28 pixels
2) Put the number in a 20x20 box inside the 28x28 image
3) Centralize the number in the box

Those tasks are done in the ConvertImage_MNIST function, that receives as input the image drawn by the user
"""
def ConvertImage_MNIST(image):
    # Resize the image to 28x28 pixels
    image = cv2.resize(image, (28,28))

    # Delete every row and column that does not contain information about the number, i.e, 
    # we transform the image to be only the 'block' that contains the number.
    while np.sum(image[0]) == 0:
        image = image[1:]
    while np.sum(image[:,0]) == 0:
        image = np.delete(image,0,1)
    while np.sum(image[-1]) == 0:
        image = image[:-1]
    while np.sum(image[:,-1]) == 0:
        image = np.delete(image,-1,1)

    # Resize the image to certify that it does no exceed the 20x20 box
    rows,cols = image.shape
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        image = cv2.resize(image, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        image = cv2.resize(image, (cols, rows))

    # Now we add zeros in the array until the image becomes again a 28x28 2D array
    colsPadding = (int(np.ceil((28-cols)/2.0)),int(np.floor((28-cols)/2.0)))
    rowsPadding = (int(np.ceil((28-rows)/2.0)),int(np.floor((28-rows)/2.0)))
    image = np.lib.pad(image,(rowsPadding,colsPadding),'constant')

    
    # Finally, we should centralize the image. To do this, we first obtain the coordinates of the center-of-mass of the
    # image using the ndimage from the scipy library, and then obtain what would be the shift to centralize the image
    rows,cols = image.shape
    cy,cx = ndimage.measurements.center_of_mass(image)
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    # Then the image can be centralized using the cv2 warpAffine function.
    M = np.float32([[1,0,shiftx],[0,1,shifty]])
    image = cv2.warpAffine(image, M, (cols,rows))    
    
    return image

"""
The output of our prediction is given as a plot. Getting user image, performing the forward propagation in the
Neural Network and showing the predictions is a job for the DigitRecognizer function.
"""
def DigitRecognizer(image):

    # Convert image to MNIST style.
    image_resized = ConvertImage_MNIST(image)
    
    # Rescale and reshape the image to use as input in the Neural Network
    image_pred = ((image_resized.reshape(1,-1) - 127.5)/ 127.5)
    
    # Perform a forward propagation in our already trained Neural Network and get the predictions
    pred = model.forward(image_pred, training=False)
    
    # Plot the results
    #fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(13,4))
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(13,4))
    ax[1].bar(range(10), pred.flatten())
    ax[0].imshow(image)
    #ax[2].imshow(image_resized)
    _ = ax[1].set_xticks(ticks=range(10))
    _ = ax[1].set_ylabel('Probability')
    _ = ax[0].set_title(f'Your Drawing!', fontsize=15)
    #_ = ax[2].set_title('Rescaled Image')
    fig.suptitle(f'You drew a {np.argmax(pred)} !', fontsize=20)
    plt.show()


"""
Here goes the app using PyQt5 to draw numbers. The coordinates of mouse clicking during the drawing is saved
to then be used to construct the image as a 2D array
"""
class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        pixmap = QtGui.QPixmap(512, 512) # Create black window of 512x512 pixels
        self.setPixmap(pixmap)

        self.last_x, self.last_y = None, None  # create variables that give the coordinates of the click
        self.coordinates_img = [] # create list to save coordinates
        self.pen_color = QtGui.QColor('blue') # set pen color to blue

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        # Draw in the black window
        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(15)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()
        # Save the coordinates
        self.coordinates_img.append([self.last_x, self.last_y])

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.canvas = Canvas()

        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)
        l.addWidget(self.canvas)
        self.setCentralWidget(w)

        # Create a button that when clicked transform coordinates_img array into a global variable that can be used
        # to construct the image as a 2D array.
        self.btn = QtWidgets.QPushButton("Predict", self)
        self.btn.move(220,500)
        self.btn.clicked.connect(self.clickMethod)

    def clickMethod(self):
        global coordinates_img 
        coordinates_img = np.array(self.canvas.coordinates_img)
        window.close()



"""
Main program
"""
# Initialize the app to draw numbers
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()

# Given the coordinates of the number, we create a 2D array that will represent our number
image = np.zeros((512,512))
for point in coordinates_img:
    for i in range(15):
        for j in range(15):
            
            if 0 <= point[1]+i <= 512 and 0 <= point[0]+j <= 512:
                image[point[1]+i, point[0]+j] = 255 

            if 0 <= point[1]-i <= 512 and 0 <= point[0]+j <= 512:
                image[point[1]-i, point[0]+j] = 255 

            if 0 <= point[1]+i <= 512 and 0 <= point[0]-j <= 512:
                image[point[1]+i, point[0]-j] = 255 

            if 0 <= point[1]-i <= 512 and 0 <= point[0]-j <= 512:
                image[point[1]-i, point[0]-j] = 255 

# Predict
DigitRecognizer(image)