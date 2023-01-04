import cv2
import numpy
import scipy.special
import matplotlib.pyplot

class webcam:
    @staticmethod
    def video():
        # Open the webcam
        cap = cv2.VideoCapture(0)

        while True:
            # Capture a single frame
            ret, frame = cap.read()

            # Crop the image to a square
            height, width = frame.shape[:2]
            cropped = frame[:, int((width-height)/2):int((width+height)/2)]

            # Display the cropped frame
            cv2.imshow("Webcam Preview", cropped)

            # Check if the user pressed the Enter key
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                # The user pressed Enter, so capture the image
                break

        # Convert the image to grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        cap.release()
        cv2.destroyAllWindows()

        # Invert the grayscale values
        inverted = 255 - gray

        # Floor the inverted grayscale values to 0 if they are smaller than 100
        floored = numpy.where(inverted < 150, 0, inverted)

        # Resize the image to 28x28 pixels
        resized = cv2.resize(floored, (28, 28))

        # Flatten the image into a 1D array
        flattened = resized.flatten()

        # Convert the array to a string in the format of the MNIST dataset
        web_cam_input = ','.join(str(x) for x in flattened)

        # Release the webcam and destroy the preview window
        return web_cam_input