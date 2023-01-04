import cv2
import numpy

class webcam:
    @staticmethod
    def video():
        # Open the webcam
        cap = cv2.VideoCapture(0)

        while True:

            # Creates cropped Webcam view
            ret, frame = cap.read()
            height, width = frame.shape[:2]
            cropped = frame[:, int((width-height)/2):int((width+height)/2)]
            cv2.imshow("Webcam Preview", cropped)

            # Checks for the Enter key
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                break
            pass

        # Close Webcam
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cap.release()
        cv2.destroyAllWindows()

        # Prepare data
        inverted = 255 - gray
        floored = numpy.where(inverted < 150, 0, inverted)
        resized = cv2.resize(floored, (28, 28))
        flattened = resized.flatten()
        web_cam_input = ','.join(str(x) for x in flattened)

        return web_cam_input
    pass