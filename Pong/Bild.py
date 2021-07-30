import mss
import cv2
import numpy

class Caster():
    def __init__(self) -> None:
        self.x = 340
        self.y = 850
        self.sct = mss.mss()
        self.dimensions = {
                'left': 660,
                'top': 350,
                'width': 600,
                'height': 400
            }
        
    def screencast(self):

        scr = numpy.array(self.sct.grab(self.dimensions))

        # Cut off alpha
        scr_remove = scr[:,:,:3]

        cv2.imshow('Screen Shot', scr)
        cv2.waitKey(1)
        return scr_remove

if __name__ == '__main__':
    # Use this to adjust the window
    cast = Caster()
    while True:
        scr_remove = cast.screencast()
        # print(scr_remove)
        # break
