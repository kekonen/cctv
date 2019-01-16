
import cv2
from datetime import datetime
import time

class MotionDetectorInstantaneous():
    
    def onChange(self, val): #callback when the user change the detection threshold
        self.threshold = val
    
    def __init__(self,threshold=8, doRecord=True, showWindows=True):
        self.writer = None
        self.font = None
        self.doRecord=doRecord #Either or not record the moving object
        self.show = showWindows #Either or not show the 2 windows
        self.frame = None
    
        self.capture=cv2.CaptureFromCAM(0)
        self.frame = cv2.QueryFrame(self.capture) #Take a frame to init recorder
        if doRecord:
            self.initRecorder()
        
        self.frame1gray = cv2.CreateMat(self.frame.height, self.frame.width, cv2.CV_8U) #Gray frame at t-1
        cv2.CvtColor(self.frame, self.frame1gray, cv2.CV_RGB2GRAY)
        
        #Will hold the thresholded result
        self.res = cv2.CreateMat(self.frame.height, self.frame.width, cv2.CV_8U)
        
        self.frame2gray = cv2.CreateMat(self.frame.height, self.frame.width, cv2.CV_8U) #Gray frame at t
        
        self.width = self.frame.width
        self.height = self.frame.height
        self.nb_pixels = self.width * self.height
        self.threshold = threshold
        self.isRecording = False
        self.trigger_time = 0 #Hold timestamp of the last detection
        
        if showWindows:
            cv2.NamedWindow("Image")
            cv2.CreateTrackbar("Detection treshold: ", "Image", self.threshold, 100, self.onChange)
        
    def initRecorder(self): #Create the recorder
        codec = cv2.CV_FOURCC('M', 'J', 'P', 'G') #('W', 'M', 'V', '2')
        self.writer=cv2.CreateVideoWriter(datetime.now().strftime("%b-%d_%H_%M_%S")+".wmv", codec, 5, cv2.GetSize(self.frame), 1)
        #FPS set to 5 because it seems to be the fps of my cam but should be ajusted to your needs
        self.font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8) #Creates a font

    def run(self):
        started = time.time()
        while True:
            
            curframe = cv2.QueryFrame(self.capture)
            instant = time.time() #Get timestamp o the frame
            
            self.processImage(curframe) #Process the image
            
            if not self.isRecording:
                if self.somethingHasMoved():
                    self.trigger_time = instant #Update the trigger_time
                    if instant > started +5:#Wait 5 second after the webcam start for luminosity adjusting etc..
                        print (datetime.now().strftime("%b %d, %H:%M:%S"), "Something is moving !")
                        if self.doRecord: #set isRecording=True only if we record a video
                            self.isRecording = True
            else:
                if instant >= self.trigger_time +10: #Record during 10 seconds
                    print (datetime.now().strftime("%b %d, %H:%M:%S"), "Stop recording")
                    self.isRecording = False
                else:
                    cv2.PutText(curframe,datetime.now().strftime("%b %d, %H:%M:%S"), (25,30),self.font, 0) #Put date on the frame
                    cv2.WriteFrame(self.writer, curframe) #Write the frame
            
            if self.show:
                cv2.ShowImage("Image", curframe)
                cv2.ShowImage("Res", self.res)
                
            cv2.Copy(self.frame2gray, self.frame1gray)
            c=cv2.WaitKey(1) % 0x100
            if c==27 or c == 10: #Break if user enters 'Esc'.
                break            
    
    def processImage(self, frame):
        cv2.CvtColor(frame, self.frame2gray, cv2.CV_RGB2GRAY)
        
        #Absdiff to get the difference between to the frames
        cv2.AbsDiff(self.frame1gray, self.frame2gray, self.res)
        
        #Remove the noise and do the threshold
        cv2.Smooth(self.res, self.res, cv2.CV_BLUR, 5,5)
        cv2.MorphologyEx(self.res, self.res, None, None, cv2.CV_MOP_OPEN)
        cv2.MorphologyEx(self.res, self.res, None, None, cv2.CV_MOP_CLOSE)
        cv2.Threshold(self.res, self.res, 10, 255, cv2.CV_THRESH_BINARY_INV)

    def somethingHasMoved(self):
        nb=0 #Will hold the number of black pixels
        min_threshold = (self.nb_pixels/100) * self.threshold #Number of pixels for current threshold
        nb = self.nb_pixels - cv2.CountNonZero(self.res)
        if (nb) > min_threshold:
           return True
        else:
           return False
        
if __name__=="__main__":
    detect = MotionDetectorInstantaneous(doRecord=True)
    detect.run()
