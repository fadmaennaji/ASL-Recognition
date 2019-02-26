#*************imports********
import cv2
import numpy as np
from keras.models import load_model
import copy
from keras.preprocessing.image import img_to_array
from imutils.video import FPS

# the lower and the upper are the range of skin color oh human in the hsv model
lower = np.array([2, 50, 50], dtype="uint8")
upper = np.array([15, 255, 255], dtype="uint8")
number_of_histogram_elements=16
track_mode =False

# Read the hand model
classifier = cv2.CascadeClassifier("hand2.xml")
roiBox = None
# Read our  model
model = load_model("model.h5")

#here we defined the types of tracker , there is many types 
tracker_types = ['BOOSTING', 'MIL','KCF']
#here we choose the tracker type that we are going to use ==> "KCF"
tracker_type = tracker_types[2]
def tracking(tracker_type):
    if tracker_type == 'BOOSTING':
       
            tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL': 
            tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()

    return tracker
tracker = tracking(tracker_type)
cam = cv2.VideoCapture(0)
fps = None

while cam.isOpened:
    # this when you click on espace the program will stop  (pause) 
    key = cv2.waitKey(10)
    if key == 27:
        break
    r,frame = cam.read()
    cv2.flip(frame,1,0)
    orig_f = frame.copy()
    #here we transfert the frame from BGR to gray
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #here we resize the frame to be small
    miniFrame = cv2.resize(frame,(int(frame.shape[1]/4),int(frame.shape[0]/4)))
    # here we apply the XML model in the real video
    hands = classifier.detectMultiScale(miniFrame,1.03,3)   
     # here just to sort all the hands available in the frame 
    hands = sorted(hands, key=lambda x: x[3])
    if track_mode == False:# track mode == false i mean we didn't begin tracking because tracking is so deffirent than detection 
        #in this block we will initialize the object that we will track )
        # we find the object using detection and after we keep trackign it using an algorithm of tracking 
        if hands:
            hand_i = hands[0]
            (x,y,w,h) = [v*4 for v in hand_i]
            hand = frame[y:y+w,x:x+h]
             #initialzing the object that will be tracked
             # the  Region of interest here is the object that will be tracked hand
            roi = orig_f[y:y+h, x:x+w]
            roiBox = (x,y-10,w,h-10)
            # this the bounding box 
            bbox=(x,y,w,h)
            #after all that steps ready to track the object
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(orig_f, bbox)
            track_mode = True 
    if track_mode == True : #tracking mode
        # Start timer
        timer = cv2.getTickCount()
        # Update tracker
        ok, bbox = tracker.update(orig_f)
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        print(ok) 
        if ok:
            # Tracking success
            cv2.putText(orig_f, "Tracking failure success ", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.80,(10,50,100))
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(orig_f, p1, p2, (255,0,0), 2, 1)
            #cv2.imshow("RoiBox,",orig_f[p1[0]:p2[0], p1[1]:p2[1]])
            img = cv2.resize(orig_f[p1[0]:p2[0], p1[1]:p2[1]],(64,64)) 
            img = img.astype("float") / 255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            classes=model.predict_proba(img)
            classes=classes.argmax(axis=-1)
            print("the output",classes)
            classes1 = model.predict(img)
            percentage = classes1[0][classes]
            print(percentage)
            # nothing  = 14 , space == 19
            if classes[0] == 14 :
                print(" NOTHING ")
                currentchar= "------NOTHING------"
            elif classes[0] == 19 :
                print(" ")
                currentchar= "       " 
            else :
                if classes[0]<14 :
                    print(chr(classes[0]+65))
                    currentchar= chr(classes[0]+65)
                elif classes[0]<19 and classes[0]>14 :
                          
                    print(chr(classes[0]+65-1))
                    currentchar= chr(classes[0]+65-1)
                else :
                    print(chr(classes[0]+65-2))
                    currentchar= chr(classes[0]+65-2)
            cv2.putText(orig_f, "The name of class :"+currentchar , (120,140), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            cv2.putText(orig_f, "The Porcentage of output :"+str(percentage) , (140,180), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
        else :
            # Tracking failure
            cv2.putText(orig_f, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            track_mode = False
        # Display tracker type on frame
        cv2.putText(orig_f,  " The Tracker Types is :"+tracker_type , (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        # Display FPS on frame
        cv2.putText(orig_f, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    cv2.imshow("orig",orig_f)

