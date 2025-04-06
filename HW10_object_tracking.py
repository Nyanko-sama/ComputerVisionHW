# ## Homework 10
# In this homework, you are going to use and compare two different trackers (of your liking) and compare the results.
# ### Step 1 ###
# Decide what video you are going to use for this homework, select an object and generate the template. 
# You can use any video you want (your own, from Youtube, etc.)
# and track any object you want (e.g. a car, a pedestrian, etc.).
import cv2
import numpy as np
from matplotlib import pyplot as plt

# read the video
video = cv2.VideoCapture("RelaxingCarDrive.mp4")
video.set(cv2.CAP_PROP_POS_FRAMES, 59)      # set to 60th frame, because opening frames are black

# read the first frame
ret, frame = video.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# select ROI (region of interest) - the object to be tracked
bbox = cv2.selectROI(frame.copy(), False)
cv2.destroyAllWindows() 

# ### Step 2 ###
# Initialize a tracker (e.g. KCF). CSRT, TLD
csrt_tracker = cv2.TrackerCSRT_create()
medianFlow_tracker = cv2.legacy.TrackerMedianFlow_create()

# ### Step 3
# Run the tracker on the video and the selected object. Run the tracker for around 10-15 frames.
# ### Step 4
# For each frame, print the bounding box on the image and save it.
def run_tracker(tracker, video, bbox, name):
    ret, frame = video.read()
    assert ret
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize tracker
    ok = tracker.init(frame, bbox)

    # run tracker for 15 frames
    for i in range(15):
        for j in range(3):          # # skip 3 frame
            ret, frame = video.read()
        if not ret:
            print('End of video.')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
            
        ok, bbox = tracker.update(frame)
        print(ok, bbox)
          
        # Draw bounding box
        x1, y1 = int(bbox[0]), int(bbox[1])
        width, height = int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)
        
        # Save the frame with bounding box
        cv2.imwrite(f"{name}_frame_{i}.jpg", frame)

        # Show the frame with bounding box
        plt.imshow(frame)
        plt.show(), plt.draw()    
        plt.waitforbuttonpress(0.1)
        plt.clf()

# ### Step 5
# Select a different tracker (e.g. CSRT) and repeat steps 2, 3 and 4.

# run the tracker with CSRT
run_tracker(csrt_tracker, video, bbox, "CSRT")

video.set(cv2.CAP_PROP_POS_FRAMES, 60)      # reset video to the 60th frame

# run the tracker with MedianFlow
run_tracker(medianFlow_tracker, video, bbox, "MedianFlow")

# ### Step 6
# Compare the results:
# * Do you see any differences? If so, what are they?
# at the beginning, both trackers perform well, but at the end there is an abrupt change in the size of frame and ROI (car), CSRT continues to
# track some object (hovewer, not the car), while MedianFlow fails to track any object.

# * Does one tracker perform better than the other? In what way?
# MedianFlow performs better in this case, although it fails to track the car at the end, it does not track any other object, 
# while CSRT continues to track some object (not the car).

# Logs: 
# CSRT
# True (205, 126, 66, 39)
# True (204, 127, 66, 39)
# True (204, 129, 63, 37)
# True (202, 129, 63, 37)
# True (198, 129, 65, 38)
# True (195, 129, 65, 38)
# True (193, 129, 63, 37)
# True (194, 129, 66, 39)
# True (191, 130, 63, 37)
# True (188, 131, 63, 37)
# True (185, 131, 63, 37)
# True (178, 133, 65, 38)
# True (176, 135, 62, 37)
# True (194, 145, 63, 37)
# True (194, 144, 63, 37)
# MedianFlow
# True (204.13193445108203, 125.65930344350605, 67.33081188396872, 39.786388840526975)
# True (201.62800946090385, 125.79154029456907, 69.65854711823133, 41.16186875168215)
# True (198.9508685783968, 127.21627953338593, 71.56989675433923, 42.291302627564086)
# True (195.3831291522844, 127.0621769836519, 72.98571007922027, 43.12791959226652)
# True (190.48681940286264, 125.89836158961769, 74.47661754193096, 44.00891036568648)
# True (188.11628838585915, 125.68592174427633, 75.19605787183636, 44.43403419699421)
# True (185.39631984836922, 125.56407951256176, 77.00833991751935, 45.504928133079616)
# True (181.42974430722933, 126.07984015582106, 78.68877493534603, 46.49791246179538)
# True (176.4550193207316, 127.24616616033731, 80.55686504506025, 47.60178389026287)
# True (171.13867440113353, 127.47866485053665, 82.34987898093607, 48.66129212509859)
# True (166.26511110181738, 127.73456897748991, 83.82974788913869, 49.53576011630923)
# True (161.06938295183932, 129.01836156304975, 85.12980404261043, 50.30397511608798)
# True (156.67158946849867, 130.6332121667327, 85.98557167335424, 50.80965598880024)
# False (0.0, 0.0, 0.0, 0.0)
# False (0.0, 0.0, 0.0, 0.0)