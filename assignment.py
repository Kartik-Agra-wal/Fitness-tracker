import cv2
import mediapipe as mp
import numpy as np
import sys
import math

import time

global startup,last_ang,count
startup = True
time_to_stop = 8 #seconds

last_ang = 0
count= 0

###############parameters for text ########################
# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 1
########################################################

img_list = []


###### intiaalizing  mediapipe pose land mark model to get poses#######
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture('E:\Computer Vision\KneeBendVideo.mp4')

if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# outdir, inputflnm = sys.argv[1][:sys.argv[1].rfind(
#     '/')+1], sys.argv[1][sys.argv[1].rfind('/')+1:]
# inflnm, inflext,_ = inputflnm.split('.')
# out_filename = f'{outdir}{inflnm}_annotated1.mp4'
out = cv2.VideoWriter('new.mp4', cv2.VideoWriter_fourcc(
*'mp4v'), 30, (frame_width, frame_height))

# gets the file name from command line argument 
# cap = cv2.VideoCapture(sys.argv[1])

# if cap.isOpened() == False:
#     print("Error opening video stream or file")
#     raise TypeError

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))


# ####### making a new video file with name filename_annotated.mp4
# outdir, inputflnm = sys.argv[1][:sys.argv[1].rfind(
#     '/')+1], sys.argv[1][sys.argv[1].rfind('/')+1:]
# inflnm, inflext,fnam = inputflnm.split('.')
# out_filename = f'{outdir}{inflext}_annotated_new.mp4'
# out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), 60, (frame_width, frame_height))


#### this func calcultes the knee angle #####
def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
 
    '''
 
    # Get the required landmarks coordinates.
    x1, y1 = landmark1.x,landmark1.y
    x2, y2 = landmark2.x,landmark2.y
    x3, y3 = landmark3.x,landmark3.y
 
    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    
    # Return the calculated angle.
    return angle


#### this func checks which knee is closer to the camera#####
"""
using just one pose as per the video would have given slightly better results as the model is not 100 percent accurate
however 
this func is created to genralize the script to work on any video 
 
"""
def knee(landmark1,landmark2,landmark1_index,landmark2_index):
    
    v1 = landmark1.z
    v2 = landmark2.z

    if v1 < v2:
        return landmark1,landmark1_index
    else:
        return landmark2,landmark2_index
    
### func to start timer 
def timer():
    timestamp = time.time()
    startup = False
    return timestamp


### this func handles the fluctuations
"""
the logic is if there is a an abrupt change of knee angle say 20 deg from one frame to another 
simply ignore such condition and resume the timer if its on 
or ask for knee to be bend 
"""
def fluctuations(angle,last_ang):
    if abs(angle-last_ang) >= 20:
        return True
    else:
        return False


##### this func checks wheter the knee is bend or not 
def is_bend(angle):
    global last_ang
    global startup
    global count
    msg = "keep your knee bent"
    if angle > -140:  ### as indicated in the problem statement 
        if startup: ### signals the starting of timer 
            count = timer()
            startup = False
            last_ang = angle
            return
        else: # once timer starts this returns the time and the angle of the knee and gives the prompt to straighten the knee after 8 sec are up
            count1 = timer()
            a = count1 - count
            #print(a)
            if a > time_to_stop:

                last_ang = angle
                return("Straighen your knee")
        
            else:
                last_ang = angle
                return str(math.trunc(a))+ str(" ") + str(math.trunc(abs(angle))) + str(" deg")
    else:
        #check if there is  abrupt change
        check = fluctuations(angle,last_ang)
        if check:
            last_ang = angle
            return 
        else:
            startup = True
            last_ang = angle
            #print(msg)
            return msg





### getting the frames
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image) ### process the image to get landdmarks

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) ### draws the poses on them image
   




    #### which knee is visible 
    try:
        """
        1. find which knee is closer 
        2. get the corresponding hip and ankle landmarks
        3. calcuate angle
        4. check if its bend
        
        """
        landmark1,index = knee(results.pose_landmarks.landmark[25],results.pose_landmarks.landmark[26],25,26)
        landmark0 = results.pose_landmarks.landmark[index-2]
        landmark2 = results.pose_landmarks.landmark[index+2]
        angle = calculateAngle(landmark0,landmark1,landmark2)
        x = is_bend(angle)

        ### write the promts, timer and the angle on the image
        image = cv2.putText(image, x, org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
        img_list.append(image)
    except:
        pass


##### write the images to the new video 

"""
note : the new video has lower fps than the original video and therefore longer in duration
however the timmer and angle measure are in sync with the actions performed in the original video 
"""
for i in range(len(img_list)):
    out.write(img_list[i])

print('done')
    
#### exit the program
pose.close()
cap.release()
out.release()
