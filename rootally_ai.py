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


# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2

img_list = []

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
   'M', 'J', 'P', 'G'), 30, (frame_width, frame_height))



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
    
    # Check if the angle is less than zero.
    # if angle < 0:
 
    #     # Add 360 to the found angle.
    #     angle += 360
    
    # Return the calculated angle.
    return angle

def knee(landmark1,landmark2,landmark1_index,landmark2_index):
    
    v1 = landmark1.z
    v2 = landmark2.z

    if v1 < v2:
        return landmark1,landmark1_index
    else:
        return landmark2,landmark2_index
    

def timer():
    timestamp = time.time()
    startup = False
    return timestamp

def fluctuations(angle,last_ang):
    if abs(angle-last_ang) >= 20:
        return True
    else:
        return False

def is_bend(angle):
    global last_ang
    global startup
    global count
    msg = "keep your knee bend"
    if angle > -140:
        if startup:
            count = timer()
            startup = False
            last_ang = angle
            return
        else:
            count1 = timer()
            a = count1 - count
            print(a)
            if a > 8:

                last_ang = angle
                return("Straighen your knee")
        
            else:
                last_ang = angle
                return str(math.trunc(a))
    else:
        #check if there is  abrupt change
        
        check = fluctuations(angle,last_ang)
        if check:
            last_ang = angle
            return 
        else:
            startup = True
            last_ang = angle
            print(msg)
            return msg






while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    
    image2 = image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    image2.flags.writeable = True




    #### which knee is visible 
    try:
        landmark1,index = knee(results.pose_landmarks.landmark[25],results.pose_landmarks.landmark[26],25,26)
        landmark0 = results.pose_landmarks.landmark[index-2]
        landmark2 = results.pose_landmarks.landmark[index+2]
        angle = calculateAngle(landmark0,landmark1,landmark2)
        x = is_bend(angle)


        image2 = cv2.putText(image2, x, org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
        img_list.append(image2)
    except:
        pass
    #print(x)

print(len(img_list))
for i in range(len(img_list)):
    out.write(img_list[i])

print('done')
    

pose.close()
cap.release()
out.release()
