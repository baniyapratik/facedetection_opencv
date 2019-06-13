#!/usr/bin/env python


#Importing the libraries
import os
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))

face_cascade_location = dir_path+'/haarcascade_frontalface_default.xml'
smile_cascade_location = dir_path+'/haarcascade_smile.xml'

# loading cascades
face_cascade = cv2.CascadeClassifier(face_cascade_location)
smile_cascade = cv2.CascadeClassifier(smile_cascade_location)

# method for face detection
min_no_neighbors = 5
scale_factor = 1.3
rectangle_thickness = 2
face_rectangle_color = (241, 113, 113)
smile_rectangle_color = (42, 138, 138)
# defining a function that will do the detection
def detect(gray, original_frame):
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_no_neighbors)
    for (x,y,w,h) in faces:
        cv2.rectangle(original_frame, (x,y), (x+w, y+h), face_rectangle_color, rectangle_thickness)
        roi_gray = gray[y:y+h, x: x+w]
        roi_color = original_frame[y:y+h, x: x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), smile_rectangle_color, rectangle_thickness)
    return original_frame    


# generate sketch 
def sketch(image):
    # convert image to gray scale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # clean up image usng gaussian blur
    img_gray_blur = cv2.GaussianBlur(image, (5,5), 0)
    # extract edges
    edges = cv2.Canny(img_gray_blur, 20, 70)
    
    ret, mask = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask

# face recognition with webcam
video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    #cv2.imshow('Our Live Sketch', sketch(frame))
    cv2.imshow('Video', canvas)
   
    if  cv2.waitKey(1) == 13: # break when enter is executed
        break

# release the camera and close the windows
video_capture.release()
cv2.destroyAllWindows()


# In[6]:


video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




