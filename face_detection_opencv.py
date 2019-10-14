#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


cap = cv2.VideoCapture(0)


# In[3]:


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# In[ ]:


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    #print(faces)
    for rect in faces:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




