#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = torch.hub.load("ultralytics/yolov5", "yolov5s")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    result = model(frame)
    out = np.squeeze(result.render())
    
    cv2.imshow("YOLO", out)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




