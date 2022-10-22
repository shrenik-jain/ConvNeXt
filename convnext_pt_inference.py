import time

import cv2
import numpy as np
import timm
import torch

labels = ['shutter_down', 'shutter_up']
id2label = {k:v for k, v in enumerate(labels)}
label2id = {v:k for k, v in enumerate(labels)}
print(id2label)

model = timm.create_model('convnext_nano', pretrained=True, num_classes=2)
checkpoint = torch.load('weights/convnext_final_weights/timm_convnext_best.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict']) 

cap = cv2.VideoCapture('videos/example.mp4')

# Mask To Apply [If Required]
contours = np.array([[262, 1], [351, 0], [351, 287], [248, 287]]) 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
t = []

while (True):
    _, frame = cap.read()
    image = frame.copy()

    # create mask
    mask = np.zeros(frame.shape, dtype = np.uint8)
    cv2.fillPoly(mask, pts = [contours], color = (255,255,255))

    # apply the mask
    frame = cv2.bitwise_and(frame, mask)

    # preprocess frame
    frame = cv2.resize(frame, (50, 50))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = (frame / 255.0 - mean) / std
    frame = np.expand_dims(frame, 0)
    frame = np.transpose(frame, [0, 3, 1, 2]) 
    frame = torch.from_numpy(frame).float()

    t1 = time.perf_counter()
    outputs = model(frame)
    t2 = time.perf_counter()
    t.append(t2 - t1)

    predicted_class_idx = outputs.argmax(-1).item()
    o = id2label[predicted_class_idx]
    print(o)
    
    image = cv2.putText(image, o, (5,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA)
    cv2.imshow('Image', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# print(t)
print(f"Avg Time Taken for Prediction: {round(np.mean(t)*1000, 2)}ms")
