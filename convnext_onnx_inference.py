import time
import numpy as np
import cv2
import onnxruntime as rt


model = 'weights/convnext_final_weights/timm_convnext_nano_best.onnx'
session = rt.InferenceSession(model, None)

labels = ['shutter_down', 'shutter_up']
id2label = {k:v for k, v in enumerate(labels)}
label2id = {v:k for k, v in enumerate(labels)}
print(id2label)

cap = cv2.VideoCapture('videos/example.mp4')
 
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
    frame = frame.astype('float32')

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    t1 = time.perf_counter()
    result = session.run([output_name], {input_name: frame})
    t2 = time.perf_counter()

    t.append(t2 - t1)

    predicted_class_idx = np.argmax(result)
    o = id2label[predicted_class_idx]
    print(o)
    
    image = cv2.putText(image, o, (5,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA)
    cv2.imshow('Image', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# print(t)
print(f"Avg Time Taken for Prediction: {round(np.mean(t)*1000, 3)}ms")