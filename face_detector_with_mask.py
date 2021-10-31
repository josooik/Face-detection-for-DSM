import cv2, sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def detect_flow(imgPath, modelDir):
    
    # read image
    img = cv2.imread(imgPath)
    (img_h, img_w) = img.shape[:2]
    print("* org img shape:", imgPath,  img.shape)
    
    # use model
    facenet = cv2.dnn.readNet(modelDir+"/deploy.prototxt",
                              modelDir+"/res10_300x300_ssd_iter_140000.caffemodel")
    
    blob = cv2.dnn.blobFromImage( img, 
                                  1.0, (300, 300),
    	                          (104.0, 177.0, 123.0))
    facenet.setInput(blob)
    detections = facenet.forward()
    print("* dections results:", detections.shape)
  
    
    # get results from detections
    
    areas = []
    for i in range(detections.shape[2]):
      conf = detections[0,0,i,2]
      if conf <0.5:
        continue
      x1 = int(detections[0,0,i,3]*img_w)
      y1 = int(detections[0,0,i,4]*img_h)
      x2 = int(detections[0,0,i,5]*img_w)
      y2 = int(detections[0,0,i,6]*img_h)
     
      areas.append([x1,y1,x2,y2])
    
    print(areas)
    return img, areas



def classify_flow(img, areas, modelDir):

    # pre-processing
    for (x1, y1, x2, y2) in areas:
        face = img[y1:y2, x1:x2]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        print(face.shape)
    
        mask_model = models.load_model(modelDir+"/mask_model.h5")
        (mask, withoutMask) = mask_model.predict(face)[0]
        print( mask, withoutMask)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(img, label, (x1, y2 - 10),
	      	      cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    cv2.imshow("out:", img)
    cv2.waitKey(0)



def main():
    imgPath = sys.argv[1]
    modelDir = sys.argv[2]
    img, areas = detect_flow(imgPath, modelDir)
    classify_flow(img, areas, modelDir)

if __name__=='__main__':
    main()
