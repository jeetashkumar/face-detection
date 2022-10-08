# face-detection
Real time face detection


cap = cv2.VideoCapture(0)

face_detection_model = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt',
                                                './models/res10_300x300_ssd_iter_140000_fp16.caffemodel')

while True:
    ret, frame = cap.read()
    
    if ret == False:
        break
        
    img_detection = face_detection_dnn(frame)
    
    cv2.imshow('Real Time Face Detection with DNN',img_detection)
    if cv2.waitKey(1) == ord('a'):
        break
        
cap.release()
cv2.destroyAllWindows()
