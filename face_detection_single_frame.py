import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp


mp_FD = mp.solutions.face_detection
FD = mp_FD.FaceDetection(min_detection_confidence = 0.5)


frame = cv2.imread('input.jpg')
if frame is None:
    print("empty frame")
    
else:
    rgb_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output_result = FD.process(rgb_frame)
    if output_result.detections:
        for i in output_result.detections:
            bboxc = i.location_data.relative_bounding_box
            h,w,_ = frame.shape
            bbox = int(bboxc.xmin *w),int(bboxc.ymin * h) ,int(bboxc.width*w),int(bboxc.height*h)
            cv2.rectangle(frame , bbox ,(0,0,255),2)

    cv2.imshow('Face Detection using mediapipe' ,frame)
    cv2.imwrite('output.jpg',frame)


cv2.destroyAllWindows()
print("Finisted Processing Frames !")