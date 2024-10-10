import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp


mp_FD = mp.solutions.face_detection
FD = mp_FD.FaceDetection(min_detection_confidence = 0.5)

cap = cv2.VideoCapture('face_detection.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_filename = "output_FD.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mprv')
out = cv2.VideoWriter(output_filename , fourcc ,fps ,(frame_width,frame_height))

while(True):
    ret,frame = cap.read()
    if not ret:
        break

    rgb_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    output_result = FD.process(rgb_frame)
    if output_result.detections:
        for i in output_result.detections:
            bboxc = i.location_data.relative_bounding_box
            h,w,_ = frame.shape
            bbox = int(bboxc.xmin *w),int(bboxc.ymin * h) ,int(bboxc.width*w),int(bboxc.height*h)
            cv2.rectangle(frame , bbox ,(0,0,255),2)

    cv2.imshow('Face Detection using mediapipe' ,frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    # cv2.imwrite('output_FD.jpg',frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("Finisted Processing Frames !")