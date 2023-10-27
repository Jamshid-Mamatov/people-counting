import cv2
from ultralytics import YOLO
import numpy as np

model=YOLO('yolov8s.pt')

def points(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('points')
cv2.setMouseCallback('points', points)


cap=cv2.VideoCapture("video_escalator.mp4")
area1=np.array([(82,602),(230,602),(230,650),(82,650)],np.int32)
area2=np.array([(370,601),(479,601),(479,657),(376,655)],np.int32)
p_up=0
p_down=0
track_history_up=[]
track_history_down=[]
while cap.isOpened():
    _,frame=cap.read()
    
    results=model.track(frame, persist=True, imgsz=320,classes=0)
    
    
        

    cv2.polylines(frame,[area1],True,(255,255,0),3)
    cv2.polylines(frame,[area2],True,(255,255,0),3)
    boxes = results[0].boxes.xyxy.cpu()
    print(results[0].boxes.id)
    track_ids = results[0].boxes.id.int().cpu().tolist()
    for box,track_id in zip(boxes, track_ids):
        # print(box,track_id)
        x,y,x1,y1=box
        x=int(x)
        y=int(y)
        x1=int(x1)
        y1=int(y1)
        # pts = area1.reshape((-1,1,2))
        # print(x1,y1)
        up=cv2.pointPolygonTest(area1, (x1, y1), True)
        down=cv2.pointPolygonTest(area2, (x1,y1), False)
        
        # print(up,down)
        if up>=1 and track_id not in track_history_up :
            cv2.rectangle(frame,(x,y),(x1,y1),(255,0,0),3)
            cv2.putText(frame,f"{track_id}",(220,380),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.circle(frame,(x1,y1),5,(0,0,255),-1)
            p_up+=1
            track_history_up.append(track_id)
        if down >=1 and track_id not in track_history_down:
            cv2.rectangle(frame,(x,y),(x1,y1),(255,0,0),3)
            cv2.putText(frame,f"{track_id}",(220,380),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.circle(frame,(x1,y1),5,(0,0,255),-1)
            p_down+=1
            track_history_down.append(track_id)
        cv2.putText(frame,f"Up : {p_up} | Down : {p_down}",(220,380),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    cv2.imshow("points",frame)

    

  
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

