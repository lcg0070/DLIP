import numpy as np
import cv2 as cv
import timeit as time
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

cap = cv.VideoCapture('DLIP_LAB_PARKING_VIDEO_21900575_LeeChanKeun.avi')

park = [80, 180, 280, 390, 480, 580, 690, 780, 880, 970, 1060, 1170, 1230]
PARK_LEN = len(park)

def find_indexes(lst, value):
    indexes = []
    for i in range(len(lst)):
        if lst[i] == value:
            indexes.append(i+1)
    return indexes

def save_to_txt(data):
    with open('counting_result.txt', 'a') as file:
        file.write(data + '\n')

frame_count = 0
if cap.isOpened():
    while True:
        k = cv.waitKey(1)
        # esc to quit
        ret, frame = cap.read()
        if ret and frame_count<=2500 and k != 27 :
        #frame_count<=2500
            height, width = frame.shape[:2]
            
            car_count = 0
            # Detect objects using YOLOv5
            results = model(frame)
            # Count number of cars
            exist =[]
            state_list = [0 for _ in range(PARK_LEN)]
            for result in results.pandas().xyxy[0].iterrows():
                if (result[1]['name'] in ('car')) and result[1]['confidence'] > 0.17:
                    #0.35, # 0.3, #0.25
                    x1, y1, x2, y2 = int(result[1]['xmin']), int(result[1]['ymin']), int(result[1]['xmax']), int(result[1]['ymax'])
                    if((y1+y2)/2 < height/2+50):
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                        for i in range(PARK_LEN):
                            if x1<park[i]<x2:
                                state_list[i] = 1
            
            cv.putText(frame,"Frame : "+ f"{frame_count}",(30,50),cv.FONT_HERSHEY_SIMPLEX,1.2,(255,0,255),2)
            cv.putText(frame,"Number of cars : "+ f"{sum(state_list)}",(30,80),cv.FONT_HERSHEY_SIMPLEX,1.2,(255,0,255),2)
            cv.putText(frame,"Blank space : " + f"{find_indexes(state_list,0)}",(30,110),cv.FONT_HERSHEY_SIMPLEX,1.2,(255,0,255),2)
            frame_with_number = f"{frame_count} {sum(state_list)}"
            save_to_txt(frame_with_number)
            frame_count += 1
            cv.imshow('video',frame)
        else:
            print("end of file")
            break
else:
    print('cannot open the file')

cap.release()
cv.destroyAllWindows()