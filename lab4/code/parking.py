import numpy as np
import cv2 as cv
import timeit as time
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv.VideoCapture('DLIP_parking_test_video.avi')

park = [80, 180, 280, 390, 480, 580, 690, 780, 880, 970, 1060, 1170, 1230]
PARK_LEN = len(park)

color=(0,255,0)
state=[]

HISTORY_MAX = 10
state_list = [0 for _ in range(PARK_LEN)]
state_history = [[] for _ in range(PARK_LEN)]

def add_history():
    for i, state in enumerate(state_list):
        state_history[i].append(state)
        if len(state_history[i]) > HISTORY_MAX: del state_history[i][0]

def set_state():
    for i in range(len(state_list)):
        state_list[i] = int(sum(state_history[i]) > (len(state_history[i]) / 2-1))

def find_indexes(lst, value):
    indexes = []
    for i in range(len(lst)):
        if lst[i] == value:
            indexes.append(i)
    return indexes

if cap.isOpened():
    while True:
        k = cv.waitKey(1)
        # esc to quit
        ret, frame = cap.read()
        if ret and k != 27:
            frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            print(frame_count)

            height, width = frame.shape[:2]
            roi = np.zeros((height, width, 3), dtype=np.uint8)
            roi[int(height/2)-70: int(height/2) + 50, 0:width] = frame[int(height/2)-70:int(height/2) + 50, 0:width]
            
            car_count = 0
            # Detect objects using YOLOv5
            results = model(roi)
            # Count number of cars
            exist =[]
            locate=[]
            for result in results.pandas().xyxy[0].iterrows():
                flag=0
                if (result[1]['name'] in ('car', 'bus', 'truck')) and result[1]['confidence'] > 0.5:
                    x1, y1, x2, y2 = int(result[1]['xmin']), int(result[1]['ymin']), int(result[1]['xmax']), int(result[1]['ymax'])
                    for i in locate:
                        if abs((x1+x2)/2 - (i[0]+i[2])/2)<25:
                            flag=1
                    if flag==1:
                        continue
                    locate.append((x1,y1,x2,y2))
            locate.sort()
            j=0
            for i in range(len(locate)):
                while j<PARK_LEN:
                    x = park[j]
                    if(locate[i][0]<x<locate[i][2]):
                        state_list[j]=1
                        j+=1
                        break
                    else:
                        state_list[j]=0
                        j+=1
            add_history()
            set_state()
            # print(state_list)
            print("Number of cars : ",sum(state_list))
            print("Blank space : ", find_indexes(state_list,0))
            j=0
            for ind,i in enumerate(state_list):
                while j <len(locate):
                    if(i != 1):
                        color = (0,255,0)
                        break
                    x1 = locate[j][0]
                    y1 = locate[j][1]
                    x2 = locate[j][2]
                    y2 = locate[j][3]
                    color = (0,0,255)
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                    j+=1
                    cv.line(frame,(park[ind], 350), (park[ind], 200),color,5)

            # print("Number of cars:", car_count)
            cv.imshow('video',frame)
            
            print(state_history[-2], ": ", state_list[-2])
        else:
            print("end of file")
            break
else:
    print('cannot open the file')

cap.release()
cv.destroyAllWindows()