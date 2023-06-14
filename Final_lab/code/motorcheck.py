import cv2 as cv
import timeit as time
import torch
import datetime
import os
import socket
import pickle
from queue import Queue
import serial
import time

area = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
AREA_NUMBER = len(area)
HISTORY_MAX = 3

def find_closest_index(target, arr = area):
    closest_diff = float('inf')
    closest_index = None

    for i, num in enumerate(arr):
        diff = abs(target - num)
        if diff < closest_diff:
            closest_diff = diff
            closest_index = i

    return closest_index

#if multiple items are detected on the same object
def remove_duplicates(data):
    indices_to_remove = set()
    for i in range(len(data)):
        x1, y1, x2, y2, prob, _ = data[i]
        avg_x = (x1 + x2) / 2
        avg_y = (y1 + y2) / 2
        for j in range(i + 1, len(data)):
            x1_j, y1_j, x2_j, y2_j, prob_j, _ = data[j]
            avg_x_j = (x1_j + x2_j) / 2
            avg_y_j = (y1_j + y2_j) / 2
            if abs(avg_x - avg_x_j) < 30 and abs(avg_y - avg_y_j) < 30:
                if prob < prob_j:
                    indices_to_remove.add(i)
                else:
                    indices_to_remove.add(j)
    result = [data[i] for i in range(len(data)) if i not in indices_to_remove]
    return result

def calculate_weighted_index(arr):
    # all_sum = sum(arr)
    left_sum = sum(arr[:4])
    right_sum = sum(arr[6:])
    direction=0
    speed=-1
    if (left_sum>right_sum):
        direction=-1
        # speed = abs(4-index)
    elif(right_sum>left_sum):
        direction=1
        # speed = abs(5-index)
    return direction, speed

def send_array_to_arduino(array):
    array_string = ','.join(map(str, array))
    try:
        arduino.write((array_string + '\n').encode())
    except Exception as e:
        print(f"Error: {e}")

#load pretrained model
cap = cv.VideoCapture(0)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#arduino
port = '/dev/tty.usbmodem11202'  
baudrate = 9600
arduino = serial.Serial(port, baudrate) 
time.sleep(3) #wait for arduino

#mode chage
mode = 1

#is program started?
start_flag=1


# save img
face_img=[]
if cap.isOpened():
    array_received=[]
    namevalue = []
    
    while True:
        
        if len(array_received):
            array_received = remove_duplicates(array_received)
        
        sorted_arr = sorted(array_received, key=lambda x: x[5])
        state_list = [0 for _ in range(AREA_NUMBER)]
        #exit
        if cv.waitKey(1) & 0xFF == 27:
            break

        ret, frame = cap.read()
        height, width = frame.shape[:2]
        #about model
        results= model(frame)
        person=[]
        
        
        for result in results.pandas().xyxy[0].iterrows():
            if (result[1]['name'] in ('person')):
                x1, y1, x2, y2 = int(result[1]['xmin']), int(result[1]['ymin']), int(result[1]['xmax']), int(result[1]['ymax'])
                person.append([x1,y1,x2,y2])
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for plot in person:
            state_list[find_closest_index((plot[0]+plot[2])/2)]+=1
        array=[0,1]
        array[0], array[1] = calculate_weighted_index(state_list)
        print(array)
        
        send_array_to_arduino(array)

        cv.imshow("Webcam", frame)
else:
    print('cannot open the camera')
cap.release()
cv.destroyAllWindows()

