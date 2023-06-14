import cv2 as cv
import timeit as time
import torch
import datetime
import os
import socket
import pickle
import serial
import time

# class_name = ['Lee_ck', 'Lee_ky', 'Oscilliscope','Red','Green','Blue','Purple','Gray','Yellow']

name = ['21900575 Lee_ck', '21800501 Lee_ky']
#450 550
area = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
AREA_NUMBER = len(area)
HISTORY_MAX = 3
state_history = [[] for _ in range(AREA_NUMBER)]
state_history_oscili = [[[1,1]] for _ in range(5)]
def add_history():
    for i, state in enumerate(state_list):
        state_history[i].append(state)
        if len(state_history[i]) > HISTORY_MAX: del state_history[i][0]

def add_history_oscili():
    for i, state in enumerate(state_list_oscili):
        state_history_oscili[i].append(state)
        if len(state_history_oscili[i]) > 2: del state_history_oscili[i][0]

def find_difference(arr1, arr2):
    different = []
    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            if arr1[i][j] != arr2[i][j]:
                different.append(i*2+j+1)
    return different

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

#attendance
def save_to_txt1(data):
    with open('attendance.txt', 'a') as file:
        file.write(data + '\n')
#watchman
def save_to_txt2(data):
    with open('watchman.txt', 'a') as file:
        file.write(data + '\n')

def array_to_string(array):
    result = ''
    for sublist in array:
        for element in sublist:
            if element == 1:
                result += 'O | '
            else:
                result += 'X | '
    result = result[:-3]
    return result

#arduino motor
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

#arduino
def send_array_to_arduino(array):
    array_string = ','.join(map(str, array))
    try:
        arduino.write((array_string + '\n').encode())
    except Exception as e:
        print(f"Error: {e}")

#data saving folder
output_folder1 = "captured_faces/attandance"
output_folder2 = "captured_faces/watchman"

#create folder
if not os.path.exists(output_folder1):
    os.makedirs(output_folder1)
if not os.path.exists(output_folder2):
    os.makedirs(output_folder2)

#load pretrained model
cap = cv.VideoCapture(0)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#arduino
port = '/dev/tty.usbmodem11202'  
baudrate = 9600
arduino = serial.Serial(port, baudrate) 
time.sleep(3) #wait for arduino

#server
host = 'localhost'
port = 12345
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    s.listen()
    conn, addr = s.accept()

#mode chage
mode = 1
flag3=0
#is program started?
start_flag=0
# start_flag=1

output=[]
# save img
face_img=[]
if cap.isOpened():
    array_received=[]
    namevalue = []
    face_img = []
    #time
    conn.settimeout(0.5)
    while True:
        #server array
        try:
            data = conn.recv(1024)
            array_received = pickle.loads(data)
        except:
            array_received=[]
        if len(array_received):
            array_received = remove_duplicates(array_received)
        if(array_received ==[]):
            continue
        sorted_arr = sorted(array_received, key=lambda x: x[5])
        # print(array_received)

        state_list = [0 for _ in range(AREA_NUMBER)]
        state_list_oscili = [[0,0] for _ in range(5)]

        #about time
        now = datetime.datetime.now()
        date, time1 = str(now.date()), str(now.strftime("%H:%M:%S"))

        #exit
        if cv.waitKey(1) & 0xFF == 27:
            break
        #attendance
        elif (cv.waitKey(1) & 0xFF == ord('a')) and mode == 1:
            flag, flag1, flag2 = 0,0,0
            start_flag=1
            mode = 0
            namevalue=[]
            face_img=[]
            #reset couning number
            count=0
            head = "========================================================================\nCheck People Attendance\t\t\t\t\t\t\t\t\tDate "+date+"\n========================================================================\nNo\t\t\t\ttime\t\t\t\t\t\tName\t\t\t\tAttendee"
            save_to_txt1(head)
        #watchman
        elif (cv.waitKey(1) & 0xFF == ord('w')) and mode == 0:
            flag = 0
            start_flag = 1
            mode = 1
            namevalue = []
            face_img = []
            #reset couning number
            count=0
            head = "======================================================================================================\nAfter Work Time															               Date "+date+"\n======================================================================================================\nNo\t\t\ttime\t\t\t\t1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10\t\t\t\tObject Change"
            save_to_txt2(head)

        ret, frame = cap.read()
        height, width = frame.shape[:2]
        #about model
        results= model(frame)
        person=[]
        #Attendance mode
        if mode == 0 and start_flag:
            for result in results.pandas().xyxy[0].iterrows():
                if (result[1]['name'] in ('person')):
                    x1, y1, x2, y2 = int(result[1]['xmin']), int(result[1]['ymin']), int(result[1]['xmax']), int(result[1]['ymax'])
                    person.append([x1,y1,x2,y2])
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for plot in person:
                state_list[find_closest_index((plot[0]+plot[2])/2)]+=1
            
            add_history()

            if(len(state_history[0])>2):
                #counting people and printing log
                people1 = sum(row[-2] for row in state_history)
                people2 = sum(state_list)
                #if people increase in area[0]
                if(people1<people2):
                    flag+=1
                if(flag>0):
                    #rightward moving
                    if (state_history[1][-2]>state_history[1][-1]) and (state_history[2][-2]<state_history[2][-1]):
                        #new people
                        flag-=1
                        flag1+=1
                        print_name='????'
                        for k in array_received:
                            if k[5] < 2 and k[4]>0.4:
                                index = find_closest_index((k[0]+k[2])/2)
                                if index < 3 :
                                    print_name=name[k[5]]
                        namevalue.append(print_name)
                        #moving point
                        face_img.append(cv.resize(frame[0:height, 0:550], (550, height)))
                        if(len(namevalue)>flag1):
                            while len(namevalue)==flag1:
                                namevalue = namevalue[:-1]
                                face_img = face_img[:-1]
                    #leftward moving
                    elif (state_history[-1][-2]>state_history[-1][-1]) and (state_history[-2][-2]<state_history[-2][-1]):
                        #existed people
                        flag-=1
                        flag2+=1
                #if rightward people exist
                if(flag1>0):
                    if (state_history[-1][-2]<state_history[-1][-1]) and (state_history[-2][-2]>state_history[-2][-1]):
                        count+=1
                        flag1-=1
                        if namevalue[0] == '????':
                            attendee ="False"
                            log = str(count) +'\t\t\t\t'+ time1 + '				    '+ namevalue[0] +'				   '+ attendee
                        else:
                            attendee ="True"
                            log = str(count) +'\t\t\t\t'+ time1 + '				'+ namevalue[0] +' 			'+ attendee
                        filename = os.path.join(output_folder1, f"{time1}_{namevalue[0]}.jpg")
                        cv.imwrite(filename, face_img[0])
                        namevalue = namevalue[1:]
                        face_img = face_img[1:]
                        save_to_txt1(log)
                elif (flag2>0):
                    #if people decrease
                    if ((state_history[1][-2]<state_history[1][-1]) and (state_history[2][-2]>state_history[2][-1])) or ((state_history[-1][-2]<state_history[-1][-1]) and (state_history[-2][-2]>state_history[-2][-1])):
                        flag2-=1
                if people2 == 0:
                    flag1 = 0
                    flag2 = 0
                    namevalue=[]
                    face_img=[]
                
        #watchman mode
        if mode == 1 and start_flag:
            ind = -1
            x1,y1,x2,y2 = 0,0,0,0
            
            for result in results.pandas().xyxy[0].iterrows():
                if (result[1]['name'] in ('person')):
                    x1, y1, x2, y2 = int(result[1]['xmin']), int(result[1]['ymin']), int(result[1]['xmax']), int(result[1]['ymax'])
                    person.append([x1,y1,x2,y2])
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for plot in person:
                ind = find_closest_index((plot[0]+plot[2])/2)
                if(ind == -1):
                    continue
                state_list[ind]+=1

            add_history()

            if(len(state_history[0])>2):
                #counting people and printing log
                people1 = sum(row[-2] for row in state_history)
                people2 = sum(state_list)
            #Red 3, Green 4, Blue 5, Purple 6, Gray 7, Yellow 8
            # locker plot number
            locate_locker = [-1,-1,-1,-1,-1,-1]
            # locker class number
            target_class_locker = [3,4,5,6,7,8]
            locate_oscili = []
            target_class_oscili = [2]
            
            for i in array_received:
                if i[5] in target_class_locker:
                    locate_locker[target_class_locker.index(i[5])] = (i[0] + i[2]) / 2
            
            for i in array_received:
                if i[5] in target_class_oscili:
                    locate_oscili.append([(i[0] + i[2]) / 2 , (i[1]+i[3]) / 2])
            
            #moving arduino
            array = [0,-1]
            #limiting phase
            red = find_closest_index(locate_locker[0])
            yellow = find_closest_index(locate_locker[5])
            
            #follow people
            if(flag == 1):
                if state_list[2] > 0:
                    face_img.append(cv.resize(frame[0:height, 0:550], (550, height)))
                elif state_list[-2] > 0:
                    face_img.append(cv.resize(frame[0:height, width-550:width], (550, height)))
                array[0], array[1] = calculate_weighted_index(state_list)

            if (1 < red < 4 ):
                if (array[0] == -1):
                    array[0] = 0
            elif (5<yellow <8):
                if (array[0] == 1):
                    array[0] = 0
            #people in
            if people1<people2:
                flag1 = 0
                flag2 = 0
                flag = 1
            #people out
            elif people1>people2:
                flag = -1
                count+=1
            if flag == -1:
                #if flag1 is 1 is left side, if 0 it has to go rightside
                if(5<yellow <8):
                    flag1=1
                    # check oscili
                    for ind,i in enumerate(locate_locker[1:]):
                        for j in locate_oscili:
                            if abs(i - j[0])<70:
                                # upper side
                                if abs(j[1]-620)<100:
                                    state_list_oscili[ind][0] = 1
                                # low side
                                elif abs(j[1]-810)<100:
                                    state_list_oscili[ind][1] = 1
                    
                    if [row[-1] for row in state_history_oscili] != state_list_oscili:
                        add_history_oscili()
                        flag3=1
                        oscili1 = [row[0] for row in state_history_oscili]
                        oscili2 = [row[1] for row in state_history_oscili]
                        output = find_difference(oscili1, oscili2)
                elif(flag1 == 1 and 1<red<4):
                    flag2 = 1
                if flag1 == 0:
                    array[0] = 1
                elif flag1 and flag2 == 0:
                    array[0] = -1
                #ready state
                elif flag1 and flag2:
                    flag1 = 0
                    flag2 = 0
                    flag = 1
                    state = array_to_string([row[-1] for row in state_history_oscili])
                    if flag3 == 0:
                        output ='none'
                    else:
                        flag3 = 0
                        output = ', '.join(map(str, output))
                    if(len(face_img) != 0):
                        log = str(count) +'\t\t\t'+ time1 + '\t\t	'+state+'\t\t\t\t'+output 
                        filename = os.path.join(output_folder2, f"{time1}_{output}.jpg")
                        cv.imwrite(filename, face_img[0])
                        save_to_txt2(log)
                    face_img=[]
                        
            
            send_array_to_arduino(array)

        cv.imshow("Webcam", frame)
        # frame_with_number = f"{frame_count} {sum(state_list)}"
        # save_to_txt(frame_with_number)
else:
    print('cannot open the camera')
cap.release()
cv.destroyAllWindows()