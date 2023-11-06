# [FINAL PROJECT] Automatic Class management System

**Date:** 2023-06-19

**Name/Student ID:** Lee-Kiyoon/ (21800501), Lee_chanKeun(21900575)

**Result Video:** 

1.[DLIP_FINAL_DEMO_for_SSL_youtube](https://www.youtube.com/watch?v=acfIznVMDbg)

2.[DLIP_FINAL_AttendanceMODE_FULL version](https://www.youtube.com/watch?v=h3dYtjUDPns)

3.[DLIP_FINAL_CCTVMODE_FULL version](https://www.youtube.com/watch?v=VDQzuvHT-hk)

4.[DLIP_FINAL_30S_DEMO](https://www.youtube.com/watch?v=6CEisOBbEq8)

**DataSet Link:** [DLIP_FINAL_datasets](http://naver.me/G5Q9u72g)





## 1. Introduction

![image](https://github.com/leekyyoon99/EC/assets/121138800/b4d8c1c9-40eb-403a-ac9c-2c139e2d16e0)

First, Existing Face detection attendance system has a weak point  that the camera only detects the front of the person's face. Second, Existing equipment management system has a similar drawbacks that the administrator has to  input using hands or keyboard to check the equipment existence in the class or lab. Finally, We want to improve these systems by making this process automatic using only one camera. That's the key idea.

![image](https://github.com/leekyyoon99/EC/assets/121138800/1862d39a-7c0c-4e06-a6ef-0a3143661df4)



## 2. Requirement 



### 2.1 required hardware and software



#### 2.1.1. Software

* YOLO v5
* Pytorch 1.6.0
* CUDA 11.6
* Python 3.9.12 (py39)
* Arduino IDE
* Fusion 360 (for gears design)



#### 2.1.2. Hardware 

* Adafruit Motor Shield v2.3

* 12V DC Motor 

* Arduino Uno WiFi

  

### 2.2 Setup



#### 2.2.1. Software Setup

**Anaconda settings**

check your cuda version and donwload nvidia driver [click here](https://developer.nvidia.com/cuda-toolkit-archive). Refer to [installation_guide](https://ykkim.gitbook.io/dlip/installation-guide/installation-guide-for-deep-learning).

```py
# create a conda env name=py39
conda create -n py39 python=3.9.12
conda activate py39
conda install -c anaconda seaborn jupyter
pip install opencv-python

# pytorch with GPU
conda install -c anaconda cudatoolkit==11.3.1 cudnn seaborn jupyter
conda install pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install opencv-python torchsummary

# Check GPU in Pytorch
conda activate py39
python
import torch
print("cuda" if torch.cuda.is_available() else "cpu"
```

 **YOLOv5 Installation**

Go to YOLOv5 github (https://github.com/ultralytics/yolov5) and download Repository as below. After entering the /`yolov5-master` folder, copy the path address. Then executing Anaconda prompt in administrator mode, execute the code below sequentially.

```
conda activate py39
cd $YOLOv5PATH$ // [ctrl+V] paste the copied yolov5 path
pip install -r requirements.txt
```

**Labeling**

The DarkLabel 2.4 program was used to generate custom data. Using this program, bounding box labeling was performed directly for each frame of the video to create an image and label dataset. Compared to other labeling programs, labeling work through images is possible, so it is possible to generate a lot of training data quickly. Go to [DarkLabel 2.4](https://github.com/darkpgmr/DarkLabel) and download the DarkLabel 2.4 program below. if it is not available, please download [here](https://darkpgmr.tistory.com/16).

After executing DarkLabel.exe, labeling is performed using the desired image or image.

1. Image file path for using image files
2. Using Darknet yolo labeling method
3. To using your customized labels for model training, the number of data and class name of coco dataset must be changed.

4. Adjust for yml file and add the custom class.![image](https://github.com/leekyyoon99/EC/assets/121138800/8825c46e-95f7-446e-8652-f64394c60828)and then when you train, you have to match yolov5 yaml to the yaml you used in darklabel 2.4.



![image](https://github.com/leekyyoon99/EC/assets/121138800/d8a012e2-a39b-4fe4-948c-dd7901accbfe)

Finally, We used the GPU server to train the datasets. If you input this in cmd, You can train

```
python -m torch.distributed.run —nproc_per_node 1 train.py —batch 8 —epochs 100 —data data.yaml —weights yolov5s.pt —device 0,1
```



#### 2.2.2. Hardware setup

First, we used **Adafruit motorshield v2.3** and **arduino uno wifi**.  you can combine motor driver and arduino by just lapping. 

![image](https://github.com/leekyyoon99/EC/assets/121138800/0ae409d7-b5dc-4415-8340-2fc366f54df3)



We used the 12V DC motor, therefore you need another power source for using this motor. It cannot be used only from arduino power source. Therefore, You need to connect 12V source from power supplier. and then we select the M4 port for motor driving.

![image](https://github.com/leekyyoon99/EC/assets/121138800/48d9b976-be9d-49e4-b19e-677552382f67)

Finally, we designed two gears. You can design two gears by using 'Fusion 360'. Below video is the motor and gear test video. 

![기어스 (2)](https://github.com/leekyyoon99/EC/assets/121138800/9c7753a9-ed8f-4762-8a2e-8fb33740a466)

Finally, Everything is ready for starting this project.



## 3. Dataset

![image](https://github.com/leekyyoon99/EC/assets/121138800/3f05534d-e657-4c85-968f-9892b6d84317)

In this project, we need to distinguish ['Lee_ck','Lee_ky'] from other people. Therefore, We trained ['Lee_ky', 'Lee_ck'] class for custom dataset and used ['person'] class from Yolov5 pretrained model. Also, We need to recognize colors to infer the oscilloscope's position and state. Therefore, We trained  [Red','Green','Blue','Purple','Gray','Yellow'] class for custom dataset. Finally we trained ['Oscilloscope'] class for custom dataset to detect the osciloscope.

We used about 2000 images for training ['Oscilloscope'] class. ['Lee_ck'] class about 2000 images and ['Lee_ck'] class about 2000 images. All images include [Red','Green','Blue','Purple','Gray','Yellow'] class. We used 729 images for Validation dataset. We explained the method how to train custom datasets in Setup part. If you want our datasets, visit this link. [DLIP_FINAL_datasets](http://naver.me/G5Q9u72g)



## 4. Flow Chart

Before explaining the flow chart, Let we explain the concept out projects.

![image](https://github.com/leekyyoon99/EC/assets/121138800/971cddc5-569f-409a-bc4b-a27b61dd0561)First, We design the attendance mode for recognizing person's face and check the time even if a number of people come in to the class. We assume that the person comes and moves from left to right and moves in a row. 

![](https://github.com/leekyyoon99/EC/assets/121138800/c6435ab8-8729-4000-b7c4-a75f02e4bfe1)Second, We design CCTV mode that checks and manages 10 oscilloscopes in the closet. In a similar to attendance mode, We recognize the person. But this mode, the camera tracks the person by using motor to effectively determine the person is in the class or not. When the person picks up some oscilloscopes and goes out the class, the camera moves back to the appointed position to check the oscilloscopes after that goes back to the original position to prepare to detect another person comings. 

Finally, We realized these concepts in code. The structure of software and  Flow chart are as follows.

![image](https://github.com/leekyyoon99/EC/assets/121138800/b6ba1bf0-658a-4903-aa8b-f145b455252f)

![image](https://github.com/leekyyoon99/EC/assets/121138800/986b5f3b-35ff-4e32-aa09-5444ffc6c9fe)





## 5. Tutorial Procedure

According to Flow Chart, We will explain First, how to activate 'Attendance mode' or 'CCTV mode'. Second, how to communicate between 'server.py' and 'client.py' and how to communicate between 'server.py' and 'motor.arduino'. Third, The core algorithm of 'Attendance mode'. and Fourth, the core algorithm of 'CCTV mode'.

### 5.1 Select Mode 

Depending on the key being pressed, it can exit the program or set the mode. By pressing key 'A'  for 'attendance mode' or 'W' for CCTV mode. 

```py
if cv.waitKey(1) & 0xFF == 27:
	break
#attendance
elif (cv.waitKey(1) & 0xFF == ord('a')) and mode == 1:
    .....
elif (cv.waitKey(1) & 0xFF == ord('w')) and mode == 0:
    .....
```



### 5.2 Communication

#### server.py (Python to Python)

**remove_duplicates** is the function that removes duplicated same labels

```python
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
```

Communicate with the server and simplify the array using the **remove_duplicates** function

```py
#server
host = 'localhost'
port = 12345
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    s.listen()
    conn, addr = s.accept()
    
    #this part goes inside of the loop
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
```

**example** 

```py
# array_received example
[[100, 200, 150, 250, 0.9776, 1]]
#x1, y1, x2, y2, prob, class
```



#### client.py (Python to Python)

The **client.py** was created by modifying the **detect.py** file of yolov5. Since the custom model was not optimized in the way of loading the existing model, the following code was added to the repetition sentence to hand over the tensor-type array to the server by adding a code communicating with the server.

```py
# added code in detect.py

data_pickled = []
def start_client(opt):
    host = 'localhost'
    port = 12345
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        run(**vars(opt),socketed=s)
        
#this code belongs to the loop
out = reversed(det).numpy()
int_data = [[int(x) for x in row[:4]] + [(row[4]), int(row[5])] for row in out]
data_pickled = pickle.dumps(int_data)
socketed.sendall(data_pickled)
```



#### Python to Arduino 

This python code is communication code that sends array from python to arduino.

```py
#(python->arduino send code)
port = '/dev/tty.usbmodem11202'  
baudrate = 9600
arduino = serial.Serial(port, baudrate) 
time.sleep(3) #wait for arduino

#(python->arduino send code)
def send_array_to_arduino(array):
    array_string = ','.join(map(str, array))
    try:
        arduino.write((array_string + '\n').encode())
    except Exception as e:
        print(f"Error: {e}")
```



This part of arduino code is communication code that receives array from python.

```cpp
#include <Wire.h>
#include <Adafruit_MotorShield.h>

// Adafruit_MotorShield object generation
Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 

// DC motor is connected to M4 Port
Adafruit_DCMotor *myMotor = AFMS.getMotor(4);
int speed = 0;

void setup() {
  Serial.begin(9600);  // Begin Serial communication
  ....
}

void loop() {
  if (Serial.available()) {  //Check if data exists in serial data
    String command = Serial.readStringUntil('\n');  //Read Serial date
    //Convert strings to array
    int array[2];
    int index = 0;
      
    //save array data from python
    for (int i = 0; i < command.length(); i++) {
        int commaIndex = command.indexOf(',', i);
        if (commaIndex == -1) {
            array[index] = command.substring(i).toInt();
            break;
        } else {
            array[index] = command.substring(i, commaIndex).toInt();
            index++;
            i = commaIndex;
        }
    }
}
```



### 5.3 Attendance mode

#### Person movement detection algorithm

![세그먼트](https://github.com/leekyyoon99/EC/assets/121138800/2be71b1c-996a-4712-ae71-f28dca7fe54f)

![image](https://github.com/leekyyoon99/EC/assets/121138800/1cdcc991-6902-4cb5-95a3-530d1cb32a87)

First, We divide the width of the frame by 10 segments and then define the area between segment and segment. Therefore, 10 areas are defined. See the **Person x-position State array**. and then we indicate when the person is in the area, the state of that area is 1. For example, the person is in the **100~300** x-pixel corrdinates then **[0]** of the **Person x-position State array** becomes 1. In this way, We can determine where the person exists in the frame. But, It is not enought to detect using only this array. Because It only captures 'current' x-position of the person. Therefore, We defined another array called **Person x-position State History array**. It stacks the states about 3 states. Watching the change of the states in this array, we can determine the direction of the person's movement more accurately. 

```python
# list of whether a person exists for a particular area
state_list = [0 for _ in range(AREA_NUMBER)]

#this is frame's area divided by x axis
area = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]

# store before frame's state_list to check differences
state_history = [[] for _ in range(AREA_NUMBER)]
def add_history():
    for i, state in enumerate(state_list):
        state_history[i].append(state)
        if len(state_history[i]) > HISTORY_MAX: del state_history[i][0]
```

Above code is the code that stacks the state history. and the person doesn't always fit the area. We have to find the closest area to determine the person's position in the frame. the function is as follows. 

```py
# find closest area of the object
def find_closest_index(target, arr = area):
    closest_diff = float('inf')
    closest_index = None

    for i, num in enumerate(arr):
        diff = abs(target - num)
        if diff < closest_diff:
            closest_diff = diff
            closest_index = i

    return closest_index
```

and then the function that saves detected Person's (x1, y1, x2, y2) in **person** array. Afterwards, the x1x2 coordinates are averaged from the stored array, the x-means coordinates for each person are calculated, and updated to the state_list by +1 at the closest position in the area. So if there are five people in the zero area and one in the third area, then the **state_list** is going to be [5,0,0,1,0,0...]. in this way, We can determine not only one person but also a number of perople's movement. This code is as follows.

```py
HISTORY_MAX = 3
AREA_NUMBER = len(area)
state_history = [[] for _ in range(AREA_NUMBER)]

def add_history():
    for i, state in enumerate(state_list):
        state_history[i].append(state)
        if len(state_history[i]) > HISTORY_MAX: del state_history[i][0]

# Save detected Person's xyxy in array and if the person is in the 
for result in results.pandas().xyxy[0].iterrows():
	if (result[1]['name'] in ('person')):
	    x1, y1, x2, y2 = int(result[1]['xmin']), int(result[1]['ymin']), int(result[1]['xmax']), int(result[1]['ymax'])
	    person.append([x1,y1,x2,y2])
	    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
for plot in person:
	state_list[find_closest_index((plot[0]+plot[2])/2)]+=1
            
add_history()
```



#### Face capture and attendance determination algorithm

![image](https://github.com/leekyyoon99/EC/assets/121138800/1c96ad7c-d122-4ba5-a80f-cb4e97a98da8)

When the person comes in [0] and [1] area, then we capture person image and then when the person goes right to [8] and [9] areas finally the out of the frame, we determine that the person is in the class and write the time and attendance or late. Finally, the code exports text log file automatically.               

```py
if(len(state_history[0])>2):
#counting people and printing log
#compare people's number with before frame and present frame
    people1 = sum(row[-2] for row in state_history)
    people2 = sum(state_list)
    #if people increase
    if(people1<people2):
        flag+=1
    if(flag>0):
        # people increased with rightward moving
        if (state_history[1][-2]>state_history[1][-1]) and (state_history[2][-2]<state_history[2][-1]):
            #new people
            flag-=1
            flag1+=1
            
             #store image and determine name
```
    #people increased with leftward moving
    elif (state_history[-1][-2]>state_history[-1][-1]) and (state_history[-2][-2]<state_history[-2][-1]):
        #existed people
        flag-=1
        flag2+=1
#if rightward moving people exist
if(flag1>0):
    # if the person fades away in the right side of frame
    if (state_history[-1][-2]<state_history[-1][-1]) and (state_history[-2][-2]>state_history[-2][-1]):
        count+=1
        flag1-=1
					```
        #determine name and attendee by using saved namevalue
					```
# Exceptional Case
elif (flag2>0):
    #if people decrease
    if ((state_history[1][-2]<state_history[1][-1]) and (state_history[2][-2]>state_history[2][-1])) or ((state_history[-1][-2]<state_history[-1][-1]) and (state_history[-2][-2]>state_history[-2][-1])):
        flag2-=1
               
# Initialization of  vartiables and array when the people don't exist
if people2 == 0:
    flag1 = 0
    flag2 = 0
    namevalue=[]
    face_img=[]
```
#### Export Result

If a person appears through the updated history state, if added, determine whether a person is new (from the left) or someone who is already in (from the right), etc. and save the image if it is a new person. It also adjusts the output name depending on whether there is a specific class (Chan-keun, Ki-yoon) that comes over from the custom data set at that time. In conclusion, when a new person enters, the time and name, attendance, and the number of times he/she enters are exported to txt, and the image is stored as the name and time in the attachment folder. This code is as follows. 

`````py
#default name
print_name='????'
for k in array_received:
    if k[5] < 2 and k[4]>0.4:
        index = find_closest_index((k[0]+k[2])/2)
        if index < 3 :
            print_name=name[k[5]]
namevalue.append(print_name)
#store img as queue
face_img.append(cv.resize(frame[0:height, 0:550], (550, height)))

# if the frame is too fast to capture too many person than expected, then delete duplicated person

if(len(namevalue)>flag1):
    while len(namevalue)==flag1:
        namevalue = namevalue[:-1]
        face_img = face_img[:-1]
````
#if rightward moving people exist
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
````

`````
This result textfile and saved images are as follows.

![image](https://github.com/leekyyoon99/EC/assets/121138800/0e199243-9711-430e-b9ec-2a1dee3c6366)

You can see that the result is exported automatically.

![ezgif com-video-to-gif (1)](https://github.com/leekyyoon99/EC/assets/121138800/44efcc99-4afe-440d-b014-a12d50c3918e)

The save path is as follows.

![image](https://github.com/leekyyoon99/EC/assets/121138800/305f02e2-0759-418e-bd7e-de8704d621e0)

### 5.4 CCTV mode

#### Person Tracking Algorithm

![찬근점프](https://github.com/leekyyoon99/EC/assets/121138800/49cfb2f4-1584-44fd-991b-1d14fed7b2d7)

![image](https://github.com/leekyyoon99/EC/assets/121138800/329cfb43-e291-4dfd-bf3d-7aad59a56d7e)

To track the person, It is important to move the camera following the person. So, using **the area state array** that we made, divide left and right. If the sum of the left states is bigger than the sum of the right states, the person is on the left of the frame. Then, the camera has to move left toward. and we didn't divide by half of the width. If we set [0]~[4] left and [5]~[6] right the motor turns left and right too fast therefore We give the [4] [5] areas empty for preventing turning too fast for the detected person. Below code is the function that calculates the sum of the states. Finally, We save the direction and velocity for **array[2]**. and using Serial communication, we are going to convey these commands to arduino finally to turn the camera to track the person.



````py
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

#in the main code, We used this functuion 
if(flag == 1):
	array[0], array[1] = calculate_weighted_index(state_list)
```
````

and next, Finally arduino receives the array from python server.py and move the motor according to the commands from the server.py

```cpp
#This is arduino code
#include <Wire.h>
#include <Adafruit_MotorShield.h>

// Adafruit_MotorShield object
Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 

// DC Motor is connected M4 Port
Adafruit_DCMotor *myMotor = AFMS.getMotor(4);
int speed = 0;

void setup() {
  Serial.begin(9600); //Start serial communication

  // Adafruit Motor Shield Start
  AFMS.begin(); 

  // DC Motor Basic velocity set
  myMotor->setSpeed(100); // Select Velocity (0~255)
}

void loop() {
  if (Serial.available())// Check if data exists in Serial Port
  {  
    String command = Serial.readStringUntil('\n');  // Read Data
    // Convert Strings to array. ',' is the divide standard
    int array[2];
    int index = 0;
    for (int i = 0; i < command.length(); i++) {
        int commaIndex = command.indexOf(',', i);
        if (commaIndex == -1) {
            array[index] = command.substring(i).toInt();
            break;
        } else {
            array[index] = command.substring(i, commaIndex).toInt();
            index++;
            i = commaIndex;
        }
    }
    int direction = array[0]; //direction command from server.py
    int velocity  = array[1]; // velocity command from server.py
      
    if( direction != 0){    
      myMotor->setSpeed(velocity);
      if(direction == 1){
        myMotor->run(BACKWARD);
      }else{
        myMotor->run(FORWARD);
      }
    }else{
      myMotor->run(RELEASE); 
    }
  }
}
```



#### Coordinate calculation algorithm

![image](https://github.com/leekyyoon99/EC/assets/121138800/0a7c0c95-34fd-462b-9cfb-bd0519149288)

The oscilloscope's coordinates and the average x-coordinates of the colors are stored in an array by receiving coordinates for each color. The below code is to save the xyxy coordinates of the oscilloscope in the array.



```python
#Red 3, Green 4, Blue 5, Purple 6, Gray 7, Yellow 8
# locker plot number
locate_locker = [-1,-1,-1,-1,-1,-1]
# locker class number
target_class_locker = [3,4,5,6,7,8]
locate_oscili = []
target_class_oscili = [2]

# enter the x-average coordinate in the locate locker according to the index when the class is in the locker target class
for i in array_received:
    if i[5] in target_class_locker:
        locate_locker[target_class_locker.index(i[5])] = (i[0] + i[2]) / 2
        
# class 2 as described above, the coordinates of the oscilloscope are added to the array.
for i in array_received:
    if i[5] in target_class_oscili:
        locate_oscili.append([(i[0] + i[2]) / 2 , (i[1]+i[3]) / 2])
        
#arduino array initialize
array = [0,-1]
#limiting phase
red = find_closest_index(locate_locker[0])
yellow = find_closest_index(locate_locker[5])
```



#### Motor movement constraints

![image](https://github.com/leekyyoon99/EC/assets/121138800/97a229e4-3c31-4aa6-b5e9-2967834e3184)

If the person goes out the class, the camera doesn't have to follow the person anymore. So, we need the constraint points that stops motor. But We don't use any Gyro or potentiometer, We used color detected xy coordinates the motor to stop. If the red color is in the appointed area, the motor doesn't move more to left toward even if person moves left. If the yellow color is in the appointed area, the motor doesn't move right even if the person moves right. The below code is to prevent the motor to rotate 360 degree and stop appointed area.

```py
#if red in area 2~3 motor stop
if (1 < red < 4 ):
    if (array[0] == -1):
        array[0] = 0
#if yellow in area 6~7 motor stop
elif (5<yellow <8):
    if (array[0] == 1):
        array[0] = 0
```



![제목 없는 동영상 - Clipchamp로 제작 (1)](https://github.com/leekyyoon99/EC/assets/121138800/5a154803-d770-40c8-a49a-3f75cf7354c4)

The code that delivers the array to Arduino is in the last line. This is because if the motor is moved using a for statement in the middle of the code, the data of the frame input through the camera will not be updated. Therefore, the motor will be controlled using a flag according to each situation instead of loop. 



**Flag**

 Therefore, when a person enters, the flag is initialized to 1 to follow the person like the code presented above, and when the person leaves, the flag is set to -1 to execute the code below. The camera continued to move to the right until it went to the right (when yellow was 6 or 7), and when this condition was met, check the condition of the oscilloscope according to the coordinates of the colors and move to the left until red is in a area 2 or 3 position. Then save the image and export the state of the oscilloscope



````py
#follow people
if(flag == 1):
    ```
		#constantly save image
		```
		# when a person exists, the camera follows the person
    array[0], array[1] = calculate_weighted_index(state_list)

```
#if red in area 2~3 or yellow in area 6~7 motor stop
```
# flag = people in or out
# flag1 = move right until yellow is in specific area
# flag2 = move left until red is in specific area
# flag3 = osciliscope history has changes or not
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
    # if flag1 is 1 is left side, if 0 it has to go rightside
    if(5<yellow <8):
        flag1=1
				```
        # check oscili
				```
		# if flag2 is 1 it's on the initial state
    elif(flag1 == 1 and 1<red<4):
        flag2 = 1
		# move right
    if flag1 == 0:
        array[0] = 1
		# move left
    elif flag1 and flag2 == 0:
        array[0] = -1
    #initialize state and flag
    elif flag1 and flag2:
        flag1 = 0
        flag2 = 0
        flag = 1
				```
				#save img and print log by logic
				```
````



**State Oscilloscope**

When the person leaves, camera goes to the right side of the frame and it should check the xy coordinates of the locker and calculate the coordinates of the oscilloscope for each coordinate to update an array.

**state_list_oscili** is declared as [[0,0] for _ in range(5)] . Array looks like this. (color)(upper, lower). Below code stores the state of the oscilloscope



```py
# check oscili
for ind,i in enumerate(locate_locker[1:]):
    for j in locate_oscili:
        if abs(i - j[0])<70:
            # upper side oscili
            if abs(j[1]-620)<100:
                state_list_oscili[ind][0] = 1
            # low side oscili
            elif abs(j[1]-810)<100:
                state_list_oscili[ind][1] = 1
```

**History Oscilloscope**

In order to know the changes in the state of the previous oscilloscope and the current oscilloscope, a code is needed to know the previous state.

However, if the state list is updated to the history variable as it is through logic using only the flag1 and 2 variables above, there is a problem because the same state lists array of 3 and 4 frames are unnecessarily updated.

Therefore, we added flag3 to check whether there is a difference between the current state and the previous state, and created a logic that updates only when it is different. Below code is to update history of oscilloscope state



````py
#initialize with [[[1,1]], [[1,1]], [[1,1]], [[1,1]], [[1,1]]]
state_history_oscili = [[[1,1]] for _ in range(5)]
def find_difference(arr1, arr2):
    different = []
    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            if arr1[i][j] != arr2[i][j]:
                different.append(i*2+j+1)
    return different

def add_history_oscili():
    for i, state in enumerate(state_list_oscili):
        state_history_oscili[i].append(state)
        if len(state_history_oscili[i]) > 2: del state_history_oscili[i][0]

```
if [row[-1] for row in state_history_oscili] != state_list_oscili:
		add_history_oscili()
		#there is changes of osciliscope state
		flag3=1
		oscili1 = [row[0] for row in state_history_oscili]
		oscili2 = [row[1] for row in state_history_oscili]
		output = find_difference(oscili1, oscili2)
```
````

Finally, the oscilloscopes state history array is defined like follows. It is 3D array, [color] [previous or current] [location]. We can finally know the needed information using this array.  

![image](https://github.com/leekyyoon99/EC/assets/121138800/4974a80f-321e-466c-a91d-627126249776)



#### Export Result

When a person takes an oscilloscope, people usually takes it with back on the screen, so we saved the picture when the person first came in and printed the picture and the state of the oscilloscope when camera returned to its original state.



````py
if(flag == 1):
  if state_list[2] > 0:
      face_img.append(cv.resize(frame[0:height, 0:550], (550, height)))
  elif state_list[-2] > 0:
      face_img.append(cv.resize(frame[0:height, width-550:width], (550, height)))

```

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
      log = str(count) +'\\t\\t\\t'+ time1 + '\\t\\t	'+state+'\\t\\t\\t\\t'+output
      filename = os.path.join(output_folder2, f"{time1}_{output}.jpg")
      cv.imwrite(filename, face_img[0])
      save_to_txt2(log)
  face_img=[]

```
````

Finally, the result is automatically exported.

![image](https://github.com/leekyyoon99/EC/assets/121138800/9942e535-1b4e-452a-8370-a389a474033c)



The result can be exported automatically.

![제목 없는 동영상 - Clipchamp로 제작 (2)](https://github.com/leekyyoon99/EC/assets/121138800/7953d3ac-276d-4bf7-82a8-f0508c6e3972)

Save path is as follows.

![image](https://github.com/leekyyoon99/EC/assets/121138800/fa41c47d-e7c4-4af1-8a18-16416825ec07)



## 6. Result and Analysis



#### Evaluation Standard

Currently, our project contains many functions, so judging the performance of an algorithm on a single basis is feared to have biased results. Therefore, the evaluation criteria were set to evaluate the algorithm from the most general point of view by setting one criterion for each algorithm and each component.

Attendance mode was based on person recognition accuracy and image accuracy. This is because the algorithm's main goal is to recognize a person based on the entry and exit of the person in attendance mode, so the recognition accuracy when a person enters and leaves and how much the captured image of the recognized person matches the actual person.

The CCTV mode was evaluated based on the location of the oscilloscope, the change in location through history, and the accuracy of the stored image. In the case of CCTV mode, the evaluation criteria were set as follows because it is a key part of the algorithm to determine who took the oscilloscope and record the history of where the oscilloscope disappeared or appeared.



#### Objective

Accuracy was targeted at more than 70 percent for both attendance mode and CCTV mode.



#### Result

![image](https://github.com/leekyyoon99/EC/assets/121138800/c819d24c-615c-4628-a4e8-f752455f16f4)



![image](https://github.com/leekyyoon99/EC/assets/121138800/4110f876-8c80-4b20-af2e-504752d9884b)

In (6/12) Demo, the attendance mode was accurately judged by all five people, and the position of the oscilloscope carried out in CCTV mode was accurately judged 6 times out of 6 changes, showing 100% accuracy. However, in  (6/16) demo the CCTV mode still identified all three of the three changes and showed 100% accuracy, while the attachment mode judged 5 out of 8 people in facial recognition and showed 62.5% accuracy.



#### Analysis

Analyzing this, first, it seemed to be the limitation of the custom model for human face recognition. [Lee_ck] and [Lee_ky] present in the train data set showed very accurate detection with more than 90% accuracy, but other people which belong to  were also recognized as Lee_ck or Lee_ky three out of eight times. Previously, while preparing for dataset learning, accessories such as headsets, hoods, and hats were added to Lee_ck or Lee_ky's face to enhance characteristics compared to others, but it was confirmed that there was insufficient to obtain high accuracy by learning a person's side profile with a custom model. Second, it was confirmed that there was a problem in the labeling process. As a result of re-checking the dataset used for training, it was confirmed that about 500 out of 6,500 sheets were lost due to errors due to the coordinates deviating between 0 and 1 during the training process. This is estimated to have shown insufficient train performance due to 10% loss of learning data as a result of turning the learning without checking it and sometimes labeling the coordinates out of the frame using the space bar in the dark label program. 

Therefore, in order to improve this, first, it is expected that performance can be improved using the yolo model, which has better performance than the existing yolov5s custom model. To distinguish a person from others with a profile, a more recent version of yolo can be used to expect improved performance with the same number of train data. Second, it seems that it can be improved algorithmically. Current algorithms capture when a person first appears in a frame, and do not capture and analyze faces separately when leaving. However, if you capture a person's image both when entering and leaving, compare the two values, and set them to be labeled if both are recognized as Lee_ky or Lee_ck, you will be able to show higher accuracy than before.



## 7. Appendix

### 7.1 client.py code

```py
import socket
import pickle
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

data_pickled = []
def start_client(opt):
    host = 'localhost'
    port = 12345
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        run(**vars(opt),socketed=s)
        

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        socketed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
):
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                #OUTPUT To SERVER
                out = reversed(det).numpy()
                int_data = [[int(x) for x in row[:4]] + [(row[4]), int(row[5])] for row in out]
                data_pickled = pickle.dumps(int_data)
                socketed.sendall(data_pickled)
                #OUTPUT To SERVER

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
    
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    
    return opt

def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    start_client(opt)
    # run(**vars(opt))
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
```



### 7.2 server.py

```py
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

            # flag -> is people come in the frame?
            # flag1 -> people entered from left
            # flag2 -> people entered from right
            # flag3 -> is there differences between state_oscili and history_oscili?
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
            
            #if red in area 2~3 motor stop
            if (1 < red < 4 ):
                if (array[0] == -1):
                    array[0] = 0
            #if yellow in area 6~7 motor stop
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
                                # upper side oscili
                                if abs(j[1]-620)<100:
                                    state_list_oscili[ind][0] = 1
                                # low side oscili
                                elif abs(j[1]-810)<100:
                                    state_list_oscili[ind][1] = 1
                    # if updated oscili state is same with history people don't update history
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
else:
    print('cannot open the camera')
cap.release()
cv.destroyAllWindows()
```



### 7.3 Arduino code

```cpp
#include <Wire.h>
#include <Adafruit_MotorShield.h>

// Adafruit_MotorShield object creation
Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 

// connect DC motor to M4 Port
Adafruit_DCMotor *myMotor = AFMS.getMotor(4);
int speed = 0;

void setup() {
  Serial.begin(9600);  //Begin Serial communication

  // Adafruit Motor Shield Set
  AFMS.begin(); 

  // DC motor basic velocity set
  myMotor->setSpeed(100); 
}

void loop() {
  if (Serial.available()) {  //Check the data exists in serial Port
    String command = Serial.readStringUntil('\n');  // Read data
    // Convert strings to array and save array
    int array[2];
    int index = 0;
    for (int i = 0; i < command.length(); i++) {
        int commaIndex = command.indexOf(',', i);
        if (commaIndex == -1) {
            array[index] = command.substring(i).toInt();
            break;
        } else {
            array[index] = command.substring(i, commaIndex).toInt();
            index++;
            i = commaIndex;
        }
    }
    int direction = array[0]; // direction command from server.py
    int velocity =  array[1]; // velocity command from server.py
      
    if( direction != 0){    
      myMotor->setSpeed(velocity);
      if(direction == 1){
        myMotor->run(BACKWARD);
      }else{
        myMotor->run(FORWARD);
      }
    }else{
      myMotor->run(RELEASE); 
    }
  }
}
```

## 8. Reference

YOLOv5 : [Click here]([ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5))

Darklabel: [Click_here](https://github.com/darkpgmr/DarkLabel)





## 9. Demo Video



1.[DLIP_FINAL_DEMO_for_SSL_youtube](https://www.youtube.com/watch?v=acfIznVMDbg)

2.[DLIP_FINAL_AttendanceMODE_FULL version](https://www.youtube.com/watch?v=h3dYtjUDPns)

3.[DLIP_FINAL_CCTVMODE_FULL version](https://www.youtube.com/watch?v=VDQzuvHT-hk)

4.[DLIP_FINAL_30S_DEMO](https://www.youtube.com/watch?v=6CEisOBbEq8)