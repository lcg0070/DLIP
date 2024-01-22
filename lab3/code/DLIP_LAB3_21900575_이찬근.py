import numpy as np
import cv2 as cv
import timeit as time

#degree to Radian
deg2Rad = lambda d: np.pi/180 * d

#y=mx+b
#get slope->m // if x2=x1 its error, so add very little value    
get_slope = lambda x1, y1, x2, y2: (y2 - y1) / (x2 - x1 + 1e-6)

#get b
get_offset = lambda slope, x1, y1: y1-slope*x1

#get fps
def cal_fps(frame, start_t, terminate_t):
    FPS = int(1./(terminate_t - start_t ))
    cv.putText(frame, "FPS : "+ str(FPS), (30,150),cv.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),2)

#get x axis if y=height
def get_x(slope,x1,y1, height):
    #y=mx+b
    b=get_offset(slope, x1, y1)
    if(slope == 0):
        return 0
    x=(height-b)/slope
    return int(x)

#get intersection of lines
def get_intersection(m1, x1, y1, m2, x2, y2):
    b1 = get_offset(m1,x1,y1)
    b2 = get_offset(m2,x2,y2)
    # intersection Point x,y
    return int((b2-b1)/(m1-m2)), int(m1*(b2-b1)/(m1-m2)+b1)

#process img
def process_image(img):
    # grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gaussianBlur
    blur_img = cv.GaussianBlur(gray_img, (7, 7), 0)
    # canny
    canny_img = cv.Canny(blur_img, 50, 100)
    return canny_img

#HoughLinesP
def hough_roi( processed_img):
    #img size
    height, width = processed_img.shape[:2]
    #triangle ROI
    vertices = np.array([[(0+100, height-70), (width / 2, height / 2+50), (width-100, height-70)]], dtype=np.int32)
    mask = np.zeros_like(processed_img)
    cv.fillPoly(mask, vertices, 255)
    roi_img = cv.bitwise_and(processed_img, mask)
    cv.imshow("what",roi_img)
    #HoughLines
    lines = cv.HoughLinesP(roi_img, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)
    return lines

# determine the lines
def cal_lines(lines):
    x_left,x_right,x_center,y_left,y_right,y_center=0,0,0,0,0,0
    max_left, max_right, max_cen= 0,0,0
    slope_right, slope_left, slope_center = 0,0,0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = get_slope(x1, y1, x2, y2)
        if slope < np.tan(deg2Rad(60)) and slope> np.tan(deg2Rad(30)):
            if(abs(y1-y2)>max_right):
                x_right=x1
                y_right=y1
                slope_right = slope
                max_right = abs(y1-y2)
        elif slope < -np.tan(deg2Rad(30)) and slope> -np.tan(deg2Rad(60)):
            if(abs(y1-y2)>max_left):
                x_left=x1
                y_left=y1
                slope_left = slope
                max_left = abs(y1-y2)
        elif abs(slope) >= np.tan(deg2Rad(60)):
            if(abs(y1-y2)>max_cen):
                x_center=x1
                y_center=y1
                slope_center = slope
                max_cen = abs(y1-y2)
        left = [int(x_left),int(y_left),slope_left]
        center = [int(x_center),int(y_center),slope_center]
        right =[int(x_right),int(y_right),slope_right]
    return left, center, right

# determine lanechange
def lanechange_detect(left, center, right):
    global previous_left, previous_cen, previous_right
    global linechange
    if center[2] != 0:
        previous_cen = center
        linechange = True
    elif left[2] != 0 and right[2] != 0:
        previous_left = left
        previous_right = right
        linechange = False
    elif right[2] != 0:
        previous_right = right
    elif left[2] != 0:
        previous_left = left

# previous or not
def select_output(frame, get_x, get_intersection, left, center, right):
    global previous_left, previous_right, previous_cen, linechange, height
    global tri_color, points, xL, xR, x2, y2
    if linechange:
        # previous
        if center[2] == 0:
            x, y, m = previous_cen
        # now
        else:
            x, y, m = center
        x = get_x(m, x, y, height)
        y = height
        x2 = get_x(m, x, y, int(height / 2) + 50)
        y2 = int(height / 2) + 50
        cv.line(frame, (x, y), (x2, y2), (0, 0, 255), 5)
        cv.circle(frame, (x2, y2), 5, (0, 0, 255), 5)
        tri_color = (0, 0, 255)
    else:
        if(left[2] == 0):
            xL, yL, mL = previous_left
            color1 = (0,255,255)
        else:
            xL, yL, mL = left
            color1 = (255,0,0)
        if(right[0] == 0):
            xR, yR, mR = previous_right
            color2 = (0,255,255)
        else:
            xR, yR, mR = right
            color2 = (255,0,0)
        x2, y2 =get_intersection(mL,xL,yL,mR,xR,yR)
        xL = get_x(mL,xL,yL,height)
        xR = get_x(mR,xR,yR,height)
        yR=yL=height
        cv.line(frame, (xL, yL), (x2, y2), color1, 5)
        cv.line(frame, (xR, yR), (x2, y2), color2, 5)
        cv.circle(frame,(x2,y2),5,(0,255,0),5)
        tri_color = (0,255,0)
        points = np.array([[x2,y2],[xL,yL],[xR,yR]])

#put text // direction, bias
def print_text(frame, xL, xR):
    global linechange, way_right
    global height, width
    bias = (width/2 - (xL+xR)/2)
    bias = (bias/(width-200))*100
    direction =" "
    if(linechange):
        lane_is ="Lane change"
        color =(0,0,255)
        if(bias<0):
            bias=-bias
        if(way_right):
            way = "->"
        else:
            way = "<-"
        cv.putText(frame,way,(int(width/2)-40, int(height/2)),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
    else:
        lane_is ="Safe"
        color =(0,255,0)
        if(bias<0):
            direction = "left "
            bias=-bias
            way_right=False
        else:
            direction = "right "
            way_right=True
        #average x of two lines
        cv.line(frame, (int((xL+xR)/2),height-20),(int((xL+xR)/2),height),(204,102,255),5)
    cv.putText(frame, "Bias : "+ direction+f"{bias:.2f}"+" %",(30,50),cv.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),3)
    cv.putText(frame, "In Line? : "+ lane_is, (30,100),cv.FONT_HERSHEY_SIMPLEX,1.3,color,3)
    cv.line(frame,(int(width/2),height-20),(int(width/2),height),(255,255,255),5)
    return way_right

def process_video(cap, lanechange_detect, process_image, hough_roi, cal_lines, print_text):
    global way_right
    global height, width
    if cap.isOpened():
        while True:
            k = cv.waitKey(30)
            start_t = time.default_timer()
            # esc to quit
            ret, frame = cap.read()
            if ret and k != 27:
                # image processing and finding lines
                processed_img = process_image(frame)
                height, width = processed_img.shape[:2]
                lines = hough_roi(processed_img)
                left, center, right = cal_lines(lines)

                # check slope
                lanechange_detect(left, center, right)

                # if slope is not detected 
                select_output(frame, get_x, get_intersection, left, center, right)

                tri_img = np.zeros((height, width, 3), dtype=np.uint8)
                cv.fillPoly(tri_img, [points],tri_color)

                # print bias and lane departure
                way_right =print_text(frame, xL, xR)
            
                cv.imshow('video',frame)
                # calculate FPS
                terminate_t = time.default_timer()
                cal_fps(frame, start_t, terminate_t)

                # add triangle shape 
                result_img = cv.addWeighted(frame, 1, tri_img, 0.3, 0)
                # 이미지를 출력합니다.
                cv.imshow('Lane Detection', result_img)
            else:
                print("end of file")
                break
    else:
        print('cannot open the file')
    cap.release()
    cv.destroyAllWindows()

# main
previous_left = [0, 0, 1e-6]
previous_right = [0, 0, 1e-6]
previous_cen = [0, 0, 1e-6]
height = 0
width = 0
points =[]
xL, xR, x2, y2 = 0,0,0,0
tri_color = (0,0,0)
linechange = False
way_right = False

# cap = cv.VideoCapture('road_straight_student.mp4')
cap = cv.VideoCapture('road_lanechange_student.mp4')

process_video(cap,lanechange_detect, process_image, hough_roi, cal_lines, print_text)