#include <ctype.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//imageprocessing
void preProcessing(Mat &input, Mat &output1, Mat &output2);

//find out the max area and index
void extractObject( vector<vector<Point> > &contours, double &max_area,int &max_index);

//color sort
void getValue(Mat &input, vector<float> &color_value);

//calculate color
float calColor(vector<float> &color_value);

//show result
void drawOutput(Mat &input, vector<vector<Point> > &contours, int &max_index, vector<float> &color_value, float &color_avg);

int main(){
   VideoCapture cap("IR_DEMO_cut.avi");
   
   if(!cap.isOpened()){
    printf("Can't open the video");
    return -1;
   }
   Mat img;
   Mat img_hsv;
   Mat img_hsv_processed;
   vector<float> color_value;
   
   while(1){
    	cap >> img;
		if (img.empty()){
			printf("Fin");
			return 0;
		}
        
        preProcessing(img, img_hsv,img_hsv_processed);

        //see processed image
        imshow("camera img1", img_hsv_processed);

        //find contour
        vector<vector<Point> > contours;
        findContours(img_hsv_processed, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        Mat output = Mat::zeros(img.size(), CV_8UC3);

        int max_index=0;
        double area =0;
        //the widest contour
        extractObject(contours,area,max_index);
        
        //minimum condition
        if(area>5000){
            Rect rect = boundingRect(contours[max_index]);
            Mat roi = img_hsv(rect);

            getValue(roi, color_value);

            float color_avg;
            color_avg = calColor(color_value);
            
            //printing output
            drawOutput(img,contours,max_index,color_value,color_avg);
        }
        imshow("camera img2", img);
		if (waitKey(25) == 27){
			break;
        }
    }     
   return 0;
}

//imageprocessing
void preProcessing(Mat &input, Mat &output1, Mat &output2) {
    Mat img_hsv, hue, saturation, value;
    vector<Mat> channels;

    cvtColor(input, output1, CV_BGR2HSV);
    //To use the channel of the most distinctive contour
    split(output1, channels);
    hue = channels[0];
    saturation = channels[1];
    value = channels[2];
    
    //hsv to binary image
    inRange(output1, Scalar(0, 0, 145), Scalar(80, 255, 255), output2);
    //dilate to make it clear
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(output2, output2, kernel, Point(-1, -1), 4);
}

//find out the max area and index
void extractObject( vector<vector<Point> > &contours, double &max_area,int &max_index) {
    double max = 0;
    double area = 0;
    //the widest contour
    for (int i = 0; i < contours.size(); i++) {
        area = contourArea(contours[i]);
        if (area > max) {
            max = area; // Max area
            max_area=max;
            max_index = i; // Max area's index
        }
    }
}

//color sort
void getValue(Mat &input, vector<float> &color_value) {
    color_value.clear();
    for (int row = 0; row < input.rows; row++) {
        for (int col = 0; col < input.cols; col++) {
            Vec3b hsv = input.at<Vec3b>(row, col);
            float h = hsv.val[2];
            //convert colorvalue 0~255 to 25~40
            color_value.push_back(25 + h / 17);
        }
    }
    sort(color_value.begin(), color_value.end());
    reverse(color_value.begin(), color_value.end());
}

//calculate color
float calColor(vector<float> &color_value) {
    float color_avg = 0;
    int k = color_value.size() * 0.03;
    for (int j = 0; j < k; j++) {
        color_avg += color_value[j];
    }
    color_avg = color_avg / k;
    return color_avg;
}

//show result
void drawOutput(Mat &input, vector<vector<Point> > &contours, int &max_index, vector<float> &color_value, float &color_avg) {
    ostringstream ss;
    ss << "Max : " << fixed << setprecision(1) << color_value[0] << "avg : " << fixed << setprecision(1) << color_avg << endl;

    drawContours(input, contours, max_index, Scalar(255, 255, 255), 2);
    Rect boxPoint = boundingRect(contours[max_index]);
    
    Scalar c;
    if (color_avg >= 38) {
        c = Scalar(0, 0, 255);
        putText(input,"Warning",Point(10,50),2,1,c);
    }else{
        c=Scalar(255,255,255);
    }
    rectangle(input, boxPoint, c, 3);
    putText(input,ss.str().substr(0,ss.str().length()-1),Point(10,20),1,1,c);
}