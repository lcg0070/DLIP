#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(){
    Mat image = imread("Lab_GrayScale_TestImage.jpg", IMREAD_GRAYSCALE);
    Mat thresholded;

	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    GaussianBlur(image,image,Size(15,15),0);
	morphologyEx(image, image, MORPH_OPEN, kernel);

    threshold(image, thresholded, 125, 255, THRESH_BINARY);

    vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
    findContours(thresholded, contours, hierarchy ,RETR_TREE, CHAIN_APPROX_SIMPLE);

	// printing processed img
	Mat fil;
	resize(thresholded, fil, Size(), 0.5,0.5, INTER_AREA);
    imshow("Outputfil", fil);

	//setting output matrix
    Mat output = Mat::zeros(image.size(), CV_8UC3);
    Scalar c; //color parameter

	int num_bolt1 = 0;
	int num_bolt2 = 0;
	int num_four_nuts = 0;
    int num_six_nuts1 = 0;
	int num_six_nuts2 = 0;

    for (int i = 0; i < contours.size(); i++) {
        //contour's area
		double area = contourArea(contours[i]);
        // contour's perimeter
		double perimeter = arcLength(contours[i], true);
        c= Scalar(255,255,255);
		
        if(area>500){
			//vertices 
			vector<Point> approx;
            approxPolyDP(contours[i], approx, 9, true);
            int vertices = approx.size();

			vector<vector<Point> > a;
            a.push_back(approx);

			//contour'hierarchy -> attached nut
			if (hierarchy[i][3] == -1){ //outer contour
            	int count = 0;
            	for (int j = hierarchy[i][2]; j != -1; j = hierarchy[j][0]) {
					if(contourArea(contours[j])<500){
						count--;
					}
                	count++;
            	}
				//count -> child
				//hex nut vertices 6~7
				//rect nut vertices 4
            	if(count>1){ //more than 2 nut is attached
					int rec=vertices%(4*count); //rect nut attached
					// int b=vertices%(5*count); //mix attached
					int hex=vertices%(6*count); //hex nut attached

					if(rec==0 || rec==1){
						c = Scalar(100,255,200);
						num_four_nuts+=2;
					}else if(hex==0 || hex==1){//area m5 hexnut:4000, m6 hexnut: 5500
						if(area>11000){
							c = Scalar(200,100,255);
							num_six_nuts2+=2;
						}else if(area>9000){
							c = Scalar(150,150,255);
							num_six_nuts1++;
							num_six_nuts2++;
						}else if(area>7000){
							c = Scalar(100,200,255);
							num_six_nuts1+=2;
						}
					}
					//couldn't implement to distinguish bolts were attached, and bolts and nuts were attached
				}else{ // single bolt, single nut
					//bolt distinguish
					if(perimeter>550){
						c = Scalar(0,255,0);
						num_bolt1++;
					}else if(perimeter>400){
						c = Scalar(255,100,0);
						num_bolt2++;
					}
					//nut distinguish
					if (vertices==4){//rect nut
						if (area > 3500 && area < 4700){
							c = Scalar(100,255,200);
							num_four_nuts++;
							}
					}else if(perimeter<350){//hex nut
						if(area>1500 && area<4000){//m6 and m5
							c = Scalar(200,100,255);
							num_six_nuts2++;
						}else if(area>4000){
							c = Scalar(100,200,255);
							num_six_nuts1++;
						}
		    		}
				}
			}
            drawContours(output, a, -1, c, 2);
        }
    }
	//Output
    cout << "Bolt M6: " << num_bolt1 << endl;
	cout << "Bolt M5: " << num_bolt2 << endl;
    cout << "Square Nut M5: " << num_four_nuts << endl;
    cout << "Hexa Nut M6: " << num_six_nuts1 << endl;
	cout << "Hexa Nut M5: " << num_six_nuts2 << endl;
	Mat fin;
	resize(output, fin, Size(), 0.5,0.5, INTER_AREA);
	imshow("Outputfin", fin);
    waitKey(0);
    return 0;
}