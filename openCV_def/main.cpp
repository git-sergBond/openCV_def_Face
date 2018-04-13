/*#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
*/

#include <opencv2\opencv.hpp>

//#include <opencv\cv.h>
//#include <opencv\highgui.h>

#include <videoInput.h>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

IplImage* image = 0;
Mat mimg;


String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
char* window_name = "Capture - Face detection";
RNG rng(12345);

void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, 6/*COLOR_BGR2GRAY*/ );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|2/*CASCADE_SCALE_IMAGE*/, Size(30, 30) );

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |2/*CASCADE_SCALE_IMAGE*/, Size(30, 30) );

        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
    }
    //-- Show what you got
    imshow( window_name, frame );
}
int main(int argc, char* argv[])
{
	
	videoInput VI;

	
   

	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	
	
	int numDevices = VI.listDevices();
	int device1=0; // первое найденое видеоустройсво из списка

	VI.setIdealFramerate(device1, 15);// частота кадров

    // указываем разрешение
    VI.setupDevice(device1, 1280, 960, VI_COMPOSITE);
    VI.showSettingsWindow(device1); 
	
	image = cvCreateImage(cvSize(VI.getWidth(device1),VI.getHeight(device1)), IPL_DEPTH_8U, 3);

	cvNamedWindow(window_name, CV_WINDOW_AUTOSIZE);
	
	int width = VI.getWidth(device1);
    int height = VI.getHeight(device1); 
	unsigned char* yourBuffer = new unsigned char[VI.getSize(device1)];

	while(1){
		if(VI.isFrameNew(device1)){
			// третий - флаг, определяющий менять ли местами B и R -составляющий
			// четвёртый - флаг, определяющий поворачивать картинку или нет
			VI.getPixels(device1, yourBuffer/*(unsigned char *)image->imageData*/, false, true); // получение пикселей в BGR
			mimg = cv::Mat(height, width, CV_8UC3, yourBuffer, Mat::AUTO_STEP);
			detectAndDisplay(mimg);
		}
		char c = cvWaitKey(33);
		if (c == 27) { // ESC
			break;}
	}

	// освобождаем ресурсы
	cvReleaseImage(&image);
	cvDestroyWindow(window_name);

	// останавливаем видеозахват
	VI.stopDevice(device1);

	return 0;
}