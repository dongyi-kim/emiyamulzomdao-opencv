#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace std;

/**
 *
 * reference
 *  - http://stackoverflow.com/questions/23001512/c-and-opencv-get-and-set-pixel-color-to-mat
 *  - http://stackoverflow.com/questions/27035672/cv-extract-differences-between-two-images
 */
cv::Mat get_diff(cv::Mat &img_before, cv::Mat &img_after){
    cv::Mat diff;
    cv::absdiff(img_before, img_after, diff);
    cv::Mat foregroundMask(img_after.rows, img_after.cols, img_after.type());

    float threshold = 30.0f;
    float dist;

    for(int j=0; j<diff.rows; ++j) {
        for (int i = 0; i < diff.cols; ++i) {
            cv::Vec3b pix = diff.at<cv::Vec3b>(i, j);
            dist = (pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2]);
            dist = sqrt(dist);
            if(dist>threshold)
            {
                foregroundMask.at<cv::Vec3b>(cv::Point(j,i)) = 0xFFFFFF;
            }else{

                foregroundMask.at<cv::Vec3b>(cv::Point(j,i)) = 0x00;
            }
        }
    }

    cv::Mat ret = foregroundMask;
    cv::fastNlMeansDenoisingColored(foregroundMask, ret, 30, 10, 10, 21);

    return ret;
}

namespace test{
    void detect_green(){
        cv::Mat img = cv::imread("img/test01/IMG_1240.jpg", CV_LOAD_IMAGE_COLOR); // 불러올 파일 이름

        if (img.empty()) // 파일이 없을 경우 에러메세지 출력
        {
            std::cout << "[!] Image load fail!" << std::endl;
            exit(-1);
        }



        // Convert input image to HSV
        cv::Mat hsv_image;
        cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
        // Threshold the HSV image, keep only the red pixels
        cv::Mat greenonly;
        int sensitivity = 30;
        cv::inRange(hsv_image, cv::Scalar(40, 50, 5), cv::Scalar(80, 255, 255), greenonly);
        cv::namedWindow("Original", CV_WINDOW_KEEPRATIO);
        cv::namedWindow("Green", CV_WINDOW_KEEPRATIO);
        cv::imshow("Original", img);
        cv::imshow("Green", greenonly);
        cv::waitKey(0);
        cv::destroyWindow("EXAMPLE01");
    }

    void diff(){
        cv::Mat img_after = cv::imread("/home/parallels/growth2.jpeg", CV_LOAD_IMAGE_COLOR);
        cv::Mat img_before = cv::imread("/home/parallels/growth1.jpeg", CV_LOAD_IMAGE_COLOR);


        cv::Mat hsv_before, hsv_after;
        cv::cvtColor(img_before, hsv_before, cv::COLOR_RGB2HSV);
        cv::cvtColor(img_after, hsv_after, cv::COLOR_RGB2HSV);

        cv::Mat diff = get_diff(img_before, img_after);
        cv::imshow("diff", diff);
        cv::imshow("before",img_before);
        cv::imshow("after", img_after);

    }

    void camera() {
        cv::VideoCapture cap(0); //capture the video from web cam

        if ( !cap.isOpened() )  // if not success, exit program
        {
            cout << "Cannot open the web cam" << endl;
            exit(-1);
        }

        cv::namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

        int iLowH = 0;
        int iHighH = 179;

        int iLowS = 0;
        int iHighS = 255;

        int iLowV = 0;
        int iHighV = 255;

        //Create trackbars in "Control" window
        cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
        cvCreateTrackbar("HighH", "Control", &iHighH, 179);

        cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
        cvCreateTrackbar("HighS", "Control", &iHighS, 255);

        cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
        cvCreateTrackbar("HighV", "Control", &iHighV, 255);

        while (true)
        {
            cv::Mat imgOriginal;

            bool bSuccess = cap.read(imgOriginal); // read a new frame from video

            if (!bSuccess) //if not success, break loop
            {
                cout << "Cannot read a frame from video stream" << endl;
                break;
            }

            cv::Mat imgHSV;

            cvtColor(imgOriginal, imgHSV, cv::COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

            cv::Mat imgThresholded;

            inRange(imgHSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

            //morphological opening (remove small objects from the foreground)
            erode(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );
            dilate( imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );

            //morphological closing (fill small holes in the foreground)
            dilate( imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );
            erode(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );

            imshow("Thresholded Image", imgThresholded); //show the thresholded image
            imshow("Original", imgOriginal); //show the original image

            if (cv::waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
            {
                cout << "esc key is pressed by user" << endl;
                break;
            }
        }

        exit(0);
    }
}
int main()
{
    std::cout << "OpenCV Version : " << CV_VERSION << std::endl;
//    cv::namedWindow("EXAMPLE01", CV_WINDOW_AUTOSIZE);

    test::camera();

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;

}