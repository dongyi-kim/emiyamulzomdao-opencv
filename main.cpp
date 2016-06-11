#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>

//background subtraction

#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

/**
 *
 * reference
 *  - http://stackoverflow.com/questions/23001512/c-and-opencv-get-and-set-pixel-color-to-mat
 *  - http://stackoverflow.com/questions/27035672/cv-extract-differences-between-two-images
 */

//namespace for testing functions
namespace test{
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
                if(dist>0)
                {
                    foregroundMask.at<cv::Vec3b>(cv::Point(j,i)) = 0xFFFFFF;
                }else{

                    foregroundMask.at<cv::Vec3b>(cv::Point(j,i)) = 0x00;
                }
            }
        }

        cv::Mat ret = foregroundMask;
        cv::fastNlMeansDenoising(foregroundMask, ret);

        return ret;
    }
    void detect_green(){
        cv::Mat img = cv::imread("img/test04/IMG_1274.jpg", CV_LOAD_IMAGE_COLOR); // 불러올 파일 이름

        cv::resize(img, img, Size( 480, 640 ));

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
        cv::inRange(hsv_image, cv::Scalar(40, 40, 60), cv::Scalar(80, 255, 255), greenonly);

       // greenonly.convertTo(greenonly, CV_8UC3);

       // cv::fastNlMeansDenoisingColored(greenonly, greenonly, 30, 10, 10, 21);

        cv::Mat result(greenonly.rows, greenonly.cols, CV_8UC3);

        double threshold =30.0;

        for(int r=0; r<result.rows; r++) {
            for (int c = 0; c < result.cols; ++c) {
                char pix = greenonly.at<char>(r, c);
                if(pix > 0){
                    result.at<cv::Vec3b>(r, c) = 0xFFFFFF;
                }else{
                    result.at<cv::Vec3b>(r, c) = 0x00;
                }
            }
        }


        cv::namedWindow("Original", CV_WINDOW_KEEPRATIO);
        cv::imshow("Original", img);

        //cv::Mat ret;



        Mat blured;

        Mat filtered = greenonly;
       // Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(5,5));
       // cv::morphologyEx(greenonly, filtered, cv::MORPH_RECT, element);
        cv::GaussianBlur(filtered, blured, cv::Size(11,11), 0);
        filtered = blured;
        for(int i = 0; i<0;i++){
            Mat temp;
            cv::fastNlMeansDenoising(filtered, temp, 7);
            filtered = temp;
        }

        vector<vector<Point> > contours;
        vector<Vec4i> hi;
        cv::findContours( filtered, contours, hi, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );

        std::vector<std::vector<cv::Point>>::iterator itc= contours.begin();

        Mat cont = (Mat)Mat::zeros(filtered.size(), CV_8UC3);

        for(int i = 0 ; i < contours.size(); i++){
            if(contours[i].size() < 100){
                continue;
            }
            int l, r, u, d;
            l = r = contours[i][0].x;
            u = d = contours[i][0].y;
            for(int j= 0 ; j < contours[i].size(); j++){
                cv::Point &pt = contours[i][j];
                l = min(l, pt.x);
                r = max(r, pt.x);
                u = max(u, pt.y);
                d = min(d, pt.y);
            }
            long long area = (long long)(r-l+1) * ( u - d + 1);
            if(area < 100){
                continue;
            }
            cv::drawContours(cont, contours, i , cv::Scalar(125, 125, 125));
            cv::Rect rect = cv::boundingRect(contours[i]);
            cv::rectangle(cont, rect.tl(), rect.br(), cv::Scalar(255,0,0));
        }

        cv::namedWindow("Green", CV_WINDOW_KEEPRATIO);
        cv::namedWindow("Filtered", CV_WINDOW_KEEPRATIO);
        cv::imshow("Green", greenonly);
        cv::imshow("Filtered", filtered);
        cv::namedWindow("Contour", CV_WINDOW_KEEPRATIO);
        cv::imshow("Contour", cont);

        cv::waitKey(0);
    }

    void detect_green_video(){
        VideoCapture cap(0);
        cv::Mat img;
        cv::namedWindow("Contour", CV_WINDOW_KEEPRATIO);
        cv::namedWindow("Green", CV_WINDOW_KEEPRATIO);
        cv::namedWindow("Filtered", CV_WINDOW_KEEPRATIO);
        cv::namedWindow("Original", CV_WINDOW_KEEPRATIO);
        cv::namedWindow("Control", CV_WINDOW_KEEPRATIO);

        int iLowH = 40;
        int iHighH = 80;

        int iLowS = 40;
        int iHighS = 255;

        int iLowV = 60;
        int iHighV = 255;



        //Create trackbars in "Control" window
        cv::createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
        cv::createTrackbar("HighH", "Control", &iHighH, 179);

        cv::createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
        cv::createTrackbar("HighS", "Control", &iHighS, 255);

        cv::createTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
        cv::createTrackbar("HighV", "Control", &iHighV, 255);




        for(;;){
            cap.read(img);
            cv::resize(img, img, Size( 480, 640 ));

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
            cv::inRange(hsv_image, cv::Scalar(iLowH, iLowH, iLowV), cv::Scalar(iHighH, iHighS, iHighV), greenonly);

            // greenonly.convertTo(greenonly, CV_8UC3);

            // cv::fastNlMeansDenoisingColored(greenonly, greenonly, 30, 10, 10, 21);

            cv::Mat result(greenonly.rows, greenonly.cols, CV_8UC3);

            double threshold =30.0;

            for(int r=0; r<result.rows; r++) {
                for (int c = 0; c < result.cols; ++c) {
                    char pix = greenonly.at<char>(r, c);
                    if(pix > 0){
                        result.at<cv::Vec3b>(r, c) = 0xFFFFFF;
                    }else{
                        result.at<cv::Vec3b>(r, c) = 0x00;
                    }
                }
            }

            cv::imshow("Original", img);

            //cv::Mat ret;



            Mat blured;

            Mat filtered = greenonly;
            // Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(5,5));
            // cv::morphologyEx(greenonly, filtered, cv::MORPH_RECT, element);
            cv::GaussianBlur(filtered, blured, cv::Size(11,11), 0);
            filtered = blured;
            for(int i = 0; i<0;i++){
                Mat temp;
                cv::fastNlMeansDenoising(filtered, temp, 7);
                filtered = temp;
            }

            vector<vector<Point> > contours;
            vector<Vec4i> hi;
            cv::findContours( filtered, contours, hi, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );

            std::vector<std::vector<cv::Point>>::iterator itc= contours.begin();

            Mat cont = (Mat)Mat::zeros(filtered.size(), CV_8UC3);

            for(int i = 0 ; i < contours.size(); i++){
                if(contours[i].size() < 100){
                    continue;
                }
                int l, r, u, d;
                l = r = contours[i][0].x;
                u = d = contours[i][0].y;
                for(int j= 0 ; j < contours[i].size(); j++){
                    cv::Point &pt = contours[i][j];
                    l = min(l, pt.x);
                    r = max(r, pt.x);
                    u = max(u, pt.y);
                    d = min(d, pt.y);
                }
                long long area = (long long)(r-l+1) * ( u - d + 1);
                if(area < 100){
                    continue;
                }
                cv::drawContours(cont, contours, i , cv::Scalar(125, 125, 125));
                cv::Rect rect = cv::boundingRect(contours[i]);
                cv::rectangle(cont, rect.tl(), rect.br(), cv::Scalar(255,0,0));
            }

            cv::imshow("Green", greenonly);
            cv::imshow("Filtered", filtered);
            cv::imshow("Contour", cont);
            int c = cv::waitKey(30);
        }
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


    void video_subtraction(){
        //global variables
        Mat frame; //current frame
        Mat resizeF;
        Mat fgMaskMOG; //fg mask generated by MOG method
        Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
        Mat fgMaskGMG; //fg mask fg mask generated by MOG2 method


        Ptr< BackgroundSubtractor> pMOG; //MOG Background subtractor
        Ptr< BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
        Ptr< BackgroundSubtractorGMG> pGMG; //MOG2 Background subtractor



        pMOG = new BackgroundSubtractorMOG();
        pMOG2 = new BackgroundSubtractorMOG2();
        pGMG = new BackgroundSubtractorGMG();


        VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera

        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1,1) );

        //unconditional loop
        while (true) {
            Mat cameraFrame;
            if(!(stream1.read(frame))) //get one frame form video
                break;

            resize(frame, resizeF, Size(frame.size().width/4, frame.size().height/4) );
            pMOG->operator()(resizeF, fgMaskMOG);
            pMOG2->operator()(resizeF, fgMaskMOG2);
            pGMG->operator()(resizeF, fgMaskGMG);
            //morphologyEx(fgMaskGMG, fgMaskGMG, CV_MOP_OPEN, element);

            imshow("Origin", resizeF);
            imshow("MOG", fgMaskMOG);
            imshow("MOG2", fgMaskMOG2);
            imshow("GMG", fgMaskGMG);


            if (waitKey(30) >= 0)
                break;
        }
    }
    void subtraction(){
        //global variables
        Ptr< BackgroundSubtractor> pMOG; //MOG Background subtractor
        Ptr< BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
        Ptr< BackgroundSubtractorGMG> pGMG; //MOG2 Background subtractor


        pMOG = new BackgroundSubtractorMOG();



        vector<cv::Mat> vf;
        vf.push_back(cv::imread("img/test04/IMG_1272.jpg", CV_LOAD_IMAGE_COLOR));
        vf.push_back(cv::imread("img/test04/IMG_1273.jpg", CV_LOAD_IMAGE_COLOR));
        vf.push_back(cv::imread("img/test04/IMG_1274.jpg", CV_LOAD_IMAGE_COLOR));
        //vf.push_back(cv::imread("img/test03/IMG_1269.jpg", CV_LOAD_IMAGE_COLOR));
        //vf.push_back(cv::imread("img/test03/IMG_1270.jpg", CV_LOAD_IMAGE_COLOR));
        //vf.push_back(cv::imread("img/test03/IMG_1271.jpg", CV_LOAD_IMAGE_COLOR));

        for(int i = 0 ; i < vf.size(); i++){
            if(vf[i].empty()){
                cout << "Empty" << endl;
                return;
            }
        }
        for(int i = 0 ; i < vf.size();i++){
            Mat resizeF;
            cv::resize(vf[i], resizeF, cv::Size( vf[i].size().width/8, vf[i].size().height/8 ));
            vf[i] = resizeF;
        }

        vector<cv::Mat> vm(vf.size()-1);
        cv::Mat temp;
        pMOG->operator()(vf[0], temp);
        for(int i = 1; i<vf.size(); i++){
            pMOG->operator()(vf[i], vm[i-1]);
        }

        while(true){

            imshow("background", vf[0]);
            for(int i = 0 ; i < vm.size() ; i++){
                char title[100] = "";
                sprintf(title, "frame %d", i+1);
                imshow(title, vm[0]);
            }
            if(waitKey(30) >= 0){
                break;
            }
        }
        return;
    }
    void camera() {

        cv::VideoCapture cap(0); //capture the video from web cam

        if ( !cap.isOpened() )  // if not success, exit program
        {
            cout << "Cannot open the web cam" << endl;
            exit(-1);
        }


        cap.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('D', 'I', 'V', '3') );
        cv::Mat frame, fgMaskMOG;

        cv::Ptr<cv::BackgroundSubtractor> pMOG = new cv::BackgroundSubtractorMOG();
        for (;;)
        {
            if(!cap.read(frame)) {
                cerr << "Unable to read next frame." << endl;
                continue;
            }
            // process the image to obtain a mask image.
            pMOG->operator()(frame, fgMaskMOG);

            cv::rectangle(frame,cv::Rect(0,cap.get(CV_CAP_PROP_FRAME_HEIGHT)-25,290,20),cv::Scalar(255,255,255),-1);

            // show image.
            imshow("Image", frame);
            imshow("Debug",fgMaskMOG);
            int c = cv::waitKey(30);
            if (c == 'q' || c == 'Q' || (c & 255) == 27)
                break;
        }

        return;
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


    void mog2(){
        VideoCapture cap(0);
        namedWindow("image", WINDOW_NORMAL);
        namedWindow("foreground mask", WINDOW_NORMAL);
        namedWindow("foreground image", WINDOW_NORMAL);
        namedWindow("mean background image", WINDOW_NORMAL);

        bool update_bg_model = true;
        int update = 100;
        BackgroundSubtractorMOG2 bg_model;//(100, 3, 0.3, 5);

        Mat img, fgmask, fgimg;

        for(;;)
        {
            cap >> img;
            cv::resize(img, img, cv::Size( img.size().width/4, img.size().height/4  ));
            if( img.empty() )
                break;

            //cvtColor(_img, img, COLOR_BGR2GRAY);

            if( fgimg.empty() )
                fgimg.create(img.size(), img.type());

            //update the model
            if(update>0){
                bg_model(img, fgmask, -1);
                update --;
                if(update == 0){
                    cout << "ok" << endl;
                }
            }else{
                bg_model(img, fgmask, 0);
            }


            fgimg = Scalar::all(0);
            img.copyTo(fgimg, fgmask);

            Mat bgimg;
            bg_model.getBackgroundImage(bgimg);
            Mat filtered;
            Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(5,5));
            cv::morphologyEx(fgmask, filtered, cv::MORPH_OPEN, element);


            imshow("image", img);
            imshow("foreground mask", filtered);
            imshow("foreground image", fgimg);
            if(!bgimg.empty())
                imshow("mean background image", bgimg );

            char k = (char)waitKey(30);
            if( k == 27 ) break;
            if( k == ' ' )
            {
                update_bg_model = !update_bg_model;
                if(update_bg_model)
                    printf("Background update is on\n");
                else
                    printf("Background update is off\n");
            }
        }
    }
}
int main()
{

    std::cout << "OpenCV Version : " << CV_VERSION << std::endl;


    test::detect_green_video();
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;

}