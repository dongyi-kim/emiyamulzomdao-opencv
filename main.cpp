#include <iostream>
#include <opencv2/opencv.hpp>

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
int main()
{
    std::cout << "OpenCV Version : " << CV_VERSION << std::endl;
//    cv::namedWindow("EXAMPLE01", CV_WINDOW_AUTOSIZE);

    cv::Mat img = cv::imread("/home/parallels/plant.jpg", CV_LOAD_IMAGE_COLOR); // 불러올 파일 이름
    cv::Mat img_after = cv::imread("/home/parallels/growth2.jpeg", CV_LOAD_IMAGE_COLOR);
    cv::Mat img_before = cv::imread("/home/parallels/growth1.jpeg", CV_LOAD_IMAGE_COLOR);

    if (img.empty() || img_after.empty() || img_before.empty()) // 파일이 없을 경우 에러메세지 출력
    {
        std::cout << "[!] Image load fail!" << std::endl;
        return -1;
    }

    cv::Mat diff = get_diff(img_before, img_after);
    cv::imshow("diff", diff);
    cv::imshow("before",img_before);
    cv::imshow("after", img_after);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;


    cv::Mat hsv_before, hsv_after;
    cv::cvtColor(img_before, hsv_before, cv::COLOR_RGB2HSV);
    cv::cvtColor(img_after, hsv_after, cv::COLOR_RGB2HSV);


    // Convert input image to HSV
    cv::Mat hsv_image;
    cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
    // Threshold the HSV image, keep only the red pixels
    cv::Mat greenonly;

    int sensitivity = 30;

    cv::inRange(hsv_image, cv::Scalar(40, 50, 5), cv::Scalar(80, 255, 255), greenonly);

    cv::imshow("EXAMPLE01", img);
    cv::imshow("EXAMPLE02", greenonly);
    cv::waitKey(0);
    cv::destroyWindow("EXAMPLE01");
    return 0;
}