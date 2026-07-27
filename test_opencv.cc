#include <opencv2/opencv.hpp>
#include <iostream>
int main() {
    cv::VideoCapture cap("test_slam.mp4");
    if(!cap.isOpened()) { std::cout << "Cannot open" << std::endl; return 1; }
    cv::Mat frame;
    if(cap.read(frame)) { std::cout << "Read frame " << frame.cols << "x" << frame.rows << std::endl; }
    else { std::cout << "Failed to read first frame" << std::endl; }
    return 0;
}
