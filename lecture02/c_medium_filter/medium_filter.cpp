#include <iostream>
#include <opencv2/opencv.hpp>


cv::Mat medium_gray(cv::Mat img, int w, int h) {
    cv::Mat new_image(img.rows, img.cols,CV_8UC1 );
    for(int i=0;i<img.cols;i++){
        for(int j=0;j<img.rows;j++){
            if(i<w/2||i>img.cols-w/2) new_image.at<uchar>(j,i)=img.at<uchar>();
            else if(j<h/2||j>img.rows-h/2) new_image.at<uchar >(j,i)=img.at<uchar>(j,i);
            else{
                std::vector<uchar> comparison(w*h);
                for(int blockW=0;blockW<w; blockW++){
                    for(int blockH=0;blockH<h;blockH++){
                        comparison.at((blockW)*h+blockH)=img.at<uchar>(j+blockW-w/2,i+blockH-h/2);
                    }
                }
                std::sort(comparison.begin(),comparison.end());
                new_image.at<uchar >(j,i)=comparison.at(w*h/2);
            }
        }
    }
    return new_image;
}

void add_salt_noise_gray(cv::Mat img) {
    int count = img.cols * img.rows / 100;
    for (int i = 0; i < count; i++) {
        int y = (int) random() % (img.rows - 1);
        int x = (int) random() % (img.cols - 1);
        img.at<uchar>(y, x) = 255;
    }
}


int main() {
    cv::Mat image;
    image = cv::imread("/home/lilinyang/CLionProjects/clion01/lena.jpg");
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(image, bgrChannels);
    for (int i = 0; i < 3; i++) {
        add_salt_noise_gray(bgrChannels[i]);
    }
    cv::Mat salt_image;
    cv::merge(bgrChannels, salt_image);
    std::vector<cv::Mat> medium_bgr(3);
    for (int i = 0; i < 3; i++) {
        medium_bgr[i]=medium_gray(bgrChannels[i],3,3);
    }
    cv::Mat medium_image;
    cv::merge(medium_bgr, medium_image);
    cv::imshow("img", image);
    cv::imshow("salt_img", salt_image);
    cv::imshow("medium_img", medium_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}


