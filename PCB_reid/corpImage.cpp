#include <iostream>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

/**
 * 这是一个测试文件，测试怎样将图片按照设定的BoundingBox进行裁剪
*/

struct BoundingBox
{
    double x;
    double y;
    double height;
    double width;
};

BoundingBox setBoundingBox(double x, double y, double height, double width){
    BoundingBox b;
    b.x = x;
    b.y = y;
    b.height = height;
    b.width = width;

    return b;
}

vector<BoundingBox> setROI(){
    vector<BoundingBox> bbox;
    BoundingBox b1 = setBoundingBox(25, 23, 401, 811);
    BoundingBox b2 = setBoundingBox(375, 180, 753, 956);
    BoundingBox b3 = setBoundingBox(659, 36, 930, 788);

    bbox.push_back(b1);
    bbox.push_back(b2);
    bbox.push_back(b3);

    return bbox;
}

cv::Mat setImageRoi(cv::Mat& img, cv::Point p1, cv::Point p2){
    int xmin = p1.x;
    int xmax = p2.x;
    int ymin = p1.y;
    int ymax = p2.y;
    cout << xmin << " " << xmax << " " << ymin << " " << ymax << endl;

    cv::Mat cropImage = img(cv::Range(ymin, ymax),cv::Range(xmin, xmax));
    return cropImage;
}
     
int main(){
    //load a image
    cv::Mat image = cv::imread("../001.jpg");
    if(image.empty())
        cout << "load image error!" << endl;
    cv::resize(image, image, cv::Size(960, 960));


    vector<BoundingBox> bbox = setROI();
    cout << bbox.size() << endl;
    for(int i=0; i<bbox.size();++i){
        cv::rectangle(image,cv::Point(bbox[i].x, bbox[i].y), cv::Point(bbox[i].height, bbox[i].width), cv::Scalar(0,0,255),2);
    }
    cv::imshow("image", image);
    cv::waitKey(0);

    //corp image
    char winName[50];
    while (1)
    {
        for(int i=0; i<bbox.size();++i){
            sprintf(winName, "No%d.jpg", i);
            cv::Mat tmp_img;
            tmp_img = setImageRoi(image, cv::Point(bbox[i].x, bbox[i].y), cv::Point(bbox[i].height, bbox[i].width));
            cv::imwrite(winName,tmp_img);
            cout << "saved" << endl;
            // cv::namedWindow(winName);
            cv::imshow(winName, tmp_img);
        }
        cv::waitKey(3);
        char key = cvWaitKey(10);
        if(key==27){
            cv::destroyAllWindows();
            break;
        }
        
    }
    return 0;
}