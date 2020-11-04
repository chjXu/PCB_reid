#include <torch/torch.h>
#include <torch/script.h>
#include <torch/data/transforms.h>

#include <iostream>
#include <stdio.h>
#include <memory>
#include <vector>
#include <string>
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

/**
 * @function: read all images in a folder
 * @param query query image
 * @param gallery gallery images
*/
void read_images(vector<cv::Mat>& query, vector<cv::Mat>& gallery){
    const int num = 7;
    char fileName[50];
    char winName[50];
    cv::Mat srcImage;
    for(int i=0; i<num; ++i){
        sprintf(fileName, "../images/0%d.jpg", i);
        sprintf(winName, "No%d", i);
        //cout << fileName << endl;
        srcImage = cv::imread(fileName);
        if(srcImage.empty()){
            cout << "read image error!" << endl;
            return ;
        }
        if(i == 0)
            query.emplace_back(srcImage);
        else
            gallery.emplace_back(srcImage);
        cv::namedWindow(winName);
        cv::imshow(winName, srcImage);
    }
    cv::waitKey(0);
}

/**
 * @function: show_image
 * @param img input
 * @param title windows name
*/
void show_image(cv::Mat& img, std::string title){
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, img);
    cv::waitKey(0);
}


/**
 * @function: trans image from tensor to cv::mat
 * @param tensor input of tensor image
 * @result return a cv::mat image
*/
auto toCvImage(at::Tensor tensor){
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];

    try
    {
        cv::Mat output_mat(cv::Size{height, width}, CV_8UC3, tensor.data_ptr<uchar>());
        show_image(output_mat, "Converted image from tensor");
        return output_mat.clone();
    }
    catch(const c10::Error& e)
    {
        cout << "Error has occured: " << e.msg() << endl;
    }
    return cv::Mat(height, width, CV_8UC3);
    
}

/**
 * @function: resize_image
 * @param src cv::mat image
 * @param size size
 * @param reuslt reuturn a new image with size is size
*/
auto resize_image(cv::Mat& src, cv::Size size){
    cv::Mat dst;
    cv::resize(src, dst, size, 0,0,CV_INTER_LINEAR);
    return dst;
}

/**
 * @function: trans the tensor image from RGB to BGR
 * @param tensor input tensor image
 * @param dims change the channel
 * @param result return a new tensor image
*/
auto transpose(at::Tensor tensor, c10::IntArrayRef dims={0, 3, 1, 2}){
    //cout << "-----------shape before: " << tensor.sizes() << endl;
    tensor = tensor.permute(dims);
    //cout << "-----------shape after: " << tensor.sizes() << endl;
    return tensor;
}

/**
 * @function: change a mat image to a tensor image
 * @param img input mat image
 * @param show_output unknow
 * @param unsqueeze bool variable to change image dim
 * @param unsqueeze_dim dim
 * @result a tensor image
*/
auto toTensor(cv::Mat& img, bool show_output = false, bool unsqueeze = false, int unsqueeze_dim = 0){
    //cout << "image shape: " << img.size() << endl;
    at::Tensor tensor_image = torch::from_blob(img.data, {img.rows, img.cols,3}, at::kByte);

    if(unsqueeze){
        tensor_image.unsqueeze_(unsqueeze_dim);
        //cout << "tensor new shape: " << tensor_image.sizes() << endl;
    }

    if(show_output){
        cout << tensor_image.slice(2,0,1) << endl;
    }

    //cout << "tensor shape: " << tensor_image.sizes() << endl;
    return tensor_image;
}

/**
 * @function: a input to net
 * @param tensor_image input tensor image
 * @result torch::jit::IValue in GPU
*/
auto toInput(at::Tensor tensor_image){
    return vector<torch::jit::IValue>{tensor_image.to(at::kCUDA)};
}

/**
 * @function: trans images to features
 * @param image input image
 * @param model module
 * @result return features
*/
auto image_to_features(cv::Mat& img, torch::jit::script::Module& model){
    //trans image to tensor
    auto new_img = resize_image(img, cv::Size(384,192));
    auto tensor_img = toTensor(new_img, false, true);
    tensor_img = tensor_img.clamp_max(c10::Scalar(50));
    tensor_img = tensor_img.toType(c10::kFloat).div(255);
    tensor_img = transpose(tensor_img);

    //创建输入
    auto input = toInput(tensor_img);

    //at::Tensor input = torch::rand({1,3,384,192}).to(at::kCUDA);
    at::Tensor output = model.forward(input).toTensor();
    
    return output;
}

/**
 * @function: compute the score between query image and gallery images
 * @param query_features query_features
 * @param gallery_features gallery_features
 * @result reutrn score
*/
auto dis_map(at::Tensor& query_features, at::Tensor& gallery_features){
    //compute the euclidean distance
    int m = query_features.sizes()[0];
    int n = gallery_features.sizes()[0];
    //cout << m << " " << n << endl;

    // cout << query_features.sizes() << endl; //[1,512]
    //query_features = query_features.view(-1);
    //gallery_features = gallery_features.view(-1);
    // cout << gallery_features.sizes() << endl;  //[512]

    auto dis = torch::pow(query_features,2).sum(1,true) + \
               torch::pow(gallery_features,2).sum(1, true).t();;
    //cout << "dis: " << dis << "and shape is: " << dis.sizes() << endl;
    dis.addmm_(query_features, gallery_features.t(), 1,-2);
    //cout << "dis: " << dis << endl;
    return dis;
}

int main(int argc, const char** argv){
    if(argc != 2){
        std::cerr << "Please enter the path of model!" << std::endl;
        return -1;
    }

    //vector<cv::Mat> query;
    //vector<cv::Mat> gallery;
    //read_images(query, gallery);
    //cout << "Now, I get " << query.size() << " query images from folder!" << endl;;
    //cout << "Now, I get " << gallery.size() << " gallery images from folder!" << endl;

    torch::Device device("cpu");
    if(torch::cuda::is_available()){
        device = torch::Device("cuda:0");
    }

    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(argv[1]);
    }
    catch(const std::exception& e)
    {
        std::cerr << "error loading the module" << '\n';
        return -1;
    }

    module.to(device);
    std::cerr << "model load successfully!\n";
    
    //vector<> results;
    vector<cv::Mat> q_img;
    cv::VideoCapture cap(0);
    while(true)
    {
        cv::Mat frame;
        cap >> frame;
        q_img.push_back(frame);
        if(q_img.size()<10)
            continue;

        for(int i=1; i<10; ++i){
            auto query_features = image_to_features(q_img[0], module);
            auto gallery_features = image_to_features(q_img[i], module);
            at::Tensor score = dis_map(query_features, gallery_features);
            //cout << "socre is: " << score << endl;
        }
        imshow("img", frame);

        q_img.erase(q_img.begin());
        char c = cv::waitKey(30);
        if(c == ' ')
            break;
    }
    
    // for(int i=0; i<query.size(); ++i){
    //     cv::Mat query_img = query[i];
    //     auto query_features = image_to_features(query_img, module);
    //     for(int j=0; j<gallery.size(); ++j){
    //         cv::Mat gallery_img = gallery[j];
    //         auto gallery_features = image_to_features(gallery_img, module);
    //         at::Tensor score = dis_map(query_features, gallery_features);
    //         cout << "socre is: " << score << endl;
    //         //results.push_back(score);
    //     }
    // }

    // for(auto result: results){
    //     cout << "The results size is: " << results.size() << "\n";
    //     cout << result << endl;    
    // }
}