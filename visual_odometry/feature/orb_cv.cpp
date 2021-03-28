#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv){
    if(argc!=3){
        cout<<"usage:feature_extraction img1 img2"<<endl;
        return 1;
    }
    //--读数图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data!=nullptr && img_2.data!=nullptr);
    
    //--初始化
    std::vector<KeyPoint> keypoint_1, keypoint_2;
    Mat descriptor_1, descriptor_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    //--第一步：检测Oriented FAST角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(img_1, keypoint_1);
    detector->detect(img_2, keypoint_2);
    
    //--第二步：根据角点位置计算BRIEF描述子
    descriptor->compute(img_1, keypoint_1, descriptor_1);
    descriptor->compute(img_2, keypoint_2, descriptor_2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"extract ORB cost = "<<time_used.count()<<" seconds."<<endl;
    
    Mat outimg1;
    drawKeypoints(img_1, keypoint_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features", outimg1);
    
    //--第三步：对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptor_1, descriptor_2, matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"match ORB cost = "<<time_used.count()<<" seconds."<<endl;
    
    //--第四步：匹配点对筛选
    //计算最小距离和最大距离
    auto min_max = minmax_element(matches.begin(), matches.end(), 
        [](const DMatch &m1, const DMatch &m2) { return m1.distance<m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    
    printf("--Max dist : %f\n", max_dist);
    printf("--Min dist : %f\n", min_dist);
    
    //当描述子之间的距离大于两倍的最小距离时，即认为匹配有误，设置一个经验值30作为下限
    std::vector<DMatch> good_matches;
    for(int i=0; i<descriptor_1.rows; i++){
        if(matches[i].distance<=max(2*min_dist,30.0)){
            good_matches.push_back(matches[i]);
        }
    }
    
    //第五步：绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoint_1, img_2, keypoint_2, matches, img_match);
    drawMatches(img_1, keypoint_1, img_2, keypoint_2, good_matches, img_goodmatch);
    imshow("all matches", img_match);
    imshow("good matches", img_goodmatch);
    waitKey();
    
    return 0;
}
