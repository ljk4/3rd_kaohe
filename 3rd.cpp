#include<iostream>
#include <vector>
#include<opencv2/opencv.hpp>

void reduceBrightness2(cv::Mat& image, int value) {
    cv::Mat brightness = cv::Mat::ones(image.size(), image.type()) * value;
    image -= brightness;
}
//只保留红色部分
void change_to_onlyred(cv::InputArray src, cv::OutputArray dst){
    // 转换为HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // 定义红色的颜色范围0, 70, 180
    cv::Scalar lower_red1(0, 100, 100);  // 较低范围的红色
    cv::Scalar upper_red1(10, 255, 255);
    cv::Scalar lower_red2(160, 100, 100);  // 较高范围的红色
    cv::Scalar upper_red2(180, 255, 255);

    // 创建红色掩码
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, lower_red1, upper_red1, mask1);
    cv::inRange(hsv, lower_red2, upper_red2, mask2);
    cv::bitwise_or(mask1, mask2, mask);

    // 提取红色区域
    cv::bitwise_and(src, src, dst, mask);
}
//得到拟合椭圆方向角
double getContourAngle(const std::vector<cv::Point>& contour) {
    if (contour.size() < 5) {
        return NAN;  // 轮廓点不足以拟合椭圆
    }
    // 将轮廓转换为浮点型
    cv::Mat contourFloat;
    cv::Mat(contour).convertTo(contourFloat, CV_32F);
    cv::RotatedRect ellipse = fitEllipse(contour);
    return ellipse.angle;  // 返回椭圆的方向角
}
//判断轮廓是否平行
bool areContoursParallel(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2, double threshold = 5.0) {
    double angle1 = getContourAngle(contour1);
    double angle2 = getContourAngle(contour2);

    if (isnan(angle1) || isnan(angle2)) {
        return false;
    }
    // 计算角度差
    double angleDiff = std::abs(angle1 - angle2);
    return angleDiff < threshold || angleDiff > 180 - threshold;
}
//计算轮廓光滑度
double calculateSmoothness(const std::vector<cv::Point>& contour) {
    // 计算轮廓的周长
    double perimeter = cv::arcLength(contour, true);
    // 计算轮廓的面积
    double area = cv::contourArea(contour);
    // 避免除以零
    if (area == 0) return 0;
    // 计算平滑度指标
    double smoothness = perimeter * perimeter / area;
    return smoothness;
}
//判断轮廓中心距离是否合适，用于判断轮廓是否组成近似矩形
bool are_rectangle(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2, double distanceThreshold) {
    // 计算轮廓中心
    cv::Moments m1 = cv::moments(contour1); //计算给定轮廓的几何矩,包含了很多信息，例如重心的位置
    cv::Moments m2 = cv::moments(contour2);
    cv::Point2f center1(m1.m10 / m1.m00, m1.m01 / m1.m00); //计算重心（中心点 center 的坐标）
    cv::Point2f center2(m2.m10 / m2.m00, m2.m01 / m2.m00);
    // 判断中心距离
    double distance = norm(center1 - center2);
    return distance < distanceThreshold;
}
std::pair<cv::Point, cv::Point> getTopBottomPoints(const std::vector<cv::Point>& contour) {
    if (contour.size() < 2) {
        throw std::runtime_error("Contour must have at least 2 points.");
    }
    cv::Point top_point, bottom_point;
    // 遍历轮廓的所有点找到最上和最下的点
    top_point = contour[0];    
    bottom_point = contour[0];

    for(const cv::Point& p : contour) {
        if(p.y < top_point.y) {
            top_point = p;
        }
        if(p.y > bottom_point.y) {
        bottom_point = p; 
        }
    }
    // 返回最上面和最下面的两个点
    return {top_point, bottom_point};
}
void draw(std::vector<cv::Point> parallelContour1,
std::vector<cv::Point> parallelContour2,
cv::Mat image){
    // 绘制每个轮廓
    std::pair<cv::Point, cv::Point> c1 = getTopBottomPoints(parallelContour1);
    cv::Point2d P1 = c1.first;
    cv::Point2d P2 = c1.second;
    std::pair<cv::Point, cv::Point> c2 = getTopBottomPoints(parallelContour2);
    cv::Point2d P3 = c2.first;
    cv::Point2d P4 = c2.second;
    cv::line(image,P1,P2,cv::Scalar(0, 255, 0),5);
    cv::line(image,P2,P4,cv::Scalar(0, 255, 0),5);
    cv::line(image,P3,P4,cv::Scalar(0, 255, 0),5);
    cv::line(image,P3,P1,cv::Scalar(0, 255, 0),5);
}
int main(int argc, char** argv)
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    cv::Mat image;
    image = cv::imread( argv[1], cv::IMREAD_COLOR );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    double smoothness_threshold = 25; //平行度阀值，越大越不平滑
    double aspectRatio_min_threshold = 0.3; //长宽比最小阀值，越小代表越瘦
    double aspectRatio_max_threshold = 0.6; //长宽比最大阀值
    double Area_max = 300; //滤去面积小于值的轮廓
    double distance_Threshold = 500;
    
    cv::Mat red_only;
    reduceBrightness2(image,30);
    change_to_onlyred(image,red_only);

    // 显示并保存结果
    cv::imshow("Red Only", red_only);
    cv::waitKey(0);

    cv::Mat gray;
    cv::cvtColor(red_only, gray, cv::COLOR_BGR2GRAY); // 转换为灰度图
    //边缘检测，增强
    cv::Canny(gray, gray, 100, 200);

    // 然后进行二值化
    cv::Mat binary;
    cv::threshold(gray, binary, 128, 255, cv::THRESH_BINARY); // 二值化，阈值可根据需要调整
    cv::imshow("binary", binary);
    //进行膨胀
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // 创建矩形结构元素
    cv::Mat opened;
    cv::dilate(binary, opened, kernel);
    cv::erode(opened, opened, kernel);
    // 显示结果
    cv::imshow("Opened", opened);
    cv::waitKey(0);
    //边缘检测并保存边缘信息
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(opened,contours,hierarchy,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
    //轮廓近似
    std::vector<std::vector<cv::Point>> approxContours;
    for (const auto& contour : contours) {
        std::vector<cv::Point> approx;
        double epsilon = 0.01 * cv::arcLength(contour, true); // 逼近精度
        cv::approxPolyDP(contour, approx, epsilon, true);
        approxContours.push_back(approx);
    }
    //判断轮廓是否为灯管状
    std::vector<std::vector<cv::Point>> filteredContours;
    for (const auto& approx : approxContours) {
        // 求轮廓面积
	    float Light_Contour_Area = cv::contourArea(approx);
	    // 去除较小轮廓&fitEllipse的限制条件
	    if (Light_Contour_Area < Area_max || approx.size() <= 5) continue;
        cv::Rect boundingBox = cv::boundingRect(approx); // 获取外接矩形
        float aspectRatio = static_cast<float>(boundingBox.width) / boundingBox.height; // 计算长宽比
        if(aspectRatio<aspectRatio_max_threshold&&aspectRatio>aspectRatio_min_threshold){
                filteredContours.push_back(approx);
        }
    }
    // 存储平行轮廓的容器
    std::vector<std::vector<std::vector<cv::Point>>> parallelContours;
    // 检查轮廓是否平行并光滑
    std::vector<int> visited(filteredContours.size(),-1);
    for (size_t i = 0; i < filteredContours.size(); ++i) {
        if (visited[i]==1) continue;  // 如果已经访问过，跳过
        if (calculateSmoothness(filteredContours[i])>smoothness_threshold) continue;//判断平滑度
        std::vector<std::vector<cv::Point>> group;  // 存储平行轮廓组
        group.push_back(filteredContours[i]);
        visited[i] = 1;
        for (size_t j = i + 1; j < filteredContours.size(); ++j) {
            if (visited[j]==1) continue;  // 如果已经访问过，跳过
            if (areContoursParallel(filteredContours[i], filteredContours[j])&&
            are_rectangle(filteredContours[i], filteredContours[j],distance_Threshold)) {
                group.push_back(filteredContours[j]);
                visited[j] = 1;  // 标记为已访问
            }
        }
        if (group.size() > 1) {  // 只存储有平行轮廓的组
            parallelContours.push_back(group);
        }
    }
    for (size_t i = 0; i < parallelContours.size(); i++) {
        draw(parallelContours[i][0],parallelContours[i][1],image);
    }
    // 显示绘制的轮廓
    cv::imshow("image_result", image);
    cv::waitKey(0);
    return 0;
}