#include "filter.hpp"

int main(int argc, char** argv)
{

    std::shared_ptr<cv::Mat> image(new cv::Mat(cv::imread(argv[1], cv::IMREAD_GRAYSCALE)));
    std::shared_ptr<cv::Mat> result(new cv::Mat(image->rows, image->cols, CV_8UC1));

    auto startTime = std::chrono::high_resolution_clock::now();
    applyFilter(image, result);
    auto endTime = std::chrono::high_resolution_clock::now();
    
    cv::imwrite("result.png", *result);

    auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    std::cout << "Elapsed time: " << elapsedTime << "s" << std::endl;
}