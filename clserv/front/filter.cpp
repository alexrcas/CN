#include "filter.hpp"


void applyFilter(std::shared_ptr<cv::Mat> source, std::shared_ptr<cv::Mat> result)
{
    int numberOfColumns = source->cols;
    int numberOfRows = source->rows;

    for (int i = 0; i < numberOfRows; i++)
        for (int j = 0; j < numberOfColumns; j++) 
        {
            float val = 0, sum = 0;
            for (int t = i - KERNEL_SIZE; t < i + KERNEL_SIZE + 1; t++)
                for (int s = j - KERNEL_SIZE; s < j + KERNEL_SIZE + 1; s++) {
                    if ((s >= 0) && (t >= 0)) {
                        int x = cv::min(numberOfColumns - 1, cv::max(0, s));
                        int y = cv::min(numberOfRows - 1, cv::max(0, t));
                        float weight = 1;
                        
                        val += source->data[y * numberOfColumns + x] * weight;
                        sum += weight;
                    }
                }
                result->data[i * numberOfColumns + j] = round(val / sum);
        }
}