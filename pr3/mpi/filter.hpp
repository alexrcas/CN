#pragma once

#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>
#include "mpi.h"

#define KERNEL_SIZE 5

void applyFilter(std::shared_ptr<cv::Mat>, std::shared_ptr<cv::Mat>);