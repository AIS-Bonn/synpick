// Calculate a pole of inaccessibility from a nonconvex polygon
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef POLE_OF_INACCESSIBILITY_H
#define POLE_OF_INACCESSIBILITY_H

#include <opencv2/opencv.hpp>

namespace arc_perception
{

cv::Point2d poleOfInaccessibility(const std::vector<std::vector<cv::Point>>& polygon, double precision = 1);

cv::Point2d polygonCentroid(const std::vector<cv::Point>& polygon);
double contourDistance(const std::vector<std::vector<cv::Point>>& polygon, const cv::Point2d& point);

}

#endif

