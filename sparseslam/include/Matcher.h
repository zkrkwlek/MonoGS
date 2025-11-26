#ifndef GAUSSIAN_SPARSE_SLAM_MATCHER_H
#define GAUSSIAN_SPARSE_SLAM_MATCHER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>

#include <../GaussianSparseSLAM/include/Types.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace EdgeSLAM {
}

namespace GaussianSparseSLAM {

	class KeyFrame;
	class Frame;
	class Map;

	class Matcher{
	public:
		static void matchEigen(std::vector<cv::DMatch>& vecMatches, const cv::Mat& feats1_cv, const cv::Mat& feats2_cv, float min_cossim= 0.9);
	};

}
#endif
