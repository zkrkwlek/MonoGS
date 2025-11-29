#ifndef GAUSSIAN_SPARSE_SLAM_FEATURE_SCALE_INFO_H
#define GAUSSIAN_SPARSE_SLAM_FEATURE_SCALE_INFO_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <../GaussianSparseSLAM/include/Types.h>

namespace GaussianSparseSLAM {


	class FeatureScaleInfo {
	public:
		FeatureScaleInfo(int _level, float _fScale);
		virtual ~FeatureScaleInfo(){}

	public:
		//Feature 관련 정보
		float mfScaleFactor;
		float mfLogScaleFactor;
		int mnLevels;
		std::vector<float> mvScaleFactor;
		std::vector<float> mvInvScaleFactor;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;
	};
}
#endif
