#ifndef GAUSSIAN_SPARSE_SLAM_OBJECT_H
#define GAUSSIAN_SPARSE_SLAM_OBJECT_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>

#include <../GaussianSparseSLAM/include/Types.h>
#include <Sensor.h>

namespace EdgeSLAM {
}

namespace GaussianSparseSLAM {

	class KeyFrame;
	 
	class FrameObject {
	public:
		FrameObject(KeyFrame* _pKF, InstanceType _type = InstanceType::SEG) : area(0.0), type(_type){}
		virtual ~FrameObject() {}
	public:
		std::vector<int> vecPointIndexes;
	public:
		InstanceType type;
		std::vector<cv::Point> contour;
		cv::Mat mask;
		cv::Rect rect;
		cv::RotatedRect rrect;//elliipse and rotated rect;
		cv::Point2f pt;
		float area;
	};

	class ObjectFrame {
	public:
		ObjectFrame():mbProcess(false){}
		virtual ~ObjectFrame(){}
	public:
		std::atomic<bool> mbProcess;
		cv::Mat mask;
		ConcurrentMap<int, FrameObject*> FrameObjects;
	};
}
#endif
