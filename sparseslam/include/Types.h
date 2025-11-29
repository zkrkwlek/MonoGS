#ifndef GAUSSIAN_SPARSE_SLAM_TYPES_H
#define GAUSSIAN_SPARSE_SLAM_TYPES_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <BaseSystem.h>
#include <Optimize/seven_dof_expmap.h>

namespace GaussianSparseSLAM {

	class KeyFrame;

	//Object
	enum class InstanceType {
		SEG, SAM, RAFT, MAP
	};
	enum class ObjectMeasureType {
		IoU, IoA, IoM
	};

	//SLAM
	enum class FrameProcessStatus {
		none, alignment, initialization, tracking, mapping, loop_closing
	};

	enum class KeyFrameType {
		coviz, gaussian
	};

	enum class MapState {
		NoImages, NotInitialized, Initialized, NeedNewKeyFrame
	};

	enum class UserState {
		NoImages, NotEstimated, Success, Failed, RECENTLY_LOST
	};

	enum class LocalMappingState {
		idle = 0,
		ongoing = 1
	};
	enum class LoopClosingState {
		idle = 0,
		ongoing = 1,
		merged = 2
	};
	
	typedef std::pair<std::set<KeyFrame*>, int> ConsistentGroup;
	typedef std::map<KeyFrame*, BaseSLAM::Optimization::Sim3, std::less<const KeyFrame*>,
		Eigen::aligned_allocator<std::pair<const KeyFrame*, BaseSLAM::Optimization::Sim3> > > KeyFrameAndPose;
}
#endif
