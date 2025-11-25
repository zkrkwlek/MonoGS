#ifndef GAUSSIAN_SPARSE_SLAM_INITIALIZER_H
#define GAUSSIAN_SPARSE_SLAM_INITIALIZER_H
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
	class Frame;
	class Map;

	class Initializer {
	public:
		Initializer(int nMinFeatures = 100, int nMinTriangulatedPoints = 100, int nMaxIdDistBetweenIntializingFrames = 15, int nNumOfFailuresAfterWichNumMinTriangulatedPointsIsHalved = 20):mpInitKeyFrame1(nullptr), mpInitKeyFrame2(nullptr),
			mnMinFeatures(nMinFeatures), mnMinTriangulatedPoints(nMinTriangulatedPoints), mnMaxIdDistBetweenIntializingFrames( ), mnNumOfFailuresAfterWichNumMinTriangulatedPointsIsHalved(nNumOfFailuresAfterWichNumMinTriangulatedPointsIsHalved), mpRef(nullptr)
		{}
		virtual ~Initializer() {}

	public:
		void UpdateReferenceFrame(int id);
		MapState MonocularInitializationWithMatch(Frame* pCur, Map* pMap, std::vector<cv::Point2i>& vecMatches);

	public:
		void Init(Frame* pRef);
		void Reset();
		void ReplaceReferenceFrame();

		MapState Initialize(BaseSLAM::CameraSensor sensor, Frame* pCur, Map* pMap);

		MapState MonocularInitialization(Frame* pCur, Map* pMap);
		MapState StereoInitialization(Frame* pCur, Map* pMap);
	public:
		Frame* mpRef;
		std::stack<Frame*> mFrameStack;
		KeyFrame *mpInitKeyFrame1, *mpInitKeyFrame2;

		//Monocular Initialization
		int mnMinFeatures, mnMinTriangulatedPoints, mnMaxIdDistBetweenIntializingFrames, mnNumOfFailuresAfterWichNumMinTriangulatedPointsIsHalved;
	};

}
#endif
