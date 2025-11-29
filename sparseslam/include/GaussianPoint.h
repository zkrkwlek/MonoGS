#ifndef GAUSSIAN_SPARSE_SLAM_GAUSSIAN_POINT_H
#define GAUSSIAN_SPARSE_SLAM_GAUSSIAN_POINT_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <ConcurrentSet.h>
#include <ConcurrentMap.h>
#include <atomic>
#include <AbstractData.h>

#include<Eigen/Dense>

namespace GaussianSparseSLAM {
	
	class Map;
	class KeyFrame;
	class Frame;

	class GaussianPoint : public BaseSLAM::AbstractData {
	public:
		GaussianPoint(const cv::Mat& Pos, KeyFrame* pRefKF, Map* pMap, long long _ts);
		GaussianPoint(const cv::Mat& Pos, Frame* pFrame, Map* pMap, const int& idxF);
		//MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF);
		virtual ~GaussianPoint();

		void Update() {}
		void SetWorldPos(const cv::Mat& Pos);
		cv::Mat GetWorldPos();

		void SetNormalVector(const cv::Mat& normal);
		cv::Mat GetNormal();
		KeyFrame* GetReferenceKeyFrame();

		//MAP
		std::map<KeyFrame*, size_t> GetObservations();
		int Observations();
		void AddObservation(KeyFrame* pKF, size_t idx);
		void EraseObservation(KeyFrame* pKF);
		int GetIndexInKeyFrame(KeyFrame* pKF);
		bool IsInKeyFrame(KeyFrame* pKF);
		
		void SetBadFlag();
		void Replace(GaussianPoint* pMP);
		GaussianPoint* GetReplaced();

		void IncreaseVisible(int n = 1);
		void IncreaseFound(int n = 1);
		float GetFoundRatio();
		inline int GetFound() {
			return mnFound;
		}

		void ComputeDistinctiveDescriptors();

		cv::Mat GetDescriptor();
		
		void UpdateNormalAndDepth();

		float GetMinDistanceInvariance();
		float GetMaxDistanceInvariance();
		int PredictScale(const float& currentDist, float logScaleFactor, int nScaleLevels);

		Map* GetMap();
		void UpdateMap(Map* pMap);

	public:
		int mnId;
		int mnFirstKFid;
		int mnFirstFrame;
		int nObs;
		std::atomic<int> nBoxObs;
		std::atomic<int> mnObjectID;
		std::atomic<int> mnLabelID;
		std::atomic<int> mnPlaneID;
		std::atomic<int> mnPlaneCount;
		std::atomic<long long> mnLastUpdatedTime;

		// Variables used by local mapping
		long unsigned int mnBALocalForKF;
		long unsigned int mnFuseCandidateForKF;

		// Variables used by loop closing
		long unsigned int mnLoopPointForKF;
		long unsigned int mnCorrectedByKF;
		long unsigned int mnCorrectedReference;
		cv::Mat mPosGBA;
		long unsigned int mnBAGlobalForKF;
		long unsigned int mnBALocalForMerge;

		// Variable used by merging
		Eigen::Vector3d mPosMerge;
		Eigen::Vector3d mNormalVectorMerge;

		static std::mutex mGlobalMutex;
	protected:

		// Position in absolute coordinates
		cv::Mat mWorldPos;

		// Keyframes observing the point and associated index in keyframe
		ConcurrentMap<KeyFrame*, size_t> mObservations;

		// Mean viewing direction
		cv::Mat mNormalVector;

		// Best descriptor to fast matching
		cv::Mat mDescriptor;

		// Reference KeyFrame
		KeyFrame* mpRefKF;

		// Tracking counters
		int mnVisible;
		int mnFound;

		// Bad flag (we do not currently erase MapPoint from memory)
		std::atomic<bool> mbBad;
		GaussianPoint* mpReplaced;

		// Scale invariance distances
		float mfMinDistance;
		float mfMaxDistance;

		Map* mpMap;
		std::mutex mMutexMap;

		std::mutex mMutexPos;
		std::mutex mMutexFeatures;
	};
}

#endif