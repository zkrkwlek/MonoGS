#ifndef GAUSSIAN_SPARSE_SLAM_KEYFRAME_H
#define GAUSSIAN_SPARSE_SLAM_KEYFRAME_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>
#include <atomic>
#include <mutex>
#include <AbstractFrame.h>
#include <Optimize/se3quat.h>

namespace EdgeSLAM {
	class Camera;
	class CameraPose;
}

namespace GaussianSparseSLAM {
	class GaussianPoint;
	class Map;
	class Frame;
	class User;
		
	class KeyFrame {
	public:
		KeyFrame(Frame* F, Map* pMap);
		virtual ~KeyFrame();
	public:
		bool is_in_image(float x, float y, float z = 1.0);
		void reset_map_points();
		void ComputeStereoFromRGBD(const cv::Mat& imDepth);
		cv::Mat UnprojectStereo(int i, const cv::Mat& Rwc, const cv::Mat& twcs);
	public:
		// Covisibility graph functions
		void AddConnection(KeyFrame* pKF, const int& weight);
		void EraseConnection(KeyFrame* pKF);
		void SetFirstConnection(bool);
		void UpdateConnections();
		void UpdateBestCovisibles();
		std::set<KeyFrame*> GetConnectedKeyFrames();
		std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
		std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int& N);
		std::vector<KeyFrame*> GetCovisiblesByWeight(const int& w);
		int GetWeight(KeyFrame* pKF);
	
	public:
		// Spanning tree functions
		void AddChild(KeyFrame* pKF);
		void EraseChild(KeyFrame* pKF);
		void ChangeParent(KeyFrame* pKF);
		std::set<KeyFrame*> GetChilds();
		KeyFrame* GetParent();
		bool hasChild(KeyFrame* pKF);
	public:
		// Loop Edges
		void AddLoopEdge(KeyFrame* pKF);
		std::set<KeyFrame*> GetLoopEdges();
		//Merge
		void AddMergeEdge(KeyFrame* pKF);
		std::set<KeyFrame*> GetMergeEdges();
	public:
		//Map Manager
		std::mutex mMutexMap;
		Map* GetMap();
		void UpdateMap(Map* pMap);

		bool mbCurrentPlaceRecognition;
	public:
		// MapPoint observation functions
		void AddGaussianPoint(GaussianPoint* pMP, const size_t& idx);
		void EraseGaussianPointMatch(const size_t& idx);
		void EraseGaussianPointMatch(GaussianPoint* pMP);
		void ReplaceGaussianPointMatch(const size_t& idx, GaussianPoint* pMP);
		std::set<GaussianPoint*> GetGaussianPoints();
		std::vector<GaussianPoint*> GetGaussianPointMatches();
		int TrackedGaussianPoints(const int& minObs);
		GaussianPoint* GetGaussianPoint(const size_t& idx);
		
		float ComputeSceneMedianDepth(const int q);
	public:
		// Enable/Disable bad flag changes
		void SetNotErase();
		void SetErase();
		// Set/check bad flag
		void SetBadFlag();
		bool isBad();

		static bool weightComp(int a, int b) {
			return a > b;
		}

		static bool lId(KeyFrame* pKF1, KeyFrame* pKF2) {
			return pKF1->mnId < pKF2->mnId;
		}

	public:
		EdgeSLAM::Camera* mpCamera;
		EdgeSLAM::CameraPose* mpCamPose;
		void SetPose(const cv::Mat& Tcw);
		cv::Mat GetPose();
		cv::Mat GetPoseInverse();
		cv::Mat GetCameraCenter();
		cv::Mat GetRotation();
		cv::Mat GetTranslation();

		std::atomic<bool> mbBA, mbFastGP;
		float mfScale;

	public:
		//static FeatureTracker* matcher;
		int mnId;
		const int mnFrameId;
		const double mdTimeStamp;

		std::atomic<int> mnConnectedDevices;

		// Variables used by the tracking
		//long unsigned int mnTrackReferenceForFrame;
		long unsigned int mnFuseTargetForKF;

		// Variables used by the local mapping
		long unsigned int mnBALocalForKF;
		long unsigned int mnBAFixedForKF;

		// Variables used by the keyframe database
		long unsigned int mnLoopQuery;
		int mnLoopWords;
		float mLoopScore;
		long unsigned int mnRelocQuery;
		int mnRelocWords;
		float mRelocScore;
		long unsigned int mnMergeQuery;
		int mnMergeWords;
		float mMergeScore;   
		long unsigned int mnPlaceRecognitionQuery;
		int mnPlaceRecognitionWords;
		float mPlaceRecognitionScore;

		// Variables used by loop closing
		cv::Mat mTcwGBA;
		cv::Mat mTcwBefGBA;
		long unsigned int mnBAGlobalForKF;

		// Calibration parameters
		const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

		// Number of KeyPoints
		const int N;

		// KeyPoints, stereo coordinate and descriptors (all associated by an index)
		const std::vector<cv::KeyPoint> mvKeys;
		const std::vector<cv::KeyPoint> mvKeysUn;
		const cv::Mat mDescriptors;

		std::vector<float> mvuRight;
		std::vector<float> mvDepth;
		std::vector<int> mvLabel;

		int mnScaleLevels;
		float mfScaleFactor;
		float mfLogScaleFactor;
		std::vector<float> mvScaleFactors;
		std::vector<float> mvInvScaleFactors;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;

		////BoW
		//DBoW2::BowVector mBowVec;
		//DBoW2::FeatureVector mFeatVec;

		// Pose relative to parent (this is computed when bad flag is activated)
		cv::Mat mTcp;
		
		const cv::Mat K;

		std::vector<bool> mvbOutliers;
		ConcurrentVector<GaussianPoint*> mvpMapPoints;
		std::vector< std::vector <std::vector<size_t> > > mGrid;

		std::map<KeyFrame*, int> mConnectedKeyFrameWeights;
		std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
		std::vector<int> mvOrderedWeights;

		// Spanning Tree and Loop Edges
		bool mbFirstConnection;
		KeyFrame* mpParent;
		std::set<KeyFrame*> mspChildrens;
		std::set<KeyFrame*> mspLoopEdges;

		/// <summary>
		/// 키프레임을 생성한 원본 기기와 관련된 정보
		/// </summary>
		std::string sourceName;
		bool mbSendLocalMap;

		// Bad flags
		bool mbNotErase;
		bool mbToBeErased;
		bool mbBad;

		//float mHalfBaseline; // Only for visualization

		Map* mpMap;
		std::mutex mMutexConnections;

		////V3
		ConcurrentSet<KeyFrame*> mspMergeEdges;
		std::vector <KeyFrame*> mvpLoopCandKFs;
		std::vector <KeyFrame*> mvpMergeCandKFs;

		// Variables used by merging
		BaseSLAM::Optimization::SE3Quat mTcwMerge;
		BaseSLAM::Optimization::SE3Quat mTcwBefMerge;
		BaseSLAM::Optimization::SE3Quat mTwcBefMerge;
		Eigen::Vector3f mVwbMerge;
		Eigen::Vector3f mVwbBefMerge;
		long unsigned int mnMergeCorrectedForKF;
		long unsigned int mnMergeForKF;
		long unsigned int mnBALocalForMerge;
		float mfScaleMerge;;
	};
}

#endif