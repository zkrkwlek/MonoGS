#ifndef GAUSSIAN_SPARSE_SLAM_FRAME_H
#define GAUSSIAN_SPARSE_SLAM_FRAME_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <mutex>
#include <AbstractFrame.h>
#include <ConcurrentVector.h>

#include <../GaussianSparseSLAM/include/Types.h>

namespace EdgeSLAM {
	class Camera;
	class CameraPose;
}

namespace GaussianSparseSLAM {

	class GaussianPoint;
	class FeatureScaleInfo;

	class Frame {
	public:
		Frame(){}
		Frame(EdgeSLAM::Camera* pCam, FeatureScaleInfo* pScaleInfo, int id, double time_stamp = 0.0);
		virtual ~Frame(){}

		void SetState(FrameProcessStatus);
		FrameProcessStatus GetState();

	public:
		int mnFrameID;
		int mnKeyFrameId;
		double mdTimeStamp;

		std::atomic<bool> mbKeypoint;
		std::atomic<bool> mbDepth,mbKF;
		float mfScale; //depth scale

		std::vector<cv::KeyPoint> mvKeys, mvKeysUn;
		cv::Mat mDescriptors;
		ConcurrentVector<GaussianPoint*> mvGaussianPoints; //->vector 
		std::vector<bool> mvbOutliers;
		std::vector<float> mvuRight;
		std::vector<float> mvDepth;

		int N;
		cv::Mat K, D, InvK;
		float fx, fy, cx, cy, invfx, invfy;
		float mb, mbf, mThDepth;
		bool mbDistorted;

		int FRAME_GRID_COLS;
		int FRAME_GRID_ROWS;
		float mfGridElementWidthInv;
		float mfGridElementHeightInv;
		std::vector<std::size_t>** mGrid;

		float mnMinX;
		float mnMaxX;
		float mnMinY;
		float mnMaxY;

		int mnScaleLevels;
		float mfScaleFactor;
		float mfLogScaleFactor;
		std::vector<float> mvScaleFactors;
		std::vector<float> mvInvScaleFactors;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;

		void check_replaced_map_points();
		void reset_map_points();
		void UndistortKeyPoints();
		void ComputeStereoFromRGBD(const cv::Mat& imDepth);
		cv::Mat UnprojectStereo(const int& i, const cv::Mat& R, const cv::Mat& t);

		void AssignFeaturesToGrid();
		bool PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY);

		EdgeSLAM::Camera* mpCamera;
		EdgeSLAM::CameraPose* mpCamPose;
		void SetPose(const cv::Mat& Tcw);
		cv::Mat GetPose();
		cv::Mat GetPoseInverse();
		cv::Mat GetCameraCenter();
		cv::Mat GetRotation();
		cv::Mat GetTranslation();
	private:
		FrameProcessStatus mFrameStatus;
		std::mutex mMutexStat;
	};
}
#endif
