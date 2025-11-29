#ifndef GAUSSIAN_SPARSE_SLAM_USER_H
#define GAUSSIAN_SPARSE_SLAM_USER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>
#include <ConcurrentMap.h>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>
#include <ConcurrentDeque.h>

#include <BaseDevice.h>
#include <../GaussianSparseSLAM/include/Types.h>
#include <../GaussianSparseSLAM/include/Serializer.h>

namespace BaseSLAM {
	class BaseMotionModel;
}

namespace EdgeSLAM {
	class Camera;
	class CameraPose;
}

namespace GaussianSparseSLAM {

	class Frame;
	class ObjectFrame;
	class KeyFrame;
	class Map;
	class GaussianPoint;

	class User : public BaseSLAM::BaseDevice {
	public:
		User();
		User(std::string _user, std::string _map, const cv::Mat& params, std::vector<bool> vbFlags);
		virtual ~User();
	public:
		bool mbMotionModel;
		void Reset();
		cv::Mat GetPosition();
		cv::Mat GetPose();
		void SetPose(cv::Mat T);
		cv::Mat GetInversePose();
		cv::Mat PredictPose();
		void UpdatePose(cv::Mat Tnew);
		void UpdatePose(cv::Mat Tnew, double ts);
		void UpdateGyro(cv::Mat _R);
		cv::Mat GetGyro();

		cv::Mat GetCameraMatrix();
		cv::Mat GetCameraInverseMatrix();
		cv::Mat GetDistortionMatrix();

		ConcurrentVector<cv::Mat> mvDeviceTrajectories;
		ConcurrentVector<double> mvDeviceTimeStamps;
		ConcurrentMap<int, KeyFrame*> KeyFrames; //Frame과 키프레임 연결

		ConcurrentMap<int, long long> mapLastSyncedMPs; //갱신 비교. 최근 전송된 시간과 마지막 갱신 시간
		ConcurrentMap<int, long long> mapLastSyncedVOs; //갱신 비교. 최근 전송된 시간과 마지막 갱신 시간
		ConcurrentMap<int, int> mapLastSendedMPs;		//전송 비교. 일정 프레임 전송 안되었으면 전체 데이터 전송
		ConcurrentMap<int, int> mapLastSendedVOs;		//전송 비교. 일정 프레임 전송 안되었으면 전체 데이터 전송

		cv::Mat GetDevicePose();
		void SetDevicePose(cv::Mat T);

		Map* GetMap();
		void SetMap(Map* pMap);
		////////////////
		////좌표계 결합용

		////////////////
	public:
		ConcurrentMap<int, cv::Mat> MapServerTrajectories;
		ConcurrentMap<int, cv::Mat> MapDeviceTrajectories;
		ConcurrentMap<int, cv::Mat> MapAlignedDeviceTrajectories;
		std::string userName;
		std::string mapName;
		static std::atomic<int> nNextId;
		std::atomic<bool> mbRequestedReset;
		int mnId;
		int mnQuality;
		int mnSkip;
		int mnContentKFs;
		double mLostTimeStamp;

		EdgeSLAM::Camera* mpCamera;
		EdgeSLAM::CameraPose* mpCamPose;
		EdgeSLAM::CameraPose* mpDevicePose;
		bool mbMapping, mbPLPOpt, mbIMU, mbDeviceTracking, mbBaseLocalMap, mbCommuTest, mbVOSyncTest, mbSaveTrajectory, mbAsyncTest, mbPlaneGBA, mbResetAR;
		bool mbSave, mbErrMsg;

		cv::Mat mLocalMapDescriptor;
		ConcurrentMap<int, std::vector<GaussianPoint*>> mmpLocalGPs;
		ConcurrentVector<GaussianPoint*> mvpLocalMapPoints;
		ConcurrentVector<KeyFrame*> mvpLocalKeyFrames;

		std::vector<cv::Mat> vecTrajectories;
		std::vector<double> vecTimestamps;
		ConcurrentMap<int, cv::Mat> mapKeyPoints; //추후 삭제
		ConcurrentSet<GaussianPoint*> mSetMapPoints;//추후 삭제
		ConcurrentSet<KeyFrame*> mSetLocalKeyFrames;//추후 삭제
		ConcurrentMap<int, Frame*> mFrames;
		ConcurrentMap<int, ObjectFrame*> mObjectFrames;

		ConcurrentMap<int, cv::Mat> ImageDatas, PoseDatas, DepthDatas; // id = frame id, 압축된 이미지와 포즈 정보를 이용하기 위해서임.
		////frame id와 키프레임 id의 대응이 필요함.
		//std::map<int, KeyFrame*> mapKeyFrames;
		KeyFrame* mpRefKF, *mpLastCreatedKF; //트래킹만 할 경우에는 가장 맵을 많이 참조하는 ref kf와 매칭. mapping도 하는 경우 기기에서 가장 마지막으로 만든 kf도 참조
		
		cv::Mat Tcoord;
		std::atomic<float> ScaleFactor;
		std::atomic<bool> mbProgress, mbRemoved, mbNewKF;
		std::atomic<int> mnUsed, mnLastRelocFrameId;
		std::atomic<int> mnDebugTrack, mnDebugSeg, mnDebugAR, mnDebugLabel, mnDebugPlane;
		std::atomic<long long> mnLastSendedTime;
		ConcurrentMap<int, std::string> QueueNotiMsg;

		cv::Mat matXcam, matXimg; //3xN 배열을 미리 생성

	public:
		UserState GetState();
		void SetState(UserState stat);
	private:
		BaseSLAM::BaseMotionModel* mpMotionModel;
		UserState mState;
		std::mutex mMutexState;
	protected:
		Map* mpMap;
		std::mutex mMutexMap;
		/////Visual ID
	public:
		void SetVisID(int id);
		int GetVisID();
	private:
		std::atomic<int> mnVisID;
		std::mutex mMutexGyro, mMutexAcc;
		cv::Mat Rgyro, tacc;

	};
}

#endif