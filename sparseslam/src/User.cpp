#include <../GaussianSparseSLAM/include/User.h>
#include <../GaussianSparseSLAM/include/Frame.h>
#include <../GaussianSparseSLAM/include/Map.h>

#include <../GaussianSparseSLAM/include/GaussianPoint.h>
#include <../GaussianSparseSLAM/include/KeyFrame.h>

#include <../EdgeSLAM/include/Camera.h>
#include <../EdgeSLAM/include/CameraPose.h>
#include <BaseMotionModel.h>

namespace GaussianSparseSLAM {

	std::atomic<int> User::nNextId = 0;

	User::User() :mbMotionModel(false), mpRefKF(nullptr), mpLastCreatedKF(nullptr), mnVisID(0), mnUsed(0), ScaleFactor(1.0), mnId(++nNextId), mbRequestedReset(false)
	{

	}
	User::User(std::string _user, std::string _map, const cv::Mat& params, std::vector<bool> vbFlags) : BaseSLAM::BaseDevice(_user, _map, params, vbFlags),
		userName(_user), mapName(_map), mState(UserState::NotEstimated), mnId(++nNextId), mbRequestedReset(false),
		Rgyro(cv::Mat::eye(3, 3, CV_32FC1)), tacc(cv::Mat::zeros(3, 1, CV_32FC1)),
		mnQuality((int)params.at<float>(11)), mnSkip((int)params.at<float>(12)), mnContentKFs((int)params.at<float>(13)),
		mbProgress(false), mbRemoved(false), mpRefKF(nullptr), mpLastCreatedKF(nullptr), mnUsed(0), mnLastRelocFrameId(-1000), mbMotionModel(false), mnVisID(0),
		mnDebugTrack(0), mnDebugSeg(0), mnDebugAR(0), mnDebugLabel(0), mnDebugPlane(0), mnLastSendedTime(0), ScaleFactor(1.0),
		mbMapping(vbFlags[0]), mbDeviceTracking(vbFlags[1]), mbIMU(vbFlags[2]), mbResetAR(vbFlags[3]), mbPLPOpt(vbFlags[9]),
		mbPlaneGBA(vbFlags[4]), mbBaseLocalMap(vbFlags[5]), mbCommuTest(vbFlags[6]), mbVOSyncTest(vbFlags[7]), mbSave(vbFlags[8]), mbErrMsg(false),
		mbSaveTrajectory(false), mbAsyncTest(false), mbNewKF(false)
	{
		int _w = (int)params.at<float>(0);
		int _h = (int)params.at<float>(1);

		mpMotionModel = new BaseSLAM::BaseMotionModel();
		mpCamPose = new EdgeSLAM::CameraPose();
		mpDevicePose = new EdgeSLAM::CameraPose();
		mpCamera = new EdgeSLAM::Camera(params);
		Tcoord = cv::Mat::eye(4, 4, CV_32FC1);
		/*mpLastFrame = nullptr;
		SetPose(cv::Mat::eye(3, 3, CV_32FC1), cv::Mat::zeros(3, 1, CV_32FC1));*/
		mpMap = nullptr;

		int nStates = 18;            // the number of states
		int nMeasurements = 6;       // the number of measured states
		int nInputs = 0;             // the number of control actions
		double dt = 0.125;           // time between measurements (1/FPS)
		//mpKalmanFilter = new KalmanFilter(nStates, nMeasurements, nInputs, dt);

		cv::Mat K = mpCamera->K;
		cv::Mat invK = mpCamera->Kinv;
		//(row*col)X3
		matXimg = cv::Mat::zeros(0, 3, CV_32FC1);

		for (int x = 0; x < _w; x++) {
			for (int y = 0; y < _h; y++) {
				cv::Mat xi = (cv::Mat_<float>(1, 3) << x, y, 1);
				matXimg.push_back(xi);
			}
		}
		matXcam = invK * matXimg.t();
		matXcam.push_back(cv::Mat::ones(1, _w * _h, CV_32FC1));
		//matXcam = matXcam.t();
	}
	User::~User() {
		//delete mpCamera;
		//delete mpCamPose;
		delete mpDevicePose;
		delete mpMotionModel;
		//delete mpKalmanFilter;
		mpMap = nullptr;

		////¹Ì±¸Çö
		/*auto setKFs = mSetLocalKeyFrames.Get();
		for (auto iter = setKFs.begin(), iend = setKFs.end(); iter != iend; iter++) {
			auto pKF = *iter;
			pKF->mnConnectedDevices--;
		}
		mSetLocalKeyFrames.Release();*/

		mFrames.Release();
		mObjectFrames.Release();
		
		mvDeviceTimeStamps.Release();
		mvDeviceTrajectories.Release();
		for (int i = 0, iend = vecTrajectories.size(); i < iend; i++)
			vecTrajectories[i].release();
		std::vector<cv::Mat>().swap(vecTrajectories);
		std::vector<double>().swap(vecTimestamps);

		mapKeyPoints.Release();
		KeyFrames.Release();
		ImageDatas.Release();
		DepthDatas.Release();
		QueueNotiMsg.Release();
		PoseDatas.Release();
		
		/*auto tempTrackingResults = mapObjectTrackingResult.Get();
		for (auto mit = tempTrackingResults.begin(), mend = tempTrackingResults.end(); mit != mend; mit++)
			delete mit->second;
		mapObjectTrackingResult.Release();
		*/

		mapLastSyncedMPs.Release();
		mapLastSyncedVOs.Release();
		mapLastSendedMPs.Release();
		mapLastSendedVOs.Release();
		//delete mapFrames;
	}

	bool mbMotionModel;

	void User::Reset()
	{
		if (mbRequestedReset) {
			mbRequestedReset = false;
			SetState(UserState::NoImages);
			SetPose(cv::Mat::eye(4, 4, CV_32FC1));
			ImageDatas.Clear();
			PoseDatas.Clear();
			DepthDatas.Clear();
			mpRefKF = nullptr;
			mpLastCreatedKF = nullptr;

			mSetLocalKeyFrames.Clear();
			mvDeviceTimeStamps.Clear();
			mvDeviceTrajectories.Clear();

			mapKeyPoints.Clear();
			KeyFrames.Clear();
			QueueNotiMsg.Clear();

			//mapObjectTrackingResult.Clear();

			mapLastSyncedMPs.Clear();
			mapLastSyncedVOs.Clear();
			mapLastSendedMPs.Clear();
			mapLastSendedVOs.Clear();

			//mnUsed = 0;
			mnLastRelocFrameId = -1000;

			mnLastSendedTime = 0;

			mnDebugTrack = 0;
			mnDebugSeg = 0;
			mnDebugLabel = 0;
			mnDebugPlane = 0;

			mbNewKF = false;
			mLostTimeStamp = DBL_MAX;

			mFrames.Clear();
			mObjectFrames.Clear();
		}
	}

	Map* User::GetMap() {
		std::unique_lock<std::mutex> lock(mMutexMap);
		return mpMap;
	}
	void User::SetMap(Map* pMap) {
		std::unique_lock<std::mutex> lock(mMutexMap);
		mpMap = pMap;
	}

	cv::Mat User::GetPosition() {
		return mpCamPose->GetCameraCenter();
	}

	cv::Mat User::GetDevicePose() {
		return mpDevicePose->GetPose();
	}
	void User::SetDevicePose(cv::Mat T) {
		mpDevicePose->SetPose(T);
	}

	cv::Mat User::GetPose() {
		return mpCamPose->GetPose();
	}
	void User::SetPose(cv::Mat T) {
		mpCamPose->SetPose(T);
	}
	cv::Mat User::GetInversePose() {
		return mpCamPose->GetInversePose();
	}
	cv::Mat User::PredictPose() {
		return mpMotionModel->predict();
	}
	void User::UpdatePose(cv::Mat Tnew) {
		mpCamPose->SetPose(Tnew);
		mpMotionModel->update(Tnew);
		vecTrajectories.push_back(Tnew);
	}
	void User::UpdatePose(cv::Mat Tnew, double ts) {
		mpCamPose->SetPose(Tnew);
		mpMotionModel->update(Tnew);
		vecTrajectories.push_back(Tnew);
		vecTimestamps.push_back(ts);
	}
	void User::UpdateGyro(cv::Mat _R) {
		std::unique_lock<std::mutex> lock(mMutexGyro);
		Rgyro = _R.clone();
	}
	cv::Mat User::GetGyro() {
		std::unique_lock<std::mutex> lock(mMutexGyro);
		return Rgyro.clone();
	}

	cv::Mat User::GetCameraMatrix() {
		return mpCamera->K;
	}
	cv::Mat User::GetCameraInverseMatrix() {
		return mpCamera->Kinv;
	}
	cv::Mat User::GetDistortionMatrix() {
		return mpCamera->D;
	}

	UserState User::GetState() {
		std::unique_lock<std::mutex> lock(mMutexState);
		return mState;
	}
	void User::SetState(UserState stat) {
		std::unique_lock<std::mutex> lock(mMutexState);
		mState = stat;
	}
	void User::SetVisID(int id) {
		mnVisID = id;
	}
	int User::GetVisID() {
		return mnVisID;
	}
}