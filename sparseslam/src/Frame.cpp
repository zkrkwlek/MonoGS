#include <../GaussianSparseSLAM/include/Frame.h>
#include <../GaussianSparseSLAM/include/GaussianPoint.h>
#include <../GaussianSparseSLAM/include/FeatureScaleInfo.h>
#include <Camera.h>
#include <CameraPose.h>

namespace GaussianSparseSLAM {
	Frame::Frame(EdgeSLAM::Camera* pCam, FeatureScaleInfo* pScaleInfo, int id, double time_stamp):
		mnFrameID(id), mdTimeStamp(time_stamp), mpCamera(pCam), mfScale(1.0f),
		K(pCam->K), D(pCam->D), InvK(pCam->Kinv), fx(pCam->fx), fy(pCam->fy), cx(pCam->cx), cy(pCam->cy), invfx(pCam->invfx), invfy(pCam->invfy), mbf(pCam->mbf), mThDepth(pCam->mThDepth), mb(pCam->mb), mbDistorted(pCam->bDistorted),
		mnMinX(pCam->u_min), mnMaxX(pCam->u_max), mnMinY(pCam->v_min), mnMaxY(pCam->v_max), mfGridElementWidthInv(pCam->mfGridElementWidthInv), mfGridElementHeightInv(pCam->mfGridElementHeightInv), FRAME_GRID_COLS(pCam->mnGridCols), FRAME_GRID_ROWS(pCam->mnGridRows), 
		mnScaleLevels(pScaleInfo->mnLevels), mfScaleFactor(pScaleInfo->mfScaleFactor), mfLogScaleFactor(pScaleInfo->mfLogScaleFactor), mvScaleFactors(pScaleInfo->mvScaleFactor), mvInvScaleFactors(pScaleInfo->mvInvScaleFactor), mvLevelSigma2(pScaleInfo->mvLevelSigma2), mvInvLevelSigma2(pScaleInfo->mvInvLevelSigma2),
		mbKeypoint(false), mbDepth(false), mbKF(false)
	{
		mpCamPose = new EdgeSLAM::CameraPose();
	}

	FrameProcessStatus Frame::GetState() {
		std::unique_lock<std::mutex> lock(mMutexStat);
		return mFrameStatus;
	}
	void Frame::SetState(FrameProcessStatus stat) {
		std::unique_lock<std::mutex> lock(mMutexStat);
		mFrameStatus = stat;
	}

	void Frame::UndistortKeyPoints() {
		cv::Mat mat(N, 2, CV_32F);
		for (int i = 0; i < N; i++)
		{
			mat.at<float>(i, 0) = mvKeys[i].pt.x;
			mat.at<float>(i, 1) = mvKeys[i].pt.y;
		}

		// Undistort points
		mat = mat.reshape(2);
		cv::undistortPoints(mat, mat, K, D, cv::Mat(), K);
		mat = mat.reshape(1);

		// Fill undistorted keypoint vector
		mvKeysUn.resize(N);
		for (int i = 0; i < N; i++)
		{
			cv::KeyPoint kp = mvKeys[i];
			kp.pt.x = mat.at<float>(i, 0);
			kp.pt.y = mat.at<float>(i, 1);
			mvKeysUn[i] = kp;
		}
	}

	void Frame::ComputeStereoFromRGBD(const cv::Mat& imDepth)
	{
		mvuRight = std::vector<float>(N, -1);
		mvDepth = std::vector<float>(N, -1);

		for (int i = 0; i < N; i++)
		{
			const cv::KeyPoint& kp = mvKeys[i];
			const cv::KeyPoint& kpU = mvKeysUn[i];

			const float& v = kp.pt.y;
			const float& u = kp.pt.x;

			const float d = imDepth.at<float>(v, u);

			if (d > 0)
			{
				mvDepth[i] = d;
				mvuRight[i] = kpU.pt.x - mbf / d;
			}
		}
	}

	void Frame::reset_map_points() {

		mvGaussianPoints.Clear();
		mvGaussianPoints.Initialize(N, nullptr);
		std::vector<bool>().swap(mvbOutliers);
		mvbOutliers = std::vector<bool>(mvKeysUn.size(), false);
	}

	void Frame::check_replaced_map_points() {
		auto vpGPs = this->mvGaussianPoints.get();
		for (int i = 0; i < N; i++) {
			auto pMP = vpGPs[i];
			if (pMP) {
				auto pMPrep = pMP->GetReplaced();
				if (pMPrep && !pMPrep->isBad())
					this->mvGaussianPoints.update(i, pMPrep);
			}
		}
	}

	cv::Mat Frame::UnprojectStereo(const int& i, const cv::Mat& R, const cv::Mat& t)
	{
		const float z = mvDepth[i];
		const float u = mvKeysUn[i].pt.x;
		const float v = mvKeysUn[i].pt.y;
		const float x = (u - cx) * z * invfx;
		const float y = (v - cy) * z * invfy;
		cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
		return R * x3Dc + t;
	}

	void Frame::AssignFeaturesToGrid() {
		int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
		for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
			for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
				mGrid[i][j].reserve(nReserve);

		for (int i = 0; i < N; i++)
		{
			const cv::KeyPoint& kp = mvKeysUn[i];

			int nGridPosX, nGridPosY;
			if (PosInGrid(kp, nGridPosX, nGridPosY))
				mGrid[nGridPosX][nGridPosY].push_back(i);
		}
	}
	bool Frame::PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY) {
		posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
		posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

		if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
			return false;

		return true;
	}

	void Frame::SetPose(const cv::Mat& Tcw) {
		mpCamPose->SetPose(Tcw);
	}
	cv::Mat Frame::GetPose() {
		return mpCamPose->GetPose();
	}
	cv::Mat Frame::GetPoseInverse() {
		return mpCamPose->GetInversePose();
	}
	cv::Mat Frame::GetCameraCenter() {
		return mpCamPose->GetCameraCenter();
	}
	cv::Mat Frame::GetRotation() {
		return mpCamPose->GetRotation();
	}
	cv::Mat Frame::GetTranslation() {
		return mpCamPose->GetTranslation();
	}
}