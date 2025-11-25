#include <../GaussianSparseSLAM/include/Initializer.h>
#include <../GaussianSparseSLAM/include/GSSLAM.h>
#include <../GaussianSparseSLAM/include/Frame.h>
#include <../GaussianSparseSLAM/include/KeyFrame.h>
#include <../GaussianSparseSLAM/include/Optimizer.h>
#include <../GaussianSparseSLAM/include/Map.h>
#include <../GaussianSparseSLAM/include/GaussianPoint.h>
#include <../GaussianSparseSLAM/include/Matcher.h>

#include <Utils.h>
#include <Utils_Geometry.h>

namespace GaussianSparseSLAM {

	void Initializer::Init(Frame* pRef) {
		if(!mpRef){
			mpRef = pRef;
		}
		mFrameStack.push(mpRef);
	}

	void Initializer::UpdateReferenceFrame(int id) {
		if (mpRef) {
			if (id - mpRef->mnFrameID > mnMaxIdDistBetweenIntializingFrames) {
				ReplaceReferenceFrame();
			}
		}
	}
	MapState Initializer::MonocularInitializationWithMatch(Frame* pCur, Map* pMap, std::vector<cv::Point2i>& vecMatches) {
		int fid = pCur->mnFrameID;
		if (pCur->N < mnMinFeatures || mpRef->N < mnMinFeatures){
			UpdateReferenceFrame(fid);
			return MapState::NotInitialized;
		}

		std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

		std::vector<cv::Point2f> pts1, pts2;

		for (auto match : vecMatches)
		{
			int idx1 = match.x;
			int idx2 = match.y;

			cv::Point2f pt1 = mpRef->mvKeysUn[idx1].pt;
			cv::Point2f pt2 = pCur->mvKeysUn[idx2].pt;
			pts1.push_back(pt1);
			pts2.push_back(pt2);
		}

		std::vector<uchar> vFInliers;
		cv::Mat Map3D;
		cv::Mat R1, t1;
		cv::Mat E12 = cv::findEssentialMat(pts1, pts2, mpRef->K, cv::FM_RANSAC, 0.999, 1.0, vFInliers);//0.0003
		int nTriangulatedPoints = cv::recoverPose(E12, pts1, pts2, mpRef->K, R1, t1, 50.0, vFInliers, Map3D);
		R1.convertTo(R1, CV_32FC1);
		t1.convertTo(t1, CV_32FC1);

		//ts
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		if (nTriangulatedPoints > mnMinTriangulatedPoints) {
			cv::Mat Tref = cv::Mat::eye(4, 4, CV_32FC1);
			cv::Mat Tcur = cv::Mat::eye(4, 4, CV_32FC1);
			R1.copyTo(Tcur.rowRange(0, 3).colRange(0, 3));
			t1.copyTo(Tcur.col(3).rowRange(0, 3));

			mpRef->SetPose(Tref);
			pCur->SetPose(Tcur);

			mpRef->reset_map_points();
			pCur->reset_map_points();

			auto pRefKeyframe = new KeyFrame(mpRef, pMap);
			auto pCurKeyframe = new KeyFrame(pCur, pMap);

			pMap->AddKeyFrame(pRefKeyframe);
			pMap->AddKeyFrame(pCurKeyframe);

			for (int i = 0, iend = vFInliers.size(); i < iend; i++) {
				if (!vFInliers[i])
					continue;

				///////////////뎁스값 체크
				cv::Mat X3D = Map3D.col(i).clone();
				X3D.convertTo(X3D, CV_32FC1);
				X3D /= X3D.at<float>(3);
				if (X3D.at<float>(2) < 0.0) {
					continue;
				}
				///////////////reprojection error
				X3D = X3D.rowRange(0, 3);
				cv::Mat proj1 = X3D.clone();
				cv::Mat proj2 = R1 * X3D + t1;
				proj1 = mpRef->K * proj1;
				proj2 = pCur->K * proj2;
				cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
				cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));
				auto pt1 = pts1[i];
				auto pt2 = pts2[i];
				auto diffPt1 = projected1 - pt1;
				auto diffPt2 = projected2 - pt2;
				float err1 = (diffPt1.dot(diffPt1));
				float err2 = (diffPt2.dot(diffPt2));
				if (err1 > 4.0 || err2 > 4.0)
					continue;
				///////////////reprojection error

				int i1 = vecMatches[i].x;
				int i2 = vecMatches[i].y;

				auto pMP = new GaussianPoint(X3D, pCurKeyframe, pMap, ts);
				pRefKeyframe->AddGaussianPoint(pMP, i1);
				pCurKeyframe->AddGaussianPoint(pMP, i2);

				pMP->AddObservation(pRefKeyframe, i1);
				pMP->AddObservation(pCurKeyframe, i2);

				pMP->ComputeDistinctiveDescriptors();
				pMP->UpdateNormalAndDepth();

				pCur->mvGaussianPoints.update(i2, pMP);
				pCur->mvbOutliers[i2] = false;

				pMap->AddGaussianPoint(pMP);
			}

			pRefKeyframe->UpdateConnections();
			pCurKeyframe->UpdateConnections();

			Optimizer::GlobalBundleAdjustemnt(pMap, 20);

			// Set median depth to 1
			float medianDepth = pRefKeyframe->ComputeSceneMedianDepth(2);
			float invMedianDepth = 1.0f / medianDepth;

			if (medianDepth < 0 || pCurKeyframe->TrackedGaussianPoints(1) < 100)
			{
				std::cout << "Wrong initialization, reseting..." << std::endl;
				pMap->Delete();
				UpdateReferenceFrame(fid);
				return MapState::NotInitialized;
			}
			std::cout << "Map Initialization Success" << std::endl;

			// Scale initial baseline
			cv::Mat Tc2w = pCurKeyframe->GetPose();
			Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
			pCurKeyframe->SetPose(Tc2w);
			pCur->SetPose(Tc2w);

			std::vector<GaussianPoint*> vpAllMapPoints = pCurKeyframe->GetGaussianPointMatches();
			for (int i = 0; i < vpAllMapPoints.size(); i++) {
				auto pMP = vpAllMapPoints[i];
				if (!pMP)
					continue;
				pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
			}

			mpInitKeyFrame1 = pRefKeyframe;
			mpInitKeyFrame2 = pCurKeyframe;

			return MapState::Initialized;
		}
		UpdateReferenceFrame(fid);
		return MapState::NotInitialized;
	}

	void Initializer::Reset() {
		mpRef = nullptr;
		mpInitKeyFrame1 = nullptr;
		mpInitKeyFrame2 = nullptr;
		while (!mFrameStack.empty()) {
			mFrameStack.pop();
		}
	}
	void Initializer::ReplaceReferenceFrame() {
		mpRef = mFrameStack.top();
		mFrameStack.pop();
	}

	MapState Initializer::Initialize(BaseSLAM::CameraSensor sensor, Frame* pCur, Map* pMap) {
		auto mapState = pMap->GetState();
		if (mapState == MapState::NoImages)
		{
			pMap->SetState(MapState::NotInitialized);
			Init(pCur);
			  
			//여기까지만.  
			return MapState::NotInitialized;
		}

		std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);
		MapState res = MapState::NotInitialized;
		if (sensor == BaseSLAM::CameraSensor::MONOCULAR) {
			res = MonocularInitialization(pCur, pMap);
		}
		if (sensor == BaseSLAM::CameraSensor::RGBD || sensor == BaseSLAM::CameraSensor::STEREO) {
			res = StereoInitialization(pCur, pMap);
		}
		return res;
	}

	MapState Initializer::MonocularInitialization(Frame* pCur, Map* pMap) {

		if (mpRef) {
			if (pCur->mnFrameID - mpRef->mnFrameID > mnMaxIdDistBetweenIntializingFrames) {
				ReplaceReferenceFrame();
			}
		}
		mFrameStack.push(pCur);
		//std::cout << "Initialize " << mpRef->mnFrameID << ", " << pCur->mnFrameID << std::endl;

		if (pCur->N < mnMinFeatures || mpRef->N < mnMinFeatures)
			return MapState::NotInitialized;

		//std::vector<int> idx1, idx2;       
		//int nMatch = mpFeatureTracker->match(mpRef->mDescriptors, pCur->mDescriptors, idx1, idx2, 0.8);
		std::vector<cv::DMatch> vecMatches;
		Matcher::matchEigen(vecMatches, mpRef->mDescriptors, pCur->mDescriptors);
		std::vector<cv::Point2f> pts1, pts2;
		
		for (auto match : vecMatches)
		{
			int idx1 = match.queryIdx;
			int idx2 = match.trainIdx;

			cv::Point2f pt1 = mpRef->mvKeysUn[idx1].pt;
			cv::Point2f pt2 = pCur->mvKeysUn[idx2].pt;
			pts1.push_back(pt1);
			pts2.push_back(pt2);
		}

		std::vector<uchar> vFInliers;
		cv::Mat Map3D;
		cv::Mat R1, t1;
		cv::Mat E12 = cv::findEssentialMat(pts1, pts2, mpRef->K, cv::FM_RANSAC, 0.999, 1.0, vFInliers);//0.0003
		int nTriangulatedPoints = cv::recoverPose(E12, pts1, pts2, mpRef->K, R1, t1, 50.0, vFInliers, Map3D);
		R1.convertTo(R1, CV_32FC1);
		t1.convertTo(t1, CV_32FC1);

		//ts
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		//std::cout << "Map = " << nTriangulatedPoints <<", "<<pts1.size()<<" "<<idx1.size()<<", "<<Map3D.cols<<" "<<vFInliers.size()<< std::endl;
		if (nTriangulatedPoints > mnMinTriangulatedPoints) {
			cv::Mat Tref = cv::Mat::eye(4, 4, CV_32FC1);
			cv::Mat Tcur = cv::Mat::eye(4, 4, CV_32FC1);
			R1.copyTo(Tcur.rowRange(0, 3).colRange(0, 3));
			t1.copyTo(Tcur.col(3).rowRange(0, 3));

			mpRef->SetPose(Tref);
			pCur->SetPose(Tcur);

			mpRef->reset_map_points();
			pCur->reset_map_points();

			auto pRefKeyframe = new KeyFrame(mpRef, pMap);
			auto pCurKeyframe = new KeyFrame(pCur, pMap);

			//pRefKeyframe->ComputeBoW();
			//pCurKeyframe->ComputeBoW();

			pMap->AddKeyFrame(pRefKeyframe);
			pMap->AddKeyFrame(pCurKeyframe);

			for (int i = 0, iend = vFInliers.size(); i < iend; i++) {
				if (!vFInliers[i])
					continue;

				///////////////뎁스값 체크
				cv::Mat X3D = Map3D.col(i).clone();
				X3D.convertTo(X3D, CV_32FC1);
				X3D /= X3D.at<float>(3);
				if (X3D.at<float>(2) < 0.0) {
					continue;
				}
				///////////////reprojection error
				X3D = X3D.rowRange(0, 3);
				cv::Mat proj1 = X3D.clone();
				cv::Mat proj2 = R1 * X3D + t1;
				proj1 = mpRef->K * proj1;
				proj2 = pCur->K * proj2;
				cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
				cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));
				auto pt1 = pts1[i];
				auto pt2 = pts2[i];
				auto diffPt1 = projected1 - pt1;
				auto diffPt2 = projected2 - pt2;
				float err1 = (diffPt1.dot(diffPt1));
				float err2 = (diffPt2.dot(diffPt2));
				if (err1 > 4.0 || err2 > 4.0)
					continue;
				///////////////reprojection error

				int i1 = vecMatches[i].queryIdx;
				int i2 = vecMatches[i].trainIdx;

				auto pMP = new GaussianPoint(X3D, pCurKeyframe, pMap, ts);
				pRefKeyframe->AddGaussianPoint(pMP, i1);
				pCurKeyframe->AddGaussianPoint(pMP, i2);

				pMP->AddObservation(pRefKeyframe, i1);
				pMP->AddObservation(pCurKeyframe, i2);

				pMP->ComputeDistinctiveDescriptors();
				pMP->UpdateNormalAndDepth();

				pCur->mvGaussianPoints.update(i2, pMP);
				pCur->mvbOutliers[i2] = false;

				pMap->AddGaussianPoint(pMP);
			}

			//std::cout << "KF = " << pRefKeyframe->mnId << ", " << pCurKeyframe->mnId << std::endl;

			pRefKeyframe->UpdateConnections();
			pCurKeyframe->UpdateConnections();

			Optimizer::GlobalBundleAdjustemnt(pMap, 20);

			// Set median depth to 1
			float medianDepth = pRefKeyframe->ComputeSceneMedianDepth(2);
			float invMedianDepth = 1.0f / medianDepth;

			if (medianDepth < 0 || pCurKeyframe->TrackedGaussianPoints(1) < 100)
			{
				std::cout << "Wrong initialization, reseting..." << std::endl;
				pMap->Delete();
				return MapState::NotInitialized;
			}
			std::cout << "Map Initialization Success" << std::endl;

			// Scale initial baseline
			cv::Mat Tc2w = pCurKeyframe->GetPose();
			Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
			pCurKeyframe->SetPose(Tc2w);
			pCur->SetPose(Tc2w);

			std::vector<GaussianPoint*> vpAllMapPoints = pCurKeyframe->GetGaussianPointMatches();
			for (int i = 0; i < vpAllMapPoints.size(); i++) {
				auto pMP = vpAllMapPoints[i];
				if (!pMP)
					continue;
				pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
			}

			mpInitKeyFrame1 = pRefKeyframe;
			mpInitKeyFrame2 = pCurKeyframe;

			/////
			/*
			mpLocalMapper->InsertKeyFrame(pKFini);
			mpLocalMapper->InsertKeyFrame(pKFcur);

			mCurrentFrame.SetPose(pKFcur->GetPose());
			mnLastKeyFrameId=mCurrentFrame.mnId;
			mpLastKeyFrame = pKFcur;

			mvpLocalKeyFrames.push_back(pKFcur);
			mvpLocalKeyFrames.push_back(pKFini);
			mvpLocalMapPoints=mpMap->GetAllMapPoints();
			mpReferenceKF = pKFcur;
			mCurrentFrame.mpReferenceKF = pKFcur;

			mLastFrame = Frame(mCurrentFrame);

			mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

			mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

			mpMap->mvpKeyFrameOrigins.push_back(pKFini);
			*/


			return MapState::Initialized;
		}
		return MapState::NotInitialized;

	}

	MapState Initializer::StereoInitialization(Frame* pCur, Map* pMap) {

		if (pCur->N < 500){
			std::cout << "???????" << std::endl;        
			return MapState::NotInitialized;
		}

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		pCur->reset_map_points();
		pCur->SetPose(cv::Mat::eye(4, 4, CV_32F));
		cv::Mat R = pCur->GetRotation();
		cv::Mat t = pCur->GetTranslation();

		// Create KeyFrame
		KeyFrame* pKFini = new KeyFrame(pCur, pMap);

		// Insert KeyFrame in the map
		pMap->AddKeyFrame(pKFini);
		
		mpInitKeyFrame1 = pKFini;

		// Create MapPoints and asscoiate to KeyFrame
		for (int i = 0; i < pCur->N; i++)
		{
			float z = pCur->mvDepth[i];
			if (z > 0)
			{
				cv::Mat x3D = pCur->UnprojectStereo(i, R, t);
				GaussianPoint* pNewMP = new GaussianPoint(x3D, pKFini, pMap, ts);
				pNewMP->AddObservation(pKFini, i);
				pKFini->AddGaussianPoint(pNewMP, i);
				pNewMP->ComputeDistinctiveDescriptors();
				//pNewMP->UpdateNormalAndDepth();
				pMap->AddGaussianPoint(pNewMP);
				pCur->mvGaussianPoints.update(i, pNewMP);
				pCur->mvbOutliers[i] = false;
			}
		}

		//mpLocalMapper->InsertKeyFrame(pKFini);

		//mLastFrame = Frame(mCurrentFrame);
		//mnLastKeyFrameId = mCurrentFrame.mnId;
		//mpLastKeyFrame = pKFini;

		//mvpLocalKeyFrames.push_back(pKFini);
		//mvpLocalMapPoints = mpMap->GetAllMapPoints();
		//mpReferenceKF = pKFini;
		//mCurrentFrame.mpReferenceKF = pKFini;

		//mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

		//mpMap->mvpKeyFrameOrigins.push_back(pKFini);

		//mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

		return MapState::Initialized;
	}

}