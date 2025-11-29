#include <../GaussianSparseSLAM/include/Visualizer.h>
#include <GSSLAM.h>
#include <../GaussianSparseSLAM/include/Map.h>
#include <../GaussianSparseSLAM/include/MapManager.h>
#include <../GaussianSparseSLAM/include/GaussianPoint.h>
#include <../GaussianSparseSLAM/include/KeyFrame.h>
#include <../GaussianSparseSLAM/include/User.h>
#include <unordered_set>

namespace GaussianSparseSLAM {
	Visualizer::Visualizer() {

	}
	Visualizer::Visualizer(GSSLAM* pSystem) :mpSystem(pSystem), mpMap(nullptr), mnVisScale(1000), mnDisplayX(0), mnDisplayY(0), mbDoingProcess(false), mnVisMode(0) {

	}
	Visualizer::~Visualizer() {}

	int mnMode = 2;
	int mnMaxMode = 3;
	int mnAxis1 = 0;
	int mnAxis2 = 2;

	bool bSaveMap = false;
	bool bLoadMap = false;
	bool bShowOnlyTrajectory = true;

	void SetAxisMode() {
		switch (mnMode) {
		case 0:
			mnAxis1 = 0;
			mnAxis2 = 2;
			break;
		case 1:
			mnAxis1 = 1;
			mnAxis2 = 2;
			break;
		case 2:
			mnAxis1 = 0;
			mnAxis2 = 1;
			break;
		}
	}
	cv::Point2f rectPt;
	void Visualizer::CallBackFunc(int event, int x, int y, int flags, void* userdata)
	{
		float* tempData = (float*)userdata;

		if (event == cv::EVENT_LBUTTONDOWN)
		{
			//std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
			tempData[2] = (float)x - tempData[0];
			tempData[3] = (float)y;

			////button interface
			if (tempData[2] < 50.0 && y < 50) {
				bSaveMap = !bSaveMap;
			}
			else if (tempData[2] < 50 && (y >= 50 && y < 100)) {
				//bShowOnlyTrajectory = !bShowOnlyTrajectory;
				bLoadMap = !bLoadMap;
			}
			////button interface
		}
		else if (event == cv::EVENT_LBUTTONUP)
		{
			//std::cout << "Left button of the mouse is released - position (" << x << ", " << y << ")" << std::endl;
			tempData[4] = (float)x - tempData[0];
			tempData[5] = (float)y;
			tempData[6] = tempData[4] - tempData[2];
			tempData[7] = tempData[5] - tempData[3];
		}
		else if (event == cv::EVENT_RBUTTONDOWN) {
			mnMode++;
			mnMode %= mnMaxMode;
			SetAxisMode();
		}
		else if (event == cv::EVENT_MOUSEWHEEL) {

			if (flags & cv::EVENT_FLAG_CTRLKEY) {
				if (flags > 0) {
					//scroll up
					tempData[8] += 0.2;
				}
				else {
					tempData[8] -= 0.2;
				}
			}
			else {
				if (flags > 0) {
					//scroll up
					tempData[1] += 2.0;
				}
				else {
					//scroll down
					tempData[1] -= 2.0;
					if (tempData[1] <= 0.0) {
						tempData[1] = 2.0;
					}
				}
			}

		}
	}

	cv::Point2f Visualizer::ConvertVisPt(cv::Mat T, cv::Mat x3D) {
		cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
		cv::Mat tempPt(tpt);
		cv::Mat aaa = T * tempPt;
		tpt.x = aaa.at<float>(0);
		tpt.y = aaa.at<float>(1);
		tpt += mVisMidPt;
		return tpt;
	}

	int nOutputImages = 0;
	void Visualizer::Init(int w, int h, bool _bSave, int _inc) {
		mnWidth = w;
		mnHeight = h;

		mbSaveVisImage = _bSave;
		mnIncForSaveImg = _inc;

		mVisPoseGraph = cv::Mat(mnHeight * 2, mnWidth * 2, CV_8UC3, cv::Scalar(255, 255, 255));
		rectangle(mVisPoseGraph, cv::Rect(0, 0, 50, 50), cv::Scalar(255, 255, 0), -1);
		rectangle(mVisPoseGraph, cv::Rect(0, 50, 50, 50), cv::Scalar(0, 255, 255), -1);
		mVisMidPt = cv::Point2f(mnHeight, mnWidth);
		mVisPrevPt = mVisMidPt;

		////맵 옆의 4개의 이미지
		//tracking, segmentation, mapping, ??
		cv::Mat leftImg1 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat leftImg2 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat leftImg3 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat leftImg4 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);

		//right image
		cv::Mat rightImg1 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat rightImg2 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat rightImg3 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat rightImg4 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);

		mSizeOutputImg = leftImg1.size();
		//맵
		cv::Mat mapImage = cv::Mat::zeros(mnHeight * 2, mnWidth * 2, CV_8UC3);

		//////sliding window
		//mnWindowImgRows = 4;
		//int nWindowSize = 8;//mpMap->mnMaxConnectedKFs + mpMap->mnHalfConnectedKFs + mpMap->mnQuarterConnectedKFs;
		//mnWindowImgCols = nWindowSize / mnWindowImgRows;
		//if (nWindowSize % 4 != 0)
		//	mnWindowImgCols++;
		//cv::Mat kfWindowImg = cv::Mat::zeros(mnWindowImgRows*mnHeight / 2, mnWindowImgCols * mnWidth / 2, CV_8UC3);

		//0 1 2 3
		mvOutputImgs.push_back((leftImg1));
		cv::Rect r1(0, 0, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r1);
		mvOutputImgs.push_back((leftImg2));
		cv::Rect r2(0, leftImg1.rows, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r2);
		mvOutputImgs.push_back((leftImg3));
		cv::Rect r3(0, leftImg1.rows * 2, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r3);
		mvOutputImgs.push_back((leftImg4));
		cv::Rect r4(0, leftImg1.rows * 3, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r4);

		//right image
		int colRight = leftImg1.cols + mapImage.cols;
		mvOutputImgs.push_back((rightImg1));
		cv::Rect r5(colRight, 0, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r5);
		mvOutputImgs.push_back((rightImg2));
		cv::Rect r6(colRight, leftImg1.rows, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r6);
		mvOutputImgs.push_back((rightImg3));
		cv::Rect r7(colRight, leftImg1.rows * 2, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r7);
		mvOutputImgs.push_back((rightImg4));
		cv::Rect r8(colRight, leftImg1.rows * 3, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r8);

		//4
		nOutputImages = mvOutputImgs.size();
		mvOutputImgs.push_back((mapImage));
		cv::Rect rMap(leftImg1.cols, 0, mapImage.cols, mapImage.rows);
		mvRects.push_back(rMap);

		////5 윈도우이미지
		//mvOutputImgs.push_back((kfWindowImg));
		//cv::Rect rWindow(leftImg1.cols + mapImage.cols, 0, kfWindowImg.cols, kfWindowImg.rows);
		//mvRects.push_back(rWindow);

		mvOutputChanged = std::vector<bool>(mvRects.size(), false);
		//map

		rectPt = cv::Point2f(r3.x, r3.y);
		int nDisRows = mnHeight * 2;
		int nDisCols = leftImg1.cols + mapImage.cols + rightImg1.cols;// +kfWindowImg.cols;
		mOutputImage = cv::Mat::zeros(nDisRows, nDisCols, CV_8UC3);


	}


	void Visualizer::Run() {

		//floor plan
		cv::Mat wean = cv::imread("../bin/data/weanhall2.png", cv::IMREAD_COLOR);
		//cv::resize(wean, wean, cv::Size(wean.cols * 1.2, wean.rows * 1.2));
		
		SetAxisMode();
		int nMapImageID = 0;

		std::stringstream ss;
		ss << "Output::Display::" << strMapName;
		std::string strWindowName = ss.str();

		//현재 여기서 시각화가 안됨
		//원인 찾는 중
		cv::imshow(strWindowName, mOutputImage);
		cv::moveWindow(strWindowName, mnDisplayX, mnDisplayY);
		float mapControlData[9] = { 0.0, };
		mapControlData[8] = 0.0;//-90.0;
		mapControlData[0] = mnWidth;
		mapControlData[1] = mnVisScale;
		cv::setMouseCallback(strWindowName, GaussianSparseSLAM::Visualizer::CallBackFunc, (void*)mapControlData);

		std::vector<cv::Scalar> planeColors(3);
		planeColors[0] = cv::Scalar(125, 0, 0);
		planeColors[1] = cv::Scalar(0, 125, 0);
		planeColors[2] = cv::Scalar(0, 0, 125);

		std::vector<cv::Scalar> userColors(6);
		userColors[0] = cv::Scalar(255, 255, 0);
		userColors[1] = cv::Scalar(0, 255, 255);
		userColors[2] = cv::Scalar(255, 0, 255);
		userColors[3] = cv::Scalar(255, 0, 0);
		userColors[4] = cv::Scalar(0, 255, 0);
		userColors[5] = cv::Scalar(0, 0, 255);

		while (true) {
			if (bSaveMap) {
				//auto vpUsers = GetUsers();
				//auto user = mpSystem->GetAllUsersInMap(strMapName)[0];
				//mpSystem->pool->EnqueueJob(Segmentator::ProcessPlanarModeling, mpSystem, user);
				//bSaveMap = false;
			}

			//////Update Visualizer 
			mVisMidPt += cv::Point2f(mapControlData[6], mapControlData[7]);
			mapControlData[6] = 0;
			mapControlData[7] = 0;
			mnVisScale = mapControlData[1];
			float radian = mapControlData[8] * CV_PI / 180.0;
			cv::Mat T = cv::Mat::eye(2, 2, CV_32FC1);
			float c = std::cosf(radian);
			float s = std::sinf(radian);
			T.at<float>(0, 0) = c;
			T.at<float>(0, 1) = -s;
			T.at<float>(1, 0) = s;
			T.at<float>(1, 1) = c;

			cv::Mat tempVis = mVisPoseGraph.clone();
			//wean.copyTo(tempVis(cv::Rect(300, 500, wean.cols, wean.rows))); //floor plan

			cv::RNG mapRNG(12345);
			auto spMaps = mpSystem->mpMapManager->GetAllMaps();
			for (auto map : spMaps) {
				if (map->IsBad())
					continue;
				cv::Scalar mapColor;
				if (map->mnId == 0)
					mapColor = cv::Scalar(0, 0, 0);
				else
					mapColor = (cv::Scalar(mapRNG.uniform(0, 255), mapRNG.uniform(0, 255), mapRNG.uniform(0, 255)));
				{
					////맵포인트 시각화
					auto mmpMap = map->mmpGaussianPoints.Get();

					for (auto pair : mmpMap)
					{
						auto pMPi = pair.second;

						if (!pMPi || pMPi->isBad())
							continue;

						cv::Mat x3D = pMPi->GetWorldPos();
						cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
						cv::Mat tempPt(tpt);
						cv::Mat aaa = T * tempPt;
						tpt.x = aaa.at<float>(0);
						tpt.y = aaa.at<float>(1);
						tpt += mVisMidPt;
						cv::circle(tempVis, tpt, 1, mapColor, -1);

					}
					
					std::map<int,GaussianPoint*>().swap(mmpMap);
				}
				//map->mlpNewMPs
				

				//keyframe
				auto KFs = map->GetAllKeyFrames();
				for (int i = 0, iend = KFs.size(); i < iend; i++) {
					auto pKFi = KFs[i];
					cv::Mat Ow = pKFi->GetCameraCenter();
					cv::Scalar color = cv::Scalar(255, 0, 0);
					if (pKFi->mnConnectedDevices > 1) {
						color.val[2] = 255;
					}
					else if (pKFi->mnConnectedDevices > 0) {
						color.val[1] = 255;
					}
					cv::Point2f tpt = cv::Point2f(Ow.at<float>(mnAxis1) * mnVisScale, Ow.at<float>(mnAxis2) * mnVisScale);
					cv::Mat tempPt(tpt);
					cv::Mat aaa = T * tempPt;
					tpt.x = aaa.at<float>(0);
					tpt.y = aaa.at<float>(1);
					tpt += mVisMidPt;
					cv::circle(tempVis, tpt, 4, color, 1);
				}

				//new mp 시각화
				auto vpNewMPs = map->mlpNewMPs.get();
				for (auto pMPi : vpNewMPs)
				{
					if (!pMPi || pMPi->isBad())
						continue;

					cv::Mat x3D = pMPi->GetWorldPos();
					cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
					cv::Mat tempPt(tpt);
					cv::Mat aaa = T * tempPt;
					tpt.x = aaa.at<float>(0);
					tpt.y = aaa.at<float>(1);
					tpt += mVisMidPt;
					cv::circle(tempVis, tpt, 2, cv::Scalar(0, 255, 0), -1);
				}
			}
			
			{
				////User 위치 시각화
				//추후 유저별 정보로 변경
				auto vpUsers = mpSystem->GetAllUsersInMap(strMapName);
				for (size_t i = 0, iend = vpUsers.size(); i < iend; i++) {
					auto user = vpUsers[i];
					if (!user)
						continue;
					user->mnUsed++;
					
					auto pos = user->GetPosition();
					cv::Point2f pt1 = cv::Point2f(pos.at<float>(mnAxis1) * mnVisScale, pos.at<float>(mnAxis2) * mnVisScale);
					cv::Mat tempPt(pt1);
					cv::Mat aaa = T * tempPt;
					pt1.x = aaa.at<float>(0);
					pt1.y = aaa.at<float>(1);

					pt1 += mVisMidPt;
					if (user->mbMapping)
						cv::circle(tempVis, pt1, 4, cv::Scalar(0, 0, 255), -1);
					else {
						cv::circle(tempVis, pt1, 4, userColors[i % 6], -1);
					}

					auto spGPs = user->mSetMapPoints.Get();
					for (auto pMPi : spGPs)
					{
						if (!pMPi || pMPi->isBad())
							continue;

						cv::Mat x3D = pMPi->GetWorldPos();
						cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
						cv::Mat tempPt(tpt);
						cv::Mat aaa = T * tempPt;
						tpt.x = aaa.at<float>(0);
						tpt.y = aaa.at<float>(1);
						tpt += mVisMidPt;
						cv::circle(tempVis, tpt, 2, cv::Scalar(0,0,255), -1);
					}


					//auto vecTrajectories = user->mvDeviceTrajectories.get();
					//for (int j = 0; j < vecTrajectories.size(); j += 1) {

					//	cv::Mat R = vecTrajectories[j].rowRange(0, 3);
					//	cv::Mat t = vecTrajectories[j].row(3).t();
					//	t = -R.t()*t;  //camera center
					//	cv::Point2f pt1 = cv::Point2f(t.at<float>(mnAxis1)* mnVisScale, t.at<float>(mnAxis2)* mnVisScale);
					//	pt1 += mVisMidPt;
					//	cv::circle(tempVis, pt1, 1, userColors[i], -1);
					//}
					user->mnUsed--;
				}
				std::vector<User*>().swap(vpUsers);
			}
			
			//최종 이미지 회전
			/*auto tempMidPt = cv::Point2f(mnHeight, mnWidth);
			cv::Mat warmImg = cv::Mat::zeros(mnWidth * 2, mnHeight * 2, CV_8UC3);
			cv::Mat MM = cv::getRotationMatrix2D(tempMidPt, 90.0, 1.0);
			cv::warpAffine(tempVis, warmImg, MM, cv::Size(mnWidth*2, mnHeight*2));*/

			//이미지 반전
			//cv::flip(tempVis, tempVis, 0);
			SetOutputImage(tempVis, nOutputImages);

			////////Update Map Visualizer
			int N = mvOutputImgs.size();
			for (int i = 0; i < N; i++) {
				if (isOutputTypeChanged(i)) {
					cv::Mat mTrackImg = GetOutputImage(i);
					mTrackImg.copyTo(mOutputImage(mvRects[i]));
				}
			}
			
			cv::Mat tempVisImage;
			//cv::resize(mOutputImage, tempVisImage, mOutputImage.size()/2);
			cv::resize(mOutputImage, tempVisImage, cv::Size(mOutputImage.cols * 0.75, mOutputImage.rows * 0.75));
			imshow(strWindowName, tempVisImage);
			if (mbSaveVisImage)
			{
				nMapImageID++;
				if (nMapImageID % mnIncForSaveImg == 0) {
					std::stringstream sss;
					sss << "../res/vis/vis_" << nMapImageID << ".jpg";
					cv::imwrite(sss.str(), mOutputImage);
				}
			}
			auto key = cv::waitKey(10);
			if (key == '1') {
				std::cout << "1" << std::endl;
			}
			if (key == '2') {
				std::cout << "2" << std::endl;
			}
		}
	}
	void Visualizer::SetBoolDoingProcess(bool b) {
		std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
		mbDoingProcess = b;
	}
	bool Visualizer::isDoingProcess() {
		std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
		return mbDoingProcess;
	}
	void Visualizer::ResizeImage(const cv::Mat& src, cv::Mat& dst) {
		cv::resize(src, dst, cv::Size(mnWidth / 2.0, mnHeight / 2.0));
	}
	void Visualizer::SetOutputImage(const cv::Mat& out, int type) {
		std::unique_lock<std::mutex> lockTemp(mMutexOutput);
		mvOutputImgs[type] = out.clone();
		mvOutputChanged[type] = true;
	}
	cv::Mat Visualizer::GetOutputImage(int type) {
		std::unique_lock<std::mutex> lockTemp(mMutexOutput);
		mvOutputChanged[type] = false;
		return mvOutputImgs[type].clone();
	}
	bool Visualizer::isOutputTypeChanged(int type) {
		std::unique_lock<std::mutex> lockTemp(mMutexOutput);
		return mvOutputChanged[type];
	}

	void Visualizer::SetMap(Map* pMap) {
		std::unique_lock<std::mutex> lockTemp(mMutexMap);
		mpMap = pMap;
	}
	Map* Visualizer::GetMap() {
		std::unique_lock<std::mutex> lockTemp(mMutexMap);
		return mpMap;
	}
}