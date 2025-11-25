#include <../GaussianSparseSLAM/include/GSSLAM.h>
#include <../GaussianSparseSLAM/include/Map.h>
#include <../GaussianSparseSLAM/include/KeyFrame.h>
#include <../GaussianSparseSLAM/include/MapManager.h>
#include <../GaussianSparseSLAM/include/Tracker.h>
#include <../GaussianSparseSLAM/include/Initializer.h>
#include <../GaussianSparseSLAM/include/Mapper.h>
#include <../GaussianSparseSLAM/include/Visualizer.h>
#include <../GaussianSparseSLAM/include/User.h>
#include <../GaussianSparseSLAM/include/FeatureScaleInfo.h>

#include <../EdgeSLAM/include/Converter.h>
#include <io.h>
#include <direct.h>


namespace GaussianSparseSLAM {
	GSSLAM::GSSLAM() :pool(), mnLevels(1), mfScaleFactor(1.2)
	{
		Init();
	}
	GSSLAM::GSSLAM(ThreadPool::ThreadPool* _pool, int _nlevel, float fscale) : pool(_pool), mnLevels(_nlevel), mfScaleFactor(fscale){
		Init();
	}
	GSSLAM::~GSSLAM() {
	}

	void GSSLAM::Init(){
		mpInitializer = new Initializer();
		mpTracker = new Tracker();
		mpMapper = new Mapper();
		mpMapManager = new MapManager();

		//feature 정보 관련
		mpFeatureScaleInfo = new FeatureScaleInfo(mnLevels, mfScaleFactor);
		
	}

	void GSSLAM::CreateMap(std::string name, int nq){
		auto pNewMap = mpMapManager->CreateNewMap();
	}
	bool GSSLAM::CheckMap(std::string str) {
		return mpMapManager->CountMaps() > 0;
	}

	void GSSLAM::AddMap(std::string name, Map* pMap) {
		if (CheckUser(name))
			mpMapManager->AddMap(name, pMap);
	}
	Map* GSSLAM::GetMap(std::string name) {
		if (CheckUser(name)) {
			return mpMapManager->GetMap(name);
		}
		return mpMapManager->GetCurrentMap();
	}
	void GSSLAM::RemoveMap(std::string name) {
		if (CheckUser(name))
			mpMapManager->EraseMap(name);
	}
	 
	void GSSLAM::CreateUser(std::string _user, std::string _map, const cv::Mat& params, std::vector<bool> vbFlags){
		auto pNewUser = new User(_user, _map, params, vbFlags);
		auto pMap = mpMapManager->GetCurrentMap();
		pNewUser->SetMap(pMap);
		pNewUser->GetMap()->AddDevice(pNewUser);
		AddMap(_user, pMap);
		AddUser(_user, pNewUser);
	}
	
	bool GSSLAM::CheckUser(std::string str){
		return Users.Count(str) > 0;
	}
	int  GSSLAM::CountUser(){
		return Users.Size();
	}
	void GSSLAM::AddUser(std::string id, User* user){
		Users.Update(id, user);
	}
	void GSSLAM::RemoveUser(std::string id){
		bool bDelete = false;
		{
			if (Users.Count(id)) {
				auto user = Users.Get(id);
				if (user->mbRemoved) {
					std::cout << "Doing removing process = " << id << std::endl;
					return;
				}
				user->mbRemoved = true;
				
				int count = 0;
				while (user->mnUsed > 0) {
					count++;
					if (count %= 200) {
						std::cout << "Tracking status = " << user->mnDebugTrack << std::endl;
						std::cout << "Sematic status = " << user->mnDebugSeg << std::endl;
						std::cout << "AR status = " << user->mnDebugAR << std::endl;
						std::cout << "Label status = " << user->mnDebugLabel << std::endl;
						std::cout << "Plane status = " << user->mnDebugPlane << std::endl;
						break;
					}
					continue;
				}
				RemoveMap(id);
				user->GetMap()->RemoveDevice(user);
				Users.Erase(id);
				delete user;
				bDelete = true;
				std::cout << "Remove user = " << id << std::endl;
			}
		}
	}
	User* GSSLAM::GetUser(std::string id){
		if (Users.Count(id)) {
			auto pUser = Users.Get(id);
			if (!pUser->mbRemoved)
				return pUser;
		}
		return nullptr;
	}
	std::vector<User*> GSSLAM::GetAllUsersInMap(std::string map) {
		std::vector<User*> res;
		if (!CheckMap(map))
			return res;
		////string compare

		std::map<std::string, User*> mapUserLists = Users.Get();
		
		for (auto iter = mapUserLists.begin(), iend = mapUserLists.end(); iter != iend; iter++) {
			auto user = iter->second;
			if (user->mapName != map || user->mbRemoved)
				continue;
			res.push_back(user);
		}
		return res;
	}

	int GSSLAM::GetConnectedDevice(){
		return CountUser();
	}
	void GSSLAM::SetUserVisID(User* user){
		std::unique_lock<std::mutex> lock(mMutexVisID);
		user->SetVisID(mnVisID);
		mnVisID++;
		std::cout << user->userName << "=" << user->GetVisID() << std::endl;
	}
	void GSSLAM::UpdateUserVisID(){
		std::unique_lock<std::mutex> lock(mMutexVisID);
		std::map<std::string, User*> mapUserLists = Users.Get();
		
		mnVisID = 0;
		for (auto iter = mapUserLists.begin(), iend = mapUserLists.end(); iter != iend; iter++) {
			auto user = iter->second;
			/*if (user->mbMapping)
				continue;*/
			user->SetVisID(mnVisID);
			mnVisID++;
		}
	}

	void GSSLAM::InitVisualizer(std::string user, std::string name, int w, int h, bool _bsave, int _inc){
	
		if (w < 1 || h < 1 || w > 10000 || h > 10000)
			return;
		auto pMap = mpMapManager->GetCurrentMap();
		std::cout << "find map " << pMap << std::endl;
		if (pMap && !pMap->mbVisualized) {
			std::cout << "Create Map Visualizer :: START" << std::endl;
			mpVisualizer = new Visualizer(this);
			mpVisualizer->SetScale(400);

			pMap->mpVisualizer = mpVisualizer;
			std::cout << "1 = " << w << " " << h << std::endl;
			pMap->mpVisualizer->Init(w, h, _bsave, _inc);
			std::cout << "2" << std::endl;
			//mptVisualizer = new std::thread(&EdgeSLAM::Visualizer::Run, mpVisualizer);
			new std::thread(&GaussianSparseSLAM::Visualizer::Run, pMap->mpVisualizer);
			std::cout << "3" << std::endl;
			pMap->mpVisualizer->SetMap(pMap);
			std::cout << "4" << std::endl;
			pMap->mpVisualizer->strMapName = name;
			std::cout << "5" << std::endl;
			pMap->mbVisualized = true;
			std::cout << "Create Map Visualizer :: END" << std::endl;
			
		}
	}

	void GSSLAM::VisualizeMatchingImage(cv::Mat& res, const cv::Mat& src1, const cv::Mat& src2, const std::vector<std::pair<cv::Point2f, cv::Point2f>>& vecMatches, std::string name, int vid, int inc, cv::Scalar color)
	{
		cv::Rect uRect(0, 0, src1.cols, src1.rows);
		cv::Rect lRect(0, src1.rows, src1.cols, src1.rows);
		res = cv::Mat::zeros(src1.rows * 2, src1.cols, src1.type());
		src1.copyTo(res(uRect));
		src2.copyTo(res(lRect));
		cv::Point2f lpt(0, src1.rows);
		for (int i = 0; i < vecMatches.size(); i += inc) {
			auto pt1 = vecMatches[i].first;
			auto pt2 = vecMatches[i].second;
			cv::line(res, pt1, pt2 + lpt, color, 3);
			cv::circle(res, pt1, 5, cv::Scalar(0, 0, 255), 2);
			cv::circle(res, pt2 + lpt, 5, cv::Scalar(0, 0, 255), 2);
		}
		if (vid < 0)
			return;
		cv::Mat vis1 = res(uRect);
		cv::Mat vis2 = res(lRect);

		VisualizeImage(name, vis1, vid);
		VisualizeImage(name, vis2, vid + 1);
	}

	void GSSLAM::VisualizeImage(std::string mapName, const cv::Mat& src, int vid) {
		//auto pMap = GetMap(mapName);
		auto pMap = mpMapManager->GetCurrentMap();
		cv::Mat dst;
		pMap->mpVisualizer->ResizeImage(src, dst);
		if (dst.channels() == 1) {
			cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
			dst.convertTo(dst, CV_8UC3);
		}
		pMap->mpVisualizer->SetOutputImage(dst, vid);
	}

	void GSSLAM::SaveTrajectory(std::string path, std::string mapname) {
		{
			auto pMap = GetMap(mapname);
			auto vpKFs = pMap->GetAllKeyFrames();
			std::cout << "save kf = " << vpKFs.size() << std::endl;
			std::ofstream file;
			file.open(path);
			for (int i = 0; i < vpKFs.size(); i++) {
				auto pKF = vpKFs[i];
				if (pKF->isBad())
					continue;
				cv::Mat R = pKF->GetRotation();
				cv::Mat t = pKF->GetTranslation();
				R = R.t(); //inverse
				t = -R * t;  //camera center
				std::vector<float> q = EdgeSLAM::Converter::toQuaternion(R);
				file << std::setprecision(16) << pKF->mdTimeStamp << std::setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
					<< " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
			}
			//file.write(ss.str().c_str(), ss.str().size());
			file.close();
		}
		return;
		auto pUser = this->GetUser(path);
		if (!pUser)
			return;
		if (!pUser->mbSaveTrajectory)
			return;
		pUser->mnUsed++;

		std::stringstream ssPath;
		ssPath << "../bin/trajectory/" << pUser->mapName;

		int resPath = _access(ssPath.str().c_str(), 0);
		if (resPath == -1)
			_mkdir(ssPath.str().c_str());



		int nMapQuality = 70;//MapQuality.Get(pUser->mapName);
		int nUserQuality = pUser->mnQuality;

		if (pUser->mbAsyncTest) {
			ssPath << "/" << pUser->mapName << "_TrackingTest_" << pUser->mnQuality;
		}
		else {
			ssPath << "/" << pUser->mapName << "_" << pUser->mnQuality;
		}
		/*if (nMapQuality != nUserQuality && user->mbDeviceTracking) {
			ssPath << "/" << user->mapName << "_TrackingTest_" << user->mnQuality;
		}
		else {
			ssPath << "/" << user->mapName << "_" << user->mnQuality;
		}*/

		resPath = _access(ssPath.str().c_str(), 0);
		if (resPath == -1)
			_mkdir(ssPath.str().c_str());

		__time64_t long_time;
		_time64(&long_time);
		struct tm newtime;
		_localtime64_s(&newtime, &long_time);

		auto pMap = GetMap(pUser->mapName);
		auto vpKFs = pMap->GetAllKeyFrames();

		std::stringstream ss;
		ss << ssPath.str() << "/" << pUser->userName << "_" << pUser->mnSkip << (pUser->mbMapping ? ("_MAPPING_") : ("_TRACKING_")) << newtime.tm_year + 1900 << "_" << newtime.tm_mon + 1 << "_" << newtime.tm_mday << "_" << newtime.tm_hour << "_" << newtime.tm_min << "_" << newtime.tm_sec << ".txt";
		std::ofstream f;
		f.open(ss.str().c_str());
		f << std::fixed;

		if (pUser->mbMapping) {
			for (int i = 0; i < vpKFs.size(); i++) {
				auto pKF = vpKFs[i];

				cv::Mat R = pKF->GetRotation();
				cv::Mat t = pKF->GetTranslation();
				R = R.t(); //inverse
				t = -R * t;  //camera center
				std::vector<float> q = EdgeSLAM::Converter::toQuaternion(R);
				f << std::setprecision(6) << pKF->mdTimeStamp << std::setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
					<< " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
			}
		}
		else {
			for (int i = 0; i < pUser->vecTrajectories.size(); i += 2) {
				cv::Mat R = pUser->vecTrajectories[i].rowRange(0, 3).colRange(0, 3);
				cv::Mat t = pUser->vecTrajectories[i].rowRange(0, 3).col(3);
				R = R.t(); //inverse
				t = -R * t;  //camera center
				std::vector<float> q = EdgeSLAM::Converter::toQuaternion(R);
				f << std::setprecision(6) << pUser->vecTimestamps[i] << std::setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
					<< " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
			}
		}
		f.close();

		if (pUser->mbDeviceTracking) {
			std::stringstream ss;
			ss << ssPath.str() << "/" << pUser->userName << "_" << pUser->mnSkip << "_DEVICE_" << newtime.tm_year + 1900 << "_" << newtime.tm_mon + 1 << "_" << newtime.tm_mday << "_" << newtime.tm_hour << "_" << newtime.tm_min << "_" << newtime.tm_sec << ".txt";
			std::ofstream f;
			f.open(ss.str().c_str());
			f << std::fixed;

			auto vecTrajectories = pUser->mvDeviceTrajectories.get();
			auto vecTimestamps = pUser->mvDeviceTimeStamps.get();

			for (int i = 0; i < vecTrajectories.size(); i += 10) {

				cv::Mat R = vecTrajectories[i].rowRange(0, 3);
				cv::Mat t = vecTrajectories[i].row(3).t();
				R = R.t(); //inverse
				t = -R * t;  //camera center
				std::vector<float> q = EdgeSLAM::Converter::toQuaternion(R);
				f << std::setprecision(6) << vecTimestamps[i] << std::setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
					<< " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
			}
			f.close();
		}
		pUser->mnUsed--;
	}
}