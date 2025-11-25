#ifndef GAUSSIAN_SPARSE_SLAM_H
#define GAUSSIAN_SPARSE_SLAM_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ThreadPool.h>
#include <ConcurrentMap.h>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>

#include <../GaussianSparseSLAM/include/Types.h>

namespace GaussianSparseSLAM {

	class Map;
	class MapManager;
	class Initializer;
	class Tracker;
	class Mapper;
	class Visualizer;
	class User;
	class FeatureScaleInfo;

	class GSSLAM{
	public:
		GSSLAM();
		GSSLAM(ThreadPool::ThreadPool* _pool, int _nlevel = 1, float fscale = 1.2);
		virtual ~GSSLAM();

	public:
		void Init();

		void CreateMap(std::string name, int nq);
		void CreateUser(std::string _user, std::string _map, const cv::Mat& params, std::vector<bool> vbFlags);
		bool CheckMap(std::string str);
		bool CheckUser(std::string str);
		int  CountUser();
		void AddUser(std::string id, User* user);
		void RemoveUser(std::string id);
		User* GetUser(std::string id);
		std::vector<User*> GetAllUsersInMap(std::string map);

		void AddMap(std::string name, Map* pMap);
		Map* GetMap(std::string name);
		void RemoveMap(std::string name);

		int GetConnectedDevice();
		void SetUserVisID(User* user);
		void UpdateUserVisID();

		void InitVisualizer(std::string user, std::string name, int w, int h, bool _bSave = false, int _inc = 20);
		void VisualizeMatchingImage(cv::Mat& res, const cv::Mat& src1, const cv::Mat& src2, const std::vector<std::pair<cv::Point2f, cv::Point2f>>& vecMatches, std::string name, int vid, int inc = 1, cv::Scalar color = cv::Scalar(255, 255, 0));
		void VisualizeImage(std::string mapName, const cv::Mat& src, int vid);

		void SaveTrajectory(std::string path, std::string mapname);
	public:
		ThreadPool::ThreadPool* pool;
		Initializer* mpInitializer;
		Tracker* mpTracker;
		Mapper* mpMapper;
		MapManager* mpMapManager;
		//LoopCloser* mpLoopCloser;
		//LoopCloserV3* mpLoopCloserV3;
		//MapManager* mpMapManager;
		//FeatureTracker* mpFeatureTracker;
		//KeyFrameDB* mpKeyFrameDB;
		Visualizer* mpVisualizer;
		std::thread* mptVisualizer;
		FeatureScaleInfo* mpFeatureScaleInfo;

		ConcurrentMap<std::string, User*> Users;

		//Feature 관련 정보
		float mfScaleFactor;
		int mnLevels;

	private:
		std::mutex mMutexVisID;
		int mnVisID;
	};

}
#endif
