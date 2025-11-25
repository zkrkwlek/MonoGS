#ifndef GAUSSIAN_SPARSE_SLAM_MAPPER_H
#define GAUSSIAN_SPARSE_SLAM_MAPPER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>
#include <ThreadPool.h>

#include <../GaussianSparseSLAM/include/Types.h>


namespace EdgeSLAM {
}

namespace GaussianSparseSLAM {

	class GSSLAM;
	class Map;
	class KeyFrame;

	class Mapper {
	public:
		Mapper() {}
		virtual ~Mapper() {}

	public:
		static void InsertNewKeyFrame(ThreadPool::ThreadPool* pool, GSSLAM* system, Map* map, const std::string& mapName, KeyFrame* targetKF, bool bBA = true);
		static void ProcessMapping(ThreadPool::ThreadPool* pool, GSSLAM* system, const std::string& mapName, KeyFrame* targetKF, bool bBA = true);

		static void ProcessNewKeyFrame(Map* map, KeyFrame* targetKF);
		static void RequestMatchForMapping(KeyFrame* targetKF, const std::string& src);
		static void MapPointCulling(Map* map, KeyFrame* targetKF);
		static void CreateNewMapPoints(Map* map, KeyFrame* targetKF, long long ts);
		static void CreateNewMapPoints(Map* map, KeyFrame* targetKF, std::vector<KeyFrame*>& vpNeighKFs, std::vector<std::vector<std::pair<int, int>>>& vecMatches, long long ts);
		static void SearchInNeighbors(Map* map, KeyFrame* targetKF, std::vector<KeyFrame*>& vpNeighKFs, std::vector<std::vector<std::pair<int, int>>>& vecMatches);
		static void KeyFrameCulling(Map* map, KeyFrame* targetKF);

		static void DownloadMatchInfos(KeyFrame* targetKF, Map* map, std::vector<KeyFrame*>& vpNeighKFs, const std::string& mapName, const std::string& src, std::vector<std::vector<std::pair<int, int>>>& vecMatches);
	public:
		static std::string ip;
		static int port;
		//ConcurrentMap<std::pair<int, int>, std::vector<int, int>> Matches;

		/*std::mutex mMutexStop;
		std::mutex mMutexReset;
		std::mutex mMutexFinish;
		std::mutex mMutexNewKFs;
		bool mbResetRequested;
		bool mbFinishRequested;
		bool mbFinished;
		bool mbAbortBA;
		bool mbStopped;
		bool mbStopRequested;
		bool mbNotStop;*/
	private:
		static void Fuse(Map* map, KeyFrame* pKF1, std::vector<KeyFrame*>& vpNeighKFs, std::vector<std::vector<std::pair<int, int>>>& vecMatches);
	};

}
#endif
