#ifndef GAUSSIAN_SPARSE_SLAM_MAP_MANAGER_H
#define GAUSSIAN_SPARSE_SLAM_MAP_MANAGER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <ConcurrentSet.h>
#include <ConcurrentMap.h>
#include <atomic>

namespace GaussianSparseSLAM {
	class Map;
	class MapManager {
	public:
		MapManager();
		MapManager(int initKFid);
		virtual ~MapManager();
	public:
		void AddMap(std::string src, Map* pMap) {
			mapDeviceMaps.Update(src, pMap);
		}
		Map* GetMap(std::string src) {
			if (mapDeviceMaps.Count(src))
				return mapDeviceMaps.Get(src);
			return nullptr;
		}
		void EraseMap(std::string src) {
			if (mapDeviceMaps.Count(src))
				mapDeviceMaps.Erase(src);
		}
		Map* GetCurrentMap();
		int CountMaps() {
			return mspMaps.Size();
		}
		Map* CreateNewMap();
		void ChangeMap(Map* pMap);
		void SetMapBad(Map* pMap);
		void RemoveBadMaps();
		std::set<Map*> GetAllMaps() {
			return mspMaps.Get();
		}
	protected:
		ConcurrentMap<std::string, Map*> mapDeviceMaps;
		ConcurrentSet<Map*> mspMaps;
		ConcurrentSet<Map*> mspBadMaps;
		Map* mpCurrentMap;
		std::atomic<unsigned long int> mnLastInitKFidMap;

		std::mutex mMutexAtlas;
	};
}

#endif