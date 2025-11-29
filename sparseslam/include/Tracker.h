#ifndef GAUSSIAN_SPARSE_SLAM_TRACKER_H
#define GAUSSIAN_SPARSE_SLAM_TRACKER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>

#include <ThreadPool.h>

#include <../GaussianSparseSLAM/include/Types.h>
#include <../GaussianSparseSLAM/include/Serializer.h>

namespace EdgeSLAM {
}

namespace GaussianSparseSLAM { 
	
	class GSSLAM;
	class User;
	class Map;
	class Mapper;
	class Frame;
	class KeyFrame;
	class GaussianPoint;

	class Tracker {
	public:
		Tracker() {}
		virtual ~Tracker() {}

	public:
		static int KeyFrameMatch(Frame* curr, KeyFrame* frame, std::vector<cv::Point2i>& vecMatches, std::set<int>& sAleadyIdxs, std::set<GaussianPoint*>& sAleadyGPs, float th = 175.0);
		static int FrameMatch(Frame* curr, Frame* frame, std::vector<cv::Point2i>& vecMatches, std::set<int>& sAleadyIdxs, std::set<GaussianPoint*>& sAleadyGPs, float th = 175.0);
		static int TrackWithFrame(ThreadPool::ThreadPool* pool, GSSLAM* system, int id, std::string user, Frame* frame, double ts);
		static int Track(ThreadPool::ThreadPool* pool, GSSLAM* system, int id, std::string user, Frame* frame, double ts);

		static bool NeedNewKeyFrame(Map* map, Frame* cur, KeyFrame* ref, int nMatchesInliers, int nLastKeyFrameId, int nLastRelocFrameID, bool bOnlyTracking = false, int nMaxFrames = 30, int nMinFrames = 0);
		static void CreateNewKeyFrame(ThreadPool::ThreadPool* pool, GSSLAM* system, Map* map, Frame* cur, User* user);

		static bool NeedTemporalKeyFrame(Map* map, Frame* cur, KeyFrame* ref, int nMatchesInliers, int nLastKeyFrameId, int nLastRelocFrameID, bool bOnlyTracking = false, int nMaxFrames = 30, int nMinFrames = 0);
		static void CreateTemporalKeyFrame(ThreadPool::ThreadPool* pool, GSSLAM* system, Map* map, Frame* cur, User* user);

		static void CreateTemporalPointsWithGridDistribution(Frame* f, Map* pMap, std::list<GaussianPoint*>& lpTemporalMPs, std::vector<cv::Point2i>& vMatches, int nMaxPoints = 100);

		static void CreateTemporalPointsWithGridDistribution(KeyFrame* kf, Frame* f, Map* pMap, std::list<GaussianPoint*>& lpTemporalMPs, std::vector<cv::Point2i>& vMatches, int nMaxPoints = 100);

		static void UpdateLocalKeyFrames(Frame* f, std::vector<KeyFrame*>& vpLocalKFs, KeyFrame* pKFmax, int th_kf = 40);
		static void UpdateLocalMapPoitns(Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<GaussianPoint*>& vpLocalMPs, cv::Mat& mLocalDesc, std::vector<LocalKeyframeData>& vecKeyframeDatas, int th_mp = 3000);
	private:
		static int UpdateFrame(Frame* frame, User* user, bool bOnlyTracking = false);
		static int  UpdateVisiblePoints(Frame* cur, std::vector<GaussianPoint*>& vpLocalMPs);
		static int  UpdateFoundPoints(Frame* cur, bool bOnlyTracking = false);
		
	};

}
#endif
