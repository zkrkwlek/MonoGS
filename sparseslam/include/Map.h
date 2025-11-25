#ifndef GAUSSIAN_SPARSE_SLAM_MAP_H
#define GAUSSIAN_SPARSE_SLAM_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>
#include <ConcurrentList.h>

#include <../GaussianSparseSLAM/include/Types.h>

namespace GaussianSparseSLAM {

	class KeyFrame;
	class GaussianPoint;
	class Visualizer;
	class User;

	class Map {
	public:
		Map();
		Map(unsigned long int initKFid = 0, bool bFixScale = false);
		virtual ~Map(){}

		void Delete();
		void Reset();

		void AddDevice(User* _user);
		void RemoveDevice(User* _user);

		//std::vector<GaussianPoint*> GetAllGaussianPoints();
		void AddGaussianPoint(GaussianPoint* pMP);
		void RemoveGaussianPoint(GaussianPoint* pMP);

		void AddKeyFrame(KeyFrame* pF);
		KeyFrame* GetKeyFrame(int id);
		void RemoveKeyFrame(KeyFrame* pF);
		std::vector<KeyFrame*> GetAllKeyFrames();
		std::vector<GaussianPoint*> GetAllPoints();
		int GetNumKeyFrames();

		void InformNewBigChange();
		int GetLastBigChangeIdx();
		
	public:
		bool mbAbortBA;
		std::atomic<int> mnBigChangeIdx;

		ConcurrentSet<User*> mSetConnectedDevices;
		//ConcurrentSet<GaussianPoint*> mspGaussianPoints;
		ConcurrentMap<int, GaussianPoint*> mmpGaussianPoints;
		ConcurrentMap<int, KeyFrame*> mmpKeyFrames;

		std::atomic<int> mnNumMappingFrames, mnNumLoopClosingFrames;
		//std::list<GaussianPoint*> mlpNewMPs;
		ConcurrentList<GaussianPoint*> mlpNewMPs;
		std::atomic<LoopClosingState> mLCState;

		std::vector<KeyFrame*> mvpKeyFrameOrigins;
		std::atomic<bool> mbResetRequested;
		std::atomic<bool> mbFinishRequested;
		std::atomic<bool> mbFinished;
		std::atomic<bool> mbStopped;
		std::atomic<bool> mbStopRequested;
		std::atomic<bool> mbNotStop;
		std::atomic<long unsigned int> mnLastKeyFrameID;
	public:
		long unsigned int mnId;
		static std::atomic<long unsigned int> mnNextKeyFrameID, mnNextGaussianPointID, mnNextMapID;

		std::atomic<bool> mbVisualized;
		Visualizer* mpVisualizer;

	
	
	public:
		//local mapper thread
		void InterruptBA() {
			mbAbortBA = true;
		}
		void RequestStop();
		bool stopRequested();
		bool Stop();
		bool isStopped();
		void Release();
		bool SetNotStop(bool flag);

		void RequestReset();
		void ResetIfRequested();

		bool CheckFinish();
		void SetFinish();
		void RequestFinish();
		bool isFinished();

		//loop closing
		bool isRunningGBA();
		bool isFinishedGBA();
		
		KeyFrame* mpMatchedKF;
		std::vector<ConsistentGroup> mvConsistentGroups;
		std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
		std::vector<KeyFrame*> mvpCurrentConnectedKFs;
		std::vector<GaussianPoint*> mvpCurrentMatchedPoints;
		std::vector<GaussianPoint*> mvpLoopMapPoints;
		cv::Mat mScw;
		BaseSLAM::Optimization::Sim3 mg2oScw;

		std::atomic<int> mnLastLoopKFid;
		std::atomic<bool> mbRunningGBA;
		std::atomic<bool> mbFinishedGBA;
		bool mbStopGBA; //for optimization
		std::mutex mMutexGBA;
		std::thread* mpThreadGBA;
		// Fix scale in the stereo/RGB-D case
		bool mbFixScale;
		int mnFullBAIdx;
		////Loop Closing
	public:
		BaseSLAM::Optimization::Sim3 mg2oLoopSlw, mg2oLoopScw;
		int mnLoopNumCoincidences;
		int mnLoopNumNotFound;
		KeyFrame* mpLoopLastCurrentKF;
		KeyFrame* mpLoopMatchedKF;
		std::vector<GaussianPoint*> mvpLoopMPs;
		std::vector<GaussianPoint*> mvpLoopMatchedMPs;
	public:

		//MapManager
	public:
		MapState GetState();
		void SetState(MapState stat);

		//V3
		long unsigned int GetId();
		void ChangeId(long unsigned int nId);
		long unsigned int GetInitKFid();
		void SetInitKFid(long unsigned int initKFif);
		long unsigned int GetMaxKFid();
		KeyFrame* GetOriginKF();
		void SetCurrentMap();
		void SetStoredMap();
		void SetBad();
		bool IsBad();
		bool IsInUse();

		void clear();
		int GetMapChangeIndex();
		void IncreaseChangeIndex();
		int GetLastMapChange();
		void SetLastMapChange(int currentChangeId);
	protected:
		std::atomic<long unsigned int> mnInitKFid, mnMaxKFid;
		std::atomic<int> mnMapChange, mnMapChangeNotified;

		std::atomic<bool> mbBad, mIsInUse;
		KeyFrame* mpKFinitial;
		KeyFrame* mpFirstRegionKF;
	public:
		std::mutex mMutexMapUpdate;
	private:
		MapState mState;
		std::mutex mMutexState, mMutexMap;
	};
	
}
#endif
