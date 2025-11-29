#include <../GaussianSparseSLAM/include/Tracker.h>

#include <../GaussianSparseSLAM/include/Frame.h>
#include <../GaussianSparseSLAM/include/KeyFrame.h>
#include <../GaussianSparseSLAM/include/GSSLAM.h>
#include <../GaussianSparseSLAM/include/User.h>
#include <../GaussianSparseSLAM/include/Map.h>
#include <../GaussianSparseSLAM/include/GaussianPoint.h>

#include <../GaussianSparseSLAM/include/Optimizer.h>
#include <../GaussianSparseSLAM/include/Mapper.h>

#include <../EdgeSLAM/include/Camera.h>
#include <Utils_Geometry.h>

namespace GaussianSparseSLAM {

	struct PointInfo {
		int idx;      // feature index
		float depth;  // depth value
		int gridX;    // grid cell X
		int gridY;    // grid cell Y
	};

	int Tracker::TrackWithFrame(ThreadPool::ThreadPool* pool, GSSLAM* system, int id, std::string user, Frame* frame, double ts){
		int nopt = Optimizer::PoseOptimization(frame);
		return nopt;
	}

	int CheckMatchesReferenceKeyFrame(KeyFrame* ref, Map* map) {
		const int nKFs = map->GetNumKeyFrames();
		int nMinObs = 3;
		if (nKFs <= 2)
			nMinObs = nKFs;
		int nRefMatches = ref->TrackedGaussianPoints(nMinObs);
		return nRefMatches;
	}

	int Tracker::Track(ThreadPool::ThreadPool* pool, GSSLAM* system, int id, std::string user, Frame* frame, double ts) {
		auto pUser = system->GetUser(user);
		if (!pUser)
			return 0;
		if (pUser->mbProgress)
			return 0;
		
		pUser->mbProgress = true;

		pUser->Reset();
		auto cam = pUser->mpCamera;
		auto map = pUser->GetMap();

		std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);

		bool bTrack;
		std::vector<GaussianPoint*> vecGPs;
		UpdateVisiblePoints(frame, vecGPs);
		int nOpt = Optimizer::PoseOptimization(frame);
		int nInliers = UpdateFrame(frame, pUser);

		if (frame->mnFrameID < pUser->mnLastRelocFrameId + pUser->mpCamera->mnMaxFrame && nInliers < 50) {
			bTrack = false;
		}
		else if (nInliers < 30) {
			bTrack = false;
		}
		else {
			bTrack = true;
		}

		auto mapState = map->GetState();
		auto userState = pUser->GetState();
		auto trackState = UserState::NotEstimated;
		
		if (!bTrack) {

			if (userState == UserState::Success || userState == UserState::NotEstimated) {
				pUser->mLostTimeStamp = frame->mdTimeStamp;
				trackState = UserState::RECENTLY_LOST;
			}
			if (userState == UserState::RECENTLY_LOST)
			{
				auto diff = frame->mdTimeStamp - pUser->mLostTimeStamp;
				//euroc ts용 임시.
				if (diff > 100000.0)
					diff *= 1e-9;
				if (diff > 5.0) {
					trackState = UserState::Failed;
				}
				else {
					trackState = UserState::RECENTLY_LOST;
				}
			}

			nInliers = 0;
			cv::Mat Rt = cv::Mat::eye(4, 4, CV_32FC1);
			frame->SetPose(Rt);

			if (userState == UserState::Failed) {
				////맵 상태 체크
				auto MAP = pUser->GetMap();
				LoopClosingState stat = map->mLCState;
				//if (bErrMsg)
					//std::cout << "Track::MapTest::" << MAP->mnId << " " << (int)stat << " " << nInliers << " " << pLocalMap->mvpLocalKFs.size() << " " << pLocalMap->mvpLocalMPs.size() << std::endl;
				if (MAP->GetNumKeyFrames() < 10) {
					//if (bErrMsg)
					//	std::cout << "MAP RESET~~~~" << std::endl;
					//system->mpKeyFrameDB->clearMap(MAP);
					MAP->Reset();
				}
				else {
					//CreateNewMap(system, user);
				}
			}
		}
		if (bTrack) {
			trackState = UserState::Success;
		}

		//remove temporal mps
		auto vpCurrGPs = frame->mvGaussianPoints.get();
		for (int i = 0; i < frame->N; i++)
		{
			auto pMP = vpCurrGPs[i];
			if (pMP)
				if (pMP->Observations() < 1)
				{
					frame->mvbOutliers[i] = false;
					frame->mvGaussianPoints.update(i, nullptr);
				}
		}

		pUser->SetState(trackState);
		if (trackState == UserState::Success) {

			//pose update
			cv::Mat T = frame->GetPose();
			cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
			cv::Mat   t = T.rowRange(0, 3).col(3);
			R.convertTo(R, CV_64FC1);
			t.convertTo(t, CV_64FC1);
			//pUser->mpKalmanFilter->fillMeasurements(t, R);
			//pUser->mpKalmanFilter->updateKalmanFilter(t, R);
			pUser->UpdatePose(T, ts);
			pUser->PoseDatas.Update(id, T);

			////check keyframe
			if (pUser->mbMapping && pUser->mpRefKF) {
				if (Tracker::NeedNewKeyFrame(map, frame, pUser->mpRefKF, nInliers, map->mnLastKeyFrameID.load(), pUser->mnLastRelocFrameId.load(), false, pUser->mpCamera->mnMaxFrame, pUser->mpCamera->mnMinFrame)) {
					Tracker::CreateNewKeyFrame(pool, system, map, frame, pUser);
					frame->SetState(FrameProcessStatus::mapping);
					frame->mbKF = true;
				}
				//else if (Tracker::NeedTemporalKeyFrame(map, frame, pUser->mpRefKF, nInliers, map->mnLastKeyFrameID.load(), pUser->mnLastRelocFrameId.load(), false, pUser->mpCamera->mnMaxFrame, pUser->mpCamera->mnMinFrame))
				//{
				//	//단일 키프레임 생성
				//	Tracker::CreateTemporalKeyFrame(pool, system, map, frame, pUser);
				//	frame->SetState(FrameProcessStatus::mapping);
				//	frame->mbKF = true;
				//}
				if (frame->mbKF)
				{
					std::cout << "New KeyFrame Generation = " << frame->mnFrameID << ", " << frame->mnKeyFrameId << std::endl;
				}
			}
			//std::cout << "tracking success " << id <<" "<<nInliers << std::endl;
		}
		else{
			//std::cout << "tracking failed " << id << std::endl;
			pUser->PoseDatas.Update(id, cv::Mat());
		}

		pUser->mbProgress = false;
		//int tempID = user->mnPrevFrameID;
		//if (mapState == MapState::Initialized && pUser->prevFrame)// && pUser->prevFrame->mnShared == 0)
		//	delete pUser->prevFrame;
		//pUser->prevFrame = frame;
		//pUser->mnDebugTrack--;
		//pUser->mnUsed--;
		return nInliers;
	}

	bool Tracker::NeedTemporalKeyFrame(Map* map, Frame* cur, KeyFrame* ref, int nMatchesInliers, int nLastKeyFrameId, int nLastRelocFrameID, bool bOnlyTracking, int nMaxFrames, int nMinFrames) {
		
		if (bOnlyTracking)
			return false;

		if (map->isStopped() || map->stopRequested()) {
			return false;
		}
		const int nKFs = map->GetNumKeyFrames();

		if (cur->mnFrameID<nLastRelocFrameID + nMaxFrames && nKFs>nMaxFrames) {
			return false;
		}
		
		int nMinObs = 3;
		if (nKFs <= 2)
			nMinObs = nKFs;
		int nRefMatches = ref->TrackedGaussianPoints(nMinObs);

		bool bLocalMappingIdle = map->mnNumMappingFrames == 0;
		const bool c1c = nMatchesInliers < nRefMatches * 0.25;

		if (c1c && !bLocalMappingIdle)
			return true;
		return false; 

	}

	bool Tracker::NeedNewKeyFrame(Map* map, Frame* cur, KeyFrame* ref, int nMatchesInliers, int nLastKeyFrameId, int nLastRelocFrameID, bool bOnlyTracking, int nMaxFrames, int nMinFrames) {
		if (bOnlyTracking)
			return false;

		// If Local Mapping is freezed by a Loop Closure do not insert keyframes
		if (map->isStopped() || map->stopRequested()) {
			return false;
		}
		const int nKFs = map->GetNumKeyFrames();

		// Do not insert keyframes if not enough frames have passed from last relocalisation
		if (cur->mnFrameID<nLastRelocFrameID + nMaxFrames && nKFs>nMaxFrames) {
			return false;
		}

		// Tracked MapPoints in the reference keyframe
		int nMinObs = 3; 
		if (nKFs <= 2)
			nMinObs = nKFs;
		int nRefMatches = ref->TrackedGaussianPoints(nMinObs);

		// Local Mapping accept keyframes?
		bool bLocalMappingIdle = map->mnNumMappingFrames == 0;
		// Thresholds
		float thRefRatio = 0.9f;

		int nNonTrackedClose = 0;
		int nTrackedClose = 0;

		bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

		// Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
		const bool c1a = cur->mnFrameID >= (nLastKeyFrameId + nMaxFrames);
		// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
		const bool c1b = (cur->mnFrameID >= (nLastKeyFrameId + nMinFrames)) && bLocalMappingIdle;
		//Condition 1c: tracking is weak
		//const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose);
		const bool c1c = nMatchesInliers < nRefMatches * 0.25;
		// Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
		const bool c1 = c1a || c1b || c1c;
		const bool c2 = (nMatchesInliers < nRefMatches* thRefRatio) || bNeedToInsertClose;
		//레퍼런스 키프레임이 강건하지 않은 경우
		//const bool c3 = nRefMatches < 300;// || nMatchesInliers < 300;
		//std::cout << "need test = " << c1<< " " << c2 << " " << bLocalMappingIdle <<" ||"<<nMatchesInliers<<" "<<nRefMatches << std::endl;
		if (c1 && (c2))
		{
			// If the mapping accepts keyframes, insert keyframe.
			// Otherwise send a signal to interrupt BA
			if (bLocalMappingIdle)
			{
				return true;
			}
			//std::cout << "InterruptBA" << std::endl;
			map->InterruptBA();
			/*if (map->mnNumMappingFrames < 3)
				return true;*/
			return false;
		}
		return false;

	}
	void Tracker::CreateTemporalKeyFrame(ThreadPool::ThreadPool* pool, GSSLAM* system, Map* map, Frame* cur, User* user) {
		if (!map->SetNotStop(true))
			return;
		KeyFrame* pKF = new KeyFrame(cur, map);
		pKF->sourceName = user->userName;
		pKF->mbSendLocalMap = user->mbBaseLocalMap;
		
		pKF->mbBA = false;
		pKF->mbFastGP = true;
		pool->EnqueueJob(Mapper::InsertNewKeyFrame, pool, system, map, user->mapName, pKF, false);

		map->SetNotStop(false);
		user->KeyFrames.Update(cur->mnFrameID, pKF);
	}
	void Tracker::CreateNewKeyFrame(ThreadPool::ThreadPool* pool, GSSLAM* system, Map* map,Frame* cur, User* user) {
		//새로운 키프레임 생성
		//매칭 요청 + 새로운 키프레임 + 옵저베이션
		//이 키프레임을 매퍼에서 읽을 방법 정리
		if (!map->SetNotStop(true))
			return;
		KeyFrame* pKF = new KeyFrame(cur, map);
		pKF->sourceName = user->userName;
		pKF->mbSendLocalMap = user->mbBaseLocalMap;
		//user->mpRefKF = pKF;
		//user->mbNewKF = true;
		//request matches

		//temporal depth update
		//depth check
		pKF->mbBA = true;
		pKF->mbFastGP = true;
		pool->EnqueueJob(Mapper::InsertNewKeyFrame, pool, system, map, user->mapName, pKF, false);

		map->SetNotStop(false);
		user->KeyFrames.Update(cur->mnFrameID, pKF);
	}

	int Tracker::UpdateFrame(Frame* cur, User* user, bool bOnlyTracking) {
		//트래킹 후에 found, reference frame을 한번에 찾기.
		auto vpGPs = cur->mvGaussianPoints.get();
		std::map<KeyFrame*, int> keyframeCounter;
		int nFrameID = cur->mnFrameID;
		int nres = 0;
		int ntemp1 = 0;
		int ntemp2 = 0;
		for (int i = 0; i < cur->N; i++)
		{
			auto pMP = vpGPs[i];
			if (pMP && !pMP->isBad() && !cur->mvbOutliers[i]) {
				pMP->IncreaseFound();
				if (!bOnlyTracking)
				{
					if (pMP->Observations() > 0)
						nres++;
					if (pMP->Observations() == 0)
						ntemp1++;
					if (pMP->Observations() == 1)
						ntemp2++;
				}
				else
					nres++;
				const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();
				for (std::map<KeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
					keyframeCounter[it->first]++;
			}
			else {
				cur->mvGaussianPoints.update(i, nullptr);
				cur->mvbOutliers[i] = false;
			}
		}
		if (user->mpRefKF)
		{
			std::cout << "tracking test = " <<cur->mnFrameID<<" = " << ntemp1 <<", "<<ntemp2 << ", " << nres << " = " << CheckMatchesReferenceKeyFrame(user->mpRefKF, user->GetMap()) << " = " << user->GetMap()->mnLastKeyFrameID << std::endl;
		}
		
		//Reference Keyframe 찾기
		if (keyframeCounter.empty())
			return nres;

		int max = 0;
		KeyFrame* pKFmax = static_cast<KeyFrame*>(nullptr);
		// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
		for (std::map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
		{
			KeyFrame* pKF = it->first;

			if (pKF->isBad())
				continue;

			if (it->second > max)
			{
				max = it->second;
				pKFmax = pKF;
			}
		}
		if (pKFmax && !pKFmax->isBad())
		{
			user->mpRefKF = pKFmax;
		}
		//std::cout << "kf test = " << keyframeCounter.size() << std::endl;
		return nres;
	}

	//local map 구축 후 거기서 visible 체크
	int Tracker::UpdateVisiblePoints(Frame* cur, std::vector<GaussianPoint*>& vpLocalMPs){
	
		auto vpGPs = cur->mvGaussianPoints.get();
		//std::set<GaussianPoint*> spGPs;
		for (int i = 0; i < cur->N; i++)
		{
			if (vpGPs[i])
			{
				auto pMP = vpGPs[i];

				if (!pMP || pMP->isBad() || cur->mvbOutliers[i]) {
					cur->mvGaussianPoints.update(i, nullptr);
					cur->mvbOutliers[i] = false;
				}
				else {
					pMP->IncreaseVisible();
				}
				//spGPs.insert(pMP);
			}
		}
		int nToMatch = 0;
		return nToMatch;
	}
	int Tracker::UpdateFoundPoints(Frame* cur, bool bOnlyTracking) {
		int nres = 0;
		auto vpGPs = cur->mvGaussianPoints.get();
		for (int i = 0; i < cur->N; i++)
		{
			auto pMPi = vpGPs[i];
			if (pMPi)
			{
				if (!cur->mvbOutliers[i])
				{
					pMPi->IncreaseFound();
					if (!bOnlyTracking)
					{
						if (pMPi->Observations() > 0)
							nres++;
					}
					else
						nres++;
				}
				//싱글 프레임에서 임시로 생성한 포인트 제거
				/*if (pMPi->Observations() < 1) {
					cur->mvbOutliers[i] = false;
					cur->mvpMapPoints[i] = nullptr;
				}*/
			}
		}
		return nres;
	}
	int Tracker::KeyFrameMatch(Frame* curr, KeyFrame * kf, std::vector<cv::Point2i>&vecMatches, std::set<int>& sAleadyIdxs, std::set<GaussianPoint*>& sAleadyGPs, float th) {
		cv::Mat R = curr->GetRotation();
		cv::Mat t = curr->GetTranslation();
		cv::Mat K = curr->K.clone();
		
		int n = 0;
		int nFailGP = 0;
		int nAlready = 0;
		int nProjection = 0;
		auto vecPrevGPs = kf->mvpMapPoints.get();
		for (int i = 0; i < vecMatches.size(); i++) {
			int pidx1 = vecMatches[i].x;
			int pidx2 = vecMatches[i].y;

			auto pPrevGP = vecPrevGPs[pidx1];
			if (!pPrevGP || pPrevGP->isBad()) {
				nFailGP++;
				continue;
			}

			if (sAleadyGPs.count(pPrevGP)){
				nAlready++;
				continue;
			}
			if (sAleadyIdxs.count(pidx2))
				continue;
			n++;
			sAleadyIdxs.insert(pidx2);
			sAleadyGPs.insert(pPrevGP);
			curr->mvGaussianPoints.update(pidx2, pPrevGP);

			/*cv::Point2f pt;
			float d;
			if (!CommonUtils::Geometry::ProjectPoint(pt, d, pPrevGP->GetWorldPos(), K, R, t))
				continue;
			cv::Point2f diff = pt - curr->mvKeysUn[pidx2];
			d = diff.x * diff.x + diff.y * diff.y;
			if (d < th){
				sAleadyGPs.insert(pPrevGP);
				curr->mvGaussianPoints.update(pidx2, pPrevGP);
				n++;
			}
			else {
				nProjection++;
			}*/
		}
		//std::cout << "keyframe match = " <<kf->mnId<<" = " << n << " " << vecMatches.size() <<"= "<<nFailGP<<" "<<nAlready<<" "<<nProjection << std::endl;
	}
	int Tracker::FrameMatch(Frame* curr, Frame* prev, std::vector<cv::Point2i>& vecMatches, std::set<int>& sAleadyIdxs, std::set<GaussianPoint*>& sAleadyGPs, float th){
		cv::Mat R = curr->GetRotation();
		cv::Mat t = curr->GetTranslation();
		cv::Mat K = curr->K.clone();

		prev->check_replaced_map_points();

		int n = 0;
		int nFailGP = 0;
		int nAlready = 0;
		int nProjection = 0;
		auto vecPrevGPs = prev->mvGaussianPoints.get();
		for (int i = 0; i < vecMatches.size(); i++) {
			int pidx1 = vecMatches[i].x;
			int pidx2 = vecMatches[i].y;

			auto pPrevGP = vecPrevGPs[pidx1];
			if (!pPrevGP || pPrevGP->isBad() || prev->mvbOutliers[pidx1]){
				nFailGP++;
				continue;
			}
			if (sAleadyGPs.count(pPrevGP)){
				nAlready++;
				continue;
			}
			if (sAleadyIdxs.count(pidx2))
				continue;
			n++;
			curr->mvGaussianPoints.update(pidx2, pPrevGP);
			sAleadyGPs.insert(pPrevGP);
			sAleadyIdxs.insert(pidx2);

			/*cv::Point2f pt;
			float d;
			if (!CommonUtils::Geometry::ProjectPoint(pt, d, pPrevGP->GetWorldPos(), K, R, t))
				continue;
			cv::Point2f diff = pt - curr->mvKeysUn[pidx2];
			d = diff.x * diff.x + diff.y * diff.y;
			if(d < th){
				curr->mvGaussianPoints.update(pidx2, pPrevGP);
				sAleadyGPs.insert(pPrevGP);
				n++;
			}
			else {
				nProjection++;
			}*/
		}
		//std::cout << "frame match = " << n <<" "<<vecMatches.size() << "= " << nFailGP << " " << nAlready << " " << nProjection << std::endl;
	}

	void Tracker::UpdateLocalKeyFrames(Frame* f, std::vector<KeyFrame*>& vpLocalKFs, KeyFrame* pKFmax, int th_kf) {
		// Each map point vote for the keyframes in which it has been observed
		std::set<KeyFrame*> spLocalKFs;
		std::map<KeyFrame*, int> keyframeCounter;
		int nFrameID = f->mnFrameID;
		auto mvpMapPoints = f->mvGaussianPoints.get();
		for (int i = 0; i < f->N; i++)
		{
			auto pMP = mvpMapPoints[i];
			if (pMP && !pMP->isBad()) {
 				const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();
				for (std::map<KeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
					keyframeCounter[it->first]++;
			}
			else {
				f->mvGaussianPoints.update(i,nullptr);
			}
		}
		if (keyframeCounter.empty())
			return;

		int max = 0;
		pKFmax = static_cast<KeyFrame*>(nullptr);

		// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
		for (std::map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
		{
			KeyFrame* pKF = it->first;

			if (pKF->isBad())
				continue;

			if (it->second > max)
			{
				max = it->second;
				pKFmax = pKF;
			}

			vpLocalKFs.push_back(it->first);
			spLocalKFs.insert(pKF);
		}

		// Include also some not-already-included keyframes that are neighbors to already-included keyframes
		//for (std::vector<KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		for (size_t i = 0, iend = vpLocalKFs.size(); i < iend; i++)
		{
			// Limit the number of keyframes
			if (vpLocalKFs.size() > th_kf)
				break;

			KeyFrame* pKF = vpLocalKFs[i];// *itKF;

			const std::vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

			for (std::vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
			{
				KeyFrame* pNeighKF = *itNeighKF;
				if (pNeighKF && !pNeighKF->isBad() && !spLocalKFs.count(pNeighKF))
				{
					spLocalKFs.insert(pNeighKF);
					break;

				}
			}

			const std::set<KeyFrame*> spChilds = pKF->GetChilds();
			for (std::set<KeyFrame*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
			{
				KeyFrame* pChildKF = *sit;
				if (pChildKF && !pChildKF->isBad() && !spLocalKFs.count(pChildKF))
				{
					spLocalKFs.insert(pChildKF);
					break;
				}
			}

			KeyFrame* pParent = pKF->GetParent();
			if (pParent && !pParent->isBad() && !spLocalKFs.count(pParent))
			{
				spLocalKFs.insert(pParent);
				break;
			}

		}
		if (!pKFmax && pKFmax->isBad())
			pKFmax = static_cast<KeyFrame*>(nullptr);
	}
	//
	void Tracker::UpdateLocalMapPoitns(Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<GaussianPoint*>& vpLocalMPs, cv::Mat& mLocalDesc, std::vector<LocalKeyframeData>& vecKeyframeDatas, int th_mp) {
		std::set<GaussianPoint*> spMPs;
		mLocalDesc = cv::Mat::zeros(0, 64, CV_32FC1);
		for (std::vector<KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			if (spMPs.size() >= th_mp)
				break;
			KeyFrame* pKF = *itKF;
			const std::vector<GaussianPoint*> vpMPs = pKF->GetGaussianPointMatches();

			LocalKeyframeData tmp;
			tmp.user = pKF->sourceName;
			tmp.keyframe_id = (int16_t)pKF->mnFrameId;

			for (int i = 0; i < vpMPs.size(); i++)
			{
				auto pMP = vpMPs[i];

				if (!pMP || pMP->isBad() || spMPs.count(pMP))
					continue;
				vpLocalMPs.push_back(pMP);
				//vpLocalTPs.push_back(new TrackPoint());
				spMPs.insert(pMP);
				//mLocalDesc.push_back(pKF->mDescriptors.row(i));

				tmp.indices.push_back((int16_t)i);
				tmp.mp_ids.push_back(pMP->mnId);
			}
			if (tmp.indices.size() > 0)
			{
				vecKeyframeDatas.push_back(tmp);
			}
		}
	}

	void Tracker::CreateTemporalPointsWithGridDistribution(KeyFrame* kf, Frame* f, Map* pMap, std::list<GaussianPoint*>& lpTemporalMPs, std::vector<cv::Point2i>& vMatches, int nMaxPoints){
		int N = vMatches.size();

		std::vector<PointInfo> vCandidates;
		vCandidates.reserve(vMatches.size());

    	auto vpPrevGPs = kf->mvpMapPoints.get();

		for (int i = 0; i < N; i++)
		{
			int idx = vMatches[i].x;

			auto pGP = vpPrevGPs[idx];

			// 이미 맵포인트가 있거나 매칭이 안된 경우 제외
			if (pGP && !pGP->isBad() && pGP->Observations() >= 1)
				continue;

			float z = kf->mvDepth[idx];
			if (z > 0)  // valid depth
			{
				PointInfo pi;
				pi.idx = idx;
				pi.depth = z;

				// 그리드 셀 계산
				const cv::KeyPoint& kp = f->mvKeysUn[idx];
				pi.gridX = round((kp.pt.x - f->mnMinX) * f->mfGridElementWidthInv);
				pi.gridY = round((kp.pt.y - f->mnMinY) * f->mfGridElementHeightInv);

				if (pi.gridX >= 0 && pi.gridX < f->FRAME_GRID_COLS && pi.gridY >= 0 && pi.gridY < f->FRAME_GRID_ROWS)
				{
					vCandidates.push_back(pi);
				}
			}
		}
		//std::cout << "Temporal Candidates = " << vMatches.size() << ", " << vCandidates.size() << std::endl;
		if (vCandidates.empty())
			return;

		std::sort(vCandidates.begin(), vCandidates.end(),
			[](const PointInfo& a, const PointInfo& b) {
				return a.depth < b.depth;
			});

		// 3. 그리드 분산을 고려하여 선택
		std::vector<std::vector<bool>> gridOccupied(f->FRAME_GRID_ROWS, std::vector<bool>(f->FRAME_GRID_COLS, false));
		std::vector<int> selectedIndices;
		selectedIndices.reserve(nMaxPoints);

		// 첫 패스: 비어있는 그리드에 우선 배치
		for (const auto& pi : vCandidates)
		{
			if (selectedIndices.size() >= nMaxPoints)
				break;

			if (!gridOccupied[pi.gridY][pi.gridX])
			{
				selectedIndices.push_back(pi.idx);
				gridOccupied[pi.gridY][pi.gridX] = true;
			}
		}

		// 두 번째 패스: 아직 maxPoints에 도달 안 했으면 나머지 추가
		if (selectedIndices.size() < nMaxPoints)
		{
			for (const auto& pi : vCandidates)
			{
				if (selectedIndices.size() >= nMaxPoints)
					break;

				if (std::find(selectedIndices.begin(), selectedIndices.end(), pi.idx)
					== selectedIndices.end())
				{
					selectedIndices.push_back(pi.idx);
				}
			}
		}

		cv::Mat Rcw1 = kf->GetRotation();
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat twc1 = kf->GetCameraCenter();

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		for (int idx : selectedIndices)
		{
			cv::Mat x3D;
			x3D = f->UnprojectStereo(idx, Rwc1, twc1);
			GaussianSparseSLAM::GaussianPoint* pMP = new GaussianSparseSLAM::GaussianPoint(x3D, kf, pMap, ts);

			kf->mvpMapPoints.update(idx, pMP);
			f->mvGaussianPoints.update(idx, pMP);
			lpTemporalMPs.push_back(pMP);

			pMP->AddObservation(kf, idx);

			pMP->ComputeDistinctiveDescriptors();
			pMP->UpdateNormalAndDepth();
			pMap->AddGaussianPoint(pMP);
			pMap->mlpNewMPs.push_back(pMP);
		}
	}

	void Tracker::CreateTemporalPointsWithGridDistribution(Frame* f, Map* pMap, std::list<GaussianPoint*>& lpTemporalMPs, std::vector<cv::Point2i>& vMatches, int nMaxPoints) {
		
		int N = vMatches.size();

		std::vector<PointInfo> vCandidates;
		vCandidates.reserve(vMatches.size());

		auto vpPrevGPs = f->mvGaussianPoints.get();

		for (int i = 0; i < N; i++)
		{
			int idx = vMatches[i].x;

			auto pGP = vpPrevGPs[idx];

			// 이미 맵포인트가 있거나 매칭이 안된 경우 제외
			if (pGP && !pGP->isBad() && pGP->Observations() >= 1 && !f->mvbOutliers[idx])
				continue;

			float z = f->mvDepth[idx];
			if (z > 0)  // valid depth
			{
				PointInfo pi;
				pi.idx = idx;
				pi.depth = z;

				// 그리드 셀 계산
				const cv::KeyPoint& kp = f->mvKeysUn[idx];
				pi.gridX = round((kp.pt.x - f->mnMinX) * f->mfGridElementWidthInv);
				pi.gridY = round((kp.pt.y - f->mnMinY) * f->mfGridElementHeightInv);
				 
				if (pi.gridX >= 0 && pi.gridX < f->FRAME_GRID_COLS && pi.gridY >= 0 && pi.gridY < f->FRAME_GRID_ROWS)
				{
					vCandidates.push_back(pi);
				}
			}
		}
		//std::cout <<"Temporal Candidates = " << vMatches.size()<<", " << vCandidates.size() << std::endl;
		if (vCandidates.empty())
			return;

		std::sort(vCandidates.begin(), vCandidates.end(),
			[](const PointInfo& a, const PointInfo& b) {
				return a.depth < b.depth;
			});

		// 3. 그리드 분산을 고려하여 선택
		std::vector<std::vector<bool>> gridOccupied(f->FRAME_GRID_ROWS, std::vector<bool>(f->FRAME_GRID_COLS, false));
		std::vector<int> selectedIndices;
		selectedIndices.reserve(nMaxPoints);

		// 첫 패스: 비어있는 그리드에 우선 배치
		for (const auto& pi : vCandidates)
		{
			if (selectedIndices.size() >= nMaxPoints)
				break;

			if (!gridOccupied[pi.gridY][pi.gridX])
			{
				selectedIndices.push_back(pi.idx);
				gridOccupied[pi.gridY][pi.gridX] = true;
			}
		}

		// 두 번째 패스: 아직 maxPoints에 도달 안 했으면 나머지 추가
		if (selectedIndices.size() < nMaxPoints)
		{
			for (const auto& pi : vCandidates)
			{
				if (selectedIndices.size() >= nMaxPoints)
					break;

				if (std::find(selectedIndices.begin(), selectedIndices.end(), pi.idx)
					== selectedIndices.end())
				{
					selectedIndices.push_back(pi.idx);
				}
			}
		}

		cv::Mat Rcw1 = f->GetRotation();
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat twc1 = f->GetCameraCenter();

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		for (int idx : selectedIndices)
		{
			cv::Mat x3D;
			x3D = f->UnprojectStereo(idx, Rwc1, twc1);
			GaussianSparseSLAM::GaussianPoint* pMP = new GaussianSparseSLAM::GaussianPoint(x3D, f, pMap, ts);

			f->mvGaussianPoints.update(idx, pMP);
			lpTemporalMPs.push_back(pMP);
		}
	}
}