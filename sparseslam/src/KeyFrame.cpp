#include <../GaussianSparseSLAM/include/KeyFrame.h>
#include <../GaussianSparseSLAM/include/Frame.h>
#include <../GaussianSparseSLAM/include/Map.h>
#include <../GaussianSparseSLAM/include/GaussianPoint.h>
#include <../GaussianSparseSLAM/include/FeatureScaleInfo.h>
#include <Camera.h>
#include <CameraPose.h>
#include <Converter.h>
#include <../GaussianSparseSLAM/include/User.h>

namespace GaussianSparseSLAM {
	KeyFrame::KeyFrame(Frame* F, Map* pMap) :
		mnId(Map::mnNextKeyFrameID++), mnFrameId(F->mnFrameID), mdTimeStamp(F->mdTimeStamp), mpCamPose(F->mpCamPose),
		mbBA(false), mbFastGP(false), mfScale(F->mfScale),
		mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),//mnTrackReferenceForFrame(0), 
		mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
		fx(F->fx), fy(F->fy), cx(F->cx), cy(F->cy), invfx(F->invfx), invfy(F->invfy),
		mb(F->mb), mbf(F->mbf), mThDepth(F->mThDepth), mvDepth(F->mvDepth), mvuRight(F->mvuRight),
		N(F->N), mvKeys(F->mvKeys), mvKeysUn(F->mvKeysUn), mDescriptors(F->mDescriptors.clone()),
		K(F->K), mnConnectedDevices(0), mbSendLocalMap(false),
		mnScaleLevels(F->mnScaleLevels), mfScaleFactor(F->mfScaleFactor),
		mfLogScaleFactor(F->mfLogScaleFactor), mvScaleFactors(F->mvScaleFactors), 
		mvLevelSigma2(F->mvLevelSigma2), mvInvLevelSigma2(F->mvInvLevelSigma2),
		mbFirstConnection(true), mpParent(nullptr), mbNotErase(false), mbCurrentPlaceRecognition(false),
		mbToBeErased(false), mbBad(false), mpMap(pMap), mpCamera(F->mpCamera), mnMergeCorrectedForKF(0), mnBALocalForMerge(0),
		mnMergeQuery(0), mnMergeWords(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0)
	{
		auto vpGPs = F->mvGaussianPoints.get();
		mvpMapPoints.Copy(vpGPs);
		F->mnKeyFrameId = this->mnId;
		mvLabel = std::vector<int>(N, 0);
		mpMap->mnLastKeyFrameID = F->mnFrameID;
	}
	KeyFrame:: ~KeyFrame() {

	}

	Map* KeyFrame::GetMap() {
		std::unique_lock<std::mutex> lock(mMutexMap);
		return mpMap;
	}
	void KeyFrame::UpdateMap(Map* pMap) {
		std::unique_lock<std::mutex> lock(mMutexMap);
		mpMap = pMap;
	}

	bool KeyFrame::is_in_image(float x, float y, float z) {
		return mpCamera->is_in_image(x, y, z);
	}
	void KeyFrame::reset_map_points() {
		mvpMapPoints.Initialize(mvKeysUn.size(), nullptr);
		mvbOutliers = std::vector<bool>(mvKeysUn.size(), false);
	}

	////Covisibility
	void KeyFrame::AddConnection(KeyFrame* pKF, const int& weight)
	{
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (!mConnectedKeyFrameWeights.count(pKF))
				mConnectedKeyFrameWeights[pKF] = weight;
			else if (mConnectedKeyFrameWeights[pKF] != weight)
				mConnectedKeyFrameWeights[pKF] = weight;
			else
				return;
		}

		UpdateBestCovisibles();
	}
	void KeyFrame::EraseConnection(KeyFrame* pKF)
	{
		bool bUpdate = false;
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (mConnectedKeyFrameWeights.count(pKF))
			{
				mConnectedKeyFrameWeights.erase(pKF);
				bUpdate = true;
			}
		}

		if (bUpdate)
			UpdateBestCovisibles();
	}

	void KeyFrame::SetFirstConnection(bool bFirst)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mbFirstConnection = bFirst;
	}


	void KeyFrame::UpdateConnections()
	{
		std::map<KeyFrame*, int> KFcounter;

		std::vector<GaussianPoint*> vpMP = mvpMapPoints.get();

		//For all map points in keyframe check in which other keyframes are they seen
		//Increase counter for those keyframes
		for (std::vector<GaussianPoint*>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
		{
			GaussianPoint* pMP = *vit;

			if (!pMP)
				continue;

			if (pMP->isBad())
				continue;

			std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

			for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				if (mit->first->mnId == mnId)
					continue;
				KFcounter[mit->first]++;
			}
		}

		// This should not happen
		if (KFcounter.empty())
			return;

		//If the counter is greater than threshold add connection
		//In case no keyframe counter is over threshold add the one with maximum counter
		int nmax = 0;
		KeyFrame* pKFmax = nullptr;
		int th = 15;

		std::vector<std::pair<int, KeyFrame*> > vPairs;
		//vPairs.reserve(KFcounter.size());
		for (std::map<KeyFrame*, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
		{
			if (mit->second > nmax)
			{
				nmax = mit->second;
				pKFmax = mit->first;
			}
			if (mit->second >= th)
			{
				vPairs.push_back(std::make_pair(mit->second, mit->first));
				(mit->first)->AddConnection(this, mit->second);
			}
		}

		if (vPairs.empty())
		{
			vPairs.push_back(std::make_pair(nmax, pKFmax));
			pKFmax->AddConnection(this, nmax);
		}

		sort(vPairs.begin(), vPairs.end());
		std::list<KeyFrame*> lKFs;
		std::list<int> lWs;
		for (size_t i = 0; i < vPairs.size(); i++)
		{
			lKFs.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		{
			std::unique_lock<std::mutex> lock(mMutexConnections);

			// mspConnectedKeyFrames = spConnectedKeyFrames;
			mConnectedKeyFrameWeights = KFcounter;
			mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
			mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

			if (mbFirstConnection && mnId != mpMap->GetInitKFid())
			{
				mpParent = mvpOrderedConnectedKeyFrames.front();
				mpParent->AddChild(this);
				mbFirstConnection = false;
			}
		}
	}

	void KeyFrame::ComputeStereoFromRGBD(const cv::Mat& imDepth)
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

	cv::Mat KeyFrame::UnprojectStereo(int i, const cv::Mat& Rwc, const cv::Mat& twc)
	{
		const float z = mvDepth[i];
		if (z > 0)
		{
			const float u = mvKeysUn[i].pt.x;
			const float v = mvKeysUn[i].pt.y;
			const float x = (u - cx) * z * invfx;
			const float y = (v - cy) * z * invfy;
			cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
			return Rwc * x3Dc + twc;
		}
		else
			return cv::Mat();
	}

	void KeyFrame::UpdateBestCovisibles()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		std::vector<std::pair<int, KeyFrame*> > vPairs;
		//vPairs.reserve(mConnectedKeyFrameWeights.size());
		for (std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
			vPairs.push_back(std::make_pair(mit->second, mit->first));

		sort(vPairs.begin(), vPairs.end());
		std::list<KeyFrame*> lKFs;
		std::list<int> lWs;
		for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
		{
			lKFs.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
		mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
	}

	std::set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		std::set<KeyFrame*> s;
		for (std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
			s.insert(mit->first);
		return s;
	}

	std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		return mvpOrderedConnectedKeyFrames;
	}

	std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int& N)
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		if ((int)mvpOrderedConnectedKeyFrames.size() < N)
			return mvpOrderedConnectedKeyFrames;
		else
			return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);

	}

	std::vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int& w)
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);

		if (mvpOrderedConnectedKeyFrames.empty())
			return std::vector<KeyFrame*>();

		std::vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, KeyFrame::weightComp);
		if (it == mvOrderedWeights.end())
			return std::vector<KeyFrame*>();
		else
		{
			int n = it - mvOrderedWeights.begin();
			return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
		}
	}

	int KeyFrame::GetWeight(KeyFrame* pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		if (mConnectedKeyFrameWeights.count(pKF))
			return mConnectedKeyFrameWeights[pKF];
		else
			return 0;
	}
	////Covisibility

	////Spanning Tree
	void KeyFrame::AddChild(KeyFrame* pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mspChildrens.insert(pKF);
	}

	void KeyFrame::EraseChild(KeyFrame* pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mspChildrens.erase(pKF);
	}

	void KeyFrame::ChangeParent(KeyFrame* pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mpParent = pKF;
		pKF->AddChild(this);
	}

	std::set<KeyFrame*> KeyFrame::GetChilds()
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mspChildrens;
	}

	KeyFrame* KeyFrame::GetParent()
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mpParent;
	}

	bool KeyFrame::hasChild(KeyFrame* pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mspChildrens.count(pKF);
	}

	////Spanning Tree

	////Loop Edges
	void KeyFrame::AddLoopEdge(KeyFrame* pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mbNotErase = true;
		mspLoopEdges.insert(pKF);
	}

	std::set<KeyFrame*> KeyFrame::GetLoopEdges()
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mspLoopEdges;
	}

	void KeyFrame::AddMergeEdge(KeyFrame* pKF) {
		{
			std::unique_lock<std::mutex> lockCon(mMutexConnections);
			mbNotErase = true;
		}
		mspMergeEdges.Update(pKF);
	}
	std::set<KeyFrame*> KeyFrame::GetMergeEdges() {
		return mspMergeEdges.Get();
	}

	////Loop Edges

	////Flag
	void KeyFrame::SetNotErase()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		mbNotErase = true;
	}

	void KeyFrame::SetErase()
	{
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (mspLoopEdges.empty())
			{
				mbNotErase = false;
			}
		}

		if (mbToBeErased)
		{
			SetBadFlag();
		}
	}

	void KeyFrame::SetBadFlag()
	{
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (mnId == mpMap->GetInitKFid())
				return;
			else if (mbNotErase)
			{
				mbToBeErased = true;
				return;
			}
		}

		std::vector<GaussianPoint*> vpMP = mvpMapPoints.get();

		for (std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
			mit->first->EraseConnection(this);

		for (size_t i = 0; i < vpMP.size(); i++)
			if (vpMP[i])
				vpMP[i]->EraseObservation(this);
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			//std::unique_lock<std::mutex> lock1(mMutexFeatures);

			mConnectedKeyFrameWeights.clear();
			mvpOrderedConnectedKeyFrames.clear();

			// Update Spanning Tree
			std::set<KeyFrame*> sParentCandidates;
			sParentCandidates.insert(mpParent);

			// Assign at each iteration one children with a parent (the pair with highest covisibility weight)
			// Include that children as new parent candidate for the rest
			while (!mspChildrens.empty())
			{
				bool bContinue = false;

				int max = -1;
				KeyFrame* pC = nullptr;
				KeyFrame* pP = nullptr;

				for (std::set<KeyFrame*>::iterator sit = mspChildrens.begin(), send = mspChildrens.end(); sit != send; sit++)
				{
					KeyFrame* pKF = *sit;
					if (pKF->isBad())
						continue;

					// Check if a parent candidate is connected to the keyframe
					std::vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
					for (size_t i = 0, iend = vpConnected.size(); i < iend; i++)
					{
						for (std::set<KeyFrame*>::iterator spcit = sParentCandidates.begin(), spcend = sParentCandidates.end(); spcit != spcend; spcit++)
						{
							if (vpConnected[i]->mnId == (*spcit)->mnId)
							{
								int w = pKF->GetWeight(vpConnected[i]);
								if (w > max)
								{
									pC = pKF;
									pP = vpConnected[i];
									max = w;
									bContinue = true;
								}
							}
						}
					}
				}

				if (bContinue && pC && pP)
				{
					pC->ChangeParent(pP);
					sParentCandidates.insert(pC);
					mspChildrens.erase(pC);
				}
				else
					break;
			}

			// If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
			if (!mspChildrens.empty())
				for (std::set<KeyFrame*>::iterator sit = mspChildrens.begin(); sit != mspChildrens.end(); sit++)
				{
					(*sit)->ChangeParent(mpParent);
				}

			mpParent->EraseChild(this);
			mTcp = mpCamPose->GetPose() * mpParent->GetPoseInverse();
			mbBad = true;
		}

		GetMap()->RemoveKeyFrame(this);
		//mpKeyFrameDB->erase(this);
	}

	bool KeyFrame::isBad()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		return mbBad;
	}
	////Flag

	////Map Point
	void KeyFrame::AddGaussianPoint(GaussianPoint* pMP, const size_t& idx)
	{
		mvpMapPoints.update(idx, pMP);
	}

	void KeyFrame::EraseGaussianPointMatch(const size_t& idx)
	{
		mvpMapPoints.update(idx, nullptr);
	}

	void KeyFrame::EraseGaussianPointMatch(GaussianPoint* pMP)
	{
		int idx = pMP->GetIndexInKeyFrame(this);
		if (idx >= 0)
			mvpMapPoints.update(idx, nullptr);
	}


	void KeyFrame::ReplaceGaussianPointMatch(const size_t& idx, GaussianPoint* pMP)
	{
		mvpMapPoints.update(idx, pMP);
	}

	std::set<GaussianPoint*> KeyFrame::GetGaussianPoints()
	{
		std::set<GaussianPoint*> s;
		for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++)
		{
			auto pMP = mvpMapPoints.get(i);
			if (pMP && !pMP->isBad())
				s.insert(pMP);
		}
		return s;
	}

	int KeyFrame::TrackedGaussianPoints(const int& minObs)
	{
		int nPoints = 0;
		auto vpMPs = mvpMapPoints.get();
		const bool bCheckObs = minObs > 0;
		for (int i = 0, iend = vpMPs.size(); i < iend; i++)
		{
			auto pMP = vpMPs[i];
			if (pMP && !pMP->isBad())
			{
				if (bCheckObs)
				{
					if (pMP->Observations() >= minObs)
						nPoints++;
				}
				else
					nPoints++;
			}
		}

		return nPoints;
	}

	std::vector<GaussianPoint*> KeyFrame::GetGaussianPointMatches()
	{
		return mvpMapPoints.get();
	}

	GaussianPoint* KeyFrame::GetGaussianPoint(const size_t& idx)
	{
		return mvpMapPoints.get(idx);
	}
	
	bool zdcompare(const std::pair<float, float>& a, const std::pair<float, float>& b) {
		if (a.first == b.first)
			a.second > b.second;
		return a.first > b.first;
	}

	float KeyFrame::ComputeSceneMedianDepth(const int q)
	{
		std::vector<GaussianPoint*> vpMapPoints = mvpMapPoints.get();
		cv::Mat Tcw_ = mpCamPose->GetPose();

		std::vector<float> vDepths;
		//vDepths.reserve(N);
		cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
		Rcw2 = Rcw2.t();
		float zcw = Tcw_.at<float>(2, 3);
		for (int i = 0; i < N; i++)
		{
			if (vpMapPoints[i])
			{
				GaussianPoint* pMP = vpMapPoints[i];
				cv::Mat x3Dw = pMP->GetWorldPos();
				float z = Rcw2.dot(x3Dw) + zcw;
				vDepths.push_back(z);
			}
		}

		sort(vDepths.begin(), vDepths.end());

		return vDepths[(vDepths.size() - 1) / q];
	}
	////Map Point

	////Camera
	void KeyFrame::SetPose(const cv::Mat& Tcw) {
		mpCamPose->SetPose(Tcw);
	}
	cv::Mat KeyFrame::GetPose() {
		return mpCamPose->GetPose();
	}
	cv::Mat KeyFrame::GetPoseInverse() {
		return mpCamPose->GetInversePose();
	}
	cv::Mat KeyFrame::GetCameraCenter() {
		return mpCamPose->GetCameraCenter();
	}
	cv::Mat KeyFrame::GetRotation() {
		return mpCamPose->GetRotation();
	}
	cv::Mat KeyFrame::GetTranslation() {
		return mpCamPose->GetTranslation();
	}
	////Camera
}
