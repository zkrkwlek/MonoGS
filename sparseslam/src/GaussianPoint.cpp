#include <../GaussianSparseSLAM/include/GaussianPoint.h>
#include <../GaussianSparseSLAM/include/Map.h>
#include <../GaussianSparseSLAM/include/Frame.h>
#include <../GaussianSparseSLAM/include/KeyFrame.h>

namespace GaussianSparseSLAM {
	std::mutex GaussianPoint::mGlobalMutex;

	GaussianPoint::GaussianPoint(const cv::Mat& Pos, KeyFrame* pRefKF, Map* pMap, long long ts) :
		mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), nBoxObs(0), //mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
		mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0), mnLabelID(0), mnObjectID(0), mnPlaneID(0), mnPlaneCount(0),
		mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
		mpReplaced(static_cast<GaussianPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap), mnId(Map::mnNextGaussianPointID++), mnLastUpdatedTime(ts)
	{
		Pos.copyTo(mWorldPos);
		mNormalVector = cv::Mat::zeros(3, 1, CV_32F);
	}

	GaussianPoint::GaussianPoint(const cv::Mat& Pos, Frame* pFrame, Map* pMap, const int& idxF)
		: mnFirstKFid(-1), mnFirstFrame(pFrame->mnFrameID), nObs(0), nBoxObs(0),
		mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
		mnLabelID(0), mnObjectID(0), mnPlaneID(0), mnPlaneCount(0),
		mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(nullptr),
		mnVisible(1), mnFound(1), mbBad(false),
		mpReplaced(static_cast<GaussianPoint*>(NULL)),
		mpMap(pMap), mnId(Map::mnNextGaussianPointID++), mnLastUpdatedTime(-1)
	{
		Pos.copyTo(mWorldPos);
		cv::Mat Ow = pFrame->GetCameraCenter();
		mNormalVector = mWorldPos - Ow;
		mNormalVector = mNormalVector / cv::norm(mNormalVector);

		//cv::Mat PC = Pos - Ow;
		//const float dist = cv::norm(PC);
		//const int level = pFrame->mvKeysUn[idxF].octave;
		//const float levelScaleFactor = pFrame->mvScaleFactors[level];
		//const int nLevels = pFrame->mnScaleLevels;

		//mfMaxDistance = dist * levelScaleFactor;
		//mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

		//pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);
	}

	GaussianPoint::~GaussianPoint() {}

	void GaussianPoint::SetWorldPos(const cv::Mat& Pos)
	{
		//std::unique_lock<std::mutex> lock2(mGlobalMutex);
		std::unique_lock<std::mutex> lock(mMutexPos);
		Pos.copyTo(mWorldPos);
	}

	cv::Mat GaussianPoint::GetWorldPos()
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		return mWorldPos.clone();
	}

	cv::Mat GaussianPoint::GetNormal()
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		return mNormalVector.clone();
	}

	void GaussianPoint::SetNormalVector(const cv::Mat& normal)
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		mNormalVector = normal;
	}

	KeyFrame* GaussianPoint::GetReferenceKeyFrame()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mpRefKF;
	}

	Map* GaussianPoint::GetMap() {
		std::unique_lock<std::mutex> lock(mMutexMap);
		return mpMap;
	}
	void GaussianPoint::UpdateMap(Map* pMap) {
		std::unique_lock<std::mutex> lock(mMutexMap);
		mpMap = pMap;
	}

	void GaussianPoint::AddObservation(KeyFrame* pKF, size_t idx)
	{
		if (mObservations.Count(pKF))
			return;
		mObservations.Update(pKF, idx);
		nObs++;
	}

	void GaussianPoint::EraseObservation(KeyFrame* pKF)
	{
		bool bBad = false;
		{
			if (mObservations.Count(pKF))
			{
				int idx = mObservations.Get(pKF);
				/*if (pKF->mvuRight[idx] >= 0)
					nObs -= 2;
				else*/
				nObs--;

				mObservations.Erase(pKF);

				{
					std::unique_lock<std::mutex> lock2(mMutexFeatures);
					if (mpRefKF == pKF) {
						auto mapObservation = mObservations.Get();
						mpRefKF = mapObservation.begin()->first;
					}
				}
				// If only 2 observations or less, discard point
				if (nObs <= 2)
					bBad = true;
			}
		}
		if (bBad)
			SetBadFlag();
	}

	std::map<KeyFrame*, size_t> GaussianPoint::GetObservations()
	{
		return mObservations.Get();
	}

	int GaussianPoint::Observations()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return nObs;
	}
		
	void GaussianPoint::SetBadFlag()
	{
		std::map<KeyFrame*, size_t> obs;
		{
			//std::unique_lock<std::mutex> lock1(mMutexFeatures);
			//std::unique_lock<std::mutex> lock2(mMutexPos);
			mbBad = true;
			obs = mObservations.Get();
			mObservations.Clear();
		}
		for (std::map<KeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
		{
			KeyFrame* pKF = mit->first;
			pKF->EraseGaussianPointMatch(mit->second);
		}

		////short-term 包府
		/*auto setUsers = mSetConnected.Get();
		for (auto iter = setUsers.begin(), iend = setUsers.end(); iter != iend; iter++) {
			auto pUser = *iter;
			mSetConnected.Erase(pUser);
		}
		mSetConnected.Release();*/
		////short-term 包府

		mpMap->RemoveGaussianPoint(this);
	}

	GaussianPoint* GaussianPoint::GetReplaced()
	{
		std::unique_lock<std::mutex> lock1(mMutexFeatures);
		std::unique_lock<std::mutex> lock2(mMutexPos);
		return mpReplaced;
	}

	void GaussianPoint::Replace(GaussianPoint* pMP)
	{
		if (pMP->mnId == this->mnId)
			return;

		int nvisible, nfound;
		std::map<KeyFrame*, size_t> obs;
		{
			//std::unique_lock<std::mutex> lock1(mMutexFeatures);
			obs = mObservations.Get();
			mObservations.Clear();
			mbBad = true;
			std::unique_lock<std::mutex> lock2(mMutexPos);
			nvisible = mnVisible;
			nfound = mnFound;
			mpReplaced = pMP;
		}

		for (std::map<KeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
		{
			// Replace measurement in keyframe
			KeyFrame* pKF = mit->first;

			if (!pMP->IsInKeyFrame(pKF))
			{
				pKF->ReplaceGaussianPointMatch(mit->second, pMP);
				pMP->AddObservation(pKF, mit->second);
			}
			else
			{
				pKF->EraseGaussianPointMatch(mit->second);
			}
		}
		pMP->IncreaseFound(nfound);
		pMP->IncreaseVisible(nvisible);
		pMP->ComputeDistinctiveDescriptors();

		mpMap->RemoveGaussianPoint(this);
	}

	void GaussianPoint::IncreaseVisible(int n)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		mnVisible += n;
	}

	void GaussianPoint::IncreaseFound(int n)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		mnFound += n;
	}

	float GaussianPoint::GetFoundRatio()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return static_cast<float>(mnFound) / mnVisible;
	}

	void GaussianPoint::ComputeDistinctiveDescriptors()
	{
		//// Retrieve all observed descriptors
		//std::vector<cv::Mat> vDescriptors;

		//std::map<KeyFrame*, size_t> observations;

		//{
		//	//std::unique_lock<std::mutex> lock(mMutexFeatures);
		//	if (mbBad)
		//		return;
		//	observations = mObservations.Get();
		//}

		//if (observations.empty())
		//	return;

		////vDescriptors.reserve(observations.size());

		//for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		//{
		//	KeyFrame* pKF = mit->first;

		//	if (!pKF->isBad())
		//		vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
		//}

		//if (vDescriptors.empty())
		//	return;

		//// Compute distances between them
		//size_t N = vDescriptors.size();
		//std::vector<std::vector<float> > Distances;
		//Distances.resize(N, std::vector<float>(N, 0));
		//for (size_t i = 0; i < N; i++)
		//{
		//	Distances[i][i] = 0;
		//	for (size_t j = i + 1; j < N; j++)
		//	{
		//		int distij = (int)mpDist->DescriptorDistance(vDescriptors[i], vDescriptors[j]);//ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
		//		Distances[i][j] = distij;
		//		Distances[j][i] = distij;
		//	}
		//}


		//// Take the descriptor with least median distance to the rest
		//int BestMedian = INT_MAX;
		//int BestIdx = 0;
		//for (size_t i = 0; i < N; i++)
		//{
		//	std::vector<int> vDists(Distances[i].begin(), Distances[i].end());
		//	sort(vDists.begin(), vDists.end());
		//	int median = vDists[0.5 * (N - 1)];

		//	if (median < BestMedian)
		//	{
		//		BestMedian = median;
		//		BestIdx = i;
		//	}
		//}

		//{
		//	std::unique_lock<std::mutex> lock(mMutexFeatures);
		//	mDescriptor = vDescriptors[BestIdx].clone();
		//}
	}

	cv::Mat GaussianPoint::GetDescriptor()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mDescriptor.clone();
	}

	int GaussianPoint::GetIndexInKeyFrame(KeyFrame* pKF)
	{
		if (mObservations.Count(pKF))
			return mObservations.Get(pKF);
		else
			return -1;
	}

	bool GaussianPoint::IsInKeyFrame(KeyFrame* pKF)
	{
		return (mObservations.Count(pKF));
	}

	void GaussianPoint::UpdateNormalAndDepth()
	{

		std::map<KeyFrame*, size_t> observations;
		KeyFrame* pRefKF;
		cv::Mat Pos;
		{
			if (mbBad)
				return;
			observations = mObservations.Get();
			{
				std::unique_lock<std::mutex> lock2(mMutexPos);
				Pos = mWorldPos.clone();
			}
			{
				std::unique_lock<std::mutex> lock2(mMutexFeatures);
				pRefKF = mpRefKF;
			}
		}

		if (observations.empty())
			return;

		cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
		int n = 0;
		for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			KeyFrame* pKF = mit->first;
			cv::Mat Owi = pKF->GetCameraCenter();
			cv::Mat normali = Pos - Owi;
			normal = normal + normali / cv::norm(normali);
			n++;
		}

		cv::Mat PC = Pos - pRefKF->GetCameraCenter();
		const float dist = cv::norm(PC);
		const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
		const float levelScaleFactor = pRefKF->mvScaleFactors[level];
		const int nLevels = pRefKF->mnScaleLevels;

		{
			std::unique_lock<std::mutex> lock(mMutexPos);
			mfMaxDistance = dist * levelScaleFactor;
			mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
			mNormalVector = normal / n;
		}
	}

	float GaussianPoint::GetMinDistanceInvariance()
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		return 0.8f * mfMinDistance;
	}   

	float GaussianPoint::GetMaxDistanceInvariance()
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		return 1.2f * mfMaxDistance;
	}
	int GaussianPoint::PredictScale(const float& currentDist, float logScaleFactor, int nScaleLevels)
	{
		float ratio;
		{
			std::unique_lock<std::mutex> lock(mMutexPos);
			ratio = mfMaxDistance / currentDist;
		}

		int nScale = ceil(log(ratio) / logScaleFactor);
		if (nScale < 0)
			nScale = 0;
		else if (nScale >= nScaleLevels)
			nScale = nScaleLevels - 1;
		return nScale;
	}
}