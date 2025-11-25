#include <../GaussianSparseSLAM/include/Map.h>

#include <../GaussianSparseSLAM/include/KeyFrame.h>
#include <../GaussianSparseSLAM/include/GaussianPoint.h>
#include <../GaussianSparseSLAM/include/Visualizer.h>
#include <../GaussianSparseSLAM/include/User.h>

#include <windows.h>

namespace GaussianSparseSLAM {

	std::atomic<long unsigned int> Map::mnNextMapID = 0;
	std::atomic<long unsigned int> Map::mnNextGaussianPointID = 0;
	std::atomic<long unsigned int> Map::mnNextKeyFrameID = 0;

	Map::Map():mState(MapState::NoImages), mLCState(LoopClosingState::idle), mnId(mnNextMapID++), mIsInUse(false), mbBad(false), mnInitKFid(0), mnMaxKFid(0)
	, mnMapChange(0), mnMapChangeNotified(0), mpFirstRegionKF(nullptr), mpKFinitial(nullptr)
	, mnNumMappingFrames(0), mnNumLoopClosingFrames(0)
	, mbResetRequested(false), mbFinishRequested(false), mbFinished(false)
	, mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false)
	, mpMatchedKF(nullptr), mnLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true)
	, mbStopGBA(false), mpThreadGBA(nullptr), mnFullBAIdx(0), mbVisualized(false)
	, mnLastKeyFrameID(-1), mnLoopNumCoincidences(0), mnLoopNumNotFound(0), mpLoopLastCurrentKF(nullptr), mpLoopMatchedKF(nullptr)
	{

	}
	Map::Map(unsigned long int initKFid, bool bFixScale) : mState(MapState::NoImages), mLCState(LoopClosingState::idle), mnId(mnNextMapID++)
		, mIsInUse(false), mbBad(false), mnInitKFid(initKFid), mnMaxKFid(initKFid)
		, mnNumMappingFrames(0), mnNumLoopClosingFrames(0)
		, mnMapChange(0), mnMapChangeNotified(0), mpFirstRegionKF(nullptr), mpKFinitial(nullptr)
		, mbResetRequested(false), mbFinishRequested(false), mbFinished(false)
		, mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false)
		, mpMatchedKF(nullptr), mnLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true)
		, mbStopGBA(false), mpThreadGBA(nullptr), mnFullBAIdx(0), mbVisualized(false)
		, mnLastKeyFrameID(-1), mnLoopNumCoincidences(0), mnLoopNumNotFound(0), mpLoopLastCurrentKF(nullptr), mpLoopMatchedKF(nullptr)
	{

	}

	void Map::Delete() {

	}

	void Map::Reset() {
		
		mmpGaussianPoints.Clear();
		mmpKeyFrames.Clear();
		mState = MapState::NoImages;

	}
	void Map::AddGaussianPoint(GaussianPoint* pMP){
		//if (!mspGaussianPoints.Count(pMP))
		//	mspGaussianPoints.Update(pMP);
		mmpGaussianPoints.Update(pMP->mnId, pMP);
	}
	void Map::RemoveGaussianPoint(GaussianPoint* pMP){
		/*if (mspGaussianPoints.Count(pMP))
			mspGaussianPoints.Erase(pMP);*/
		mmpGaussianPoints.Erase(pMP->mnId);
	}
	//std::map<int,GaussianPoint*> Map::GetAllGaussianPoints() {
	//	//return mspGaussianPoints.ConvertVector();
	//	mmpGaussianPoints.Get();
	//}

	void Map::AddDevice(User* _user) {
		if (!mSetConnectedDevices.Count(_user))
			mSetConnectedDevices.Update(_user);
	}
	void Map::RemoveDevice(User* _user) {
		if (mSetConnectedDevices.Count(_user))
			mSetConnectedDevices.Erase(_user);
	}

	void Map::AddKeyFrame(KeyFrame* pF) {
		int id = pF->mnId;
		if (mmpKeyFrames.Size() == 0) {
			mpKFinitial = pF;
			mnInitKFid = id;
		}
		if (!mmpKeyFrames.Count(id)) {
			mmpKeyFrames.Update(id, pF);
		}
		if (pF->mnId > mnMaxKFid)
			mnMaxKFid = pF->mnId;
	}

	KeyFrame* Map::GetKeyFrame(int id) {
		if (mmpKeyFrames.Count(id))
			return mmpKeyFrames.Get(id);
		return nullptr;
	}
	void Map::RemoveKeyFrame(KeyFrame* pF) {
		int id = pF->mnId;
		if (mmpKeyFrames.Count(id))
			mmpKeyFrames.Erase(id);
	}
	std::vector<KeyFrame*> Map::GetAllKeyFrames() {
		auto mapKFs = mmpKeyFrames.Get();
		std::vector<KeyFrame*> res;
		for (auto pair : mapKFs) {
			auto pKF = pair.second;
			res.push_back(pKF);
		}
		return std::vector<KeyFrame*>(res.begin(), res.end());
	}
	std::vector<GaussianPoint*> Map::GetAllPoints() {
		auto mapMPs = mmpGaussianPoints.Get();
		std::vector<GaussianPoint*> res;
		for (auto pair : mapMPs) {
			auto pMP = pair.second;
			res.push_back(pMP);
		}
		return std::vector<GaussianPoint*>(res.begin(), res.end());
	}
	int Map::GetNumKeyFrames() {
		return mmpKeyFrames.Size();
	}

	MapState Map::GetState() {
		std::unique_lock<std::mutex> lock(mMutexState);
		return mState;
	}

	void Map::SetState(MapState stat) {
		std::unique_lock<std::mutex> lock(mMutexState);
		mState = stat;
	}

	void Map::InformNewBigChange()
	{
		mnBigChangeIdx++;
	}

	int Map::GetLastBigChangeIdx()
	{
		return mnBigChangeIdx.load();
	}

	long unsigned int Map::GetId(){
		return mnId;
	}
	void Map::ChangeId(long unsigned int nId){}
	long unsigned int Map::GetInitKFid(){
		return mnInitKFid;
	}
	void Map::SetInitKFid(long unsigned int initKFif){
		mnInitKFid = initKFif;
	}
	long unsigned int Map::GetMaxKFid(){
		return mnMaxKFid;
	}
	KeyFrame* Map::GetOriginKF(){
		return mpKFinitial;
	}
	void Map::SetCurrentMap(){
		mIsInUse = false;
	}
	void Map::SetStoredMap(){
		mIsInUse = false;
	}
	void Map::SetBad(){
		mbBad = true;
	}
	bool Map::IsBad(){
		return mbBad;
	}
	bool Map::IsInUse(){
		return mIsInUse;
	}


	void Map::RequestStop()
	{
		mbStopRequested = true;
		mbAbortBA = true;
	}
	bool Map::Stop()
	{
		if (mbStopRequested && !mbNotStop)
		{
			mbStopped = true;
			return true;
		}
		return false;
	}
	bool Map::isStopped()
	{
		return mbStopped;
	}
	bool Map::stopRequested()
	{
		return mbStopRequested.load();
	}
	bool Map::SetNotStop(bool flag)
	{

		if (flag && mbStopped)
			return false;
		mbNotStop = flag;
		return true;
	}
	void Map::Release()
	{
		if (mbFinished)
			return;
		mbStopped = false;
		mbStopRequested = false;
		/*for (std::list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
		delete *lit;
		mlNewKeyFrames.clear();*/
	}
	void Map::RequestReset()
	{
		{
			mbResetRequested = true;
		}

		while (1)
		{
			{
				if (!mbResetRequested)
					break;
			}
			Sleep(3000);
		}
	}

	void Map::ResetIfRequested()
	{
		if (mbResetRequested)
		{
			//mlNewKeyFrames.clear();
			//mlpRecentAddedMapPoints.clear();
			mbResetRequested = false;
		}
	}

	void Map::RequestFinish()
	{
		mbFinishRequested = true;
	}

	bool Map::CheckFinish()
	{
		return mbFinishRequested.load();
	}

	void Map::SetFinish()
	{
		mbFinished = true;
		mbStopped = true;
	}

	bool Map::isFinished()
	{
		return mbFinished.load();
	}

	bool Map::isRunningGBA() {
		return mbRunningGBA.load();
	}
	bool Map::isFinishedGBA() {
		return mbFinishedGBA.load();
	}
}