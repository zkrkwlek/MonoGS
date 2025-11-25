#include <../GaussianSparseSLAM/include/Mapper.h>

#include <../GaussianSparseSLAM/include/GSSLAM.h>
#include <../GaussianSparseSLAM/include/Optimizer.h>
#include <../GaussianSparseSLAM/include/GaussianPoint.h>
#include <../GaussianSparseSLAM/include/Map.h>
#include <../GaussianSparseSLAM/include/Frame.h>
#include <../GaussianSparseSLAM/include/KeyFrame.h>
#include <../GaussianSparseSLAM/include/User.h>

#include <../EdgeSLAM/include/Camera.h>
#include <WebAPI.h>
#include <Utils.h>
#include <Utils_Geometry.h>

namespace GaussianSparseSLAM {

	std::string GaussianSparseSLAM::Mapper::ip;
	int GaussianSparseSLAM::Mapper::port;

	ConcurrentMap<int, std::chrono::high_resolution_clock::time_point> mapTimes;

	void Mapper::InsertNewKeyFrame(ThreadPool::ThreadPool* pool, GSSLAM* system, Map* map, const std::string& mapName, KeyFrame* targetKF, bool bBA) {
		map->mnNumMappingFrames++;

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		mapTimes.Update(targetKF->mnId, start);

		//std::cout << "InsertNewKeyFrame " <<targetKF->mnId<< std::endl;
		ProcessNewKeyFrame(map, targetKF);
		//std::cout << "InsertNewKeyFrame=1 " << targetKF->mnId << std::endl;
		MapPointCulling(map, targetKF);
		//std::cout << "InsertNewKeyFrame=2 " << targetKF->mnId << std::endl;
		std::stringstream ss;
		ss << mapName <<"." << targetKF->sourceName << "." << targetKF->mnFrameId;
		RequestMatchForMapping(targetKF, ss.str());

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - start).count();
		std::cout << "Mapping Time = Request = " << targetKF->mnId << " = " << du_test1 << std::endl;

		//std::cout << "InsertNewKeyFrame=3 " << targetKF->mnId << std::endl;
		
		//ProcessNewKeyFrame
		//인접 키프레임과 매칭 정보를 받기
		//새 키프레임 생성
		//맵포인트 컬링
		//키프레임 컬링
		//ba 수행
		//루프 클로징은 보류

		//map->mnNumMappingFrames--;
	}
	void Mapper::ProcessMapping(ThreadPool::ThreadPool* pool, GSSLAM* system, const std::string& mapName, KeyFrame* targetKF, bool bBA) {
		//std::cout << "Mapper::ProcessMapping" << std::endl;
		auto map = targetKF->mpMap;
		//map->mnNumMappingFrames++;

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		//std::cout << "ProcessMapping=0 " << targetKF->mnId << std::endl;
		
		//download matching result.
		//가우시안 포인트 체크. 에피폴라 제약 체크 필요
		std::vector<KeyFrame*> vpNeighKFs;// = targetKF->GetBestCovisibilityKeyFrames(nn);
		std::stringstream ss;
		ss << targetKF->sourceName << "." << targetKF->mnFrameId;
		std::string src = ss.str();
		std::vector<std::vector<std::pair<int, int>>> vecMatches;
		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
		DownloadMatchInfos(targetKF, map, vpNeighKFs, mapName, src, vecMatches);

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		CreateNewMapPoints(map, targetKF, vpNeighKFs, vecMatches, ts);
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		//std::cout <<"create::new::mp::" << du_test1/1000.0 << std::endl;

		//std::cout << "ProcessMapping=created new map" << std::endl;
		//if (map->mnNumMappingFrames == 1)
			//SearchInNeighbors(map, targetKF, vpNeighKFs, vecMatches);
			Fuse(map, targetKF, vpNeighKFs, vecMatches);

		std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
		std::chrono::high_resolution_clock::time_point t4 = t3;
		//User의 마지막 생성 KF 교체
		if (system->CheckUser(targetKF->sourceName))
		{
			auto User = system->GetUser(targetKF->sourceName);
			if (User && User->mbMapping)
			{
				User->mnUsed++;
				User->mpLastCreatedKF = targetKF;
				User->mnUsed--;
			}
		}

		//if (map->mnNumMappingFrames == 1)
		//	pMapper->SearchInNeighbors(map, targetKF);
		//std::cout << "ProcessMapping=1 " << map->mnNumMappingFrames <<", "<< map->GetNumKeyFrames() << std::endl;
		map->mbAbortBA = false;   
		if (map->mnNumMappingFrames == 1 && !map->stopRequested())
		{
			if (map->GetNumKeyFrames() > 2 && bBA) {
				Optimizer::LocalBundleAdjustment(targetKF, &map->mbAbortBA, map, ts);
				//std::cout << "ProcessMapping=2 " << targetKF->mnId << std::endl;
			}
			t4 = std::chrono::high_resolution_clock::now();
			KeyFrameCulling(map, targetKF);
			//std::cout << "ProcessMapping=3 " << targetKF->mnId << std::endl;
		}

		std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();

		std::chrono::high_resolution_clock::time_point end= std::chrono::high_resolution_clock::now();
		std::chrono::high_resolution_clock::time_point astart= mapTimes.Get(targetKF->mnId);
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - astart).count();
		auto du_test0 = std::chrono::duration_cast<std::chrono::milliseconds>(start - astart).count();

		auto du_test_1 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
		auto du_test_2 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		auto du_test_3 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
		auto du_test_4 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
		auto du_test_5 = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();

		std::cout << "MappingTime = " << targetKF->mnId << "::" <<du_test0<<" , " << du_test1 <<" == "<<du_test_1<<", "<<du_test_2<<" "<<du_test_3<<" "<<du_test_4 <<" "<<du_test_5 << std::endl;

		//시각화
		{
			auto User = system->GetUser(targetKF->sourceName);
			User->mnUsed++;

			auto encoded = User->ImageDatas.Get(targetKF->mnFrameId);
			cv::Mat img1 = cv::imdecode(encoded, cv::IMREAD_COLOR);
			cv::Mat visImg1 = img1.clone();
			std::string mapName = User->mapName;
			User->mnUsed--;

			auto vpGPs = targetKF->mvpMapPoints.get();

			for (int i = 0; i < targetKF->N; i++)
			{
				cv::circle(visImg1, targetKF->mvKeys[i].pt , 1, cv::Scalar(0, 0, 255), -1);
				auto pGPi = vpGPs[i];
				if (!pGPi || pGPi->isBad())
					continue;
				cv::circle(visImg1, targetKF->mvKeys[i].pt, 3, cv::Scalar(0, 255, 0), -1);
			}
			system->VisualizeImage(mapName, visImg1, 0);
			
		}

		map->mnNumMappingFrames--;
	}

	void Mapper::ProcessNewKeyFrame(Map* map, KeyFrame* targetKF) {
		
		const std::vector<GaussianPoint*> vpMapPointMatches = targetKF->GetGaussianPointMatches();

		for (size_t i = 0; i < vpMapPointMatches.size(); i++)
		{
			auto pMP = vpMapPointMatches[i];
			if (pMP)
			{
				if (!pMP->isBad())
				{
					if (!pMP->IsInKeyFrame(targetKF))
					{
						pMP->AddObservation(targetKF, i);
						pMP->UpdateNormalAndDepth();
						pMP->ComputeDistinctiveDescriptors();
					}
				}
			}
		}
		 /*if (targetKF->mpCamera->mSensor == Sensor::RGBD) {
			targetKF->ScaleAdjustment();
		}*/
		// Update links in the Covisibility Graph
		targetKF->UpdateConnections();

		// Insert Keyframe in Map
		map->AddKeyFrame(targetKF);
	}

	void Mapper::DownloadMatchInfos(KeyFrame* targetKF, Map* map, std::vector<KeyFrame*>& vpNeighKFs, const std::string& mapName, const std::string& src, std::vector<std::vector<std::pair<int, int>>>& vecMatches)
	{
		int nTemp = 0;
		WebAPI API(ip, port);

		/*std::cout << "download matching = " << targetKF->mnId<<" = ";
		for (size_t i = 0; i < vpNeighKFs.size(); i++)
			std::cout << vpNeighKFs[i]->mnId<<",";
		std::cout << std::endl;*/

		std::vector<int> vecNeighIDs;
		{
			std::stringstream ss;
			ss << "/Download?keyword=" << "reqkfmatches" << "&id=" << targetKF->mnId << "&src=" << mapName+"."+src;
			auto res = API.Send(ss.str(), "");
			int num_ids = res.size() / sizeof(int);
			const int* ptr = reinterpret_cast<const int*>(res.data());
			vecNeighIDs = std::vector<int>(ptr, ptr + num_ids);
		}

		for (int i = 0; i < vecNeighIDs.size(); i++)
		{
			int kf_id = vecNeighIDs[i];

			auto pKF2 = map->GetKeyFrame(kf_id);
			if (!pKF2)
				continue;

			vpNeighKFs.push_back(pKF2);

			std::vector<std::pair<int, int>> vMatchedIndices;

			std::stringstream ss;
			ss << "/Download?keyword=" << "reskfmatch" << "&id=" << kf_id << "&src=" << src;
			auto res = API.Send(ss.str(), "");

			int num_pts = res.size() / sizeof(cv::Point2i);
			const cv::Point2i* ptr = reinterpret_cast<const cv::Point2i*>(res.data());
			std::vector<cv::Point2i> vec(ptr, ptr + num_pts);

			auto vpGP1 = targetKF->mvpMapPoints.get();
			
			auto vpGP2 = pKF2->mvpMapPoints.get();

			for (auto pt : vec)
			{
				int idx1 = pt.x;
				int idx2 = pt.y;
			
				vMatchedIndices.push_back(std::make_pair(idx1, idx2));
			}
			nTemp = vec.size();
			vecMatches.push_back(vMatchedIndices);

		}

		//for (size_t i = 0; i < vpNeighKFs.size(); i++)
		//{
		//	KeyFrame* pKF2 = vpNeighKFs[i];
		//	std::vector<std::pair<int, int>> vMatchedIndices;

		//	std::stringstream ss;
		//	ss << "/Download?keyword=" << "reskfmatch" << "&id=" << pKF2->mnId << "&src=" << src;
		//	auto res = API.Send(ss.str(), "");

		//	int num_pts = res.size() / sizeof(cv::Point2i);
		//	const cv::Point2i* ptr = reinterpret_cast<const cv::Point2i*>(res.data());
		//	std::vector<cv::Point2i> vec(ptr, ptr + num_pts);

		//	/*cv::Mat mat = cv::Mat(res.size() / 8, 1, CV_32SC2, (void*)res.data());
		//	std::vector<cv::Point2i> vec;
		//	vec.assign((cv::Point2i*)mat.datastart, (cv::Point2i*)mat.dataend);*/
		//	//중복 포인트 제거 필요

		//	auto vpGP1 = targetKF->mvpMapPoints.get();
		//	auto vpGP2 = pKF2->mvpMapPoints.get();

		//	for (auto pt : vec)
		//	{
		//		int idx1 = pt.x;
		//		int idx2 = pt.y;

		//		/*
		//		auto pGP1 = vpGP1[idx1];
		//		if (pGP1 && !pGP1->isBad())
		//			continue;
		//		auto pGP2 = vpGP2[idx2];
		//		if (pGP2 && !pGP2->isBad())
		//			continue;
		//		if (CommonUtils::Geometry::CheckDistEpipolarLine(targetKF->mvKeysUn[idx1], pKF2->mvKeysUn[idx2], F12, 1.0)) {

		//		}
		//		*/
		//		vMatchedIndices.push_back(std::make_pair(idx1, idx2));
		//	}
		//	nTemp = vec.size();
		//	vecMatches.push_back(vMatchedIndices);
		//}
	}

	void Mapper::RequestMatchForMapping(KeyFrame* targetKF, const std::string& src)
	{
		int nn = 20; 
		const std::vector<KeyFrame*> vpNeighKFs = targetKF->GetBestCovisibilityKeyFrames(nn);
		
		//{
		//	//keyframe
		//	std::vector < cv::Vec2i> vecDatas;
		//	auto vpGPs = targetKF->mvpMapPoints.get();
		//	for (int i = 0; i < targetKF->N; i++)
		//	{
		//		auto pGP = vpGPs[i];
		//		if (!pGP || pGP->isBad())
		//			continue;
		//		vecDatas.push_back(cv::Vec2i(i, pGP->mnId)); 
		//	}
		//	//std::cout << "Request Mapping Test " << vecDatas.size() << std::endl;
		//	cv::Mat tmp = cv::Mat(vecDatas.size(), 1, CV_32SC2, vecDatas.data());
		//	Utils::SendData("kf_gp_ids", src, tmp, targetKF->mnId, 4, 2);
		//}

		std::vector<int> vecDatas;
		for (auto pKF : vpNeighKFs) {
			if (pKF->isBad())
				continue;
			vecDatas.push_back(pKF->mnId);
		}
		cv::Mat tmp = cv::Mat(vecDatas.size(), 1, CV_32SC1, vecDatas.data());
		cv::Mat tmp2 = cv::Mat::ones(1000 - vecDatas.size(), 1, CV_32SC1) * -1;
		//std::cout << "request kf match " <<targetKF->mnId<<" = "<<vecDatas .size()<<", " << tmp.t() << std::endl;
		Utils::SendData("reqkfmatches", src, tmp, targetKF->mnId, 4, 1);
	}

	void Mapper::MapPointCulling(Map* map, KeyFrame* targetKF)
	{
		// Check Recent Added MapPoints
		//Concurrent Vector로 변경해야 할 듯.
		
		
		std::list<GaussianPoint*>::iterator lit = map->mlpNewMPs.Begin();
		const unsigned long int nCurrentKFid = targetKF->mnId;

		const int cnThObs = 2;
		
		while (lit != map->mlpNewMPs.End())
		{
			auto pMP = *lit;
			if (pMP->isBad())
			{
				lit = map->mlpNewMPs.Erase(lit);
			}
			else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->GetFoundRatio() < 0.25f)
			{
				pMP->SetBadFlag();
				lit = map->mlpNewMPs.Erase(lit);
			}
			else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs)
			{
				pMP->SetBadFlag();
				lit = map->mlpNewMPs.Erase(lit);
			}
			else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
				lit = map->mlpNewMPs.Erase(lit);
			else
				lit++;
		}
	}

	void Mapper::CreateNewMapPoints(Map* map, KeyFrame* targetKF, std::vector<KeyFrame*>& vpNeighKFs, std::vector<std::vector<std::pair<int, int>>>& vecMatches, long long ts) {
		cv::Mat Rcw1 = targetKF->GetRotation();
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat tcw1 = targetKF->GetTranslation();
		cv::Mat Tcw1(3, 4, CV_32F);
		Rcw1.copyTo(Tcw1.colRange(0, 3));
		tcw1.copyTo(Tcw1.col(3));
		cv::Mat Ow1 = targetKF->GetCameraCenter();

		const float& fx1 = targetKF->fx;
		const float& fy1 = targetKF->fy;
		const float& cx1 = targetKF->cx;
		const float& cy1 = targetKF->cy;
		const float& invfx1 = targetKF->invfx;
		const float& invfy1 = targetKF->invfy;

		const float ratioFactor = 1.5f * targetKF->mfScaleFactor;
		bool mbMonocular = true;

		int nnew = 0;
		int nnew_stereo = 0;
		int nmatch = 0;
		int nfail0 = 0;
		int nfail1 = 0;
		int nfail2 = 0;
		int nfail3 = 0;
		int nfail4 = 0;

		for (size_t i = 0; i < vpNeighKFs.size(); i++)
		{
			if (i > 0 && map->mnNumMappingFrames > 1)
				return;

			KeyFrame* pKF2 = vpNeighKFs[i];

			// Check first that baseline is not too short
			cv::Mat Ow2 = pKF2->GetCameraCenter();
			cv::Mat vBaseline = Ow2 - Ow1;
			const float baseline = cv::norm(vBaseline);

			if (!mbMonocular) {
				if (baseline < pKF2->mb) {
					continue;
				}
			}
			else {
				const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
				const float ratioBaselineDepth = baseline / medianDepthKF2;

				if (ratioBaselineDepth < 0.01)
					continue;
			}

			// Compute Fundamental Matrix
			cv::Mat R2 = pKF2->GetRotation();
			cv::Mat t2 = pKF2->GetTranslation();
			cv::Mat F12 = CommonUtils::Geometry::ComputeF12(Rcw1, tcw1, R2, t2, targetKF->K, pKF2->K);

			// Search matches that fullfil epipolar constraint
			auto vMatchedIndices = vecMatches[i];

			cv::Mat Rcw2 = pKF2->GetRotation();
			cv::Mat Rwc2 = Rcw2.t();
			cv::Mat tcw2 = pKF2->GetTranslation();
			cv::Mat Tcw2(3, 4, CV_32F);
			Rcw2.copyTo(Tcw2.colRange(0, 3));
			tcw2.copyTo(Tcw2.col(3));

			const float& fx2 = pKF2->fx;
			const float& fy2 = pKF2->fy;
			const float& cx2 = pKF2->cx;
			const float& cy2 = pKF2->cy;
			const float& invfx2 = pKF2->invfx;
			const float& invfy2 = pKF2->invfy;

			// Triangulate each match
			const int nmatches = vMatchedIndices.size();

			for (int ikp = 0; ikp < nmatches; ikp++)

			{
				const int& idx1 = vMatchedIndices[ikp].first;
				const int& idx2 = vMatchedIndices[ikp].second;

				auto pGP1 = targetKF->mvpMapPoints.get(idx1);
				if (pGP1 && !pGP1->isBad())
					continue;
				auto pGP2 = pKF2->mvpMapPoints.get(idx2);
				if (pGP2 && !pGP2->isBad())
					continue;
				nmatch++;

				const cv::KeyPoint& kp1 = targetKF->mvKeysUn[idx1];
				const cv::KeyPoint& kp2 = pKF2->mvKeysUn[idx2];
				bool bStereo1 = targetKF->mvuRight[idx1] >= 0;
				bool bStereo2 = pKF2->mvuRight[idx2] >= 0;

				// Check parallax between rays
				cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
				cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

				cv::Mat ray1 = Rwc1 * xn1;
				cv::Mat ray2 = Rwc2 * xn2;
				const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

				float cosParallaxStereo = cosParallaxRays + 1;
				float cosParallaxStereo1 = cosParallaxStereo;
				float cosParallaxStereo2 = cosParallaxStereo;

				if (bStereo1)
					cosParallaxStereo1 = cos(2 * atan2(targetKF->mb / 2, targetKF->mvDepth[idx1]));
				if (bStereo2)
					cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));
				cosParallaxStereo = std::min(cosParallaxStereo1, cosParallaxStereo2);

				cv::Mat x3D;
				if (cosParallaxRays < cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays < 0.9998))
				{
					// Linear Triangulation Method
					cv::Mat A(4, 4, CV_32F);
					A.row(0) = (xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0));
					A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
					A.row(2) = (xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0));
					A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

					cv::Mat w, u, vt;
					cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

					x3D = vt.row(3).t();

					if (x3D.at<float>(3) == 0){
						nfail3++;
						continue;
					}
					// Euclidean coordinates
					x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

				}
				else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2)
				{
					nnew_stereo++;
					x3D = targetKF->UnprojectStereo(idx1, Rwc1, Ow1);
				}
				else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
				{
					nnew_stereo++;
					x3D = pKF2->UnprojectStereo(idx2, Rwc2, Ow2);
				}
				else{
					nfail4++;
					continue; //No stereo and very low parallax
				}
				cv::Mat x3Dt = x3D.t();

				//Check triangulation in front of cameras
				float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
				if (z1 <= 0){
					nfail0++;
					continue;
				}

				float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
				if (z2 <= 0){
					nfail0++;
					continue;
				}

				//Check reprojection error in first keyframe
				const float& sigmaSquare1 = targetKF->mvLevelSigma2[kp1.octave];
				const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
				const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
				const float invz1 = 1.0 / z1;

				float u1 = fx1 * x1 * invz1 + cx1;
				float v1 = fy1 * y1 * invz1 + cy1;
				float errX1 = u1 - kp1.pt.x;
				float errY1 = v1 - kp1.pt.y;
				if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1){
					nfail1++;
					continue;
				}
				//Check reprojection error in second keyframe
				const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
				const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
				const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
				const float invz2 = 1.0 / z2;
				float u2 = fx2 * x2 * invz2 + cx2;
				float v2 = fy2 * y2 * invz2 + cy2;
				float errX2 = u2 - kp2.pt.x;
				float errY2 = v2 - kp2.pt.y;
				if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2){
					nfail1++;
					continue;
				}
				//Check scale consistency
				cv::Mat normal1 = x3D - Ow1;
				float dist1 = cv::norm(normal1);

				cv::Mat normal2 = x3D - Ow2;
				float dist2 = cv::norm(normal2);

				if (dist1 == 0 || dist2 == 0){
					nfail2++;
					continue;
				}

				const float ratioDist = dist2 / dist1;
				const float ratioOctave = targetKF->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

				if (ratioDist * ratioFactor<ratioOctave || ratioDist>ratioOctave * ratioFactor)
					continue;

				// Triangulation is succesfull
				GaussianPoint* pMP = new GaussianPoint(x3D, targetKF, map, ts);

				pMP->AddObservation(targetKF, idx1);
				pMP->AddObservation(pKF2, idx2);

				targetKF->AddGaussianPoint(pMP, idx1);
				pKF2->AddGaussianPoint(pMP, idx2);

				pMP->ComputeDistinctiveDescriptors();

				pMP->UpdateNormalAndDepth();

				map->AddGaussianPoint(pMP);
				map->mlpNewMPs.push_back(pMP);

				nnew++;
			}
		}
		auto vpMPs = targetKF->mvpMapPoints.get();
		int nMP = 0;
		for (auto pMP : vpMPs)
		{
			if (!pMP || pMP->isBad())
				continue;
			nMP++;
		}
		std::cout << "Mapping::NewMP::" << targetKF->mnId << " == " << targetKF->N <<":"<<nMP << ", " << nnew << " | " << nmatch << "::" << nfail0 << "," << nfail1 << "," << nfail2 << "," << nfail3 << "," << nfail4 << std::endl;
	}

	void Mapper::CreateNewMapPoints(Map* map, KeyFrame* targetKF, long long ts) {
		int nn = 20;
		const std::vector<KeyFrame*> vpNeighKFs = targetKF->GetBestCovisibilityKeyFrames(nn);

		cv::Mat Rcw1 = targetKF->GetRotation();
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat tcw1 = targetKF->GetTranslation();
		cv::Mat Tcw1(3, 4, CV_32F);
		Rcw1.copyTo(Tcw1.colRange(0, 3));
		tcw1.copyTo(Tcw1.col(3));
		cv::Mat Ow1 = targetKF->GetCameraCenter();

		const float& fx1 = targetKF->fx;
		const float& fy1 = targetKF->fy;
		const float& cx1 = targetKF->cx;
		const float& cy1 = targetKF->cy;
		const float& invfx1 = targetKF->invfx;
		const float& invfy1 = targetKF->invfy;

		const float ratioFactor = 1.5f * 1.2;// targetKF->mfScaleFactor;
		bool mbMonocular = true;

		int nnew = 0;

		std::stringstream ss;
		ss << targetKF->sourceName << "." << targetKF->mnFrameId;
		std::string src = ss.str();

		for (size_t i = 0; i < vpNeighKFs.size(); i++)
		{
			if (i > 0 && map->mnNumMappingFrames > 1)
				return;

			KeyFrame* pKF2 = vpNeighKFs[i];

			// Check first that baseline is not too short
			cv::Mat Ow2 = pKF2->GetCameraCenter();
			cv::Mat vBaseline = Ow2 - Ow1;
			const float baseline = cv::norm(vBaseline);

			if (!mbMonocular) {
				if (baseline < pKF2->mb) {
					continue;
				}
			}
			else {
				const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
				const float ratioBaselineDepth = baseline / medianDepthKF2;

				if (ratioBaselineDepth < 0.01)
					continue;
			}

			// Compute Fundamental Matrix
			cv::Mat R2 = pKF2->GetRotation();
			cv::Mat t2 = pKF2->GetTranslation();
			cv::Mat F12 = CommonUtils::Geometry::ComputeF12(Rcw1, tcw1, R2, t2, targetKF->K, pKF2->K);

			// Search matches that fullfil epipolar constraint
			std::vector<std::pair<size_t, size_t> > vMatchedIndices;

			//매칭 정보 획득하기
			{
				WebAPI API(ip, port);
				std::stringstream ss;
				ss << "/Download?keyword=" << "reskfmatch" << "&id=" << pKF2->mnId << "&src=" << src;
				auto res = API.Send(ss.str(), "");
				cv::Mat mat = cv::Mat(res.size() / 8, 1, CV_32SC2, (void*)res.data());
				std::vector<cv::Point2i> vec;
				vec.assign((cv::Point2i*)mat.datastart, (cv::Point2i*)mat.dataend);
				//중복 포인트 제거 필요

				auto vpGP1 = targetKF->mvpMapPoints.get();
				auto vpGP2 = pKF2->mvpMapPoints.get();

				for (auto pt : vec)
				{
					int idx1 = pt.x;
					int idx2 = pt.y;

					auto pGP1 = vpGP1[idx1];
					if (pGP1 && !pGP1->isBad())
						continue;
					auto pGP2 = vpGP2[idx2];
					if (pGP2 && !pGP2->isBad())
						continue;
					auto kp1 = targetKF->mvKeysUn[idx1];
					auto kp2 = pKF2->mvKeysUn[idx2];
					if (CommonUtils::Geometry::CheckDistEpipolarLine(kp1.pt, kp2.pt, F12, pKF2->mvLevelSigma2[kp2.octave])) {
						vMatchedIndices.push_back(std::make_pair(idx1, idx2));
					}
				}
			}

			cv::Mat Rcw2 = pKF2->GetRotation();
			cv::Mat Rwc2 = Rcw2.t();
			cv::Mat tcw2 = pKF2->GetTranslation();
			cv::Mat Tcw2(3, 4, CV_32F);
			Rcw2.copyTo(Tcw2.colRange(0, 3));
			tcw2.copyTo(Tcw2.col(3));

			const float& fx2 = pKF2->fx;
			const float& fy2 = pKF2->fy;
			const float& cx2 = pKF2->cx;
			const float& cy2 = pKF2->cy;
			const float& invfx2 = pKF2->invfx;
			const float& invfy2 = pKF2->invfy;

			// Triangulate each match
			const int nmatches = vMatchedIndices.size();

			for (int ikp = 0; ikp < nmatches; ikp++)

			{
				const int& idx1 = vMatchedIndices[ikp].first;
				const int& idx2 = vMatchedIndices[ikp].second;

				const cv::KeyPoint& kp1 = targetKF->mvKeysUn[idx1];
				const cv::KeyPoint& kp2 = pKF2->mvKeysUn[idx2];
				bool bStereo1 = targetKF->mvuRight[idx1] >= 0;
				bool bStereo2 = pKF2->mvuRight[idx2] >= 0;

				// Check parallax between rays
				cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
				cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

				cv::Mat ray1 = Rwc1 * xn1;
				cv::Mat ray2 = Rwc2 * xn2;
				const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

				float cosParallaxStereo = cosParallaxRays + 1;
				float cosParallaxStereo1 = cosParallaxStereo;
				float cosParallaxStereo2 = cosParallaxStereo;

				if (bStereo1)
					cosParallaxStereo1 = cos(2 * atan2(targetKF->mb / 2, targetKF->mvDepth[idx1]));
				if (bStereo2)
					cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));
				cosParallaxStereo = std::min(cosParallaxStereo1, cosParallaxStereo2);

				cv::Mat x3D;
				//if(cosParallaxRays > 0 && cosParallaxRays < 0.9998)
				if (cosParallaxRays < cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays < 0.9998))
				{
					// Linear Triangulation Method
					cv::Mat A(4, 4, CV_32F);
					A.row(0) = (xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0));
					A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
					A.row(2) = (xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0));
					A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

					cv::Mat w, u, vt;
					cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

					x3D = vt.row(3).t();

					if (x3D.at<float>(3) == 0)
						continue;

					// Euclidean coordinates
					x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

				}
				else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2)
				{
					x3D = targetKF->UnprojectStereo(idx1, Rwc1, Ow1);
				}
				else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
				{
					x3D = pKF2->UnprojectStereo(idx2, Rwc2, Ow2);
				}
				else
					continue; //No stereo and very low parallax

				cv::Mat x3Dt = x3D.t();

				//Check triangulation in front of cameras
				float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
				if (z1 <= 0)
					continue;

				float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
				if (z2 <= 0)
					continue;

				//Check reprojection error in first keyframe
				const float& sigmaSquare1 = targetKF->mvLevelSigma2[kp1.octave];
				const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
				const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
				const float invz1 = 1.0 / z1;

				float u1 = fx1 * x1 * invz1 + cx1;
				float v1 = fy1 * y1 * invz1 + cy1;
				float errX1 = u1 - kp1.pt.x;
				float errY1 = v1 - kp1.pt.y;
				if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
					continue;

				//Check reprojection error in second keyframe
				const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
				const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
				const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
				const float invz2 = 1.0 / z2;
				float u2 = fx2 * x2 * invz2 + cx2;
				float v2 = fy2 * y2 * invz2 + cy2;
				float errX2 = u2 - kp2.pt.x;
				float errY2 = v2 - kp2.pt.y;
				if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
					continue;

				//Check scale consistency
				cv::Mat normal1 = x3D - Ow1;
				float dist1 = cv::norm(normal1);

				cv::Mat normal2 = x3D - Ow2;
				float dist2 = cv::norm(normal2);

				if (dist1 == 0 || dist2 == 0)
					continue;

				const float ratioDist = dist2 / dist1;
				const float ratioOctave = targetKF->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

				if (ratioDist * ratioFactor<ratioOctave || ratioDist>ratioOctave * ratioFactor)
					continue;
								
				// Triangulation is succesfull
				GaussianPoint* pMP = new GaussianPoint(x3D, targetKF, map, ts);

				pMP->AddObservation(targetKF, idx1);
				pMP->AddObservation(pKF2, idx2);

				targetKF->AddGaussianPoint(pMP, idx1);
				pKF2->AddGaussianPoint(pMP, idx2);

				pMP->ComputeDistinctiveDescriptors();

				pMP->UpdateNormalAndDepth();

				map->AddGaussianPoint(pMP);
				map->mlpNewMPs.push_back(pMP);

				nnew++;
			}
		}
		//std::cout << "Create New MP = " << nnew << std::endl;
	}

	void Mapper::SearchInNeighbors(Map* map, KeyFrame* targetKF, std::vector<KeyFrame*>& vpNeighKFs, std::vector<std::vector<std::pair<int, int>>>& vecMatches) {

		std::set<KeyFrame*> spTargetKFs;
		std::vector<KeyFrame*> vpTargetKFs;
		for (std::vector<KeyFrame*>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;
			/*if (pKFi->isBad() || pKFi->mnFuseTargetForKF == targetKF->mnId)
				continue;*/
			if (pKFi->isBad() || spTargetKFs.count(pKFi))
				continue;
			spTargetKFs.insert(pKFi);
			vpTargetKFs.push_back(pKFi);
			pKFi->mnFuseTargetForKF = targetKF->mnId;

			// Extend to some second neighbors
			/*const std::vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
			for (std::vector<KeyFrame*>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
			{
				KeyFrame* pKFi2 = *vit2;
				if (pKFi2->isBad() || spTargetKFs.count(pKFi2) || pKFi2->mnId == targetKF->mnId)
					continue;
				pKFi2->mnFuseTargetForKF = targetKF->mnId;
				spTargetKFs.insert(pKFi);
				vpTargetKFs.push_back(pKFi2);
			}*/
		}
		int Na = 0;
		int Nb = 0;
		// Search matches by projection from current KF in target KFs
		std::vector<GaussianPoint*> vpMapPointMatches = targetKF->GetGaussianPointMatches();
		std::set<GaussianPoint*> spNewDepthMPs;
		for (int i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
		{
			auto pMP = vpMapPointMatches[i];
			if (!pMP || pMP->isBad())
				continue;
			if (pMP->Observations() == 1 && pMP->mnFirstKFid == targetKF->mnId)
			{
				spNewDepthMPs.insert(pMP);
			}
		}

		for (std::vector<KeyFrame*>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;
			//Na += SearchPoints::Fuse(pKFi, vpMapPointMatches);
		}

		// Search matches by projection from target KFs in current KF
		std::vector<GaussianPoint*> vpFuseCandidates;
		//vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());
		for (std::vector<KeyFrame*>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
		{
			KeyFrame* pKFi = *vitKF;

			std::vector<GaussianPoint*> vpMapPointsKFi = pKFi->GetGaussianPointMatches();

			for (std::vector<GaussianPoint*>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
			{
				auto pMP = *vitMP;
				if (!pMP)
					continue;
				if (pMP->isBad() || pMP->mnFuseCandidateForKF == targetKF->mnId)
					continue;
				pMP->mnFuseCandidateForKF = targetKF->mnId;
				vpFuseCandidates.push_back(pMP);
			}
		}

		//int nFused = SearchPoints::Fuse(targetKF, vpFuseCandidates);
		// Update points
		vpMapPointMatches = targetKF->GetGaussianPointMatches();
		for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
		{
			auto pMP = vpMapPointMatches[i];
			if (pMP)
			{
				if (!pMP->isBad())
				{
					pMP->ComputeDistinctiveDescriptors();
					pMP->UpdateNormalAndDepth();
				}
			}
		}
		for (auto iter = spNewDepthMPs.begin(), iend = spNewDepthMPs.end(); iter != iend; iter++) {
			auto pMPi = *iter;
			if (!pMPi || pMPi->isBad())
				continue;
			if (pMPi->Observations() == 1) {
				pMPi->SetBadFlag();	
			}
		}
		targetKF->UpdateConnections();
	}

	void Mapper::KeyFrameCulling(Map* map, KeyFrame* targetKF)
	{
		// Check redundant keyframes (only local keyframes)
		// A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
		// in at least other 3 keyframes (in the same or finer scale)
		// We only consider close stereo points
		std::vector<KeyFrame*> vpLocalKeyFrames = targetKF->GetVectorCovisibleKeyFrames();

		int nObs = 3;
		const int thObs = nObs;

		for (std::vector<KeyFrame*>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
		{
			KeyFrame* pKF = *vit;
			if (pKF->mnId == map->GetInitKFid() || pKF->isBad())
				continue;
			const std::vector<GaussianPoint*> vpMapPoints = pKF->GetGaussianPointMatches();
			
			int nRedundantObservations = 0;
			int nMPs = 0;
			for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
			{
				auto pMP = vpMapPoints[i];
				if (pMP)
				{
					if (!pMP->isBad())
					{
						nMPs++;
						if (pMP->Observations() > thObs)
						{
							const int& scaleLevel = pKF->mvKeysUn[i].octave;
							const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();
							int tmp_nObs = 0;
							for (std::map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
							{
								KeyFrame* pKFi = mit->first;
								if (pKFi == pKF)
									continue;
								const int& scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

								if (scaleLeveli <= scaleLevel + 1)
								{
									tmp_nObs++;
									if (tmp_nObs >= thObs)
										break;
								}
							}
							if (tmp_nObs >= thObs)
							{
								nRedundantObservations++;
							}
						}
					}
				}
			}

			if (nRedundantObservations > 0.9 * nMPs){
				pKF->SetBadFlag();
			}
		}
	}
		
	void Mapper::Fuse(Map* map, KeyFrame* targetKF, std::vector<KeyFrame*>& vpNeighKFs, std::vector<std::vector<std::pair<int, int>>>& vecMatches){
		cv::Mat Rcw1 = targetKF->GetRotation();
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat tcw1 = targetKF->GetTranslation();
		cv::Mat Tcw1(3, 4, CV_32F);
		Rcw1.copyTo(Tcw1.colRange(0, 3));
		tcw1.copyTo(Tcw1.col(3));
		cv::Mat Ow1 = targetKF->GetCameraCenter();

		const float& fx1 = targetKF->fx;
		const float& fy1 = targetKF->fy;
		const float& cx1 = targetKF->cx;
		const float& cy1 = targetKF->cy;
		const float& invfx1 = targetKF->invfx;
		const float& invfy1 = targetKF->invfy;
		const cv::Mat K = targetKF->mpCamera->K;

		const float ratioFactor = 1.5f * targetKF->mfScaleFactor;
		bool mbMonocular = true;

		int nnew = 0;

		std::stringstream ss;
		ss << targetKF->sourceName << "." << targetKF->mnFrameId;
		std::string src = ss.str();

		const float th = 5.99;

		int n1 = 0;
		int n2 = 0;

		for (size_t i = 0; i < vpNeighKFs.size(); i++)
		{
			KeyFrame* pKF2 = vpNeighKFs[i];

			// Check first that baseline is not too short
			cv::Mat Ow2 = pKF2->GetCameraCenter();
			cv::Mat vBaseline = Ow2 - Ow1;

			// Compute Fundamental Matrix
			cv::Mat R2 = pKF2->GetRotation();
			cv::Mat t2 = pKF2->GetTranslation();
			cv::Mat F12 = CommonUtils::Geometry::ComputeF12(Rcw1, tcw1, R2, t2, targetKF->K, pKF2->K);

			auto vpGPs1 = targetKF->mvpMapPoints.get();
			auto vpGPs2 = pKF2->mvpMapPoints.get();

			auto vMatchedIndices = vecMatches[i];

			for (int ikp = 0; ikp < vMatchedIndices.size(); ikp++)

			{
				const int& idx1 = vMatchedIndices[ikp].first;
				const int& idx2 = vMatchedIndices[ikp].second;

				//epipolar constraints
				/*if (!CommonUtils::Geometry::CheckDistEpipolarLine(targetKF->mvKeysUn[idx1], pKF2->mvKeysUn[idx2], F12, 1.0)) {
					continue;
				}*/

				auto kp1 = targetKF->mvKeysUn[idx1];
				auto kp2 = pKF2->mvKeysUn[idx2];

				auto pGP1 = vpGPs1[idx1];
				bool bGP1 = false;
				if (pGP1 && !pGP1->isBad() && !pGP1->IsInKeyFrame(pKF2))
					bGP1 = true;
				auto pGP2 = vpGPs2[idx2];
				bool bGP2 = false;
				if (pGP2 && !pGP2->isBad() && !pGP2->IsInKeyFrame(targetKF))
					bGP2 = true;
				
				if (!pGP1 && !pGP2)
					continue;
				if (pGP1 && pGP2 && (pGP1->mnId == pGP2->mnId))
					continue;

				//reprojection
				cv::Point2f pt1, pt2;
				if (bGP1)
				{
					cv::Mat p3Dw = pGP1->GetWorldPos();
					float d1 = 0.0;
					CommonUtils::Geometry::ProjectPoint(pt1, d1, p3Dw, K, R2, t2);
					
					// Depth must be positive
					/*if (d1 < 0.0f)
						bGP1 = false;*/
					
					float u = pt1.x;
					float v = pt1.y;
					// Point must be inside the image
					//if (!targetKF->is_in_image(u, v))
					//	continue;

					const float& kpx = kp2.pt.x;
					const float& kpy = kp2.pt.y;
					const float ex = u - kpx;
					const float ey = v - kpy;
					const float e2 = ex * ex + ey * ey;

					if (e2 > 5.99)
						bGP1 = false;
				}
				if (bGP2)
				{
					cv::Mat p3Dw = pGP2->GetWorldPos();
					float d2 = 0.0;
					CommonUtils::Geometry::ProjectPoint(pt2, d2, p3Dw, K, Rcw1, tcw1);

					// Depth must be positive
					/*if (d2 < 0.0f)
						bGP1 = false;*/

					float u = pt2.x;
					float v = pt2.y;
					// Point must be inside the image
					//if (!targetKF->is_in_image(u, v))
					//	continue;

					const float& kpx = kp1.pt.x;
					const float& kpy = kp1.pt.y;
					const float ex = u - kpx;
					const float ey = v - kpy;
					const float e2 = ex * ex + ey * ey;

					if (e2 > 5.99)
						bGP2 = false;
				}
				n1++;
				if (bGP1 && !bGP2)
				{
					pKF2->AddGaussianPoint(pGP1, idx2);
					pGP1->AddObservation(pKF2, idx2);
					n2++;
				}
				else if (bGP2 && !bGP1)
				{
					targetKF->AddGaussianPoint(pGP2, idx1);
					pGP2->AddObservation(targetKF, idx1);
					n2++;
				}
				else if ((bGP1 && bGP2) && (pGP1->mnId != pGP2->mnId))
				{
					if (pGP2->Observations() > pGP1->Observations())
						pGP1->Replace(pGP2);
					else
						pGP2->Replace(pGP1);
					n2++;
				}
			}
		}
		std::cout << "Mapper::Fuse::Test::" << n1 << " " << n2 << std::endl; 
	}
}