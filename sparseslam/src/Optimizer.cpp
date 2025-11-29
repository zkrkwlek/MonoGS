#include <../GaussianSparseSLAM/include/Optimizer.h>
#include <../GaussianSparseSLAM/include/Frame.h>
#include <../GaussianSparseSLAM/include/KeyFrame.h>
#include <../GaussianSparseSLAM/include/Map.h>
#include <../GaussianSparseSLAM/include/GaussianPoint.h>
#include <Converter.h>
#include <DataConverter.h>

#include <g2o/core/base_edge.h>
#include "g2o/core/base_binary_edge.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/core/block_solver.h"
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"

#include <Optimize/VertexPoint.h>
#include <Optimize/VertexFrame.h>
#include <Optimize/EdgeSE3MonoFrame.h>
#include <Optimize/EdgeSE3MonoFramePoint.h>
#include <Optimize/EdgeSE3StereoFrame.h>
#include <Optimize/EdgeSE3StereoFramePoint.h>
#include <Optimize/seven_dof_expmap.h>

namespace GaussianSparseSLAM {

	void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
	{
		std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
		std::vector<GaussianPoint*> vpMP = pMap->GetAllPoints();
		BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
	}

	void Optimizer::BundleAdjustment(const std::vector<KeyFrame*>& vpKFs, const std::vector<GaussianPoint*>& vpMP,
		int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
	{
		std::vector<bool> vbNotIncludedMP;
		vbNotIncludedMP.resize(vpMP.size());

		g2o::SparseOptimizer optimizer;
		//g2o::BlockSolver_6_3::LinearSolverType * linearSolver = linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
		//g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

		auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
		auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
		auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
		optimizer.setAlgorithm(algorithm);

		if (pbStopFlag)
			optimizer.setForceStopFlag(pbStopFlag);

		long unsigned int maxKFid = 0;

		// Set KeyFrame vertices
		for (size_t i = 0; i < vpKFs.size(); i++)
		{
			KeyFrame* pKF = vpKFs[i];
			if (pKF->isBad())
				continue;
			auto vSE3 = new BaseSLAM::Optimization::VertexFrame();
			vSE3->setEstimate(CommonUtils::DataConverter::toSE3Quat(pKF->GetPose()));
			vSE3->setId(pKF->mnId);
			vSE3->setFixed(pKF->mnId == pKF->GetMap()->GetInitKFid());
			optimizer.addVertex(vSE3);
			if (pKF->mnId > maxKFid)
				maxKFid = pKF->mnId;
		}

		const float thHuber2D = sqrt(5.99);
		const float thHuber3D = sqrt(7.815);

		// Set MapPoint vertices
		for (size_t i = 0; i < vpMP.size(); i++)
		{
			GaussianPoint* pMP = vpMP[i];
			if (pMP->isBad())
				continue;
			auto vPoint = new BaseSLAM::Optimization::VertexPoint();
			vPoint->setEstimate(EdgeSLAM::Converter::toVector3d(pMP->GetWorldPos()));
			const int id = pMP->mnId + maxKFid + 1;
			vPoint->setId(id);
			vPoint->setMarginalized(true);
			optimizer.addVertex(vPoint);

			const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

			int nEdges = 0;
			//SET EDGES
			for (std::map<KeyFrame*, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
			{

				KeyFrame* pKF = mit->first;
				if (pKF->isBad() || pKF->mnId > maxKFid)
					continue;

				nEdges++;

				const cv::KeyPoint& kpUn = pKF->mvKeysUn[mit->second];

				if (pKF->mvuRight[mit->second] < 0) {
					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					auto e = new BaseSLAM::Optimization::EdgeSE3MonoFramePoint();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));

					e->setMeasurement(obs);
					const float& invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

					if (bRobust)
					{
						g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
						e->setRobustKernel(rk);
						rk->setDelta(thHuber2D);
					}

					e->fx_ = pKF->fx;
					e->fy_ = pKF->fy;
					e->cx_ = pKF->cx;
					e->cy_ = pKF->cy;

					optimizer.addEdge(e);
				}
				else {
					Eigen::Matrix<double, 3, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y, pKF->mvuRight[mit->second];

					auto e = new BaseSLAM::Optimization::EdgeSE3StereoFramePoint();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));

					e->setMeasurement(obs);
					const float& invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

					if (bRobust)
					{
						g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
						e->setRobustKernel(rk);
						rk->setDelta(thHuber3D);
					}

					e->fx_ = pKF->fx;
					e->fy_ = pKF->fy;
					e->cx_ = pKF->cx;
					e->cy_ = pKF->cy;
					e->bf_ = pKF->mbf;
					optimizer.addEdge(e);
				}
			}

			if (nEdges == 0)
			{
				optimizer.removeVertex(vPoint);
				vbNotIncludedMP[i] = true;
			}
			else
			{
				vbNotIncludedMP[i] = false;
			}
		}

		// Optimize!
		optimizer.initializeOptimization();
		optimizer.optimize(nIterations);

		// Recover optimized data

		//Keyframes
		for (size_t i = 0; i < vpKFs.size(); i++)
		{
			KeyFrame* pKF = vpKFs[i];
			if (pKF->isBad())
				continue;
			auto vSE3 = static_cast<BaseSLAM::Optimization::VertexFrame*>(optimizer.vertex(pKF->mnId));
			auto SE3quat = vSE3->estimate();
			if (nLoopKF == 0)
			{
				pKF->SetPose(CommonUtils::DataConverter::toCvMat(SE3quat));
			}
			else
			{
				pKF->mTcwGBA.create(4, 4, CV_32F);
				CommonUtils::DataConverter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
				pKF->mnBAGlobalForKF = nLoopKF;
			}
		}

		//Points
		for (size_t i = 0; i < vpMP.size(); i++)
		{
			if (vbNotIncludedMP[i])
				continue;

			GaussianPoint* pMP = vpMP[i];

			if (pMP->isBad())
				continue;
			auto vPoint = static_cast<BaseSLAM::Optimization::VertexPoint*>(optimizer.vertex(pMP->mnId + maxKFid + 1));

			if (nLoopKF == 0)
			{
				pMP->SetWorldPos(EdgeSLAM::Converter::toCvMat(vPoint->estimate()));
				pMP->UpdateNormalAndDepth();
			}
			else
			{
				pMP->mPosGBA.create(3, 1, CV_32F);
				EdgeSLAM::Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
				pMP->mnBAGlobalForKF = nLoopKF;
			}
		}

	}


	int Optimizer::PoseOptimization(Frame* pFrame)
	{
		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

		auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
		auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
		auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
		optimizer.setAlgorithm(algorithm);

		int nInitialCorrespondences = 0;

		// Set Frame vertex
		auto vSE3 = new BaseSLAM::Optimization::VertexFrame();
		vSE3->setEstimate(CommonUtils::DataConverter::toSE3Quat(pFrame->GetPose()));
		vSE3->setId(0);
		vSE3->setFixed(false);
		optimizer.addVertex(vSE3);

		// Set MapPoint vertices
		int N = pFrame->N;

		std::vector<BaseSLAM::Optimization::EdgeSE3MonoFrame*> vpEdgesMono;
		std::vector<size_t> vnIndexEdgeMono;
		vpEdgesMono.reserve(N);
		vnIndexEdgeMono.reserve(N);

		std::vector<BaseSLAM::Optimization::EdgeSE3StereoFrame*> vpEdgesStereo;
		std::vector<size_t> vnIndexEdgeStereo;
		vpEdgesStereo.reserve(N);
		vnIndexEdgeStereo.reserve(N);

		float deltaMono = sqrt(5.991);
		const float deltaStereo = sqrt(7.815);
		{
			//std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

			auto vpGPs = pFrame->mvGaussianPoints.get();

			for (int i = 0; i < N; i++)
			{
				auto pMP = vpGPs[i];
				if (pMP)
				{
					////다이나믹 오브젝트는 포함 안함
					//if (pMP->mnObjectID == 1 || pMP->mnObjectID == 57){
					//	pFrame->mvbOutliers[i] = true;
					//	continue;
					//}
					if (pFrame->mvuRight[i] < 0) {
						nInitialCorrespondences++;
						pFrame->mvbOutliers[i] = false;

						Eigen::Matrix<double, 2, 1> obs;
						cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
						obs << kpUn.pt.x, kpUn.pt.y;

						auto e = new BaseSLAM::Optimization::EdgeSE3MonoFrame();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
						e->setRobustKernel(rk);
						rk->setDelta(deltaMono);

						e->fx_ = pFrame->fx;
						e->fy_ = pFrame->fy;
						e->cx_ = pFrame->cx;
						e->cy_ = pFrame->cy;
						cv::Mat Xw = pMP->GetWorldPos();
						e->pos_w[0] = Xw.at<float>(0);
						e->pos_w[1] = Xw.at<float>(1);
						e->pos_w[2] = Xw.at<float>(2);

						optimizer.addEdge(e);

						vpEdgesMono.push_back(e);
						vnIndexEdgeMono.push_back(i);
					}
					else {
						nInitialCorrespondences++;
						pFrame->mvbOutliers[i] = false;

						Eigen::Matrix<double, 3, 1> obs;
						cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
						obs << kpUn.pt.x, kpUn.pt.y, pFrame->mvuRight[i];

						auto e = new BaseSLAM::Optimization::EdgeSE3StereoFrame();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
						e->setMeasurement(obs);
						float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

						g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
						e->setRobustKernel(rk);
						rk->setDelta(deltaStereo);

						e->fx_ = pFrame->fx;
						e->fy_ = pFrame->fy;
						e->cx_ = pFrame->cx;
						e->cy_ = pFrame->cy;
						e->bf_ = pFrame->mbf;

						cv::Mat Xw = pMP->GetWorldPos();
						e->pos_w[0] = Xw.at<float>(0);
						e->pos_w[1] = Xw.at<float>(1);
						e->pos_w[2] = Xw.at<float>(2);

						optimizer.addEdge(e);
						vpEdgesStereo.push_back(e);
						vnIndexEdgeStereo.push_back(i);
					}
				}//pMP

			}
		}

		if (nInitialCorrespondences < 3) {
			//delete linearSolver;
			//delete vSE3;
			return 0;
		}
		// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
		// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
		float chi2Mono[4] = { 5.991,5.991,5.991,5.991 };
		const float chi2Stereo[4] = { 7.815,7.815,7.815, 7.815 };
		int its[4] = { 10,10,10,10 };

		int nTest = 0;
		int nBad = 0;
		for (size_t it = 0; it < 4; it++)
		{

			vSE3->setEstimate(CommonUtils::DataConverter::toSE3Quat(pFrame->GetPose()));
			optimizer.initializeOptimization(0);
			optimizer.optimize(its[it]);

			nBad = 0;
			for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
			{
				auto e = vpEdgesMono[i];

				size_t idx = vnIndexEdgeMono[i];

				if (pFrame->mvbOutliers[idx])
				{
					e->computeError();
				}

				float chi2 = e->chi2();
				
				if (chi2 > chi2Mono[it])
				{
					pFrame->mvbOutliers[idx] = true;
					e->setLevel(1);
					nBad++;
					if (chi2 < chi2Stereo[it])
						nTest++;
				}
				else
				{
					pFrame->mvbOutliers[idx] = false;
					e->setLevel(0);
				}

				if (it == 2)
					e->setRobustKernel(0);
			}
			for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
			{
				auto e = vpEdgesStereo[i];

				const size_t idx = vnIndexEdgeStereo[i];

				if (pFrame->mvbOutliers[idx])
				{
					e->computeError();
				}

				const float chi2 = e->chi2();

				if (chi2 > chi2Stereo[it])
				{
					pFrame->mvbOutliers[idx] = true;
					e->setLevel(1);
					nBad++;
				}
				else
				{
					e->setLevel(0);
					pFrame->mvbOutliers[idx] = false;
				}

				if (it == 2)
					e->setRobustKernel(0);
			}

			if (optimizer.edges().size() < 10)
				break;
		}
		// Recover optimized pose and return number of inliers
		auto vSE3_recov = static_cast<BaseSLAM::Optimization::VertexFrame*>(optimizer.vertex(0));
		auto SE3quat_recov = vSE3_recov->estimate();
		cv::Mat pose = CommonUtils::DataConverter::toCvMat(SE3quat_recov);
		pFrame->SetPose(pose);
		//std::cout << "tracking test = " << nInitialCorrespondences <<" "<<nBad << " = "<<vpEdgesMono.size()<<" "<<vpEdgesStereo.size() << std::endl;
		return nInitialCorrespondences - nBad;
	}

	void Optimizer::LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap, long long ts)
	{
		// Local KeyFrames: First Breath Search from Current Keyframe
		std::list<KeyFrame*> lLocalKeyFrames;

		lLocalKeyFrames.push_back(pKF);
		pKF->mnBALocalForKF = pKF->mnId;

		const std::vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
		for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
		{
			KeyFrame* pKFi = vNeighKFs[i];
			pKFi->mnBALocalForKF = pKF->mnId;
			if (!pKFi->isBad())
				lLocalKeyFrames.push_back(pKFi);
		}

		// Local MapPoints seen in Local KeyFrames
		std::list<GaussianPoint*> lLocalMapPoints;
		for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			std::vector<GaussianPoint*> vpMPs = (*lit)->GetGaussianPointMatches();
			for (std::vector<GaussianPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
			{
				auto pMP = *vit;
				if (pMP)
					if (!pMP->isBad()) {
						/*if (pMP->mnObjectID == 1 || pMP->mnObjectID == 57)
							continue;*/
						if (pMP->mnBALocalForKF != pKF->mnId)
						{
							lLocalMapPoints.push_back(pMP);
							pMP->mnBALocalForKF = pKF->mnId;
						}
					}
			}
		}

		// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
		std::list<KeyFrame*> lFixedCameras;
		for (std::list<GaussianPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
		{
			std::map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
			for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				KeyFrame* pKFi = mit->first;

				if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
				{
					pKFi->mnBAFixedForKF = pKF->mnId;
					if (!pKFi->isBad())
						lFixedCameras.push_back(pKFi);
				}
			}
		}

		// Setup optimizer
		g2o::SparseOptimizer optimizer;
		auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
		auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
		auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
		optimizer.setAlgorithm(algorithm);

		if (pbStopFlag)
			optimizer.setForceStopFlag(pbStopFlag);

		unsigned long maxKFid = 0;

		// Set Local KeyFrame vertices
		for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			KeyFrame* pKFi = *lit;
			auto vSE3 = new BaseSLAM::Optimization::VertexFrame();
			vSE3->setEstimate(CommonUtils::DataConverter::toSE3Quat(pKFi->GetPose()));
			vSE3->setId(pKFi->mnId);
			vSE3->setFixed(pKFi->mnId == pKFi->GetMap()->GetInitKFid());
			optimizer.addVertex(vSE3);
			if (pKFi->mnId > maxKFid)
				maxKFid = pKFi->mnId;
		}

		// Set Fixed KeyFrame vertices
		for (std::list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
		{
			KeyFrame* pKFi = *lit;
			auto vSE3 = new BaseSLAM::Optimization::VertexFrame();
			vSE3->setEstimate(CommonUtils::DataConverter::toSE3Quat(pKFi->GetPose()));
			vSE3->setId(pKFi->mnId);
			vSE3->setFixed(true);
			optimizer.addVertex(vSE3);
			if (pKFi->mnId > maxKFid)
				maxKFid = pKFi->mnId;
		}

		// Set MapPoint vertices
		const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

		std::vector<BaseSLAM::Optimization::EdgeSE3MonoFramePoint*> vpEdgesMono;
		vpEdgesMono.reserve(nExpectedSize);

		std::vector<KeyFrame*> vpEdgeKFMono;
		vpEdgeKFMono.reserve(nExpectedSize);

		std::vector<GaussianPoint*> vpMapPointEdgeMono;
		vpMapPointEdgeMono.reserve(nExpectedSize);

		std::vector<BaseSLAM::Optimization::EdgeSE3StereoFramePoint*> vpEdgesStereo;
		vpEdgesStereo.reserve(nExpectedSize);

		std::vector<KeyFrame*> vpEdgeKFStereo;
		vpEdgeKFStereo.reserve(nExpectedSize);

		std::vector<GaussianPoint*> vpMapPointEdgeStereo;
		vpMapPointEdgeStereo.reserve(nExpectedSize);

		const float thHuberMono = sqrt(5.991);
		const float thHuberStereo = sqrt(7.815);

		for (std::list<GaussianPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
		{
			auto pMP = *lit;
			auto vPoint = new BaseSLAM::Optimization::VertexPoint();
			vPoint->setEstimate(EdgeSLAM::Converter::toVector3d(pMP->GetWorldPos()));
			int id = pMP->mnId + maxKFid + 1;
			vPoint->setId(id);
			vPoint->setMarginalized(true);
			optimizer.addVertex(vPoint);

			const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

			//Set edges
			for (std::map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				KeyFrame* pKFi = mit->first;

				if (!pKFi->isBad())
				{
					const cv::KeyPoint& kpUn = pKFi->mvKeysUn[mit->second];

					if (pKFi->mvuRight[mit->second] < 0) {
						Eigen::Matrix<double, 2, 1> obs;
						obs << kpUn.pt.x, kpUn.pt.y;

						auto e = new BaseSLAM::Optimization::EdgeSE3MonoFramePoint();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
						e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));

						e->setMeasurement(obs);
						const float& invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

						g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
						e->setRobustKernel(rk);
						rk->setDelta(thHuberMono);

						e->fx_ = pKFi->fx;
						e->fy_ = pKFi->fy;
						e->cx_ = pKFi->cx;
						e->cy_ = pKFi->cy;

						optimizer.addEdge(e);
						vpEdgesMono.push_back(e);
						vpEdgeKFMono.push_back(pKFi);
						vpMapPointEdgeMono.push_back(pMP);
					}
					else {
						Eigen::Matrix<double, 3, 1> obs;
						obs << kpUn.pt.x, kpUn.pt.y, pKFi->mvuRight[mit->second];

						auto e = new BaseSLAM::Optimization::EdgeSE3StereoFramePoint();

						e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
						e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));

						e->setMeasurement(obs);
						const float& invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
						e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

						g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
						e->setRobustKernel(rk);
						rk->setDelta(thHuberStereo);

						e->fx_ = pKFi->fx;
						e->fy_ = pKFi->fy;
						e->cx_ = pKFi->cx;
						e->cy_ = pKFi->cy;
						e->bf_ = pKFi->mbf;

						optimizer.addEdge(e);
						vpEdgesStereo.push_back(e);
						vpEdgeKFStereo.push_back(pKFi);
						vpMapPointEdgeStereo.push_back(pMP);
					}
				}
			}
		}

		if (pbStopFlag)
			if (*pbStopFlag)
				return;

		optimizer.initializeOptimization();
		optimizer.optimize(5);

		bool bDoMore = true;

		if (pbStopFlag)
			if (*pbStopFlag)
				bDoMore = false;

		if (bDoMore)
		{

			// Check inlier observations
			for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
			{
				auto e = vpEdgesMono[i];
				auto pMP = vpMapPointEdgeMono[i];

				if (pMP->isBad())
					continue;

				if (e->chi2() > 5.991 || !e->isDepthPositive())
				{
					e->setLevel(1);
				}

				e->setRobustKernel(0);
			}
			for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
			{
				auto e = vpEdgesStereo[i];
				auto pMP = vpMapPointEdgeStereo[i];

				if (pMP->isBad())
					continue;

				if (e->chi2() > 7.815 || !e->isDepthPositive())
				{
					e->setLevel(1);
				}

				e->setRobustKernel(0);
			}
			// Optimize again without the outliers
			optimizer.initializeOptimization(0);
			optimizer.optimize(10);

		}

		std::vector<std::pair<KeyFrame*, GaussianPoint*> > vToErase;
		vToErase.reserve(vpEdgesMono.size());

		// Check inlier observations       
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
		{
			auto e = vpEdgesMono[i];
			auto pMP = vpMapPointEdgeMono[i];

			if (pMP->isBad())
				continue;

			if (e->chi2() > 5.991 || !e->isDepthPositive())
			{
				KeyFrame* pKFi = vpEdgeKFMono[i];
				vToErase.push_back(std::make_pair(pKFi, pMP));
			}
		}
		for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
		{
			auto e = vpEdgesStereo[i];
			auto pMP = vpMapPointEdgeStereo[i];

			if (pMP->isBad())
				continue;

			if (e->chi2() > 7.815 || !e->isDepthPositive())
			{
				KeyFrame* pKFi = vpEdgeKFStereo[i];
				vToErase.push_back(std::make_pair(pKFi, pMP));
			}
		}

		// Get Map Mutex
		std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

		if (!vToErase.empty())
		{
			for (size_t i = 0; i < vToErase.size(); i++)
			{
				auto pKFi = vToErase[i].first;
				auto pMPi = vToErase[i].second;
				pKFi->EraseGaussianPointMatch(pMPi);
				pMPi->EraseObservation(pKFi);
			}
		}

		// Recover optimized data

		//Keyframes
		for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			KeyFrame* pKF = *lit;
			auto vSE3 = static_cast<BaseSLAM::Optimization::VertexFrame*>(optimizer.vertex(pKF->mnId));
			auto SE3quat = vSE3->estimate();
			pKF->SetPose(CommonUtils::DataConverter::toCvMat(SE3quat));
		}

		//Points
		for (std::list<GaussianPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
		{
			auto pMP = *lit;
			cv::Mat Xold = pMP->GetWorldPos();
			auto vPoint = static_cast<BaseSLAM::Optimization::VertexPoint*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
			pMP->SetWorldPos(EdgeSLAM::Converter::toCvMat(vPoint->estimate()));
			pMP->UpdateNormalAndDepth();
			pMP->mnLastUpdatedTime = ts;

			//std::cout << Xold.t() << " " << pMP->GetWorldPos().t() << std::endl;
		}
		//std::cout << "ba = " << vpEdgeKFMono.size() << " " << vpEdgeKFStereo.size() << std::endl;
	}


}