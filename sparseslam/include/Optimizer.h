#ifndef GAUSSIAN_SPARSE_SLAM_OPTIMIZER_H
#define GAUSSIAN_SPARSE_SLAM_OPTIMIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>

#include <../GaussianSparseSLAM/include/Types.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/dense/linear_solver_dense.h"

#include <Optimize/Sim3.h>

namespace EdgeSLAM {
}

namespace GaussianSparseSLAM {

	class KeyFrame;
	class Frame;
	class Map;
	class GaussianPoint;

	class Optimizer {
	public:
		void static BundleAdjustment(const std::vector<KeyFrame*>& vpKF, const std::vector<GaussianPoint*>& vpMP, int nIterations = 5, bool* pbStopFlag = NULL, const unsigned long nLoopKF = 0, const bool bRobust = true);
		void static GlobalBundleAdjustemnt(Map* pMap, int nIterations = 5, bool* pbStopFlag = NULL, const unsigned long nLoopKF = 0, const bool bRobust = true);
		static int PoseOptimization(Frame* pFrame);
		static void LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap, long long ts);
	};

}
#endif
