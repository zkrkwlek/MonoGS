#ifndef GAUSSIAN_SPARSE_SLAM_VISUALIZER_H
#define GAUSSIAN_SPARSE_SLAM_VISUALIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>
#include <ConcurrentList.h>

#include <../GaussianSparseSLAM/include/Types.h>

namespace GaussianSparseSLAM {

	class GSSLAM;
	class Map;

	class Visualizer {
	public:
		Visualizer();
		Visualizer(GSSLAM* pSystem);
		virtual ~Visualizer();

	public:
		void Init(int w, int h, bool _bSave = false, int _inc = 20);
		void Run();
		void SetBoolDoingProcess(bool b);
		bool isDoingProcess();
		static void CallBackFunc(int event, int x, int y, int flags, void* userdata);
		void SetMap(Map* pMap);
		Map* GetMap();
		std::string strMapName;
	public:
		void SetScale(int _n) {
			mnVisScale = _n;
		}
		void ResizeImage(const cv::Mat& src, cv::Mat& dst);
		void SetOutputImage(const cv::Mat& out, int type);
		cv::Mat GetOutputImage(int type);
		bool isOutputTypeChanged(int type);
		int mnWindowImgCols, mnWindowImgRows;
		GSSLAM* mpSystem;
		Map* mpMap;
		int mnVisMode;
		int mnVisScale;
		int mnDisplayX, mnDisplayY;
		int mnWidth, mnHeight;
		cv::Mat mVisPoseGraph;
		cv::Point2f mVisMidPt, mVisPrevPt;
		cv::Size mSizeOutputImg;

	public:
		
	private:
		cv::Point2f ConvertVisPt(cv::Mat T, cv::Mat x3D);
		std::vector<cv::Mat> mvOutputImgs;
		std::vector<cv::Rect> mvRects;
		cv::Mat mOutputImage;
		std::vector<bool> mvOutputChanged;
		std::mutex mMutexOutput;
		////output½Ã°¢È­
		//////////////////////

		std::mutex mMutexMap;
		//std::mutex mMutexUserList;
		//std::set<User*> mspUserLists;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;

		bool mbSaveVisImage;
		int mnIncForSaveImg;

	};

}
#endif
