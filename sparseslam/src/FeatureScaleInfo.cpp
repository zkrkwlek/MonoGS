#include <FeatureScaleInfo.h>

namespace GaussianSparseSLAM {
	FeatureScaleInfo::FeatureScaleInfo(int _level, float _fScale) :mnLevels(_level), mfScaleFactor(_fScale)
	{
		mfLogScaleFactor = log(mfScaleFactor);
		mvScaleFactor.resize(mnLevels);
		mvLevelSigma2.resize(mnLevels);
		mvScaleFactor[0] = 1.0f;
		mvLevelSigma2[0] = 1.0f;
		for (int i = 1; i < mnLevels; i++)
		{
			mvScaleFactor[i] = mvScaleFactor[i - 1] * mfScaleFactor;
			mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
		}

		mvInvScaleFactor.resize(mnLevels);
		mvInvLevelSigma2.resize(mnLevels);
		for (int i = 0; i < mnLevels; i++)
		{
			mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
			mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
		}
	}
}