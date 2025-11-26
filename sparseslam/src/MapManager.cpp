#include <../GaussianSparseSLAM/include/MapManager.h>
#include <../GaussianSparseSLAM/include/Map.h>

namespace GaussianSparseSLAM {

    MapManager::MapManager()
        :mpCurrentMap(nullptr)
    {}
    MapManager::MapManager(int initKFid)
        : mnLastInitKFidMap(initKFid), mpCurrentMap(nullptr)
    {}
    MapManager::~MapManager()
    {}

    Map* MapManager::CreateNewMap()
    {
        std::unique_lock<std::mutex> lock(mMutexAtlas);

        if (mpCurrentMap) {
            if (mspMaps.Size() > 0 && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
                mnLastInitKFidMap = mpCurrentMap->GetMaxKFid() + 1; //The init KF is the next of current maximum

            mpCurrentMap->SetStoredMap();

            //if(mHasViewer)
            //    mpViewer->AddMapToCreateThumbnail(mpCurrentMap);
        }

        mpCurrentMap = new Map(mnLastInitKFidMap);
        mpCurrentMap->SetCurrentMap();
        mspMaps.Update(mpCurrentMap);
        return mpCurrentMap;
    }
    Map* MapManager::GetCurrentMap() {
        std::unique_lock<std::mutex> lock(mMutexAtlas);
        return mpCurrentMap;
    }
    void MapManager::ChangeMap(Map* pMap)
    {
        std::unique_lock<std::mutex> lock(mMutexAtlas);

        if (mpCurrentMap) {
            mpCurrentMap->SetStoredMap();
        }

        mpCurrentMap = pMap;
        mpCurrentMap->SetCurrentMap();
    }
    void MapManager::SetMapBad(Map* pMap)
    {
        mspMaps.Erase(pMap);
        pMap->SetBad();
        mspBadMaps.Update(pMap);
    }

    void MapManager::RemoveBadMaps()
    {
        /*for(Map* pMap : mspBadMaps)
        {
            delete pMap;
            pMap = static_cast<Map*>(NULL);
        }*/
        mspBadMaps.Clear();
    }
}