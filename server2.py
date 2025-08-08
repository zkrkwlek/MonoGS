import threading
import ujson
import time
import numpy as np
import requests
import cv2
from socket import *
import argparse

import os
#os.environ['TORCH_USE_CUDA_DSA'] = "1"

import torch

#처리 시간 등 기록용
import csv
import os.path
import keyboard
keyboard.add_hotkey("ctrl+s",lambda: savecsv())

##MonoGS
#import torch.multiprocessing as mp
#from multiprocessing import Process, Queue
from utils.config_utils import load_config
from edgeslam import EdgeGSSLAM
#from utils.objslsam_backend import ObjectBackEnd
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2

##test
from python_orb_slam3 import ORBExtractor

num_device = 1
strmodel = "temp"
path_sv = "./evaluation/"+strmodel+"_"+str(num_device)+".csv"
if(not os.path.exists(path_sv)):
    with open(path_sv, 'w', newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(["processing","download","upload"])
        csvfile.close()

f = open(path_sv, 'a', newline='')
wr = csv.writer(f)
csvdatas=[]

def savecsv():
    if len(csvdatas)==0:
        return
    with open(path_sv, 'a', newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerows(csvdatas)

        n = len(csvdatas)

        csvfile.close()
        csvdatas.clear()
        print("save csv", n, len(csvdatas))
# 처리 시간 등 기록용

#데이터 처리 부분 : 구현되어야 하는 부분
def predict(message):
    # 구현 부분
    #데이터 다운로드

    #처리

    #업로드

    return

def yolosegc(id, src):
    res = sess.post(FACADE_SERVER_ADDR + "/Download?keyword=" + "yolosegc" + "&id=" + str(id) + "&src=" + src,"")
    contour_array = np.frombuffer(res.content, dtype=np.uint16)

    n_array = len(contour_array)
    idx = 0
    contours = []
    while True:
        iid = (contour_array[idx])
        idx += 1
        ni = (contour_array[idx])
        idx += 1

        contour = []
        for ti in range(ni):
            x = (contour_array[idx])
            idx += 1
            y = (contour_array[idx])
            idx += 1
            contour.append((x, y))
        contours.append(contour)
        if idx == n_array:
            break
    slam.AddContours(id, contours)

def resdepthanything(id,src):

    #if not slam.CheckFrame(id):
        #print("resdepthanything", id)
        try:
            res_pose = sess.post(
                FACADE_SERVER_ADDR + "/Download?keyword=" + "FrameUpdate" + "&id=" + str(id) + "&src=" + src, "")
            if len(res_pose.content) == 0:
                print("?????????????")
            array = np.frombuffer(res_pose.content, dtype=np.float32)
            fid = int(array[1])

            res_rgb = sess.post(
                FACADE_SERVER_ADDR + "/Download?keyword=" + datatype + "&id=" + str(id) + "&src=" + src, "")
            img_array = np.frombuffer(res_rgb.content, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img_cv is not None:
                t = array[6:9]
                R = array[9:18].reshape(3, 3)

                image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                # gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

                res_depth = sess.post(
                    FACADE_SERVER_ADDR + "/Download?keyword=" + "resdepthanything" + "&id=" + str(id) + "&src=" + src,
                    "")
                depth_array = np.frombuffer(res_depth.content, dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                depth = depth.astype(np.float64) / 1000.0

                frame = slam.AddFrame(fid, image, R, t, depth=depth)
                # slam.SetDepth(id, depth)
                slam.edge_queue.put(id)

                # ss = time.time()
                # list<cv2.Keypoint?>, numpy.ndarray
                #frame.keypoints, frame.descriptors = orb_extractor.detectAndCompute(img_cv)
                #print(frame.keypoints[0].pt,frame.keypoints[0].pt[0],frame.keypoints[0].pt[1])
                """
                image_with_keypoints = cv2.drawKeypoints(
                    img_cv,
                    source_keypoints,
                    None,
                    color=(0, 255, 0),  # 녹색
                    # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # 크기와 방향도 함께 표시
                )
                cv2.imshow('ORB Keypoints', image_with_keypoints)
                cv2.waitKey(1)
                # ee = time.time()
                # print("test ", ee - ss)
                """
        except ValueError as e:
            #print(f"error : {e}")
            pass
    #else:
    #print("?????",id)
    
        """
        ##depth 시각화
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_AUTUMN)
        cv2.imshow("depth", colored)
        cv2.waitKey(1)
        """

def ObjectMapCreation(id,src):
    res = sess.post(FACADE_SERVER_ADDR + "/Download?keyword=" + "ObjectMapCreation" + "&id=" + str(id) + "&src=" + src,"")
    array = np.frombuffer(res.content, dtype=np.float32)
    t = array[:3]
    R = np.array(array[3:12]).reshape(3, 3)
    oid = id
    print("Add Object", oid)
    #slam.AddObject(oid,R,t)

    #slam.backend.ObjectMapInitialization()

def ObjectMapUpdate(id,src):
    #print("ObjectMapUpdate")
    """
    data = ujson.loads(msg.decode())
    id = data['id']
    src = data['src']
    ts = data['ts']
    ss = src.split('.')
    """
    res = sess.post(FACADE_SERVER_ADDR + "/Download?keyword=" + "ObjectMapUpdate" + "&id=" + str(id) + "&src=" + src, "")
    array = np.frombuffer(res.content, dtype=np.float32)
    oid = int(array[0])
    fid = int(array[1])
    bbox = array[2:6]
    slam.AddObjectBBox(fid, oid, bbox)


    """
    if not slam.CheckFrame(fid):
        #Frame 추가
        t = array[6:9]
        R = array[9:18].reshape(3, 3)

        res = sess.post(
        FACADE_SERVER_ADDR + "/Download?keyword=" + datatype + "&id=" + str(id) + "&src=" + src.split('.')[0], "")
        img_array = np.frombuffer(res.content, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_cv is not None:
            image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            #gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

            slam.AddFrame(fid, image, R, t)
            #slam.edge_queue.put(fid)

            #slam.ConvertViewePoint(fid, projection_matrix)
            #print('add frame', fid, src, slam.running)

    else:
        print('already exist ',fid, src)
    """
    #print(src, id, array[2:6], array[6:9],array[9:18])
    #print("Find object", oid)
    #object = slam.objects[oid]
    """
    if not object.initialized and not object.used:
        slam.queue.put(["init", oid, fid, None])
    elif not object.used:
        slam.queue.put(["frame", oid, fid, bbox])
    """

    """
    if not slam.IsInitialized(oid):
        if not slam.IsUsed(oid):
            slam.queue.put(["init", oid, fid, None])
        else:
            slam.queue.put(["frame", oid, fid, bbox])
    else:
        #if not slam.IsUsed(oid):
        slam.queue.put(["frame", oid, fid, bbox])
    """
    #slam.FrameQueue.put(["f", oid, fid, bbox])

#데이터 처리 쓰레드
bufferSize = 1024
def udpthread():

    while True:
        bytesAddressPair = ECHO_SOCKET.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        data = ujson.loads(message.decode())
        id = data['id']
        src = data['src']
        keyword = data['keyword']

        if keyword in globals():
            globals()[keyword](id,src)
        #predict(message)

def start_slam_process(slam_instance, queue):
    print("1312341234")
    slam_instance.queue = queue
    slam_instance.running = True
    print("???????")
    slam_instance.run()

if __name__ == '__main__':
    ##################################################
    ##arguments parsing
    parser = argparse.ArgumentParser(
        description='Object Map Generation Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ##클라이언트에서 생성한 데이터의 알림을 받는 키워드
    ##클라이언트와 키워드 일치 필요
    ##','으로 연결하여 다중 키워드 등록
    ##ex)'image,segmentation'
    parser.add_argument(
        '--RKeywords', type=str,default='ObjectMapCreation,ObjectMapUpdate,resdepthanything,yolosegc',
        help='Received keyword lists')
    ##서버에서 생성한 데이터를 등록하는 키워드
    ##유니크 키워드 생성 필요
    ##다른 서버 또는 기기에서 해당 데이터 이용 가능
    parser.add_argument(
        '--SKeywords', type=str,default='resobjrecon',
        help='Sendeded keyword lists')
    ##전송받는 데이터의 타입 설정.
    parser.add_argument(
        '--DataType', type=str, default='Image',
        help='Data type')
    ##서버 아이피.
    ##추후 서버 아이피와 포트가 변경되면 수정
    parser.add_argument(
        '--FACADE_SERVER_ADDR', type=str, default='http://143.248.6.25:35005',
        help='facade server address')
    parser.add_argument(
        '--ECHO_SERVER_IP', type=str, default='143.248.6.25',
        help='ip address')
    parser.add_argument(
        '--ECHO_SERVER_PORT', type=int, default=35001,
        help='port number')
    ##MonoGS
    parser.add_argument(
        '--config', type=str, default='./configs/rgbd/tum/fr2_desk.yaml',
        help='port number')
    parser.add_argument("--eval", action="store_true")
    ##MonoGS
    opt = parser.parse_args()
    ##arguments parsing

    ###CONNECT
    Data = {}

    capacity = 0
    ##Echo server

    ##통신 서버와 주고받을 키워드 등록 과정
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    ReceivedKeywords = opt.RKeywords.split(',')
    SendKeywords = opt.SKeywords
    datatype = opt.DataType

    sess = requests.Session()
    strServerName = 'ObjectGaussianSplattingServer'
    sess.post(FACADE_SERVER_ADDR + "/Connect", ujson.dumps({
        # 'port':opt.port,'key': keyword, 'prior':opt.prior, 'ratio':opt.ratio
        'src': strServerName, 'type1': 'server', 'type2': 'test', 'keyword': SendKeywords, 'capacity':capacity, 'Additional': None
    }))
    ECHO_SERVER_ADDR = (opt.ECHO_SERVER_IP, opt.ECHO_SERVER_PORT)
    ECHO_SOCKET = socket(AF_INET, SOCK_DGRAM)
    for keyword in ReceivedKeywords:
        temp = ujson.dumps({'type1': 'connect', 'keyword': keyword, 'src': strServerName, 'type2': 'all'})
        ECHO_SOCKET.sendto(temp.encode(), ECHO_SERVER_ADDR)
        Data[keyword] = {}
    # Echo server connect
    ######################

    ####LOAD MODEL
    ##구현한 모델 로드
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Load model")

    sensor_type = "rgbd" #"mono
    dataset ="tum"
    sequence="fr2_desk.yaml"
    config_path = f"./configs/{sensor_type}/{dataset}/{sequence}"
    config = load_config(config_path)
    bTrack = True
    bPoseUpdate = True
    slam = EdgeGSSLAM(config, tracking_mode=bTrack, mapping_update_pose = bPoseUpdate)

    ##test
    orb_extractor = ORBExtractor()

    """
    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=slam.dataset.fx,
        fy=slam.dataset.fy,
        cx=slam.dataset.cx,
        cy=slam.dataset.cy,
        W=slam.dataset.width,
        H=slam.dataset.height,
    ).transpose(0, 1)
    projection_matrix = projection_matrix.to(device=device)
    """

    #print(slam.dataset.poses[0], type(slam.dataset.poses[0]), slam.dataset.poses[0].dtype)

    #task_queue = Queue()
    #slam.MappingQueue = Queue()
    #slam.FrameQueue = Queue()

    #slam.running = True
    #mapping_process = threading.Thread(target=slam.run)
    #mapping_process.start()
    slam.run()
    print("slam.run start")

    th1 = threading.Thread(target=udpthread)
    th1.start()
    print("thread start")
    #mapping_process.join()