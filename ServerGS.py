import threading
import ujson
import time
import numpy as np
import requests
import cv2
from socket import *
import argparse
import gzip
import psutil

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

def datakf(id, src, ts = '0.0'):
    strsplit = src.split('.')
    map = strsplit[0]
    kf_id = id
    device_name = strsplit[1]
    data_type = strsplit[2]
    frame_id = (strsplit[3])

    a = time.time()
    # image
    res_rgb = sess.post(
        FACADE_SERVER_ADDR + "/Download?keyword=" + data_type + "&id=" + frame_id + "&src=" + device_name, "")
    img_array = np.frombuffer(res_rgb.content, dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # if img_cv is not None:
    image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    #image = torch.from_numpy(image)

    # keypoint
    res_kps = sess.post(
        FACADE_SERVER_ADDR + "/Download?keyword=" + "resdetect" + "&id=" + frame_id + "&src=" + device_name, "")
    keypoints = torch.from_numpy(np.frombuffer(res_kps.content, dtype=np.float32).copy())
    keypoints = keypoints.reshape(-1, 2)

    # depth
    res_depth = sess.post(
        FACADE_SERVER_ADDR + "/Download?keyword=" + "resdepthanything" + "&id=" + frame_id + "&src=" + device_name, "")
    depth_array = np.frombuffer(res_depth.content, dtype=np.uint8)
    depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
    depth = (depth.astype(np.float64) / 1000.0) #torch.from_numpy

    # pose
    src2 = map + '.' + device_name + '.' + frame_id
    res_pose = sess.post(
        FACADE_SERVER_ADDR + "/Download?keyword=" + "datakfpose" + "&id=" + str(kf_id) + "&src=" + src2, "")
    pose_array = np.frombuffer(res_pose.content[:48], dtype=np.float32).copy()
    #pose_array = torch.from_numpy(pose_array)
    pose_array = pose_array.reshape(-1, 3)

    T = np.zeros((4, 4), dtype=np.float32)
    T[:3, :3] = pose_array[:3, :3]
    T[:3, 3] = pose_array[3, :].T
    T[3, 3] = 1

    # sparse map
    res_tmp_map = sess.post(
        FACADE_SERVER_ADDR + "/Download?keyword=" + "datasparsemap" + "&id=" + str(kf_id) + "&src=" + src2, "")
    tmp_map_array = torch.from_numpy(np.frombuffer(res_tmp_map.content, dtype=np.float32).copy())
    tmp_map_array = tmp_map_array.reshape(-1, 4)
    b = time.time()

    slam.AddKeyFrame(kf_id, image, keypoints, depth, T, device_name)
    print("Add Keyframe", b-a, kf_id)

def reqgsmapping(id, src, ts = '0.0'):
    strsplit = src.split('.')
    map = strsplit[0]
    kf_id = id
    device_name = strsplit[1]
    data_type = strsplit[2]
    frame_id = (strsplit[3])
    if len(strsplit) > 4:
        neighbor_kfs = [int(x) for x in strsplit[4:]]
    else:
        neighbor_kfs = None

    a = time.time()
    #image
    res_rgb = sess.post(
        FACADE_SERVER_ADDR + "/Download?keyword=" + data_type + "&id=" + frame_id + "&src=" + device_name, "")
    img_array = np.frombuffer(res_rgb.content, dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    #if img_cv is not None:
    image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    #image = torch.from_numpy(image)

    #keypoint
    res_kps = sess.post(
        FACADE_SERVER_ADDR + "/Download?keyword=" + "resdetect" + "&id=" + frame_id + "&src=" + device_name, "")
    keypoints = torch.from_numpy(np.frombuffer(res_kps.content, dtype=np.float32).copy())
    keypoints = keypoints.reshape(-1,2)

    #depth
    res_depth = sess.post(
        FACADE_SERVER_ADDR + "/Download?keyword=" + "resdepthanything" + "&id=" + frame_id + "&src=" + device_name, "")
    depth_array = np.frombuffer(res_depth.content, dtype=np.uint8)
    depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
    depth = (depth.astype(np.float64) / 1000.0) #torch.from_numpy

    #pose
    src2 = map+'.'+device_name+'.'+frame_id
    res_pose = sess.post(
        FACADE_SERVER_ADDR + "/Download?keyword=" + "datakfpose" + "&id=" + str(kf_id) + "&src=" + src2, "")
    pose_array = np.frombuffer(res_pose.content[:48], dtype=np.float32).copy()
    #pose_array = torch.from_numpy(pose_array)
    pose_array = pose_array.reshape(-1, 3)
    T = np.zeros((4,4),dtype=np.float32)
    T[:3,:3] = pose_array[:3,:3]
    T[:3,3] = pose_array[3,:].T
    T[3,3] = 1
    #R = pose_array[:3,:3]
    #t = pose_array[3,:].unsqueeze(1)

    #sparse map
    res_tmp_map = sess.post(
        FACADE_SERVER_ADDR + "/Download?keyword=" + "datasparsemap" + "&id=" + str(kf_id) + "&src=" + src2, "")
    tmp_map_array = torch.from_numpy(np.frombuffer(res_tmp_map.content, dtype=np.float32).copy())
    tmp_map_array = tmp_map_array.reshape(-1, 4)

    #gaussian selection
    #gaussian generation
    #optimization
    #return

    nw = int(image.shape[1]/4)
    nh = int(image.shape[0]/4)

    image = cv2.resize(image, (nw,nh))
    depth = cv2.resize(depth, (nw,nh))

    slam.AddKeyFrame(kf_id, image, keypoints, depth, T, device_name)

    #observation
    b = time.time()
    print('Gaussian Splatting Mapping', b-a, map, kf_id, device_name, data_type, frame_id,":",image.shape, depth.shape, keypoints.shape, tmp_map_array.shape)
    print('neighbor', neighbor_kfs)

    # for neigh_id in neighbor_kfs:
    #    print(slam.keyframes[neigh_id].id)

    #thread??
    slam.GenerateLocalMap(kf_id, neighbor_kfs, device_name)

def GSDeviceConnect(id, src, ts = '0.0'):
    res = sess.post(FACADE_SERVER_ADDR + "/Download?keyword=" + "GSDeviceConnect" + "&id=" + str(id) + "&src=" + src, "")
    cam_data = np.frombuffer(res.content[:80], dtype=np.float32)
    k = 4
    w = int(int(cam_data[0]) / k)
    h = int(int(cam_data[1]) / k)
    fx = cam_data[2] / k
    fy = cam_data[3] / k
    cx = cam_data[4] / k
    cy = cam_data[5] / k
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    D = cam_data[6:11]
    #bMapper = bool(cam_data[18])
    slam.AddDevice(src, K, D, w, h, False)
    print("GSDeviceConnect", src)

bufferSize = 1024
def udpthread():

    while True:
        bytesAddressPair = ECHO_SOCKET.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        data = ujson.loads(message.decode())
        id = data['id']
        src = data['src']
        keyword = data['keyword']
        ts = data['ts2'] if 'ts2' in data else None

        #cpu_usage = p.cpu_percent(interval=1)
        #memory_usage = p.memory_info().rss
        #print(f"Server = CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage} bytes = cores ", os.cpu_count())

        if keyword in globals():
            globals()[keyword](id,src, ts)


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
        '--RKeywords', type=str,default='datakf,reqgsmapping,GSDeviceConnect',
        help='Received keyword lists')
    ##서버에서 생성한 데이터를 등록하는 키워드
    ##유니크 키워드 생성 필요
    ##다른 서버 또는 기기에서 해당 데이터 이용 가능
    parser.add_argument(
        '--SKeywords', type=str,default='requnidepth,resobjrecon,reqsalad',
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

    sensor_type = "rgbd"  # "mono
    dataset = "tum"
    sequence = "fr2_desk.yaml"
    config_path = f"./configs/{sensor_type}/{dataset}/{sequence}"
    config = load_config(config_path)
    bTrack = True
    bPoseUpdate = True
    slam = EdgeGSSLAM(config, tracking_mode=bTrack, mapping_update_pose=bPoseUpdate)

    print("slam.run start")

    th1 = threading.Thread(target=udpthread)
    th1.start()
    print("thread start")