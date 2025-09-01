import ujson
import requests
import numpy as np
import threading
from socket import *

#데이터 처리 쓰레드

def resllm(id, src):
    # 이벤트가 있으면 set
    if id in response_events:
        response_events[id].set()

    res_llm = sess.post(FACADE_SERVER_ADDR + "/Download?keyword=" + "resllm" + "&id=" + str(id) + "&src=" + src, "")
    string_data = res_llm.content.decode('utf-8')
    #json_object = ujson.loads(string_data)
    print(id, string_data)



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

if __name__ == '__main__':

    #서버 설정 및 키워드 등록
    FACADE_SERVER_ADDR = 'http://143.248.6.25:35005'
    strServerName = 'LLMClient'
    SendKeywords = 'reqllm'
    ReceivedKeywords = ['resllm']

    Data = {}
    capacity = 0
    sess = requests.Session()
    sess.post(FACADE_SERVER_ADDR + "/Connect", ujson.dumps({
        # 'port':opt.port,'key': keyword, 'prior':opt.prior, 'ratio':opt.ratio
        'src': strServerName, 'type1': 'server', 'type2': 'test', 'keyword': SendKeywords, 'capacity': capacity,
        'Additional': None
    }))
    ECHO_SERVER_ADDR = ('143.248.6.25', 35001)
    ECHO_SOCKET = socket(AF_INET, SOCK_DGRAM)
    for keyword in ReceivedKeywords:
        temp = ujson.dumps({'type1': 'connect', 'keyword': keyword, 'src': strServerName, 'type2': 'all'})
        ECHO_SOCKET.sendto(temp.encode(), ECHO_SERVER_ADDR)
        Data[keyword] = {}

    #Load Dataset
    keydataset = "TUM"
    scene_id = "2_desk"
    quality = 70
    skip = 5
    srcc = keydataset+scene_id+"_"+str(quality)+".color"
    srct = keydataset+scene_id+".ts"
    sess.post(FACADE_SERVER_ADDR + "/Load?keyword="+keydataset+"&src=" + srcc, "")

    #이미지 시컨스 id 다운로드,
    res_ts = sess.post(FACADE_SERVER_ADDR+"/Get?keyword="+keydataset+"&src=" + srct,"")
    ts_array = np.frombuffer(res_ts.content, dtype=np.int32)
    start_id = ts_array[0]
    end_id = ts_array[-1]

    #LLM 처리 쓰레드
    th1 = threading.Thread(target=udpthread)
    th1.start()

    # 이벤트 핸들러 저장용 전역 사전
    response_events = {}

    #LLM 요청
    for id in range(start_id, end_id, skip):
        # 이벤트 생성 후 등록
        response_events[id] = threading.Event()

        #서버의 이미지 시컨스에서 이미지 다운로드
        res_img = sess.post(FACADE_SERVER_ADDR + "/Download?keyword="+keydataset+"&id=" + str(id) + "&src=" + srcc,"")
        #데이터 처리를 위한 이미지 업로드
        sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=Image&id=" + str(id) + "&src=" + strServerName,res_img.content)
        #LLM 요청
        sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=reqllm&id=" + str(id) + "&src=" + strServerName,"")

        # resllm 콜백 호출 시까지 대기 (최대 30초 타임아웃 권장)
        response_events[id].wait(timeout=30)

        # 대기 완료 후 이벤트 제거
        del response_events[id]
