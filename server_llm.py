import threading
import ujson
import time
import numpy as np
import requests
import cv2
from socket import *
import argparse
import gzip

import os
import json
import torch
import base64
import numpy as np
from openai import OpenAI
import time
# OpenAI 설정
client = OpenAI(api_key="AAA")
# 멀티뷰 경로 설정
base_dir = "/home/daringspirit/SegAnyGaussians/datasets/lerf/figurines/furniture"
output_path = os.path.join(base_dir, "gaussian_physics_metadata.json")
# GPT 프롬프트
prompt_text = """
You are a material property estimation assistant.
From the image, estimate the **detailed physical material properties** of the object using visual cues like size, material texture, structure, and common sense.
Do NOT use any fixed density or mass lookup table.
Estimate based on:
- visible material proportions (e.g., thick wood frame vs thin foam cushion)
- realistic density ranges of materials (e.g., wood > foam)
- typical object scale (e.g., a chair < bed in volume)
Steps:
1. Guess the object category. Be specific (e.g., 'kingsize bed', 'dining table').
2. Estimate the total mass and density of the object.
3. Break down into major parts and estimate each part’s:
   - name
   - material
   - density
   - static friction coefficient
   - mass
Constraints:
- The sum of part masses must match the total mass (±1.5kg allowed).
- All values must be physically consistent.
- Ensure values make sense even if object scale is ambiguous.
Return valid JSON only, like:
[
  {
    "part": "entire object",
    "category": "tea table",
    "density": 500,
    "mass": 12.0
  },
  {
    "part": "legs",
    "composition": "metal",
    "density": 7800,
    "mass": 8.0,
    "staticFriction": 0.3
  },
  {
    "part": "top",
    "composition": "wood",
    "density": 650,
    "mass": 4.0,
    "staticFriction": 0.5
  }
]
"""
# clip 함수: 가장 중심에 가까운 대표 이미지 선택
# def get_representative_image(instance_imgs, all_embeddings, all_filenames):
#     indices = [all_filenames.index(img) for img in instance_imgs]
#     vectors = all_embeddings[indices]
#     center = np.mean(vectors, axis=0)
#     dists = np.linalg.norm(vectors - center, axis=1)
#     closest_idx = np.argmin(dists)
#     return instance_imgs[closest_idx]
# 함수: LLM 추론
def query_llm_with_image(image_path):
    with open(image_path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that estimates physical properties of objects."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content
# 텍스트만으로 LLM 다시 호출
def query_llm_with_text_only(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
def validate_physics_sanity(json_text):
    sanity_check_prompt = f"""
You are a validator of physical plausibility.
Given the following estimated material properties in JSON, check:
- Are the density and mass values realistic for such an object?
- Are part materials reasonable?
- Is the sum of masses consistent with total?
Only return:
- 'valid' if plausible
- or a string error message if not.
Input:
{json_text}
"""
    response = query_llm_with_text_only(sanity_check_prompt)
    return response.strip()

#이미지 쿼리 처리
def reqllm(id, src):

    res_rgb = sess.post(FACADE_SERVER_ADDR + "/Download?keyword=" + datatype + "&id=" + str(id) + "&src=" + src, "")
    #jpeg 인코딩 된 이미지 다운로드
    img_array = np.frombuffer(res_rgb.content, dtype=np.uint8)
    #base64로 변환
    img_base64 = base64.b64encode(img_array).decode('utf-8')

    a = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that estimates physical properties of objects."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]}
        ],
        temperature=0.3
    )
    #response.choices[0].message.content
    b = time.time()
    res_llm = sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=" + "resllm" + "&id=" + str(id) + "&src=" + src, response.choices[0].message.content)
    print(id, b - a, res_llm)

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

        #cpu_usage = p.cpu_percent(interval=1)
        #memory_usage = p.memory_info().rss
        #print(f"Server = CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage} bytes = cores ", os.cpu_count())

        if keyword in globals():
            globals()[keyword](id,src)
        #predict(message)

if __name__ == '__main__':
    ##################################################
    ##arguments parsing
    parser = argparse.ArgumentParser(
        description='LLM Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ##클라이언트에서 생성한 데이터의 알림을 받는 키워드
    ##클라이언트와 키워드 일치 필요
    ##','으로 연결하여 다중 키워드 등록
    ##ex)'image,segmentation'
    parser.add_argument(
        '--RKeywords', type=str,default='reqllm',
        help='Received keyword lists')
    ##서버에서 생성한 데이터를 등록하는 키워드
    ##유니크 키워드 생성 필요
    ##다른 서버 또는 기기에서 해당 데이터 이용 가능
    parser.add_argument(
        '--SKeywords', type=str,default='resllm',
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
    strServerName = 'LLMServer'
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

    th1 = threading.Thread(target=udpthread)
    th1.start()
    print("thread start")
    #mapping_process.join()
