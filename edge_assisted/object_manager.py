

class Object:
    def __init__(self, id):
        self.id = id
        self._frames = {}

    def __contains__(self, fid):
        return fid in self._objects
    def __setitem__(self, fid, bbox):
        #네트워크로 전송받은 데이터를 추가함.
        self._frames[fid] = bbox;
    def __getitem__(self, fid):
        if fid in self._frames:
            return self._frames[fid]
        else:
            None

class ObjectManager:
    def __init__(self):
        self._objects = {}

        """
        ids, bboxs = object[fid] 
        오브젝트 ID
        오브젝트와 연결 된 프레임
        오브젝트의 바운딩 박스 등이 있음.
        프레임 id 입력시 프레임 내의 객체 정보가 나오는게 좋음.
        """
    def __contains__(self, item):
        #데이터를 전송받았는지 확인함.
        return item in self._objects

    def __setitem__(self, key, value):
        #네트워크로 전송받은 데이터를 추가함.
        self._objects[key] = value;

    def __getitem__(self, item):
        if item in self._objects:
            return self._objects[item]
        else:
            None