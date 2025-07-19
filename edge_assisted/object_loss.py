import torch

class ObjectLoss:
    def __init__(self):
        pass

    def convert_index_to_box(self, indices, ids, boxes):
        indices = indices.view(-1)  # (N,)
        ids = ids.view(-1)  # (M,)

        N = indices.shape[0]
        result = torch.zeros((N, 4), dtype=boxes.dtype, device=boxes.device)

        valid_mask = indices >= 0
        if valid_mask.any():
            # (N_valid, M): indices[valid_mask] == ids[None, :]
            # broadcasting 비교로 일치하는 위치 찾기
            matches = (indices[valid_mask].unsqueeze(1) == ids.unsqueeze(0))  # (N_valid, M)
            # 일치하는 첫 번째 인덱스(없으면 -1)
            has_match = matches.any(dim=1)
            idx_in_boxes = matches.float().cumsum(dim=1).argmax(dim=1)
            # 매칭이 없는 경우 -1로
            idx_in_boxes = torch.where(has_match, idx_in_boxes, torch.full_like(idx_in_boxes, -1))

            # 실제로 매칭된 것만 복사
            valid_box_mask = idx_in_boxes >= 0
            if valid_box_mask.any():
                result_idx = valid_mask.nonzero(as_tuple=True)[0][valid_box_mask]
                box_idx = idx_in_boxes[valid_box_mask]
                result[result_idx] = boxes[box_idx]
        return result


        """
        N = indices.shape[0]
        result = torch.zeros((N, 4), dtype=boxes.dtype, device=boxes.device)  # 기본값 [0,0,0,0]

        # 유효한 인덱스만 골라서 박스 복사
        valid_mask = indices >= 0
        if valid_mask.any():
            result[valid_mask] = boxes[indices[valid_mask]]
        return result
        """

    def select_object_gaussians(self, projection, ids, boxes):
        #boxes는 cuda 로 올려 놓기

        N = projection.shape[0]
        M = boxes.shape[0]

        # 각 박스의 (x1, y1, x2, y2)로 변환
        x1y1 = boxes[:, :2]  # (M, 2)
        x2y2 = boxes[:, :2] + boxes[:, 2:]  # (M, 2)

        # 포인트를 (N, 1, 2)로, 박스를 (1, M, 2)로 브로드캐스팅
        pts = projection.unsqueeze(1)  # (N, 1, 2)
        x1y1 = x1y1.unsqueeze(0)  # (1, M, 2)
        x2y2 = x2y2.unsqueeze(0)  # (1, M, 2)

        # 포함 조건: x1 <= px < x2, y1 <= py < y2
        in_box = ((pts >= x1y1) & (pts < x2y2)).all(dim=2)  # (N, M)

        # 포인트별로, 포함되는 첫 번째 박스 인덱스 (없으면 -1)
        any_in_box = in_box.any(dim=1)
        # in_box에서 True인 첫 번째 인덱스 찾기 (없으면 0)
        first_idx = in_box.float().cumsum(dim=1).argmax(dim=1)

        # 포함되는 박스가 없었던 경우는 -1로 설정
        indices = torch.where(any_in_box, first_idx, torch.full_like(first_idx, -1))

        # 인덱스에 따라 ids 값 선택, -1이면 -1 리턴
        mapped_ids = torch.full((N,), -1, dtype=ids.dtype, device=ids.device)
        valid = indices >= 0
        if valid.any():
            mapped_ids[valid] = ids[indices[valid]]

        return mapped_ids #indices : 박스의 인덱스에서 객체 id로 변경

        ##박스 직접 대응 하는 코드
        """
        # 포함되는 박스가 없었던 경우는 무효화
        first_idx = torch.where(any_in_box, first_idx, torch.zeros_like(first_idx))

        # 결과 박스값 뽑기
        selected_boxes = boxes[first_idx]  # (N, 4)
        # 포함되는 박스가 없었던 경우 [0,0,0,0]으로 덮어쓰기
        selected_boxes = torch.where(
            any_in_box.unsqueeze(1),
            selected_boxes,
            torch.zeros_like(selected_boxes)
        )
        return selected_boxes, any_in_box
        """

    def calculate_distance(self, points, boxes):

        bx, by, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        px, py = points[:, 0], points[:, 1]

        # 박스 내 여부: x <= px < x+w, y <= py < y+h
        in_box = (bx <= px) & (px < bx + bw) & (by <= py) & (py < by + bh)

        # 박스 중심 좌표
        cx = bx + bw / 2
        cy = by + bh / 2

        # 중심과의 거리
        dist = torch.sqrt((px - cx) ** 2 + (py - cy) ** 2)

        # 박스 안이면 0, 아니면 거리
        result = torch.where(in_box, torch.zeros_like(dist), dist)
        return result