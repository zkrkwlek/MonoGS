import torch
from utils.ObjectMap.bbox_utils import BoundingBox
from typing import Dict, List, Tuple, Optional, Set

class MultiViewBBoxMapping:
    """Multi-view bounding box mapping for consistent tracking"""
    track_id: str
    gaussian_indices: torch.Tensor
    bbox_history: Dict[int, BoundingBox]  # view_id -> bbox
    creation_frame: int
    last_update_frame: int
    confidence_history: List[float]
    active_views: Set[int]

    def add_view_bbox(self, view_id: int, bbox: BoundingBox):
        """Add bounding box for specific view"""
        self.bbox_history[view_id] = bbox
        self.active_views.add(view_id)
        self.confidence_history.append(bbox.confidence)
        if len(self.confidence_history) > 20:
            self.confidence_history = self.confidence_history[-20:]

    def get_bbox_for_view(self, view_id: int) -> Optional[BoundingBox]:
        """Get bounding box for specific view"""
        return self.bbox_history.get(view_id, None)

    def is_consistent_across_views(self, new_view_id: int, new_bbox: BoundingBox,
                                   iou_threshold: float = 0.3) -> bool:
        """Check if new bbox is consistent with existing views"""
        if not self.active_views:
            return True

        # Check consistency with recent views
        recent_views = [v for v in self.active_views if v >= new_view_id - 5]
        if not recent_views:
            return True

        for view_id in recent_views:
            existing_bbox = self.bbox_history[view_id]
            if new_bbox.iou(existing_bbox) < iou_threshold:
                return False
        return True