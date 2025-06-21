import uuid
from typing import Dict, List, Tuple, Optional, Set

class BoundingBox:
    """Enhanced Bounding box class with tracking capabilities"""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    confidence: float
    label: str = ""
    track_id: Optional[str] = None
    frame_id: Optional[int] = None
    view_id: Optional[int] = None  # 뷰 식별자

    def __post_init__(self):
        if self.track_id is None:
            self.track_id = str(uuid.uuid4())

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains_point_2d(self, x: float, y: float) -> bool:
        """Check if 2D point is inside bounding box"""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def expand(self, factor: float = 1.1) -> 'BoundingBox':
        """Expand bounding box by factor"""
        expand_x = self.width * (factor - 1.0) / 2
        expand_y = self.height * (factor - 1.0) / 2
        return BoundingBox(
            x1=max(0, int(self.x1 - expand_x)),
            y1=max(0, int(self.y1 - expand_y)),
            x2=int(self.x2 + expand_x),
            y2=int(self.y2 + expand_y),
            class_id=self.class_id,
            confidence=self.confidence,
            label=self.label,
            track_id=self.track_id,
            frame_id=self.frame_id,
            view_id=self.view_id
        )

    def iou(self, other: 'BoundingBox') -> float:
        """Calculate IoU with another bounding box"""
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)

        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0