from cog import BaseModel, Path
from typing import Optional

class Output(BaseModel):
    coco_keypoints: str
    facs: str
    fullbodyfacs: str
    debug_media: Path
    hand_landmarks: Optional[str]
    num_people: int
    media_type: str
    total_frames: Optional[int] = None
