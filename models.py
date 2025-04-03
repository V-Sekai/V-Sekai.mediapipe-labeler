from cog import BaseModel
from typing import Optional
from pathlib import Path


class Output(BaseModel):
    annotations: Path
    debug_media: Path
    num_people: int
    media_type: str
    total_frames: Optional[int] = None
