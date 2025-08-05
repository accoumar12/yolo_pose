from enum import Enum

from pydantic import BaseModel, Field


class KeypointLabel(str, Enum):
    RIGHT_ANKLE = "right_ankle"
    RIGHT_KNEE = "right_knee"
    RIGHT_HIP = "right_hip"
    RIGHT_ELBOW = "right_elbow"
    RIGHT_SHOULDER = "right_shoulder"
    RIGHT_WRIST = "right_wrist"


KEYPOINT_LABEL_TO_INDEX = {
    KeypointLabel.RIGHT_ANKLE: 16,
    KeypointLabel.RIGHT_KNEE: 14,
    KeypointLabel.RIGHT_HIP: 12,
    KeypointLabel.RIGHT_ELBOW: 8,
    KeypointLabel.RIGHT_SHOULDER: 6,
    KeypointLabel.RIGHT_WRIST: 10,
}


class Position(BaseModel):
    x: int
    y: int


class FrameData(BaseModel):
    frame: int
    keypoints: dict[KeypointLabel, Position] = Field(description="Keypoints detected in the frame.")


FramesData = list[FrameData]