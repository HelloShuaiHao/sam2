# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional

import strawberry
from app_conf import API_URL
from data.resolver import resolve_videos
from dataclasses_json import dataclass_json
from strawberry import relay


@strawberry.type
class Video(relay.Node):
    """Core type for video."""

    code: relay.NodeID[str]
    path: str
    poster_path: Optional[str]
    width: int
    height: int

    @strawberry.field
    def url(self) -> str:
        return f"{API_URL}/{self.path}"

    @strawberry.field
    def poster_url(self) -> str:
        return f"{API_URL}/{self.poster_path}"

    @classmethod
    def resolve_nodes(
        cls,
        *,
        info: relay.PageInfo,
        node_ids: Iterable[str],
        required: bool = False,
    ):
        return resolve_videos(node_ids, required)


@strawberry.type
class RLEMask:
    """Core type for Onevision GraphQL RLE mask."""

    size: List[int]
    counts: str
    order: str


@strawberry.type
class RLEMaskForObject:
    """Type for RLE mask associated with a specific object id."""

    object_id: int
    rle_mask: RLEMask


@strawberry.type
class RLEMaskListOnFrame:
    """Type for a list of object-associated RLE masks on a specific video frame."""

    frame_index: int
    rle_mask_list: List[RLEMaskForObject]


@strawberry.input
class StartSessionInput:
    path: str


@strawberry.type
class StartSession:
    session_id: str


@strawberry.input
class PingInput:
    session_id: str


@strawberry.type
class Pong:
    success: bool


@strawberry.input
class CloseSessionInput:
    session_id: str


@strawberry.type
class CloseSession:
    success: bool


@strawberry.input
class AddPointsInput:
    session_id: str
    frame_index: int
    clear_old_points: bool
    object_id: int
    labels: List[int]
    points: List[List[float]]


@strawberry.input
class ClearPointsInFrameInput:
    session_id: str
    frame_index: int
    object_id: int


@strawberry.input
class ClearPointsInVideoInput:
    session_id: str


@strawberry.type
class ClearPointsInVideo:
    success: bool


@strawberry.input
class RemoveObjectInput:
    session_id: str
    object_id: int


@strawberry.input
class PropagateInVideoInput:
    session_id: str
    start_frame_index: int


@strawberry.input
class CancelPropagateInVideoInput:
    session_id: str


@strawberry.type
class CancelPropagateInVideo:
    success: bool


@strawberry.type
class SessionExpiration:
    session_id: str
    expiration_time: int
    max_expiration_time: int
    ttl: int


# Export-related types
@strawberry.enum
class ExportJobStatus(str, Enum):
    """Status of an export job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@strawberry.input
class ExportVideoAnnotationsInput:
    """Input for exporting video annotations."""
    session_id: str
    target_fps: float


@strawberry.type
class ExportResult:
    """Result of an export request."""
    job_id: str
    status: ExportJobStatus
    message: Optional[str] = None
    download_url: Optional[str] = None
    progress: Optional[float] = None
    estimated_frames: Optional[int] = None


@strawberry.type
class ExportJobInfo:
    """Detailed information about an export job."""
    job_id: str
    session_id: str
    status: ExportJobStatus
    target_fps: float
    total_frames: int
    processed_frames: int
    progress: float
    created_at: str
    completed_at: Optional[str] = None
    download_url: Optional[str] = None
    file_size_mb: Optional[float] = None
    error_message: Optional[str] = None
