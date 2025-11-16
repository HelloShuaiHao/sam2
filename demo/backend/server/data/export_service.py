# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Export service for managing video annotation export jobs.

Handles export job creation, processing, status tracking, and file generation.
"""

import os
import json
import uuid
import zipfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Thread
import traceback

import numpy as np
import cv2
import torch
from PIL import Image
from pycocotools.mask import decode as decode_rle_mask

from app_conf import DATA_PATH, API_URL
from data.data_types import ExportResult, ExportJobInfo, ExportJobStatus
from utils.frame_sampler import FrameSampler
from utils.annotation_serializer import AnnotationSerializer, create_metadata_file
from inference.predictor import InferenceAPI

logger = logging.getLogger(__name__)


class ExportService:
    """
    Service for managing video annotation exports.
    """

    # In-memory job storage (for simplicity; could use Redis/DB in production)
    _jobs: Dict[str, Dict[str, Any]] = {}

    # Export storage directory
    EXPORT_DIR = Path(DATA_PATH) / "exports"

    def __init__(self):
        """Initialize export service and ensure export directory exists."""
        self.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Export service initialized. Export dir: {self.EXPORT_DIR}")

    def create_export_job(
        self,
        session_id: str,
        target_fps: float,
        inference_api: InferenceAPI,
        object_names: Optional[Dict[str, str]] = None
    ) -> ExportResult:
        """
        Create a new export job and start processing in background.

        Args:
            session_id: Video session ID
            target_fps: Target export frame rate
            inference_api: Reference to inference API for getting masks

        Returns:
            ExportResult with job ID and initial status
        """
        try:
            # Get session state to access video metadata
            session = inference_api._InferenceAPI__get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            inference_state = session["state"]

            # Get video metadata from inference state
            total_frames = inference_state["num_frames"]

            # Assuming 30 FPS by default (could be extracted from video file)
            # In production, this should be read from video metadata
            source_fps = 30.0  # TODO: Get from video metadata
            duration_sec = total_frames / source_fps

            # Create job ID
            job_id = str(uuid.uuid4())

            # Estimate frames to export
            sampler = FrameSampler(source_fps, total_frames, duration_sec)
            estimated_frames = sampler.estimate_export_count(target_fps)

            # Create job record
            job = {
                "job_id": job_id,
                "session_id": session_id,
                "status": ExportJobStatus.PENDING,
                "target_fps": target_fps,
                "source_fps": source_fps,
                "total_frames": total_frames,
                "duration_sec": duration_sec,
                "processed_frames": 0,
                "progress": 0.0,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "completed_at": None,
                "download_url": None,
                "file_size_mb": None,
                "error_message": None,
                "estimated_frames": estimated_frames,
                "object_names": object_names or {}
            }

            self._jobs[job_id] = job

            # Start background processing
            thread = Thread(
                target=self._process_export_job,
                args=(job_id, inference_api, session["video_path"], object_names or {}),
                daemon=True
            )
            thread.start()

            logger.info(
                f"Created export job {job_id} for session {session_id} "
                f"(target: {target_fps} FPS, estimated: {estimated_frames} frames)"
            )

            return ExportResult(
                job_id=job_id,
                status=ExportJobStatus.PENDING,
                message="Export job created successfully",
                estimated_frames=estimated_frames
            )

        except Exception as e:
            logger.error(f"Failed to create export job: {e}")
            logger.error(traceback.format_exc())
            raise

    def _process_export_job(self, job_id: str, inference_api: InferenceAPI, video_path: str, object_names: Dict[str, str]):
        """
        Process export job in background thread.

        Args:
            job_id: Job ID
            inference_api: Inference API instance
            video_path: Path to original video file
            object_names: Mapping of object_id (as string) to custom name
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        try:
            # Update status to processing
            job["status"] = ExportJobStatus.PROCESSING
            logger.info(f"Starting export processing for job {job_id}")

            # Get session
            session_id = job["session_id"]
            session = inference_api._InferenceAPI__get_session(session_id)
            inference_state = session["state"]

            # Calculate frame sampling
            sampler = FrameSampler(
                job["source_fps"],
                job["total_frames"],
                job["duration_sec"]
            )
            frame_indices = sampler.calculate_frame_indices(job["target_fps"])

            logger.info(f"Job {job_id}: Will export {len(frame_indices)} frames")

            # Initialize annotation serializer
            video_metadata = {
                "filename": f"session_{session_id}.mp4",
                "width": inference_state.get("video_width", 1920),
                "height": inference_state.get("video_height", 1080),
                "fps": job["source_fps"],
                "total_frames": job["total_frames"],
                "duration_sec": job["duration_sec"]
            }

            serializer = AnnotationSerializer(video_metadata)

            # Store visualized frames
            visualized_frames = {}

            # Process each sampled frame
            for idx, frame_index in enumerate(frame_indices):
                try:
                    # Get masks for this frame from inference state
                    objects = self._get_frame_annotations(
                        inference_api,
                        inference_state,
                        frame_index,
                        object_names
                    )

                    # Add to serializer
                    timestamp_sec = frame_index / job["source_fps"]
                    serializer.add_frame_annotation(
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        objects=objects
                    )

                    # Generate visualization with masks
                    if len(objects) > 0:  # Only visualize if there are objects
                        vis_image = self._visualize_frame_with_masks(
                            video_path=video_path,
                            frame_index=frame_index,
                            objects=objects,
                            video_width=video_metadata["width"],
                            video_height=video_metadata["height"]
                        )
                        visualized_frames[frame_index] = vis_image

                    # Update progress
                    job["processed_frames"] = idx + 1
                    job["progress"] = (idx + 1) / len(frame_indices)

                    if (idx + 1) % 10 == 0:
                        logger.info(
                            f"Job {job_id}: Processed {idx + 1}/{len(frame_indices)} frames "
                            f"({job['progress']*100:.1f}%)"
                        )

                except Exception as e:
                    logger.error(f"Error processing frame {frame_index}: {e}")
                    # Continue with other frames

            # Generate export files
            export_path = self._generate_export_files(
                job_id,
                serializer,
                job["target_fps"],
                frame_indices,
                visualized_frames
            )

            # Calculate file size
            file_size_mb = os.path.getsize(export_path) / (1024 * 1024)

            # Update job as completed
            job["status"] = ExportJobStatus.COMPLETED
            job["completed_at"] = datetime.utcnow().isoformat() + "Z"
            job["download_url"] = f"{API_URL}/api/download/export/{job_id}"
            job["file_size_mb"] = round(file_size_mb, 2)
            job["progress"] = 1.0

            logger.info(
                f"Job {job_id} completed successfully. "
                f"File size: {file_size_mb:.2f} MB"
            )

        except Exception as e:
            # Update job as failed
            job["status"] = ExportJobStatus.FAILED
            job["error_message"] = str(e)
            job["completed_at"] = datetime.utcnow().isoformat() + "Z"

            logger.error(f"Job {job_id} failed: {e}")
            logger.error(traceback.format_exc())

    def _get_frame_annotations(
        self,
        inference_api: InferenceAPI,
        inference_state: Any,
        frame_index: int,
        object_names: Dict[str, str]
    ) -> list:
        """
        Get all object annotations for a specific frame.

        Args:
            inference_api: Inference API
            inference_state: Inference state
            frame_index: Frame index
            object_names: Mapping of object_id (as string) to custom name

        Returns:
            List of object dictionaries with masks
        """
        objects = []

        # Get object IDs from inference state
        # The predictor stores tracked objects in the state
        obj_ids = inference_state.get("obj_id_to_idx", {}).keys()

        # 🔍 DEBUG: Check what we have in inference_state
        logger.info(f"🔍 DEBUG - Frame {frame_index}:")
        logger.info(f"  - obj_id_to_idx keys: {list(obj_ids)}")
        logger.info(f"  - inference_state keys: {list(inference_state.keys())}")
        if "output_dict_per_obj" in inference_state:
            logger.info(f"  - output_dict_per_obj length: {len(inference_state['output_dict_per_obj'])}")

        with inference_api.autocast_context():
            for obj_id in obj_ids:
                logger.info(f"  Processing object_id: {obj_id}")
                try:
                    # Get the object index
                    obj_idx = inference_state["obj_id_to_idx"][obj_id]
                    logger.info(f"    - obj_idx: {obj_idx}")

                    # Access the per-object output dictionary
                    obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                    logger.info(f"    - cond_frame_outputs frames: {list(obj_output_dict['cond_frame_outputs'].keys())}")
                    logger.info(f"    - non_cond_frame_outputs frames: {list(obj_output_dict['non_cond_frame_outputs'].keys())}")

                    # Try to get output from cond_frame_outputs first, then non_cond_frame_outputs
                    output = obj_output_dict["cond_frame_outputs"].get(frame_index)
                    if output is None:
                        output = obj_output_dict["non_cond_frame_outputs"].get(frame_index)
                        logger.info(f"    - Using non_cond_frame_outputs for frame {frame_index}")
                    else:
                        logger.info(f"    - Using cond_frame_outputs for frame {frame_index}")

                    # If we found an output for this frame and it has masks
                    if output is not None and output.get("pred_masks") is not None:
                        logger.info(f"    - Found mask! Shape: {output['pred_masks'].shape}")
                    else:
                        logger.warning(f"    - No mask found for frame {frame_index}, output: {output is not None}")

                    if output is not None and output.get("pred_masks") is not None:
                        # Extract the mask (pred_masks shape can be 2D, 3D, or 4D)
                        pred_masks = output["pred_masks"]

                        # Handle different tensor shapes
                        if pred_masks.dim() == 4:
                            # Shape: [batch, num_masks, H, W] -> take [0, 0] to get [H, W]
                            mask = pred_masks[0, 0].cpu().numpy()
                        elif pred_masks.dim() == 3:
                            # Shape: [num_masks, H, W] -> take [0] to get [H, W]
                            mask = pred_masks[0].cpu().numpy()
                        elif pred_masks.dim() == 2:
                            # Shape: [H, W] -> use directly
                            mask = pred_masks.cpu().numpy()
                        else:
                            logger.warning(
                                f"Unexpected mask shape for object {obj_id} "
                                f"at frame {frame_index}: {pred_masks.shape}"
                            )
                            continue

                        # Convert to binary mask
                        mask_binary = (mask > 0).astype(np.uint8)

                        # Get custom name if available, otherwise use default
                        custom_name = object_names.get(str(obj_id))
                        label = custom_name if custom_name else f"object_{obj_id}"

                        objects.append({
                            "object_id": obj_id,
                            "label": label,
                            "mask": mask_binary,
                            "confidence": 0.95  # Default confidence
                        })

                except Exception as e:
                    logger.warning(
                        f"Could not get mask for object {obj_id} "
                        f"at frame {frame_index}: {e}"
                    )
                    import traceback
                    logger.debug(traceback.format_exc())

        return objects

    def _visualize_frame_with_masks(
        self,
        video_path: str,
        frame_index: int,
        objects: list,
        video_width: int,
        video_height: int
    ) -> np.ndarray:
        """
        Create visualization of frame with mask overlays.

        Args:
            video_path: Path to video file
            frame_index: Frame index
            objects: List of objects with masks (from _get_objects_for_frame)
            video_width: Original video width
            video_height: Original video height

        Returns:
            RGB image array (H, W, 3) with mask overlays
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)

        # Seek to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the frame
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret:
            logger.warning(f"Could not read frame {frame_index} from {video_path}")
            # Return a blank frame
            return np.zeros((video_height, video_width, 3), dtype=np.uint8)

        # Resize to target dimensions if needed
        if frame_bgr.shape[0] != video_height or frame_bgr.shape[1] != video_width:
            frame_bgr = cv2.resize(frame_bgr, (video_width, video_height))

        # Define colors for different objects (BGR format for OpenCV)
        colors = [
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 255, 0),    # Light green
            (255, 128, 0),    # Orange
        ]

        # Create overlay
        overlay = frame_bgr.copy()

        for idx, obj in enumerate(objects):
            mask = obj["mask"]

            # Resize mask to original video dimensions if needed
            if mask.shape[0] != video_height or mask.shape[1] != video_width:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (video_width, video_height),
                    interpolation=cv2.INTER_NEAREST
                )

            # Get color for this object
            color = colors[idx % len(colors)]

            # Create colored mask
            colored_mask = np.zeros_like(frame_bgr)
            colored_mask[mask > 0] = color

            # Blend with overlay (30% opacity)
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.3, 0)

            # Draw contours
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

            # Add label
            label = obj.get("label", f"object_{obj['object_id']}")
            # Find top-left point of mask for label placement
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                label_x = int(x_indices.min())
                label_y = int(y_indices.min()) - 10
                label_y = max(20, label_y)  # Ensure label is visible

                # Add background rectangle for text
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    overlay,
                    (label_x, label_y - text_h - 5),
                    (label_x + text_w, label_y + 5),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    overlay,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        # Convert back to RGB
        result_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        return result_rgb

    def _generate_export_files(
        self,
        job_id: str,
        serializer: AnnotationSerializer,
        target_fps: float,
        frame_indices: list,
        visualized_frames: Dict[int, np.ndarray]
    ) -> Path:
        """
        Generate export ZIP file with annotations, metadata, and visualized images.

        Args:
            job_id: Job ID
            serializer: Annotation serializer
            target_fps: Target FPS
            frame_indices: List of exported frame indices
            visualized_frames: Dict mapping frame_index to RGB image array

        Returns:
            Path to generated ZIP file
        """
        # Create job-specific directory
        job_dir = self.EXPORT_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        # Create images subdirectory
        images_dir = job_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Save annotations JSON
        annotations_path = job_dir / "annotations.json"
        serializer.to_json_file(
            str(annotations_path),
            target_fps,
            frame_indices
        )

        # Save metadata JSON
        stats = serializer.get_statistics()
        metadata = create_metadata_file(
            serializer.video_metadata,
            {"target_fps": target_fps, "frame_indices": frame_indices},
            stats
        )

        metadata_path = job_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save visualized images
        image_paths = []
        for frame_idx, vis_image in visualized_frames.items():
            # Save as PNG
            image_filename = f"frame_{frame_idx:06d}.png"
            image_path = images_dir / image_filename

            # Convert RGB to PIL Image and save
            pil_image = Image.fromarray(vis_image)
            pil_image.save(image_path, format='PNG')
            image_paths.append((image_path, f"images/{image_filename}"))

        logger.info(f"Saved {len(image_paths)} visualized frames")

        # Create ZIP archive
        zip_path = self.EXPORT_DIR / f"{job_id}.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(annotations_path, "annotations.json")
            zipf.write(metadata_path, "metadata.json")

            # Add all visualized images
            for image_path, archive_name in image_paths:
                zipf.write(image_path, archive_name)

        logger.info(f"Generated export archive: {zip_path}")

        return zip_path

    def get_job_status(self, job_id: str) -> Optional[ExportJobInfo]:
        """
        Get status of an export job.

        Args:
            job_id: Job ID

        Returns:
            ExportJobInfo or None if job doesn't exist
        """
        job = self._jobs.get(job_id)
        if not job:
            return None

        return ExportJobInfo(
            job_id=job["job_id"],
            session_id=job["session_id"],
            status=job["status"],
            target_fps=job["target_fps"],
            total_frames=job.get("estimated_frames", job["total_frames"]),
            processed_frames=job["processed_frames"],
            progress=job["progress"],
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
            download_url=job.get("download_url"),
            file_size_mb=job.get("file_size_mb"),
            error_message=job.get("error_message")
        )

    def get_export_file_path(self, job_id: str) -> Optional[Path]:
        """
        Get file path for a completed export.

        Args:
            job_id: Job ID

        Returns:
            Path to export ZIP file or None
        """
        zip_path = self.EXPORT_DIR / f"{job_id}.zip"

        if zip_path.exists():
            return zip_path

        return None

    def cleanup_old_exports(self, max_age_hours: int = 24):
        """
        Clean up old export files (to prevent disk space issues).

        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for export_file in self.EXPORT_DIR.glob("*.zip"):
            file_age = current_time - export_file.stat().st_mtime

            if file_age > max_age_seconds:
                try:
                    export_file.unlink()
                    logger.info(f"Cleaned up old export: {export_file}")
                except Exception as e:
                    logger.error(f"Failed to clean up {export_file}: {e}")
