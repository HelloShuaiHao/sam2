# Specification: Video Annotation Export

## ADDED Requirements

### Requirement: Export video annotations with tracked object masks

The system SHALL export video annotations with tracked object masks in JSON format to capture all tracking information for downstream use in ML pipelines and dataset creation.

#### Scenario: User exports annotations for a fully annotated video

**Given** a user has annotated a video with 2 objects tracked across 900 frames
**And** the video is 30 seconds long at 30 FPS
**When** the user clicks the "Export Annotations" button
**And** selects an export frame rate of 5 FPS
**Then** the system shall sample 150 frames (5 frames × 30 seconds)
**And** generate a JSON file containing:
- Video metadata (filename, resolution, source FPS, duration)
- Export configuration (target FPS, frame indices)
- Per-frame annotations with object masks in RLE format
- Bounding boxes and confidence scores for each object
**And** package the JSON into a downloadable ZIP archive
**And** initiate download within 5 seconds of export completion

#### Scenario: User exports annotations for partially annotated video

**Given** a user has annotated only frames 0-300 of a 900-frame video
**When** the user exports annotations at 5 FPS
**Then** the system shall only include frames with valid tracking data
**And** indicate in the export metadata which frame range contains annotations
**And** omit frames without tracking information from the export

#### Scenario: Export fails due to processing error

**Given** a user initiates an export
**When** the backend encounters an error during mask generation
**Then** the system shall display a user-friendly error message
**And** log the detailed error for debugging
**And** allow the user to retry the export
**And** not leave any partial download files

---

### Requirement: Provide progress feedback during export processing

The system SHALL provide progress feedback during export processing, including percentage complete, current stage, and estimated time remaining.

#### Scenario: Export shows progress for long video

**Given** a user exports a 60-second video at 10 FPS (600 frames to process)
**When** the export starts
**Then** the system shall display a progress indicator showing:
- Percentage complete
- Current processing stage (e.g., "Sampling frames", "Generating masks", "Creating archive")
- Estimated time remaining
**And** update progress at least every 2 seconds
**And** keep the UI responsive during processing

#### Scenario: Export completes successfully

**Given** an export is in progress
**When** the backend completes all processing
**Then** the progress indicator shall show 100% complete
**And** automatically trigger the download
**And** display a success message with file size information
**And** clear the progress indicator after 3 seconds

---

### Requirement: Generate compact JSON output with RLE-encoded masks

The system SHALL generate compact JSON output with RLE-encoded masks to minimize export file sizes and download times.

#### Scenario: RLE encoding reduces mask size

**Given** a video frame with a 1920×1080 mask (2,073,600 pixels)
**When** the system encodes the mask using RLE
**Then** the encoded mask size shall be at least 10× smaller than raw binary format
**And** the RLE data shall be losslessly decodable to the original mask
**And** follow the COCO RLE format specification for interoperability

#### Scenario: JSON export stays within size limits

**Given** a 30-second video with 2 objects exported at 5 FPS
**When** the system generates the JSON export
**Then** the total JSON file size shall not exceed 50MB
**And** if the export would exceed 100MB, display a warning before processing
**And** recommend reducing the frame rate to decrease size

---

### Requirement: Include comprehensive metadata in export

The system SHALL include comprehensive metadata in export files, containing all information needed to reproduce or understand the annotation process.

#### Scenario: Export includes complete video metadata

**Given** a user exports a video
**Then** the JSON shall include:
- Original filename and upload timestamp
- Video dimensions (width, height)
- Source frame rate and total frame count
- Video duration in seconds
- Export timestamp and exporter version

#### Scenario: Export includes object tracking information

**Given** a video with 3 tracked objects
**Then** each annotation frame shall include for each object:
- Unique object ID (consistent across frames)
- Object label (if provided by user)
- RLE-encoded mask
- Bounding box coordinates [x, y, width, height]
- Mask area in pixels
- Confidence score (from SAM2 model)

#### Scenario: Export records configuration used

**Given** a user exports at a specific frame rate
**Then** the export metadata shall document:
- Target export FPS
- Total number of exported frames
- Exact frame indices included in export
- Frame sampling method (time-based)
