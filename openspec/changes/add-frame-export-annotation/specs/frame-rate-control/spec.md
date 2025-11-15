# Specification: Frame Rate Control

## ADDED Requirements

### Requirement: Allow users to configure export frame rate

The system SHALL allow users to configure export frame rate to balance between annotation completeness and dataset size, with sensible defaults for common use cases.

#### Scenario: User selects from preset frame rates

**Given** a user is preparing to export annotations
**When** the user opens the export configuration dialog
**Then** the system shall present frame rate options: 0.5, 1, 2, 5, 10, 15, 30 FPS
**And** default to 5 FPS for videos with source FPS ≥ 30
**And** default to 1 FPS for videos with source FPS < 30
**And** display the estimated number of frames for each option based on video duration

#### Scenario: Frame rate selection shows export size estimate

**Given** a user selects different frame rate options
**When** the frame rate changes
**Then** the system shall immediately update the display to show:
- Estimated number of frames to export
- Approximate export file size (in MB)
- Percentage of source frames included (e.g., "16.7% of frames" for 5 FPS from 30 FPS source)
**And** warn if estimated size exceeds 50MB

#### Scenario: System prevents invalid frame rate selections

**Given** a video with source frame rate of 24 FPS
**When** the user selects an export frame rate
**Then** the system shall not allow export FPS higher than source FPS
**And** display a message if user attempts invalid selection
**And** auto-adjust to maximum valid FPS if needed

---

### Requirement: Sample frames accurately based on time intervals

The system SHALL sample frames accurately based on time intervals (time-based) rather than index-based to ensure consistent results across videos with variable frame rates or dropped frames.

#### Scenario: Time-based sampling for 30 FPS video at 5 FPS export

**Given** a 30-second video recorded at 30 FPS (900 frames)
**When** the user exports at 5 FPS
**Then** the system shall sample frames at time intervals of 0.2 seconds
**And** select the frame closest to each target timestamp: [0.0s, 0.2s, 0.4s, ...]
**And** result in exactly 150 exported frames (5 × 30 seconds)
**And** handle any missing or duplicate frames gracefully

#### Scenario: Time-based sampling for variable frame rate video

**Given** a video with variable frame rate (VFR) averaging 30 FPS
**When** the user exports at 1 FPS
**Then** the system shall calculate target timestamps: [0.0s, 1.0s, 2.0s, ...]
**And** select the available frame with timestamp closest to each target
**And** not skip any target timestamps even if precise frames are unavailable
**And** document in export metadata which actual frames were used

#### Scenario: Fractional frame rates work correctly

**Given** a user selects 0.5 FPS (1 frame every 2 seconds)
**When** exporting a 60-second video
**Then** the system shall sample 30 frames
**And** space them evenly at 2-second intervals
**And** round timestamps to nearest available frame

---

### Requirement: Display frame rate impact on export quality

The system SHALL display frame rate impact on export quality to help users understand the trade-off between export size and temporal resolution when making frame rate selections.

#### Scenario: System explains frame rate recommendations

**Given** a user hovers over frame rate options
**Then** the system shall display tooltips explaining:
- "0.5-1 FPS: Sparse sampling for object detection datasets"
- "5 FPS: Balanced for motion tracking datasets"
- "10-15 FPS: Dense sampling for detailed motion analysis"
- "30 FPS: Full temporal resolution (large export size)"

#### Scenario: Warning for very high frame rates

**Given** a 5-minute video
**When** the user selects 30 FPS export
**Then** the system shall display a warning:
- "This will export 9,000 frames (~200MB)"
- "Consider 5 FPS for a smaller dataset (1,500 frames)"
**And** require explicit confirmation before proceeding

#### Scenario: Recommendation for very low frame rates

**Given** a user selects 0.5 FPS for a 10-second video
**Then** the system shall show a notice:
- "Only 5 frames will be exported"
- "This may be too sparse for motion tracking"
**And** suggest a higher frame rate if video duration is short
