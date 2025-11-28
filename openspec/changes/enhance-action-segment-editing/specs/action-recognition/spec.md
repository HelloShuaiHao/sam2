# Action Recognition Specification

## ADDED Requirements

### Requirement: Action Segment Object Editing
Users SHALL be able to edit objects within action segments with the same capabilities as global object editing, including label modification, position adjustment, tracking control, and deletion.

#### Scenario: Edit object label within action segment
- **WHEN** user clicks on an object within an action segment in the action segment panel
- **THEN** an inline label editor appears allowing the user to rename the object
- **AND** the new label is immediately reflected in the UI and stored in the action segment state

#### Scenario: Adjust object annotation position
- **WHEN** user selects an object within an action segment
- **THEN** the video canvas displays the object's mask/points for the current frame
- **AND** user can add/remove points or adjust the mask boundary
- **AND** changes are saved to the action segment object's annotation data

#### Scenario: Enable object tracking within segment
- **WHEN** user clicks a "track" button for an action segment object
- **THEN** the system propagates the object's mask across all frames within the segment's time range
- **AND** tracking is limited to frames between `frameStart` and `frameEnd` of the segment
- **AND** a progress indicator shows tracking status

#### Scenario: Delete action segment object
- **WHEN** user clicks delete button on an action segment object
- **THEN** a confirmation dialog appears
- **AND** upon confirmation, the object is removed from the segment's object list
- **AND** the object's masks are cleared from the video canvas

#### Scenario: Copy action segment object
- **WHEN** user clicks copy button on an action segment object
- **THEN** a new object is created within the same segment with identical annotations
- **AND** the new object has a unique ID and default color
- **AND** the new object's label is suffixed with " (Copy)"

### Requirement: Action Segment Panel Object Management UI
The action segment panel SHALL display each action segment's objects with inline editing controls consistent with the global object toolbar UI.

#### Scenario: View objects within action segment
- **WHEN** user expands an action segment in the panel
- **THEN** all objects within that segment are listed with their thumbnails, labels, and colors
- **AND** each object shows edit controls (rename, delete, copy buttons)

#### Scenario: Add object to action segment
- **WHEN** user clicks "添加物体" button within an action segment
- **THEN** a new empty object is created in that segment
- **AND** the user can immediately annotate the object on the current frame
- **AND** the object is added to the segment's `objects` array

#### Scenario: Select action segment object for editing
- **WHEN** user clicks on an object in the action segment panel
- **THEN** the object becomes active (highlighted in the panel)
- **AND** the video timeline jumps to the segment's start frame if not already within segment range
- **AND** the object's annotations are displayed on the video canvas

### Requirement: Segment-Scoped Swimlane Visualization
Action segment objects SHALL be displayed in a swimlane timeline showing their presence only within the segment's time range.

#### Scenario: Display action object presence in timeline
- **WHEN** an action segment contains objects
- **THEN** each object displays a swimlane bar spanning from the segment's `frameStart` to `frameEnd`
- **AND** swimlane bars are visually distinct from global object swimlanes (e.g., different styling or section)

#### Scenario: Indicate annotated frames within segment
- **WHEN** user has manually annotated specific frames for an action segment object
- **THEN** those frames are marked with dots/indicators on the swimlane
- **AND** tracked frames (propagated by SAM2) are shown with a continuous bar

#### Scenario: Navigate to segment frame via swimlane
- **WHEN** user clicks on an action object's swimlane
- **THEN** the video timeline seeks to the clicked frame within the segment
- **AND** the action segment panel highlights the corresponding segment

### Requirement: Action Segment Export Format
The system SHALL support exporting action segment annotations with metadata including segment name, time range, and segment-scoped object masks.

#### Scenario: Export single action segment
- **WHEN** user clicks "导出" button on a specific action segment
- **THEN** the system generates a ZIP file containing:
  - `action_segment_<id>.json` with segment metadata (id, name, frameStart, frameEnd)
  - `annotations.json` with object annotations only for frames within the segment range
  - `metadata.json` with video information and export settings
  - `images/` folder with visualized frames (optional, based on user preference)

#### Scenario: Export action segment annotation data structure
- **WHEN** exporting an action segment
- **THEN** the `action_segment_<id>.json` file contains:
```json
{
  "segment_id": "uuid-string",
  "segment_name": "剪断胆囊管",
  "frame_start": 120,
  "frame_end": 180,
  "created_at": 1234567890,
  "objects": [
    {
      "object_id": 1,
      "label": "剪刀",
      "color": "#FF5733"
    }
  ]
}
```
- **AND** the `annotations.json` contains per-frame annotations only for frames `[120, 180]`

#### Scenario: Export multiple action segments
- **WHEN** user clicks "导出全部动作片段" button
- **THEN** the system generates a ZIP file containing:
  - A separate folder for each action segment (named `segment_<name>_<id>/`)
  - Each folder contains `annotations.json` and `metadata.json` for that segment
  - A root-level `segments_index.json` listing all exported segments

#### Scenario: Distinguish action export from global export
- **WHEN** viewing the export UI in action mode
- **THEN** two export buttons are visible:
  - "导出动作片段标注" - exports selected or all action segments
  - "导出全部标注" - exports all global objects (existing behavior)
- **AND** each button's tooltip clearly describes what will be exported

### Requirement: Segment-Scoped Object Tracking
The system SHALL support tracking objects within action segments, limiting propagation to the segment's frame range.

#### Scenario: Track object within segment bounds
- **WHEN** user initiates tracking for an action segment object at frame F
- **THEN** the tracking operation propagates masks from frame `max(F, frameStart)` to `frameEnd`
- **AND** no masks are generated outside the segment's frame range
- **AND** tracking progress shows frames processed relative to segment length

#### Scenario: Handle tracking interruption
- **WHEN** tracking is in progress for an action segment object
- **AND** user clicks "stop" or navigates away
- **THEN** tracking halts immediately
- **AND** already-computed masks within the segment are preserved
- **AND** user can resume tracking from the current frame

#### Scenario: Display tracking status for segment objects
- **WHEN** an action segment object has tracking enabled
- **THEN** the object in the action segment panel shows a tracking indicator (e.g., icon or badge)
- **AND** the swimlane displays the tracked frame range within the segment

### Requirement: UI Mode Separation
The system SHALL clearly distinguish between object annotation mode and action annotation mode, preventing confusion between global objects and action segment objects.

#### Scenario: Switch from object mode to action mode
- **WHEN** user toggles from "物体标注" mode to "动作标注" mode
- **THEN** the action segment panel becomes visible
- **AND** global object toolbar is hidden or collapsed
- **AND** timeline displays action segment bars
- **AND** clicking on video canvas while an action segment is selected adds objects to that segment

#### Scenario: Switch from action mode to object mode
- **WHEN** user toggles from "动作标注" mode to "物体标注" mode
- **THEN** the action segment panel is hidden or collapsed
- **AND** global object toolbar becomes visible
- **AND** action segment bars remain visible on timeline but are non-interactive
- **AND** clicking on video canvas creates/edits global objects

#### Scenario: Prevent cross-mode object editing conflicts
- **WHEN** in object mode
- **THEN** users cannot edit or select action segment objects
- **WHEN** in action mode
- **THEN** users cannot edit or select global objects
- **AND** an informational message guides users to switch modes if they attempt cross-mode editing

### Requirement: Export Backward Compatibility
Action segment export functionality SHALL maintain backward compatibility with existing global object export format.

#### Scenario: Export format versioning
- **WHEN** exporting action segments
- **THEN** the `metadata.json` includes a `format_version` field set to `"2.0"`
- **AND** existing global object exports continue to use `format_version: "1.0"`
- **AND** export parsing tools can distinguish between the two formats

#### Scenario: Reuse existing export infrastructure
- **WHEN** implementing action segment export
- **THEN** the system reuses existing RLE encoding, mask serialization, and visualization utilities
- **AND** only adds action-specific metadata wrapping and frame filtering logic
- **AND** existing export jobs for global objects remain unaffected
