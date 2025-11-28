# Proposal: Enhance Action Segment Object Editing and Export

## Why

The action recognition module currently supports creating time segments (动作片段) on the video timeline and adding objects within those segments. However, objects added to action segments lack the comprehensive editing capabilities that are available for global objects (物体标注). Users cannot modify labels, adjust positions, enable tracking, or export action-scoped annotations, creating an inconsistent and incomplete user experience.

This proposal aims to bring feature parity between global object tracking and action segment object management, ensuring users can efficiently annotate, edit, track, and export action-scoped data.

## What Changes

### Frontend Changes
- **Action segment object editing UI**: Add full editing controls (label rename, position adjustment, tracking toggle, delete/copy) for objects within action segments, mirroring the UI/UX of global object editing
- **Unified editing experience**: Reuse existing tracklet editing components to maintain consistency between object mode and action mode
- **Action segment panel improvements**: Display object editing controls directly within the action segment panel for easy access
- **Swimlane visualization**: Extend the swimlane timeline component to support action-scoped object visualization (show object presence only within segment time range)
- **Clear UI separation**: Distinguish between "导出全部标注" (export all annotations) and "导出动作片段标注" (export action segment annotations) with separate export buttons

### Backend Changes
- **Action segment data model**: Formalize the action segment data structure in backend inference state
- **Segment-scoped tracking**: Support object tracking operations limited to action segment frame ranges
- **Action segment export endpoint**: New export format that includes action metadata (segment name, time range) and only includes objects/masks within segment boundaries
- **Batch export support**: Enable exporting multiple action segments in a single operation

### Data Format Changes
- **Export schema extension**: Add action segment metadata to annotation export format:
  - Segment ID, name, frame range
  - Nested object annotations (only frames within segment)
  - Maintain backward compatibility with existing object-only export format

## Impact

### Affected Specs
- **New capability**: `action-recognition` (adding comprehensive requirements for action segment editing and export)

### Affected Code

**Frontend**:
- `/demo/frontend/src/demo/atoms.ts` - Action segment state management
- `/demo/frontend/src/common/components/actions/ActionSegmentPanel.tsx` - Add object editing UI
- `/demo/frontend/src/common/components/annotations/ToolbarObject.tsx` - Reuse for action objects
- `/demo/frontend/src/common/components/export/ExportButton.tsx` - Add action export mode
- `/demo/frontend/src/common/components/export/useExport.ts` - Support action segment export
- `/demo/frontend/src/common/tracker/Tracker.ts` - Add segment-scoped tracking methods

**Backend**:
- `/demo/backend/server/data/data_types.py` - Add action segment export types
- `/demo/backend/server/data/export_service.py` - Implement action segment export logic
- `/demo/backend/server/utils/annotation_serializer.py` - Extend serialization for action segments
- `/demo/backend/server/inference/predictor.py` - Support segment-scoped operations

### Breaking Changes
None - this is purely additive functionality. Existing global object tracking and export remain unchanged.

## Success Criteria

- [ ] Users can edit action segment objects with the same UI/UX as global objects (rename, reposition, track, delete)
- [ ] Action segment objects display in swimlane visualization scoped to segment time range
- [ ] Users can export individual action segments or all segments with proper metadata
- [ ] Export format includes action segment information and only segment-scoped annotations
- [ ] UI clearly distinguishes between global object export and action segment export
- [ ] No regression in existing global object tracking functionality
