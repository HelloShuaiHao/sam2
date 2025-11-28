# Implementation Tasks

## 1. Frontend State Management
- [x] 1.1 Extend `ActionSegmentObject` type in atoms.ts to support tracking state (`isTracking`, `trackingProgress`)
- [x] 1.2 Add atom for selected action segment object (`activeActionSegmentObjectIdAtom`)
- [x] 1.3 Create derived atom for filtering action segment objects by current segment
- [x] 1.4 Add atom for action segment export configuration (`actionExportConfigAtom`)

## 2. Action Segment Panel Object Editing UI
- [x] 2.1 Create `ActionSegmentObjectItem.tsx` component with inline edit controls (similar to `ToolbarObject.tsx`)
- [x] 2.2 Add object list rendering within `ActionSegmentPanel.tsx` for each expanded segment
- [x] 2.3 Implement object label rename inline editing with validation
- [x] 2.4 Add delete object confirmation dialog and state cleanup logic
- [x] 2.5 Implement copy object functionality with unique ID generation
- [x] 2.6 Add "添加物体" button within each action segment with object creation logic (already exists)

## 3. Segment-Scoped Object Tracking
- [x] 3.1 Extend `Tracker.ts` interface with segment-scoped methods:
  - `trackObjectInSegment(objectId, segmentId, startFrame, endFrame)`
  - `stopSegmentTracking(objectId, segmentId)`
- [x] 3.2 Implement segment-bound tracking logic in tracker (limit propagation to frame range)
- [x] 3.3 Add tracking progress indicator for action segment objects
- [ ] 3.4 Update `DemoVideoEditor.tsx` to handle segment-scoped tracking commands (TODO: wire to backend)
- [x] 3.5 Store tracking state per action segment object (tracked frames array)

## 4. Swimlane Visualization for Action Objects
- [x] 4.1 Create `ActionSegmentSwimlane.tsx` component extending existing swimlane logic
- [x] 4.2 Render swimlanes scoped to segment time range (visual bounds from `frameStart` to `frameEnd`)
- [x] 4.3 Display annotation points and tracked bars within segment bounds
- [x] 4.4 Add click-to-navigate functionality for segment swimlanes
- [x] 4.5 Style action swimlanes distinctly from global object swimlanes (e.g., different background color, border)

## 5. UI Mode Separation and Consistency
- [x] 5.1 Update `AnnotationModeToggle.tsx` to clearly switch UI context (already exists)
- [x] 5.2 Conditionally show/hide global object toolbar vs action segment panel based on mode (already implemented in ActionSegmentPanel)
- [ ] 5.3 Add mode-specific canvas interaction handlers in `InteractionLayer.tsx` (TODO: optional enhancement)
- [ ] 5.4 Add informational tooltips/messages when users attempt cross-mode editing (TODO: optional enhancement)
- [x] 5.5 Ensure timeline action segment bars are visible but non-interactive in object mode (already implemented)

## 6. Backend Action Segment Data Model
- [x] 6.1 Define `ActionSegment` and `ActionSegmentObject` types in `data_types.py`
- [ ] 6.2 Extend `StartSessionRequest` to optionally accept action segment metadata (TODO: future enhancement)
- [ ] 6.3 Update inference state structure to store action segments alongside global tracklets (TODO: future enhancement)
- [ ] 6.4 Add GraphQL mutations for action segment operations (TODO: required for export to work)

## 7. Segment-Scoped Backend Tracking
- [x] 7.1 Create `propagate_in_segment` method in `predictor.py` with frame_start and frame_end parameters
- [x] 7.2 Filter propagation output to only include frames within specified range
- [x] 7.3 Add `propagate_in_segment` method wrapping SAM2 propagation with segment bounds
- [ ] 7.4 Update session state to track segment-scoped objects separately (TODO: future enhancement)

## 8. Action Segment Export Backend
- [x] 8.1 Create `ExportActionSegmentsInput` type in `data_types.py`
- [ ] 8.2 Add `exportActionSegments` GraphQL mutation (TODO: required for frontend to work)
- [x] 8.3 Implement `create_action_segment_export_job` method in `export_service.py`
- [x] 8.4 Add `segments_index.json` generation for batch exports listing all segments
- [x] 8.5 Extend annotation serializer to include action segment metadata in output format

## 9. Frontend Export Integration
- [x] 9.1 Create `useActionSegmentExport.ts` hook mirroring `useExport.ts` for segment-specific export
- [x] 9.2 Add `ExportActionSegmentButton.tsx` component with dropdown and batch selection
- [ ] 9.3 Update `ExportConfigModal.tsx` to show action export options when in action mode (TODO: optional)
- [x] 9.4 Add "导出动作片段标注" button with clear labels and tooltips
- [x] 9.5 Implement batch export UI showing progress for multiple segments

## 10. Export Format Versioning
- [x] 10.1 Add `format_version: "2.0"` to action segment export metadata
- [x] 10.2 Maintain `format_version: "1.0"` for existing global object exports
- [x] 10.3 Update export documentation to describe both formats (see IMPLEMENTATION_SUMMARY.md)
- [x] 10.4 Ensure backward compatibility: existing tools can still parse v1.0 exports

## 11. Testing and Validation
- [ ] 11.1 Test action segment object creation and editing workflow
- [ ] 11.2 Verify segment-scoped tracking correctly limits frame range
- [ ] 11.3 Test export of single action segment with visualization
- [ ] 11.4 Test batch export of multiple action segments
- [ ] 11.5 Verify export format compatibility with existing annotation parsers (for global objects)
- [ ] 11.6 Test UI mode switching preserves state correctly
- [ ] 11.7 Validate swimlane visualization for action objects
- [ ] 11.8 Test edge cases:
  - Empty action segments (no objects)
  - Overlapping action segments
  - Action segments at video start/end boundaries
  - Tracking interruption and resume

## 12. Documentation
- [ ] 12.1 Update user documentation describing action segment editing workflow
- [ ] 12.2 Document export format v2.0 schema with examples
- [ ] 12.3 Add inline code comments for complex segment-scoped logic
- [ ] 12.4 Create example JSON files showing action segment export structure

## Notes
- Tasks 1-5 focus on frontend UI/UX parity with global object editing
- Tasks 6-8 implement backend support for action segment operations and export
- Tasks 9-10 integrate export functionality with version management
- Tasks 11-12 ensure quality and maintainability
- Many tasks can be parallelized (e.g., 1-5 can progress alongside 6-8)
