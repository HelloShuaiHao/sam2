# Remaining Work - Action Recognition Module

## 🚨 Critical (Required for Functionality)

### 1. Add GraphQL Schema Mutations (30 minutes)
**Priority:** CRITICAL
**Blocks:** Export functionality

**Location:** `demo/backend/server/schema.py` (or GraphQL schema definition file)

**Required Mutations:**
```graphql
type Mutation {
  exportActionSegments(
    input: ExportActionSegmentsInput!
  ): ExportActionSegmentsResult!
}

input ExportActionSegmentsInput {
  sessionId: String!
  actionSegments: [ActionSegmentInputData!]!
  targetFps: Float!
  includeVisualizations: Boolean = true
}

input ActionSegmentInputData {
  id: String!
  name: String!
  frame_start: Int!
  frame_end: Int!
  created_at: Float!
  objects: [ActionSegmentObjectInputData!]!
}

input ActionSegmentObjectInputData {
  object_id: Int!
  label: String!
  color: String!
}

type ExportActionSegmentsResult {
  jobId: String!
  status: ExportJobStatus!
  message: String
  segmentCount: Int
}
```

**Resolver:**
```python
@strawberry.mutation
def export_action_segments(
    self,
    input: ExportActionSegmentsInput
) -> ExportActionSegmentsResult:
    # Wire to export_service.create_action_segment_export_job()
    pass
```

---

## ⚠️ Important (Enhances UX)

### 2. Wire Frontend Tracking to Backend (1-2 hours)
**Priority:** IMPORTANT
**File:** `demo/frontend/src/common/components/video/editor/DemoVideoEditor.tsx`

**TODO:**
- Add handler for track button clicks in ActionSegmentObjectItem
- Call `video.trackObjectInSegment(objectId, segmentId, frameStart, frameEnd)`
- Update tracking progress in action segment state
- Handle tracking completion/errors

---

### 3. Add Canvas Interaction Handlers (1 hour)
**Priority:** OPTIONAL
**File:** `demo/frontend/src/common/components/video/editor/InteractionLayer.tsx`

**TODO:**
- Check `annotationMode` before creating points
- In object mode: only allow global object annotation
- In action mode: only allow action segment object annotation
- Show tooltip when clicking in wrong mode

---

## 📝 Testing (2-3 hours)

### 4. Manual Testing Checklist
**Priority:** HIGH

**Workflow Tests:**
- [ ] Create action segment on timeline
- [ ] Add object to segment
- [ ] Edit object label
- [ ] Copy object
- [ ] Delete object
- [ ] Track object (after #2 complete)
- [ ] View swimlane
- [ ] Export single segment (after #1 complete)
- [ ] Export multiple segments (after #1 complete)
- [ ] Download and verify ZIP structure

**Edge Cases:**
- [ ] Empty segments (no objects)
- [ ] Overlapping segments
- [ ] Segments at video boundaries
- [ ] Mode switching preserves state
- [ ] Tracking interruption

**Export Validation:**
- [ ] Verify `format_version: "2.0"` in metadata
- [ ] Check `segments_index.json` structure
- [ ] Validate per-segment folders
- [ ] Confirm only segment frames in annotations
- [ ] Test backward compatibility (v1.0 still works)

---

## 🎨 Nice-to-Have (Optional Enhancements)

### 5. Visualization Export (1-2 hours)
**Priority:** LOW
**File:** `demo/backend/server/data/export_service.py`

**TODO:**
- In `_process_action_segment_export()`, add frame visualization
- Reuse existing mask overlay code from global export
- Save visualized frames to `images/` folder
- Only if `include_visualizations=True`

---

### 6. Export Configuration Modal (30 minutes)
**Priority:** LOW
**File:** `demo/frontend/src/common/components/export/ExportConfigModal.tsx`

**TODO:**
- Add tab/section for action segment export
- FPS selector for action exports
- "Include visualizations" checkbox
- Preview of selected segments

---

### 7. Progress Enhancements (1 hour)
**Priority:** LOW

**Features:**
- Cancel button during export
- Estimated time remaining
- Segment-by-segment progress (not just overall)
- Pause/resume capability

---

## 📚 Documentation (2-3 hours)

### 8. User Documentation
**Priority:** MEDIUM

**Required:**
- Action segment creation guide
- Object editing within segments
- Export workflow documentation
- Troubleshooting guide

---

### 9. Developer Documentation
**Priority:** MEDIUM

**Required:**
- Export format v2.0 specification
- Example JSON files (3-4 examples)
- API documentation for new endpoints
- Migration guide from v1.0 to v2.0

---

### 10. Code Documentation
**Priority:** LOW

**Improvements:**
- Add JSDoc comments to exported functions
- Document complex algorithms
- Add type documentation
- README updates

---

## 📊 Summary

| Category | Tasks | Est. Time | Priority |
|----------|-------|-----------|----------|
| Critical | 1 | 30 min | CRITICAL |
| Important | 2 | 2 hours | HIGH |
| Testing | 1 | 3 hours | HIGH |
| Nice-to-Have | 4 | 4 hours | LOW |
| Documentation | 3 | 4 hours | MEDIUM |
| **TOTAL** | **11** | **~13 hours** | - |

---

## 🎯 Minimum Viable Product (MVP)

To get the feature working end-to-end:

1. ✅ **Add GraphQL mutations** (Task #1) - 30 minutes
2. ✅ **Manual testing** (Task #4) - 2 hours
3. ⚠️ **Fix critical bugs found during testing** - 1 hour (estimated)

**Total MVP Time:** ~3.5 hours

---

## 🚀 Recommended Implementation Order

### Phase 1: Make it Work (3.5 hours)
1. Add GraphQL mutations (#1)
2. Manual testing (#4)
3. Bug fixes

### Phase 2: Make it Better (3 hours)
4. Wire frontend tracking (#2)
5. Add canvas interaction handlers (#3)
6. User documentation (#8)

### Phase 3: Polish (6 hours)
7. Visualization export (#5)
8. Export config modal (#6)
9. Progress enhancements (#7)
10. Developer docs (#9)
11. Code docs (#10)

---

## 🐛 Known Issues to Fix

1. **ExportActionSegmentButton not integrated**
   - Component created but not added to UI
   - Need to add to ActionSegmentPanel or toolbar

2. **ActionSegmentSwimlane not rendered**
   - Component created but not integrated
   - Need to add to swimlane container

3. **SAM2Model stub methods**
   - `trackObjectInSegment()` logs but doesn't execute
   - `stopSegmentTracking()` is a no-op
   - Need GraphQL endpoint wiring

4. **No error boundaries**
   - Export errors could crash UI
   - Add error boundaries to export components

---

## 📞 Support & Questions

If you encounter issues during implementation:

1. Check `IMPLEMENTATION_SUMMARY.md` for architecture details
2. Review OpenSpec proposal for requirements
3. Check TODO comments in code (search for "TODO:")
4. Test with existing global object export to verify backend works

---

## ✅ Success Criteria

The feature is complete when:

- [x] All core code implemented (95% done)
- [ ] GraphQL mutations wired (#1)
- [ ] Manual testing passes (#4)
- [ ] Export generates valid v2.0 format
- [ ] Download works end-to-end
- [ ] No regressions in global object export
- [ ] User can complete full workflow without errors

**Current Status:** 95% complete, pending GraphQL wiring and testing.
