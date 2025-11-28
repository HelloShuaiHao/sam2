/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  ActionSegmentObject,
  actionSegmentsAtom,
  activeActionSegmentObjectIdAtom,
  frameIndexAtom,
} from '@/demo/atoms';
import stylex from '@stylexjs/stylex';
import {useAtom, useSetAtom} from 'jotai';
import {useState, useCallback} from 'react';
import useVideo from '@/common/components/video/editor/useVideo';
import {TrashCan} from '@carbon/icons-react';

const styles = stylex.create({
  container: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '4px',
    padding: '8px',
    marginTop: '4px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    transition: 'all 0.2s',
    ':hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.08)',
      borderColor: 'rgba(255, 255, 255, 0.2)',
    },
  },
  containerActive: {
    backgroundColor: 'rgba(59, 130, 246, 0.15)',
    borderColor: 'rgba(59, 130, 246, 0.5)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '8px',
    cursor: 'pointer',
  },
  thumbnail: {
    width: '32px',
    height: '32px',
    borderRadius: '4px',
    flexShrink: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '10px',
    fontWeight: 'bold',
    color: '#fff',
  },
  labelContainer: {
    flex: 1,
    minWidth: 0,
  },
  label: {
    fontSize: '13px',
    fontWeight: '500',
    color: '#fff',
    cursor: 'text',
    padding: '2px 4px',
    borderRadius: '2px',
    ':hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.1)',
    },
  },
  input: {
    width: '100%',
    padding: '2px 4px',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    border: '1px solid rgba(255, 255, 255, 0.3)',
    borderRadius: '2px',
    color: '#fff',
    fontSize: '13px',
    outline: 'none',
  },
  actions: {
    display: 'flex',
    gap: '4px',
  },
  actionButton: {
    padding: '4px',
    backgroundColor: 'transparent',
    border: 'none',
    borderRadius: '2px',
    cursor: 'pointer',
    color: 'rgba(255, 255, 255, 0.6)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s',
    ':hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.1)',
      color: '#fff',
    },
  },
  trackButton: {
    fontSize: '11px',
    padding: '4px 8px',
    backgroundColor: 'rgba(34, 197, 94, 0.2)',
    color: 'rgb(34, 197, 94)',
    border: '1px solid rgba(34, 197, 94, 0.4)',
    borderRadius: '3px',
    cursor: 'pointer',
    fontWeight: '500',
    ':hover': {
      backgroundColor: 'rgba(34, 197, 94, 0.3)',
    },
  },
  tracking: {
    backgroundColor: 'rgba(59, 130, 246, 0.2)',
    color: 'rgb(59, 130, 246)',
    borderColor: 'rgba(59, 130, 246, 0.4)',
  },
  progressBar: {
    marginTop: '4px',
    height: '3px',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: '2px',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: 'rgb(59, 130, 246)',
    transition: 'width 0.3s',
  },
  info: {
    fontSize: '10px',
    color: 'rgba(255, 255, 255, 0.5)',
    marginTop: '4px',
  },
});

type Props = {
  object: ActionSegmentObject;
  segmentId: string;
  isActive: boolean;
};

export default function ActionSegmentObjectItem({
  object,
  segmentId,
  isActive,
}: Props) {
  const video = useVideo();
  const [actionSegments, setActionSegments] = useAtom(actionSegmentsAtom);
  const [activeObjectId, setActiveObjectId] = useAtom(
    activeActionSegmentObjectIdAtom,
  );
  const setFrameIndex = useSetAtom(frameIndexAtom);

  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(object.name);
  const [isDeleting, setIsDeleting] = useState(false);

  // Select this object
  const handleSelect = useCallback(() => {
    setActiveObjectId(object.id);
    // Jump to segment start if needed
    const segment = actionSegments.find(s => s.id === segmentId);
    if (segment && video) {
      const currentFrame = video.frame;
      if (currentFrame < segment.frameStart || currentFrame > segment.frameEnd) {
        video.frame = segment.frameStart;
        setFrameIndex(segment.frameStart);
      }
    }
  }, [object.id, segmentId, actionSegments, setActiveObjectId, setFrameIndex, video]);

  // Edit label
  const handleLabelClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      setEditValue(object.name);
      setIsEditing(true);
    },
    [object.name],
  );

  const handleLabelSave = useCallback(() => {
    if (editValue.trim()) {
      setActionSegments(segments =>
        segments.map(seg =>
          seg.id === segmentId
            ? {
                ...seg,
                objects: seg.objects.map(obj =>
                  obj.id === object.id
                    ? {...obj, name: editValue.trim()}
                    : obj,
                ),
              }
            : seg,
        ),
      );
    }
    setIsEditing(false);
  }, [editValue, object.id, segmentId, setActionSegments]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        handleLabelSave();
      } else if (e.key === 'Escape') {
        setIsEditing(false);
      }
    },
    [handleLabelSave],
  );

  // Delete object
  const handleDelete = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (!confirm(`确定要删除物体 "${object.name}" 吗？`)) return;

      setIsDeleting(true);
      try {
        // Remove from segment's object list
        setActionSegments(segments =>
          segments.map(seg =>
            seg.id === segmentId
              ? {
                  ...seg,
                  objects: seg.objects.filter(obj => obj.id !== object.id),
                }
              : seg,
          ),
        );

        // Clear selection if this object was active
        if (activeObjectId === object.id) {
          setActiveObjectId(null);
        }

        // TODO: Call backend to clear masks from video state
        // await video?.deleteActionSegmentObject(segmentId, object.id);
      } catch (error) {
        console.error('Failed to delete action segment object:', error);
      } finally {
        setIsDeleting(false);
      }
    },
    [object, segmentId, activeObjectId, setActionSegments, setActiveObjectId],
  );

  // Track object within segment
  const handleTrack = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();

      const segment = actionSegments.find(s => s.id === segmentId);
      if (!segment) return;

      // TODO: Implement segment-scoped tracking
      // For now, just toggle tracking state
      setActionSegments(segments =>
        segments.map(seg =>
          seg.id === segmentId
            ? {
                ...seg,
                objects: seg.objects.map(obj =>
                  obj.id === object.id
                    ? {...obj, isTracking: !obj.isTracking}
                    : obj,
                ),
              }
            : seg,
        ),
      );

      // Call backend tracking method
      // await video?.trackObjectInSegment(
      //   object.id,
      //   segmentId,
      //   segment.frameStart,
      //   segment.frameEnd
      // );
    },
    [object.id, segmentId, actionSegments, setActionSegments],
  );

  // Count annotated frames
  const annotatedFrames = object.points.filter(p => p && p.length > 0).length;
  const trackedFrames = object.masks.filter(m => !m.isEmpty).length;

  return (
    <div
      {...stylex.props(styles.container, isActive && styles.containerActive)}
      onClick={handleSelect}>
      <div {...stylex.props(styles.header)}>
        {/* Thumbnail */}
        <div
          {...stylex.props(styles.thumbnail)}
          style={{backgroundColor: object.color}}>
          {object.thumbnail ? (
            <img
              src={object.thumbnail}
              alt={object.name}
              style={{width: '100%', height: '100%', objectFit: 'cover'}}
            />
          ) : (
            object.name[0]?.toUpperCase() || '?'
          )}
        </div>

        {/* Label */}
        <div {...stylex.props(styles.labelContainer)}>
          {isEditing ? (
            <input
              {...stylex.props(styles.input)}
              type="text"
              value={editValue}
              onChange={e => setEditValue(e.target.value)}
              onBlur={handleLabelSave}
              onKeyDown={handleKeyDown}
              autoFocus
              onClick={e => e.stopPropagation()}
            />
          ) : (
            <div {...stylex.props(styles.label)} onClick={handleLabelClick}>
              {object.name}
            </div>
          )}
        </div>

        {/* Actions */}
        {!isEditing && (
          <div {...stylex.props(styles.actions)}>
            <button
              {...stylex.props(styles.actionButton)}
              onClick={handleDelete}
              disabled={isDeleting}
              title="Delete object"
              style={{color: isDeleting ? 'gray' : 'rgba(239, 68, 68, 0.8)'}}>
              <TrashCan size={16} />
            </button>
          </div>
        )}
      </div>

      {/* Track button and progress */}
      <div>
        <button
          {...stylex.props(
            styles.trackButton,
            object.isTracking && styles.tracking,
          )}
          onClick={handleTrack}>
          {object.isTracking ? '⏸ 停止追踪' : '▶ 追踪'}
        </button>

        {object.isTracking && (
          <div {...stylex.props(styles.progressBar)}>
            <div
              {...stylex.props(styles.progressFill)}
              style={{width: `${object.trackingProgress * 100}%`}}
            />
          </div>
        )}

        <div {...stylex.props(styles.info)}>
          标注: {annotatedFrames} 帧 | 追踪: {trackedFrames} 帧
        </div>
      </div>
    </div>
  );
}
