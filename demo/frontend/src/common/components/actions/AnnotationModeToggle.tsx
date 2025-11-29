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
  annotationModeAtom,
  activeActionSegmentObjectIdAtom,
  activeActionObjectIdAtom,
  isAddingActionObjectAtom,
  actionSegmentsAtom,
  globalTrackletObjectsAtom,
} from '@/demo/atoms';
import useVideo from '@/common/components/video/editor/useVideo';
import stylex from '@stylexjs/stylex';
import {useAtom, useSetAtom, useAtomValue} from 'jotai';
import {useCallback} from 'react';

const styles = stylex.create({
  container: {
    display: 'flex',
    gap: '4px',
    backgroundColor: 'rgba(30, 30, 30, 0.9)',
    borderRadius: '8px',
    padding: '6px',
    border: '2px solid rgba(59, 130, 246, 0.5)',
  },
  button: {
    padding: '10px 20px',
    fontSize: '15px',
    fontWeight: '600',
    color: 'rgba(255, 255, 255, 0.9)',
    backgroundColor: 'rgba(50, 50, 50, 0.8)',
    border: '1px solid rgba(100, 100, 100, 0.5)',
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'all 0.2s',
    ':hover': {
      color: '#fff',
      backgroundColor: 'rgba(70, 70, 70, 0.9)',
      borderColor: 'rgba(150, 150, 150, 0.7)',
    },
  },
  buttonActive: {
    color: '#fff',
    backgroundColor: 'rgba(59, 130, 246, 0.8)',
    borderColor: 'rgba(59, 130, 246, 1)',
    ':hover': {
      backgroundColor: 'rgba(59, 130, 246, 0.9)',
    },
  },
});

export default function AnnotationModeToggle() {
  const [mode, setMode] = useAtom(annotationModeAtom);
  const setActiveActionSegmentObjectId = useSetAtom(activeActionSegmentObjectIdAtom);
  const setActiveActionObjectId = useSetAtom(activeActionObjectIdAtom);
  const setIsAddingActionObject = useSetAtom(isAddingActionObjectAtom);
  const actionSegments = useAtomValue(actionSegmentsAtom);
  const setGlobalTracklets = useSetAtom(globalTrackletObjectsAtom);
  const video = useVideo();

  const handleModeChange = useCallback(
    (newMode: 'object' | 'action') => {
      // Clear action-related state when switching to object mode
      if (newMode === 'object') {
        setActiveActionSegmentObjectId(null);
        setActiveActionObjectId(null);
        setIsAddingActionObject(false);

        // Remove action segment object tracklets from global tracklet list
        const actionObjectIds = new Set<number>();
        actionSegments.forEach(segment => {
          segment.objects.forEach(obj => {
            actionObjectIds.add(obj.id);
          });
        });

        if (actionObjectIds.size > 0) {
          // Remove these tracklets from global list and video
          setGlobalTracklets(prev =>
            prev.filter(tracklet => !actionObjectIds.has(tracklet.id)),
          );

          // Also remove these tracklets from video
          actionObjectIds.forEach(id => {
            video?.deleteTracklet(id);
          });
        }
      }
      setMode(newMode);
    },
    [
      setMode,
      setActiveActionSegmentObjectId,
      setActiveActionObjectId,
      setIsAddingActionObject,
      actionSegments,
      setGlobalTracklets,
      video,
    ],
  );

  return (
    <div {...stylex.props(styles.container)}>
      <button
        {...stylex.props(styles.button, mode === 'object' && styles.buttonActive)}
        onClick={() => handleModeChange('object')}>
        物体标注
      </button>
      <button
        {...stylex.props(styles.button, mode === 'action' && styles.buttonActive)}
        onClick={() => handleModeChange('action')}>
        动作标注
      </button>
    </div>
  );
}
