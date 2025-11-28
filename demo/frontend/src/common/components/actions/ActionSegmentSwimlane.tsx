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
import useSelectedFrameHelper from '@/common/components/video/filmstrip/useSelectedFrameHelper';
import {ActionSegmentObject, ActionSegment} from '@/demo/atoms';
import {spacing, w} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';
import {useMemo} from 'react';

const styles = stylex.create({
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: spacing[4],
    width: '100%',
  },
  trackletNameContainer: {
    width: w[12],
    textAlign: 'center',
    fontSize: '10px',
    color: 'rgba(34, 197, 94, 0.9)', // Green tint for action objects
    fontWeight: '500',
  },
  swimlaneContainer: {
    flexGrow: 1,
    position: 'relative',
    display: 'flex',
    height: 12,
    marginVertical: '0.25rem' /* 4px */,
    '@media screen and (max-width: 768px)': {
      marginVertical: 0,
    },
  },
  swimlane: {
    position: 'absolute',
    left: 0,
    top: '50%',
    width: '100%',
    height: 1,
    transform: 'translate3d(0, -50%, 0)',
    opacity: 0.3,
  },
  segmentBounds: {
    position: 'absolute',
    top: 0,
    height: '100%',
    backgroundColor: 'rgba(34, 197, 94, 0.1)',
    border: '1px dashed rgba(34, 197, 94, 0.4)',
    borderRadius: '2px',
  },
  segment: {
    position: 'absolute',
    top: '50%',
    height: 2,
    transform: 'translate3d(0, -50%, 0)',
  },
  segmentationPoint: {
    position: 'absolute',
    top: '50%',
    transform: 'translate3d(0, -50%, 0)',
    borderRadius: '50%',
    cursor: 'pointer',
    width: 12,
    height: 12,
    border: '1px solid rgba(255, 255, 255, 0.3)',
    '@media screen and (max-width: 768px)': {
      width: 8,
      height: 8,
    },
  },
});

type SwimlineSegment = {
  left: number;
  width: number;
};

type Props = {
  object: ActionSegmentObject;
  segment: ActionSegment;
  onSelectFrame: (object: ActionSegmentObject, index: number) => void;
};

function getSwimlaneSegments(
  masks: Array<{isEmpty: boolean}>,
  frameStart: number,
  frameEnd: number,
): SwimlineSegment[] {
  if (masks.length === 0) {
    return [];
  }

  const swimlineSegments: SwimlineSegment[] = [];
  let left = -1;

  // Only process frames within the segment range
  const startIdx = Math.max(0, frameStart);
  const endIdx = Math.min(masks.length - 1, frameEnd);

  for (let frameIndex = startIdx; frameIndex <= endIdx; ++frameIndex) {
    const isEmpty = masks?.[frameIndex]?.isEmpty ?? true;
    if (left === -1 && !isEmpty) {
      left = frameIndex;
    } else if (left !== -1 && (isEmpty || frameIndex === endIdx)) {
      swimlineSegments.push({
        left,
        width: frameIndex - left + (!isEmpty && frameIndex === endIdx ? 1 : 0),
      });
      left = -1;
    }
  }

  return swimlineSegments;
}

export default function ActionSegmentSwimlane({
  object,
  segment,
  onSelectFrame,
}: Props) {
  const selection = useSelectedFrameHelper();

  const segments = useMemo(() => {
    return getSwimlaneSegments(object.masks, segment.frameStart, segment.frameEnd);
  }, [object.masks, segment.frameStart, segment.frameEnd]);

  const framesWithPoints = useMemo(() => {
    return object.points.reduce<number[]>((frames, pts, frameIndex) => {
      // Only include frames within segment range
      if (
        pts != null &&
        pts.length > 0 &&
        frameIndex >= segment.frameStart &&
        frameIndex <= segment.frameEnd
      ) {
        frames.push(frameIndex);
      }
      return frames;
    }, []);
  }, [object.points, segment.frameStart, segment.frameEnd]);

  if (selection === null) {
    return;
  }

  return (
    <div {...stylex.props(styles.container)}>
      <div {...stylex.props(styles.trackletNameContainer)}>
        {object.name}
      </div>
      <div {...stylex.props(styles.swimlaneContainer)}>
        {/* Segment time bounds background */}
        <div
          {...stylex.props(styles.segmentBounds)}
          style={{
            left: selection.toPosition(segment.frameStart),
            width: selection.toPosition(segment.frameEnd - segment.frameStart + 1),
          }}
        />

        {/* Base swimlane - only visible within segment bounds */}
        <div
          {...stylex.props(styles.swimlane)}
          style={{
            backgroundColor: object.color,
            left: selection.toPosition(segment.frameStart),
            width: selection.toPosition(segment.frameEnd - segment.frameStart + 1),
          }}
        />

        {/* Tracked segments */}
        {segments.map(seg => {
          return (
            <div
              key={seg.left}
              {...stylex.props(styles.segment)}
              style={{
                backgroundColor: object.color,
                left: selection.toPosition(seg.left),
                width: selection.toPosition(seg.width),
              }}
            />
          );
        })}

        {/* Annotation points */}
        {framesWithPoints.map(index => {
          return (
            <div
              key={`frame${index}`}
              onClick={() => {
                onSelectFrame?.(object, index);
              }}
              {...stylex.props(styles.segmentationPoint)}
              style={{
                left: Math.floor(selection.toPosition(index) - 4),
                backgroundColor: object.color,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
