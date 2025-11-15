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

import {color, spacing} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';
import {useMemo} from 'react';

const styles = stylex.create({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: spacing[3],
  },
  label: {
    fontSize: 14,
    fontWeight: 500,
    color: color['gray-100'],
    marginBottom: spacing[2],
  },
  optionsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))',
    gap: spacing[2],
  },
  optionButton: {
    padding: `${spacing[2]}px ${spacing[3]}px`,
    border: `1px solid ${color['gray-600']}`,
    borderRadius: 6,
    background: color['gray-800'],
    color: color['gray-100'],
    cursor: 'pointer',
    fontSize: 14,
    fontWeight: 500,
    transition: 'all 0.2s ease',
    ':hover': {
      background: color['gray-700'],
      borderColor: color['blue-500'],
    },
  },
  optionButtonSelected: {
    background: color['blue-600'],
    borderColor: color['blue-500'],
    color: 'white',
  },
  estimateText: {
    fontSize: 12,
    color: color['gray-400'],
    marginTop: spacing[1],
  },
  warningText: {
    fontSize: 12,
    color: color['yellow-500'],
    marginTop: spacing[1],
  },
});

export type FrameRateOption = 0.5 | 1 | 2 | 5 | 10 | 15 | 30;

const FRAME_RATE_OPTIONS: FrameRateOption[] = [0.5, 1, 2, 5, 10, 15, 30];

const FPS_DESCRIPTIONS: Record<FrameRateOption, string> = {
  0.5: 'Sparse sampling (1 frame per 2 sec)',
  1: 'Low density (1 frame per sec)',
  2: 'Basic sampling',
  5: 'Balanced (recommended)',
  10: 'Medium density',
  15: 'High density',
  30: 'Full resolution (large file)',
};

type Props = {
  selectedFps: FrameRateOption;
  onSelectFps: (fps: FrameRateOption) => void;
  sourceFps?: number;
  videoDuration?: number; // in seconds
  totalFrames?: number;
};

export default function FrameRateSelector({
  selectedFps,
  onSelectFps,
  sourceFps = 30,
  videoDuration = 0,
  totalFrames = 0,
}: Props) {
  const estimates = useMemo(() => {
    return FRAME_RATE_OPTIONS.map(fps => {
      const effectiveFps = Math.min(fps, sourceFps);
      const estimatedFrames = Math.floor(videoDuration * effectiveFps);
      const percentage = totalFrames > 0
        ? ((estimatedFrames / totalFrames) * 100).toFixed(1)
        : '0';

      return {
        fps,
        estimatedFrames,
        percentage,
        isWarning: estimatedFrames > 1000 || fps > sourceFps,
      };
    });
  }, [sourceFps, videoDuration, totalFrames]);

  const selectedEstimate = estimates.find(e => e.fps === selectedFps);

  return (
    <div {...stylex.props(styles.container)}>
      <label {...stylex.props(styles.label)}>
        Export Frame Rate
      </label>

      <div {...stylex.props(styles.optionsGrid)}>
        {FRAME_RATE_OPTIONS.map(fps => {
          const isSelected = fps === selectedFps;
          const isDisabled = fps > sourceFps;

          return (
            <button
              key={fps}
              onClick={() => !isDisabled && onSelectFps(fps)}
              disabled={isDisabled}
              {...stylex.props(
                styles.optionButton,
                isSelected && styles.optionButtonSelected
              )}
              style={{
                opacity: isDisabled ? 0.5 : 1,
                cursor: isDisabled ? 'not-allowed' : 'pointer',
              }}
              title={FPS_DESCRIPTIONS[fps]}
            >
              {fps} FPS
            </button>
          );
        })}
      </div>

      {selectedEstimate && (
        <div>
          <p {...stylex.props(styles.estimateText)}>
            ~{selectedEstimate.estimatedFrames} frames ({selectedEstimate.percentage}% of video)
          </p>
          {selectedEstimate.isWarning && (
            <p {...stylex.props(styles.warningText)}>
              {selectedFps > sourceFps
                ? `Frame rate exceeds source video (${sourceFps} FPS)`
                : 'Large export size - consider lower frame rate'}
            </p>
          )}
        </div>
      )}

      <p {...stylex.props(styles.estimateText)}>
        {FPS_DESCRIPTIONS[selectedFps]}
      </p>
    </div>
  );
}
