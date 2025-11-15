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

import FrameRateSelector, {FrameRateOption} from './FrameRateSelector';
import {color, spacing} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';
import {useState} from 'react';

const styles = stylex.create({
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  modal: {
    background: color['gray-900'],
    borderRadius: 12,
    padding: spacing[6],
    maxWidth: 600,
    width: '90%',
    maxHeight: '80vh',
    overflow: 'auto',
    border: `1px solid ${color['gray-700']}`,
  },
  header: {
    marginBottom: spacing[4],
  },
  title: {
    fontSize: 20,
    fontWeight: 600,
    color: color['gray-100'],
    marginBottom: spacing[2],
  },
  subtitle: {
    fontSize: 14,
    color: color['gray-400'],
    lineHeight: 1.5,
  },
  content: {
    marginBottom: spacing[5],
  },
  metadataGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: spacing[3],
    marginBottom: spacing[4],
    padding: spacing[3],
    background: color['gray-800'],
    borderRadius: 8,
  },
  metadataItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: spacing[1],
  },
  metadataLabel: {
    fontSize: 12,
    color: color['gray-500'],
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  metadataValue: {
    fontSize: 16,
    color: color['gray-100'],
    fontWeight: 500,
  },
  footer: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: spacing[3],
    paddingTop: spacing[4],
    borderTop: `1px solid ${color['gray-700']}`,
  },
  button: {
    padding: `${spacing[2]}px ${spacing[4]}px`,
    borderRadius: 6,
    fontSize: 14,
    fontWeight: 500,
    cursor: 'pointer',
    border: 'none',
    transition: 'all 0.2s ease',
  },
  cancelButton: {
    background: color['gray-700'],
    color: color['gray-100'],
    ':hover': {
      background: color['gray-600'],
    },
  },
  exportButton: {
    background: color['blue-600'],
    color: 'white',
    ':hover': {
      background: color['blue-500'],
    },
    ':disabled': {
      opacity: 0.5,
      cursor: 'not-allowed',
    },
  },
});

type Props = {
  isOpen: boolean;
  onClose: () => void;
  onExport: (targetFps: FrameRateOption) => void;
  videoMetadata: {
    duration: number; // seconds
    fps: number;
    totalFrames: number;
    width: number;
    height: number;
  };
  isExporting?: boolean;
};

export default function ExportConfigModal({
  isOpen,
  onClose,
  onExport,
  videoMetadata,
  isExporting = false,
}: Props) {
  const [selectedFps, setSelectedFps] = useState<FrameRateOption>(5);

  if (!isOpen) {
    return null;
  }

  const handleExport = () => {
    onExport(selectedFps);
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div {...stylex.props(styles.overlay)} onClick={onClose}>
      <div
        {...stylex.props(styles.modal)}
        onClick={e => e.stopPropagation()}
      >
        <div {...stylex.props(styles.header)}>
          <h2 {...stylex.props(styles.title)}>Export Video Annotations</h2>
          <p {...stylex.props(styles.subtitle)}>
            Export your video annotations as JSON format with frame sampling.
            Choose a frame rate to balance between dataset size and temporal resolution.
          </p>
        </div>

        <div {...stylex.props(styles.content)}>
          <div {...stylex.props(styles.metadataGrid)}>
            <div {...stylex.props(styles.metadataItem)}>
              <span {...stylex.props(styles.metadataLabel)}>Duration</span>
              <span {...stylex.props(styles.metadataValue)}>
                {formatDuration(videoMetadata.duration)}
              </span>
            </div>

            <div {...stylex.props(styles.metadataItem)}>
              <span {...stylex.props(styles.metadataLabel)}>Source FPS</span>
              <span {...stylex.props(styles.metadataValue)}>
                {videoMetadata.fps} FPS
              </span>
            </div>

            <div {...stylex.props(styles.metadataItem)}>
              <span {...stylex.props(styles.metadataLabel)}>Total Frames</span>
              <span {...stylex.props(styles.metadataValue)}>
                {videoMetadata.totalFrames.toLocaleString()}
              </span>
            </div>

            <div {...stylex.props(styles.metadataItem)}>
              <span {...stylex.props(styles.metadataLabel)}>Resolution</span>
              <span {...stylex.props(styles.metadataValue)}>
                {videoMetadata.width} × {videoMetadata.height}
              </span>
            </div>
          </div>

          <FrameRateSelector
            selectedFps={selectedFps}
            onSelectFps={setSelectedFps}
            sourceFps={videoMetadata.fps}
            videoDuration={videoMetadata.duration}
            totalFrames={videoMetadata.totalFrames}
          />
        </div>

        <div {...stylex.props(styles.footer)}>
          <button
            {...stylex.props(styles.button, styles.cancelButton)}
            onClick={onClose}
            disabled={isExporting}
          >
            Cancel
          </button>
          <button
            {...stylex.props(styles.button, styles.exportButton)}
            onClick={handleExport}
            disabled={isExporting}
          >
            {isExporting ? 'Exporting...' : 'Export Annotations'}
          </button>
        </div>
      </div>
    </div>
  );
}
