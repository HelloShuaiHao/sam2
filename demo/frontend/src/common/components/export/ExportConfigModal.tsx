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
    background: 'rgba(0, 0, 0, 0.6)',
    backdropFilter: 'blur(8px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  modal: {
    background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
    borderRadius: 16,
    padding: spacing[6],
    maxWidth: 600,
    width: '90%',
    maxHeight: '80vh',
    overflow: 'auto',
    border: '1px solid rgba(255, 255, 255, 0.3)',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3), 0 0 1px rgba(255, 255, 255, 0.5) inset',
  },
  header: {
    marginBottom: spacing[4],
  },
  title: {
    fontSize: 22,
    fontWeight: 700,
    color: '#2d3748',
    marginBottom: spacing[2],
    letterSpacing: '-0.5px',
  },
  subtitle: {
    fontSize: 14,
    color: '#718096',
    lineHeight: 1.6,
  },
  content: {
    marginBottom: spacing[5],
  },
  metadataGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: spacing[3],
    marginBottom: spacing[4],
    padding: spacing[4],
    background: 'rgba(255, 255, 255, 0.5)',
    borderRadius: 12,
    border: '1px solid rgba(255, 255, 255, 0.6)',
  },
  metadataItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: spacing[1],
  },
  metadataLabel: {
    fontSize: 11,
    color: '#718096',
    textTransform: 'uppercase',
    letterSpacing: '1px',
    fontWeight: 600,
  },
  metadataValue: {
    fontSize: 17,
    color: '#2d3748',
    fontWeight: 700,
  },
  footer: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: spacing[3],
    paddingTop: spacing[4],
    borderTop: '1px solid rgba(0, 0, 0, 0.08)',
  },
  button: {
    padding: `${spacing[2]}px ${spacing[5]}px`,
    borderRadius: 10,
    fontSize: 15,
    fontWeight: 600,
    cursor: 'pointer',
    border: 'none',
    transition: 'all 0.2s ease',
  },
  cancelButton: {
    background: 'rgba(0, 0, 0, 0.05)',
    color: '#4a5568',
    border: '1px solid rgba(0, 0, 0, 0.1)',
    ':hover': {
      background: 'rgba(0, 0, 0, 0.08)',
      borderColor: 'rgba(0, 0, 0, 0.15)',
    },
  },
  exportButton: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    boxShadow: '0 4px 12px rgba(102, 126, 234, 0.3)',
    ':hover': {
      transform: 'translateY(-2px)',
      boxShadow: '0 6px 16px rgba(102, 126, 234, 0.4)',
    },
    ':disabled': {
      opacity: 0.5,
      cursor: 'not-allowed',
      transform: 'none',
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
