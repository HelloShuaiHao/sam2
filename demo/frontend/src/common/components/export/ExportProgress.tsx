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

import {spacing} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';

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
  container: {
    background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
    border: '1px solid rgba(255, 255, 255, 0.3)',
    borderRadius: 16,
    padding: spacing[6],
    minWidth: 420,
    maxWidth: 500,
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3), 0 0 1px rgba(255, 255, 255, 0.5) inset',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing[4],
  },
  title: {
    fontSize: 18,
    fontWeight: 600,
    color: '#2d3748',
    letterSpacing: '-0.5px',
  },
  closeButton: {
    background: 'rgba(0, 0, 0, 0.05)',
    border: 'none',
    color: '#718096',
    cursor: 'pointer',
    fontSize: 20,
    width: 28,
    height: 28,
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s ease',
    ':hover': {
      background: 'rgba(0, 0, 0, 0.1)',
      color: '#2d3748',
    },
  },
  progressBar: {
    width: '100%',
    height: 10,
    background: 'rgba(0, 0, 0, 0.08)',
    borderRadius: 8,
    overflow: 'hidden',
    marginBottom: spacing[3],
    boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.1)',
  },
  progressFill: {
    height: '100%',
    background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
    transition: 'width 0.3s ease',
    borderRadius: 8,
    boxShadow: '0 2px 8px rgba(102, 126, 234, 0.4)',
  },
  status: {
    fontSize: 13,
    color: '#718096',
    marginBottom: spacing[1],
    fontWeight: 500,
  },
  percentage: {
    fontSize: 24,
    fontWeight: 700,
    color: '#2d3748',
    marginBottom: spacing[2],
    letterSpacing: '-1px',
  },
  successMessage: {
    fontSize: 15,
    color: '#38a169',
    marginBottom: spacing[3],
    fontWeight: 600,
    display: 'flex',
    alignItems: 'center',
    gap: spacing[2],
  },
  errorMessage: {
    fontSize: 15,
    color: '#e53e3e',
    marginBottom: spacing[3],
    fontWeight: 600,
  },
  downloadButton: {
    width: '100%',
    padding: `${spacing[3]}px ${spacing[4]}px`,
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: 10,
    fontSize: 15,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    boxShadow: '0 4px 12px rgba(102, 126, 234, 0.3)',
    ':hover': {
      transform: 'translateY(-2px)',
      boxShadow: '0 6px 16px rgba(102, 126, 234, 0.4)',
    },
  },
  retryButton: {
    width: '100%',
    padding: `${spacing[3]}px ${spacing[4]}px`,
    background: 'rgba(0, 0, 0, 0.05)',
    color: '#4a5568',
    border: '1px solid rgba(0, 0, 0, 0.1)',
    borderRadius: 10,
    fontSize: 15,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    ':hover': {
      background: 'rgba(0, 0, 0, 0.08)',
      borderColor: 'rgba(0, 0, 0, 0.15)',
    },
  },
});

export type ExportStatus = 'pending' | 'processing' | 'completed' | 'failed';

type Props = {
  isVisible: boolean;
  status: ExportStatus;
  progress: number; // 0 to 1
  processedFrames?: number;
  totalFrames?: number;
  downloadUrl?: string | null;
  errorMessage?: string | null;
  fileSizeMb?: number | null;
  onClose: () => void;
  onDownload?: () => void;
  onRetry?: () => void;
};

export default function ExportProgress({
  isVisible,
  status,
  progress,
  processedFrames,
  totalFrames,
  downloadUrl,
  errorMessage,
  fileSizeMb,
  onClose,
  onDownload,
  onRetry,
}: Props) {
  // Removed auto-hide functionality - user must manually close

  if (!isVisible) {
    return null;
  }

  const percentage = Math.round(progress * 100);

  const getStatusText = () => {
    switch (status) {
      case 'pending':
        return 'Preparing export...';
      case 'processing':
        return processedFrames && totalFrames
          ? `Processing frame ${processedFrames} of ${totalFrames}`
          : 'Processing frames...';
      case 'completed':
        return 'Export completed!';
      case 'failed':
        return 'Export failed';
      default:
        return '';
    }
  };

  return (
    <div {...stylex.props(styles.overlay)} onClick={onClose}>
      <div
        {...stylex.props(styles.container)}
        onClick={e => e.stopPropagation()}
      >
        <div {...stylex.props(styles.header)}>
          <h3 {...stylex.props(styles.title)}>Exporting Annotations</h3>
          <button
            {...stylex.props(styles.closeButton)}
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {status === 'processing' || status === 'pending' ? (
          <>
            <div {...stylex.props(styles.progressBar)}>
              <div
                {...stylex.props(styles.progressFill)}
                style={{width: `${percentage}%`}}
              />
            </div>
            <p {...stylex.props(styles.percentage)}>{percentage}%</p>
            <p {...stylex.props(styles.status)}>{getStatusText()}</p>
          </>
        ) : status === 'completed' ? (
          <>
            <p {...stylex.props(styles.successMessage)}>
              <span>✓</span>
              <span>
                {getStatusText()}
                {fileSizeMb && ` (${fileSizeMb.toFixed(2)} MB)`}
              </span>
            </p>
            {downloadUrl && onDownload && (
              <button
                {...stylex.props(styles.downloadButton)}
                onClick={onDownload}
              >
                Download Export
              </button>
            )}
          </>
        ) : status === 'failed' ? (
          <>
            <p {...stylex.props(styles.errorMessage)}>
              ✗ {errorMessage || 'An error occurred during export'}
            </p>
            {onRetry && (
              <button
                {...stylex.props(styles.retryButton)}
                onClick={onRetry}
              >
                Retry Export
              </button>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
