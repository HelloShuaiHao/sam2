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
import {useEffect, useState} from 'react';

const styles = stylex.create({
  container: {
    position: 'fixed',
    bottom: spacing[4],
    right: spacing[4],
    background: color['gray-900'],
    border: `1px solid ${color['gray-700']}`,
    borderRadius: 12,
    padding: spacing[4],
    minWidth: 320,
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
    zIndex: 999,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing[3],
  },
  title: {
    fontSize: 14,
    fontWeight: 600,
    color: color['gray-100'],
  },
  closeButton: {
    background: 'none',
    border: 'none',
    color: color['gray-400'],
    cursor: 'pointer',
    fontSize: 18,
    padding: 0,
    ':hover': {
      color: color['gray-100'],
    },
  },
  progressBar: {
    width: '100%',
    height: 8,
    background: color['gray-700'],
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: spacing[2],
  },
  progressFill: {
    height: '100%',
    background: color['blue-500'],
    transition: 'width 0.3s ease',
    borderRadius: 4,
  },
  status: {
    fontSize: 12,
    color: color['gray-400'],
    marginBottom: spacing[1],
  },
  percentage: {
    fontSize: 18,
    fontWeight: 600,
    color: color['gray-100'],
    marginBottom: spacing[2],
  },
  successMessage: {
    fontSize: 14,
    color: color['green-500'],
    marginBottom: spacing[2],
  },
  errorMessage: {
    fontSize: 14,
    color: color['red-500'],
    marginBottom: spacing[2],
  },
  downloadButton: {
    width: '100%',
    padding: `${spacing[2]}px ${spacing[3]}px`,
    background: color['blue-600'],
    color: 'white',
    border: 'none',
    borderRadius: 6,
    fontSize: 14,
    fontWeight: 500,
    cursor: 'pointer',
    ':hover': {
      background: color['blue-500'],
    },
  },
  retryButton: {
    width: '100%',
    padding: `${spacing[2]}px ${spacing[3]}px`,
    background: color['gray-700'],
    color: color['gray-100'],
    border: 'none',
    borderRadius: 6,
    fontSize: 14,
    fontWeight: 500,
    cursor: 'pointer',
    ':hover': {
      background: color['gray-600'],
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
  downloadUrl?: string;
  errorMessage?: string;
  fileSizeMb?: number;
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
  const [autoHideTimer, setAutoHideTimer] = useState<number | null>(null);

  useEffect(() => {
    if (status === 'completed' && autoHideTimer === null) {
      // Auto-hide after 3 seconds on completion
      const timer = window.setTimeout(() => {
        onClose();
      }, 3000);
      setAutoHideTimer(timer);
    }

    return () => {
      if (autoHideTimer !== null) {
        window.clearTimeout(autoHideTimer);
      }
    };
  }, [status, autoHideTimer, onClose]);

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
    <div {...stylex.props(styles.container)}>
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
            ✓ {getStatusText()}
            {fileSizeMb && ` (${fileSizeMb.toFixed(2)} MB)`}
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
  );
}
