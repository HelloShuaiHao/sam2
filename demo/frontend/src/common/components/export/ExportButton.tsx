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

import ExportConfigModal from './ExportConfigModal';
import ExportProgress from './ExportProgress';
import useExport from './useExport';
import {color, spacing} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';
import {useState} from 'react';
import {FrameRateOption} from './FrameRateSelector';

const styles = stylex.create({
  button: {
    display: 'flex',
    alignItems: 'center',
    gap: spacing[2],
    padding: `${spacing[2]}px ${spacing[3]}px`,
    background: color['blue-600'],
    color: 'white',
    border: 'none',
    borderRadius: 6,
    fontSize: 14,
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    ':hover': {
      background: color['blue-500'],
    },
    ':disabled': {
      opacity: 0.5,
      cursor: 'not-allowed',
    },
  },
  icon: {
    width: 16,
    height: 16,
  },
});

type Props = {
  sessionId: string | null;
  videoMetadata: {
    duration: number;
    fps: number;
    totalFrames: number;
    width: number;
    height: number;
  };
  hasTrackedObjects?: boolean;
};

export default function ExportButton({
  sessionId,
  videoMetadata,
  hasTrackedObjects = false,
}: Props) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const {exportState, startExport, downloadExport, resetExport} = useExport(sessionId);

  const handleOpenModal = () => {
    if (hasTrackedObjects && sessionId) {
      setIsModalOpen(true);
    }
  };

  const handleExport = async (targetFps: FrameRateOption) => {
    setIsModalOpen(false);
    await startExport(targetFps);
  };

  const handleCloseProgress = () => {
    resetExport();
  };

  return (
    <>
      <button
        {...stylex.props(styles.button)}
        onClick={handleOpenModal}
        disabled={!hasTrackedObjects || !sessionId || exportState.isExporting}
        title={
          !hasTrackedObjects
            ? 'Add object annotations before exporting'
            : !sessionId
            ? 'Session not initialized'
            : 'Export video annotations'
        }
      >
        <svg
          {...stylex.props(styles.icon)}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
          />
        </svg>
        Export
      </button>

      <ExportConfigModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onExport={handleExport}
        videoMetadata={videoMetadata}
        isExporting={exportState.isExporting}
      />

      <ExportProgress
        isVisible={
          exportState.isExporting ||
          exportState.status === 'completed' ||
          exportState.status === 'failed'
        }
        status={exportState.status}
        progress={exportState.progress}
        processedFrames={exportState.processedFrames}
        totalFrames={exportState.totalFrames}
        downloadUrl={exportState.downloadUrl}
        errorMessage={exportState.errorMessage}
        fileSizeMb={exportState.fileSizeMb}
        onClose={handleCloseProgress}
        onDownload={downloadExport}
        onRetry={() => isModalOpen && handleExport(5)}
      />
    </>
  );
}
