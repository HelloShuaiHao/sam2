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
import {Export} from '@carbon/icons-react';
import ExportConfigModal from '@/common/components/export/ExportConfigModal';
import ExportProgress from '@/common/components/export/ExportProgress';
import useExport from '@/common/components/export/useExport';
import {FrameRateOption} from '@/common/components/export/FrameRateSelector';
import {sessionAtom, trackletObjectsAtom} from '@/demo/atoms';
import {useAtomValue} from 'jotai';
import {useState, useMemo} from 'react';
import OptionButton from './OptionButton';
import useVideo from '@/common/components/video/editor/useVideo';

export default function ExportOption() {
  const session = useAtomValue(sessionAtom);
  const trackletObjects = useAtomValue(trackletObjectsAtom);
  const video = useVideo();
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Get video metadata
  const videoMetadata = useMemo(() => {
    // These values will be updated when video is decoded
    // For now, provide defaults that will work with the modal
    return {
      duration: 0, // Will be calculated from video
      fps: 30,
      totalFrames: 0,
      width: 1920,
      height: 1080,
    };
  }, []);

  const {exportState, startExport, downloadExport, resetExport} = useExport(
    session?.id || null,
  );

  const hasTrackedObjects = trackletObjects.length > 0;

  const handleOpenModal = () => {
    if (hasTrackedObjects && session) {
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
      <OptionButton
        title="Export Annotations"
        Icon={Export}
        loadingProps={{
          loading: exportState.isExporting,
          label: 'Exporting...',
        }}
        onClick={handleOpenModal}
        disabled={!hasTrackedObjects || !session}
      />

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
