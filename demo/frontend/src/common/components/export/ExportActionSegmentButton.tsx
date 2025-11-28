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

import {useState, useCallback} from 'react';
import {useAtomValue} from 'jotai';
import {actionSegmentsAtom, sessionAtom} from '@/demo/atoms';
import useActionSegmentExport from './useActionSegmentExport';
import {FrameRateOption} from './FrameRateSelector';
import stylex from '@stylexjs/stylex';
import PrimaryCTAButton from '@/common/components/button/PrimaryCTAButton';

const styles = stylex.create({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  button: {
    padding: '10px 16px',
    fontSize: '14px',
    fontWeight: '600',
    color: '#fff',
    backgroundColor: 'rgba(34, 197, 94, 0.8)',
    border: '1px solid rgba(34, 197, 94, 1)',
    borderRadius: '6px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    transition: 'all 0.2s',
    ':hover': {
      backgroundColor: 'rgba(34, 197, 94, 0.9)',
    },
    ':disabled': {
      opacity: 0.5,
      cursor: 'not-allowed',
    },
  },
  dropdown: {
    position: 'relative',
  },
  dropdownButton: {
    padding: '10px 16px',
    fontSize: '14px',
    fontWeight: '600',
    color: '#fff',
    backgroundColor: 'rgba(34, 197, 94, 0.8)',
    border: '1px solid rgba(34, 197, 94, 1)',
    borderRadius: '6px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    transition: 'all 0.2s',
    ':hover': {
      backgroundColor: 'rgba(34, 197, 94, 0.9)',
    },
  },
  dropdownMenu: {
    position: 'absolute',
    top: '100%',
    left: 0,
    marginTop: '4px',
    backgroundColor: 'rgba(30, 30, 30, 0.95)',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    borderRadius: '6px',
    padding: '8px',
    minWidth: '200px',
    zIndex: 1000,
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
  },
  menuItem: {
    padding: '8px 12px',
    fontSize: '13px',
    color: '#fff',
    cursor: 'pointer',
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    transition: 'background-color 0.2s',
    ':hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.1)',
    },
  },
  checkbox: {
    width: '16px',
    height: '16px',
    accentColor: 'rgb(34, 197, 94)',
  },
  progressContainer: {
    padding: '12px',
    backgroundColor: 'rgba(30, 30, 30, 0.9)',
    borderRadius: '6px',
    border: '1px solid rgba(59, 130, 246, 0.5)',
  },
  progressBar: {
    width: '100%',
    height: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: '4px',
    overflow: 'hidden',
    marginTop: '8px',
  },
  progressFill: {
    height: '100%',
    backgroundColor: 'rgb(34, 197, 94)',
    transition: 'width 0.3s',
  },
  progressText: {
    fontSize: '13px',
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: '4px',
  },
  downloadButton: {
    padding: '10px 16px',
    fontSize: '14px',
    fontWeight: '600',
    color: '#fff',
    backgroundColor: 'rgba(59, 130, 246, 0.8)',
    border: '1px solid rgba(59, 130, 246, 1)',
    borderRadius: '6px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: '8px',
    ':hover': {
      backgroundColor: 'rgba(59, 130, 246, 0.9)',
    },
  },
  errorText: {
    fontSize: '13px',
    color: 'rgb(239, 68, 68)',
    marginTop: '4px',
  },
});

type Props = {
  targetFps?: FrameRateOption;
};

export default function ExportActionSegmentButton({targetFps = 30}: Props) {
  const session = useAtomValue(sessionAtom);
  const actionSegments = useAtomValue(actionSegmentsAtom);
  const {exportState, startExport, downloadExport, resetExport} =
    useActionSegmentExport(session?.id || null);

  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedSegmentIds, setSelectedSegmentIds] = useState<string[]>([]);

  const handleExportAll = useCallback(() => {
    startExport(targetFps, undefined, true);
    setShowDropdown(false);
  }, [startExport, targetFps]);

  const handleExportSelected = useCallback(() => {
    if (selectedSegmentIds.length > 0) {
      startExport(targetFps, selectedSegmentIds, true);
      setShowDropdown(false);
      setSelectedSegmentIds([]);
    }
  }, [startExport, targetFps, selectedSegmentIds]);

  const toggleSegmentSelection = useCallback((segmentId: string) => {
    setSelectedSegmentIds(prev =>
      prev.includes(segmentId)
        ? prev.filter(id => id !== segmentId)
        : [...prev, segmentId]
    );
  }, []);

  const handleDownload = useCallback(() => {
    downloadExport();
    resetExport();
  }, [downloadExport, resetExport]);

  if (actionSegments.length === 0) {
    return null;
  }

  // Show progress if exporting
  if (exportState.isExporting || exportState.status === 'completed') {
    return (
      <div {...stylex.props(styles.progressContainer)}>
        <div {...stylex.props(styles.progressText)}>
          {exportState.status === 'pending' && '准备导出动作片段...'}
          {exportState.status === 'processing' &&
            `导出进度: ${Math.round(exportState.progress * 100)}%`}
          {exportState.status === 'completed' && '导出完成!'}
        </div>

        {exportState.status !== 'completed' && (
          <div {...stylex.props(styles.progressBar)}>
            <div
              {...stylex.props(styles.progressFill)}
              style={{width: `${exportState.progress * 100}%`}}
            />
          </div>
        )}

        {exportState.status === 'completed' && exportState.downloadUrl && (
          <button
            {...stylex.props(styles.downloadButton)}
            onClick={handleDownload}>
            📥 下载动作片段标注 ({exportState.segmentCount} 个片段)
          </button>
        )}

        {exportState.errorMessage && (
          <div {...stylex.props(styles.errorText)}>
            错误: {exportState.errorMessage}
          </div>
        )}
      </div>
    );
  }

  return (
    <div {...stylex.props(styles.container)}>
      <div {...stylex.props(styles.dropdown)}>
        <PrimaryCTAButton
          onClick={() => setShowDropdown(!showDropdown)}>
          导出标注
        </PrimaryCTAButton>

        {showDropdown && (
          <div {...stylex.props(styles.dropdownMenu)}>
            <div
              {...stylex.props(styles.menuItem)}
              onClick={handleExportAll}>
              导出全部片段 ({actionSegments.length} 个)
            </div>

            <div
              style={{
                height: '1px',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                margin: '8px 0',
              }}
            />

            <div style={{maxHeight: '200px', overflowY: 'auto'}}>
              {actionSegments.map(segment => (
                <div
                  key={segment.id}
                  {...stylex.props(styles.menuItem)}
                  onClick={() => toggleSegmentSelection(segment.id)}>
                  <input
                    type="checkbox"
                    {...stylex.props(styles.checkbox)}
                    checked={selectedSegmentIds.includes(segment.id)}
                    onChange={() => {}}
                    onClick={e => e.stopPropagation()}
                  />
                  {segment.name}
                </div>
              ))}
            </div>

            {selectedSegmentIds.length > 0 && (
              <>
                <div
                  style={{
                    height: '1px',
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    margin: '8px 0',
                  }}
                />
                <div
                  {...stylex.props(styles.menuItem)}
                  onClick={handleExportSelected}
                  style={{
                    backgroundColor: 'rgba(34, 197, 94, 0.2)',
                    fontWeight: '600',
                  }}>
                  导出选中的 {selectedSegmentIds.length} 个片段
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
