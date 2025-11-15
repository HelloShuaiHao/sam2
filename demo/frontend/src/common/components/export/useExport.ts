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

import {useState, useCallback, useEffect, useRef} from 'react';
import type {ExportStatus} from './ExportProgress';
import {FrameRateOption} from './FrameRateSelector';

type ExportState = {
  isExporting: boolean;
  jobId: string | null;
  status: ExportStatus;
  progress: number;
  processedFrames: number;
  totalFrames: number;
  downloadUrl: string | null;
  errorMessage: string | null;
  fileSizeMb: number | null;
};

const INITIAL_STATE: ExportState = {
  isExporting: false,
  jobId: null,
  status: 'pending',
  progress: 0,
  processedFrames: 0,
  totalFrames: 0,
  downloadUrl: null,
  errorMessage: null,
  fileSizeMb: null,
};

export default function useExport(sessionId: string | null) {
  const [exportState, setExportState] = useState<ExportState>(INITIAL_STATE);
  const pollingIntervalRef = useRef<number | null>(null);

  const startExport = useCallback(async (targetFps: FrameRateOption) => {
    if (!sessionId) {
      console.error('No session ID available for export');
      return;
    }

    try {
      setExportState(prev => ({
        ...prev,
        isExporting: true,
        status: 'pending',
        progress: 0,
        errorMessage: null,
      }));

      // Call GraphQL mutation to start export
      const response = await fetch('/graphql', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: `
            mutation ExportVideoAnnotations($input: ExportVideoAnnotationsInput!) {
              exportVideoAnnotations(input: $input) {
                jobId
                status
                message
                estimatedFrames
              }
            }
          `,
          variables: {
            input: {
              sessionId,
              targetFps,
            },
          },
        }),
      });

      const data = await response.json();

      if (data.errors) {
        throw new Error(data.errors[0]?.message || 'Export failed');
      }

      const result = data.data.exportVideoAnnotations;

      if (result.status === 'FAILED') {
        throw new Error(result.message || 'Export failed');
      }

      setExportState(prev => ({
        ...prev,
        jobId: result.jobId,
        status: 'processing',
        totalFrames: result.estimatedFrames || 0,
      }));

      // Start polling for status
      startPolling(result.jobId);

    } catch (error) {
      console.error('Export error:', error);
      setExportState(prev => ({
        ...prev,
        isExporting: false,
        status: 'failed',
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
      }));
    }
  }, [sessionId]);

  const startPolling = useCallback((jobId: string) => {
    // Clear any existing polling
    if (pollingIntervalRef.current) {
      window.clearInterval(pollingIntervalRef.current);
    }

    // Poll every 1 second
    pollingIntervalRef.current = window.setInterval(async () => {
      try {
        const response = await fetch('/graphql', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: `
              query ExportJobStatus($jobId: String!) {
                exportJobStatus(jobId: $jobId) {
                  jobId
                  status
                  progress
                  processedFrames
                  totalFrames
                  downloadUrl
                  fileSizeMb
                  errorMessage
                }
              }
            `,
            variables: {
              jobId,
            },
          }),
        });

        const data = await response.json();

        if (data.errors) {
          throw new Error(data.errors[0]?.message || 'Status check failed');
        }

        const jobInfo = data.data.exportJobStatus;

        if (!jobInfo) {
          throw new Error('Job not found');
        }

        setExportState(prev => ({
          ...prev,
          status: jobInfo.status.toLowerCase() as ExportStatus,
          progress: jobInfo.progress,
          processedFrames: jobInfo.processedFrames,
          totalFrames: jobInfo.totalFrames,
          downloadUrl: jobInfo.downloadUrl,
          fileSizeMb: jobInfo.fileSizeMb,
          errorMessage: jobInfo.errorMessage,
          isExporting: jobInfo.status === 'PROCESSING' || jobInfo.status === 'PENDING',
        }));

        // Stop polling if completed or failed
        if (jobInfo.status === 'COMPLETED' || jobInfo.status === 'FAILED') {
          if (pollingIntervalRef.current) {
            window.clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
        }

      } catch (error) {
        console.error('Polling error:', error);
        if (pollingIntervalRef.current) {
          window.clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
        setExportState(prev => ({
          ...prev,
          isExporting: false,
          status: 'failed',
          errorMessage: error instanceof Error ? error.message : 'Polling failed',
        }));
      }
    }, 1000);
  }, []);

  const downloadExport = useCallback(() => {
    if (exportState.downloadUrl) {
      // Trigger download
      window.location.href = exportState.downloadUrl;
    }
  }, [exportState.downloadUrl]);

  const resetExport = useCallback(() => {
    if (pollingIntervalRef.current) {
      window.clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    setExportState(INITIAL_STATE);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        window.clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  return {
    exportState,
    startExport,
    downloadExport,
    resetExport,
  };
}
