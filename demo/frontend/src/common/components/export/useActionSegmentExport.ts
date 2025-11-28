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
import useSettingsContext from '@/settings/useSettingsContext';
import type {ExportStatus} from './ExportProgress';
import {FrameRateOption} from './FrameRateSelector';
import {useAtomValue} from 'jotai';
import {actionSegmentsAtom} from '@/demo/atoms';

type ActionExportState = {
  isExporting: boolean;
  jobId: string | null;
  status: ExportStatus;
  progress: number;
  segmentCount: number;
  downloadUrl: string | null;
  errorMessage: string | null;
};

const INITIAL_STATE: ActionExportState = {
  isExporting: false,
  jobId: null,
  status: 'pending',
  progress: 0,
  segmentCount: 0,
  downloadUrl: null,
  errorMessage: null,
};

export default function useActionSegmentExport(sessionId: string | null) {
  const {settings} = useSettingsContext();
  const [exportState, setExportState] = useState<ActionExportState>(INITIAL_STATE);
  const pollingIntervalRef = useRef<number | null>(null);
  const actionSegments = useAtomValue(actionSegmentsAtom);

  const startExport = useCallback(
    async (
      targetFps: FrameRateOption,
      segmentIds?: string[],
      includeVisualizations: boolean = true
    ) => {
      if (!sessionId) {
        console.error('No session ID available for export');
        return;
      }

      if (actionSegments.length === 0) {
        console.error('No action segments to export');
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

        // Prepare segments data for export
        const segmentsToExport = segmentIds
          ? actionSegments.filter(seg => segmentIds.includes(seg.id))
          : actionSegments;

        const segmentsData = segmentsToExport.map(seg => ({
          id: seg.id,
          name: seg.name,
          frameStart: seg.frameStart,
          frameEnd: seg.frameEnd,
          createdAt: seg.createdAt,
          objects: seg.objects.map(obj => ({
            objectId: obj.id,
            label: obj.name,
            color: obj.color,
          })),
        }));

        // Call GraphQL mutation to start export
        const graphqlUrl = `${settings.inferenceAPIEndpoint}/graphql`;
        const response = await fetch(graphqlUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: `
              mutation ExportActionSegments($input: ExportActionSegmentsInput!) {
                exportActionSegments(input: $input) {
                  jobId
                  status
                  message
                  segmentCount
                }
              }
            `,
            variables: {
              input: {
                sessionId,
                actionSegments: segmentsData,
                targetFps,
                includeVisualizations,
              },
            },
          }),
        });

        const data = await response.json();

        if (data.errors) {
          throw new Error(data.errors[0]?.message || 'Export failed');
        }

        const result = data.data.exportActionSegments;

        if (result.status === 'FAILED') {
          throw new Error(result.message || 'Export failed');
        }

        setExportState(prev => ({
          ...prev,
          jobId: result.jobId,
          status: 'processing',
          segmentCount: result.segmentCount || segmentsToExport.length,
        }));

        // Start polling for status
        startPolling(result.jobId);
      } catch (error) {
        console.error('Action segment export error:', error);
        setExportState(prev => ({
          ...prev,
          isExporting: false,
          status: 'failed',
          errorMessage:
            error instanceof Error ? error.message : 'Unknown error',
        }));
      }
    },
    [sessionId, settings.inferenceAPIEndpoint, actionSegments]
  );

  const startPolling = useCallback(
    (jobId: string) => {
      // Clear any existing polling
      if (pollingIntervalRef.current) {
        window.clearInterval(pollingIntervalRef.current);
      }

      // Poll every 1 second
      pollingIntervalRef.current = window.setInterval(async () => {
        try {
          const graphqlUrl = `${settings.inferenceAPIEndpoint}/graphql`;
          const response = await fetch(graphqlUrl, {
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
                    downloadUrl
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
            downloadUrl: jobInfo.downloadUrl,
            errorMessage: jobInfo.errorMessage,
            isExporting:
              jobInfo.status === 'PROCESSING' || jobInfo.status === 'PENDING',
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
            errorMessage:
              error instanceof Error ? error.message : 'Polling failed',
          }));
        }
      }, 1000);
    },
    [settings.inferenceAPIEndpoint]
  );

  const downloadExport = useCallback(async () => {
    if (exportState.downloadUrl && exportState.jobId) {
      // Trigger download
      window.location.href = exportState.downloadUrl;

      // Wait a bit to ensure download starts, then delete the export file
      setTimeout(async () => {
        try {
          const graphqlUrl = `${settings.inferenceAPIEndpoint}/graphql`;
          const response = await fetch(graphqlUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              query: `
                mutation DeleteExport($input: DeleteExportInput!) {
                  deleteExport(input: $input) {
                    success
                    message
                  }
                }
              `,
              variables: {
                input: {
                  jobId: exportState.jobId,
                },
              },
            }),
          });

          const data = await response.json();

          if (data.errors) {
            console.error('Failed to delete export:', data.errors);
          } else if (data.data?.deleteExport?.success) {
            console.log('Export file deleted successfully');
          } else {
            console.warn(
              'Export deletion failed:',
              data.data?.deleteExport?.message
            );
          }
        } catch (error) {
          console.error('Error deleting export:', error);
        }
      }, 2000); // Wait 2 seconds for download to start
    }
  }, [
    exportState.downloadUrl,
    exportState.jobId,
    settings.inferenceAPIEndpoint,
  ]);

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
