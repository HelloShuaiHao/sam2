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
import TrackletsAnnotation from '@/common/components/annotations/TrackletsAnnotation';
import ActionSegmentsAnnotation from '@/common/components/actions/ActionSegmentsAnnotation';
import useCloseSessionBeforeUnload from '@/common/components/session/useCloseSessionBeforeUnload';
import MessagesSnackbar from '@/common/components/snackbar/MessagesSnackbar';
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import {OBJECT_TOOLBAR_INDEX} from '@/common/components/toolbar/ToolbarConfig';
import useToolbarTabs from '@/common/components/toolbar/useToolbarTabs';
import VideoFilmstripWithPlayback from '@/common/components/video/VideoFilmstripWithPlayback';
import {
  FrameUpdateEvent,
  RenderingErrorEvent,
  SessionStartedEvent,
  TrackletsEvent,
} from '@/common/components/video/VideoWorkerBridge';
import VideoEditor from '@/common/components/video/editor/VideoEditor';
import useResetDemoEditor from '@/common/components/video/editor/useResetEditor';
import useVideo from '@/common/components/video/editor/useVideo';
import InteractionLayer from '@/common/components/video/layers/InteractionLayer';
import {PointsLayer} from '@/common/components/video/layers/PointsLayer';
import LoadingStateScreen from '@/common/loading/LoadingStateScreen';
import UploadLoadingScreen from '@/common/loading/UploadLoadingScreen';
import useScreenSize from '@/common/screen/useScreenSize';
import {SegmentationPoint} from '@/common/tracker/Tracker';
import {
  activeTrackletObjectIdAtom,
  frameIndexAtom,
  isAddObjectEnabledAtom,
  isPlayingAtom,
  isVideoLoadingAtom,
  pointsAtom,
  sessionAtom,
  streamingStateAtom,
  trackletObjectsAtom,
  uploadingStateAtom,
  VideoData,
  annotationModeAtom,
  activeActionSegmentAtom,
  actionSegmentsAtom,
  activeActionObjectIdAtom,
  isAddingActionObjectAtom,
  activeActionObjectPointsAtom,
} from '@/demo/atoms';
import useSettingsContext from '@/settings/useSettingsContext';
import {color, spacing} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';
import {useAtom, useAtomValue, useSetAtom} from 'jotai';
import {useEffect, useState} from 'react';
import type {ErrorObject} from 'serialize-error';

const styles = stylex.create({
  container: {
    display: 'flex',
    flexDirection: 'column',
    overflow: 'auto',
    width: '100%',
    borderColor: color['gray-800'],
    backgroundColor: color['gray-800'],
    borderWidth: 8,
    borderRadius: 12,
    '@media screen and (max-width: 768px)': {
      // on mobile, we want to grow the editor container so that the editor
      // fills the remaining vertical space between the navbar and bottom
      // of the page
      flexGrow: 1,
      borderWidth: 0,
      borderRadius: 0,
      paddingBottom: spacing[4],
    },
  },
  loadingScreenWrapper: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    background: 'white',
    overflow: 'hidden',
    overflowY: 'auto',
    zIndex: 999,
  },
});

type Props = {
  video: VideoData;
};

export default function DemoVideoEditor({video: inputVideo}: Props) {
  const {settings} = useSettingsContext();
  const video = useVideo();

  const [isSessionStartFailed, setIsSessionStartFailed] =
    useState<boolean>(false);

  const [session, setSession] = useAtom(sessionAtom);

  const [activeTrackletId, setActiveTrackletObjectId] = useAtom(
    activeTrackletObjectIdAtom,
  );
  const setTrackletObjects = useSetAtom(trackletObjectsAtom);
  const setFrameIndex = useSetAtom(frameIndexAtom);
  const points = useAtomValue(pointsAtom);
  const actionObjectPoints = useAtomValue(activeActionObjectPointsAtom);
  const isAddObjectEnabled = useAtomValue(isAddObjectEnabledAtom);
  const streamingState = useAtomValue(streamingStateAtom);
  const isPlaying = useAtomValue(isPlayingAtom);
  const isVideoLoading = useAtomValue(isVideoLoadingAtom);
  const uploadingState = useAtomValue(uploadingStateAtom);

  // 动作片段相关状态
  const annotationMode = useAtomValue(annotationModeAtom);
  const activeActionSegment = useAtomValue(activeActionSegmentAtom);
  const [actionSegments, setActionSegments] = useAtom(actionSegmentsAtom);
  const [activeActionObjectId, setActiveActionObjectId] = useAtom(
    activeActionObjectIdAtom,
  );
  const [isAddingActionObject, setIsAddingActionObject] = useAtom(
    isAddingActionObjectAtom,
  );

  const [renderingError, setRenderingError] = useState<ErrorObject | null>(
    null,
  );

  const {isMobile} = useScreenSize();

  const [tabIndex] = useToolbarTabs();
  const {enqueueMessage} = useMessagesSnackbar();

  useCloseSessionBeforeUnload();

  const {resetEditor, resetSession} = useResetDemoEditor();
  useEffect(() => {
    resetEditor();
  }, [inputVideo, resetEditor]);

  useEffect(() => {
    function onFrameUpdate(event: FrameUpdateEvent) {
      setFrameIndex(event.index);
    }

    // Listen to frame updates to fetch the frame index in the main thread,
    // which is then used downstream to render points per frame.
    video?.addEventListener('frameUpdate', onFrameUpdate);

    function onSessionStarted(event: SessionStartedEvent) {
      setSession({id: event.sessionId, ranPropagation: false});
    }

    video?.addEventListener('sessionStarted', onSessionStarted);

    function onSessionStartFailed() {
      setIsSessionStartFailed(true);
    }

    video?.addEventListener('sessionStartFailed', onSessionStartFailed);

    function onTrackletsUpdated(event: TrackletsEvent) {
      const tracklets = event.tracklets;
      if (tracklets.length === 0) {
        resetSession();
        return;
      }
      // 直接更新，过滤逻辑在下面的 useEffect 中处理
      setTrackletObjects(tracklets);
    }

    video?.addEventListener('trackletsUpdated', onTrackletsUpdated);

    function onRenderingError(event: RenderingErrorEvent) {
      setRenderingError(event.error);
    }

    video?.addEventListener('renderingError', onRenderingError);

    video?.initializeTracker('SAM 2', {
      inferenceEndpoint: settings.inferenceAPIEndpoint,
    });

    video?.startSession(inputVideo.path);

    return () => {
      // Don't close session here - it needs to remain active for export functionality
      // Session will be closed on browser/tab close via useCloseSessionBeforeUnload
      // video?.closeSession();
      video?.removeEventListener('frameUpdate', onFrameUpdate);
      video?.removeEventListener('sessionStarted', onSessionStarted);
      video?.removeEventListener('sessionStartFailed', onSessionStartFailed);
      video?.removeEventListener('trackletsUpdated', onTrackletsUpdated);
      video?.removeEventListener('renderingError', onRenderingError);
    };
  }, [
    setFrameIndex,
    setSession,
    setTrackletObjects,
    resetSession,
    inputVideo,
    video,
    settings.inferenceAPIEndpoint,
    settings.videoAPIEndpoint,
  ]);

  async function handleOptimisticPointUpdate(newPoints: SegmentationPoint[]) {
    if (session == null) {
      return;
    }

    async function createActiveTracklet() {
      if (!isAddObjectEnabled || newPoints.length === 0) {
        return;
      }
      const tracklet = await video?.createTracklet();
      if (tracklet != null && newPoints.length > 0) {
        setActiveTrackletObjectId(tracklet.id);
        video?.updatePoints(tracklet.id, [newPoints[newPoints.length - 1]]);
      }
    }

    if (activeTrackletId != null) {
      video?.updatePoints(activeTrackletId, newPoints);
    } else {
      await createActiveTracklet();
    }
    enqueueMessage('pointClick');
  }

  async function handleAddPoint(point: SegmentationPoint) {
    if (streamingState === 'partial' || streamingState === 'requesting') {
      return;
    }
    if (isPlaying) {
      return video?.pause();
    }
    handleOptimisticPointUpdate([...points, point]);
  }

  // === 动作片段模式：添加物体点击 ===
  async function handleAddPointInActionSegment(point: SegmentationPoint) {
    if (streamingState === 'partial' || streamingState === 'requesting') {
      return;
    }
    if (isPlaying) {
      video?.pause();
      return;
    }

    // 检查 session 是否有效
    if (!session) {
      console.error('[DemoVideoEditor] Session 不存在或已过期，请刷新页面');
      alert('Session 已过期，请刷新页面重新开始');
      setIsAddingActionObject(false);
      return;
    }

    // 必须有活跃的片段
    if (!activeActionSegment) {
      console.warn('[DemoVideoEditor] 没有活跃的动作片段，无法添加物体');
      setIsAddingActionObject(false);
      return;
    }

    const frameIndex = video?.frame ?? 0;

    // 检查当前帧是否在片段范围内
    if (
      frameIndex < activeActionSegment.frameStart ||
      frameIndex > activeActionSegment.frameEnd
    ) {
      console.warn(
        `[DemoVideoEditor] 当前帧 ${frameIndex} 不在片段范围 [${activeActionSegment.frameStart}, ${activeActionSegment.frameEnd}] 内`,
      );
      alert(
        `请将视频帧移动到时间段范围内 (${activeActionSegment.frameStart}-${activeActionSegment.frameEnd})`
      );
      return;
    }

    // 如果有选中的物体（编辑模式），向该物体添加点
    if (activeActionObjectId !== null) {
      const existingObject = activeActionSegment.objects.find(
        obj => obj.id === activeActionObjectId,
      );
      if (existingObject) {
        // 获取当前帧的现有点
        const currentFramePoints = existingObject.points[frameIndex] ?? [];
        const newPoints = [...currentFramePoints, point];

        // 更新视频中的点
        video?.updatePoints(activeActionObjectId, newPoints);

        // 更新状态中的点
        setActionSegments(segments =>
          segments.map(seg =>
            seg.id === activeActionSegment.id
              ? {
                  ...seg,
                  objects: seg.objects.map(obj =>
                    obj.id === activeActionObjectId
                      ? {
                          ...obj,
                          points: (() => {
                            const updatedPoints = [...obj.points];
                            updatedPoints[frameIndex] = newPoints;
                            return updatedPoints;
                          })(),
                        }
                      : obj,
                  ),
                }
              : seg,
          ),
        );
        enqueueMessage('pointClick');
      }
      return;
    }

    // 如果没有活跃的片段物体，创建新物体（仅在isAddingActionObject为true时）
    if (activeActionObjectId === null && isAddingActionObject) {
      // 创建新的 tracklet（临时的，仅用于该片段）
      const tracklet = await video?.createTracklet();
      if (tracklet != null) {
        // 生成随机颜色
        const color = `#${Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0')}`;

        const newObject = {
          id: tracklet.id,
          name: `物体 ${activeActionSegment.objects.length + 1}`,
          color,
          points: (() => {
            const pts: SegmentationPoint[][] = [];
            pts[frameIndex] = [point];
            return pts;
          })(), // 初始化points数组，在当前帧索引设置点
          masks: [],
          isTracking: false,
          trackingProgress: 0,
          thumbnail: null,
        };

        // 添加第一个点
        video?.updatePoints(tracklet.id, [point]);
        setActiveActionObjectId(tracklet.id);

        // 更新片段的物体列表
        setActionSegments(segments =>
          segments.map(seg =>
            seg.id === activeActionSegment.id
              ? {...seg, objects: [...seg.objects, newObject]}
              : seg,
          ),
        );

        enqueueMessage('pointClick');

        // 成功添加物体后，退出添加模式
        setIsAddingActionObject(false);
      }
    }
  }

  // === 动作片段模式：删除物体点 ===
  function handleRemoveActionObjectPoint(point: SegmentationPoint) {
    if (
      isPlaying ||
      streamingState === 'partial' ||
      streamingState === 'requesting'
    ) {
      return;
    }

    // 必须有活跃的片段和选中的物体
    if (!activeActionSegment || activeActionObjectId === null) {
      return;
    }

    const frameIndex = video?.frame ?? 0;
    const existingObject = activeActionSegment.objects.find(
      obj => obj.id === activeActionObjectId,
    );

    if (existingObject) {
      const currentFramePoints = existingObject.points[frameIndex] ?? [];
      const newPoints = currentFramePoints.filter(p => p !== point);

      // 更新视频中的点
      video?.updatePoints(activeActionObjectId, newPoints);

      // 更新状态中的点
      setActionSegments(segments =>
        segments.map(seg =>
          seg.id === activeActionSegment.id
            ? {
                ...seg,
                objects: seg.objects.map(obj =>
                  obj.id === activeActionObjectId
                    ? {
                        ...obj,
                        points: (() => {
                          const updatedPoints = [...obj.points];
                          updatedPoints[frameIndex] = newPoints;
                          return updatedPoints;
                        })(),
                      }
                    : obj,
                ),
              }
            : seg,
        ),
      );
    }
  }

  function handleRemovePoint(point: SegmentationPoint) {
    if (
      isPlaying ||
      streamingState === 'partial' ||
      streamingState === 'requesting'
    ) {
      return;
    }
    handleOptimisticPointUpdate(points.filter(p => p !== point));
  }

  // The interaction layer handles clicks onto the video canvas. It is used
  // to get absolute point clicks within the video's coordinate system.
  // The PointsLayer handles rendering of input points and allows removing
  // individual points by clicking on them.
  const layers = (
    <>
      {/* 物体标注模式 */}
      {annotationMode === 'object' && tabIndex === OBJECT_TOOLBAR_INDEX && (
        <>
          <InteractionLayer
            key="interaction-layer"
            onPoint={point => handleAddPoint(point)}
          />
          <PointsLayer
            key="points-layer"
            points={points}
            onRemovePoint={handleRemovePoint}
          />
        </>
      )}
      {/* 动作标注模式 - 添加物体或编辑选中物体时显示 InteractionLayer */}
      {annotationMode === 'action' &&
        activeActionSegment != null &&
        (isAddingActionObject || activeActionObjectId !== null) && (
          <>
            <InteractionLayer
              key="action-interaction-layer"
              onPoint={point => handleAddPointInActionSegment(point)}
            />
            {/* 显示选中动作物体的点 */}
            {activeActionObjectId !== null && (
              <PointsLayer
                key="action-points-layer"
                points={actionObjectPoints}
                onRemovePoint={handleRemoveActionObjectPoint}
              />
            )}
          </>
        )}
      {!isMobile && <MessagesSnackbar key="snackbar-layer" />}
    </>
  );

  return (
    <>
      {(isVideoLoading || session === null) && !isSessionStartFailed && (
        <div {...stylex.props(styles.loadingScreenWrapper)}>
          <LoadingStateScreen
            title="Loading demo..."
            description="This may take a few moments, you're almost there!"
          />
        </div>
      )}
      {isSessionStartFailed && (
        <div {...stylex.props(styles.loadingScreenWrapper)}>
          <LoadingStateScreen
            title="Did we just break the internet?"
            description={
              <>Uh oh, it looks like there was an issue starting a session.</>
            }
            linkProps={{to: '..', label: 'Back to homepage'}}
          />
        </div>
      )}
      {isMobile && renderingError != null && (
        <div {...stylex.props(styles.loadingScreenWrapper)}>
          <LoadingStateScreen
            title="Well, this is embarrassing..."
            description="This demo is not optimized for your device. Please try again on a different device with a larger screen."
            linkProps={{to: '..', label: 'Back to homepage'}}
          />
        </div>
      )}
      {uploadingState !== 'default' && (
        <div {...stylex.props(styles.loadingScreenWrapper)}>
          <UploadLoadingScreen />
        </div>
      )}
      <div {...stylex.props(styles.container)}>
        <VideoEditor
          video={inputVideo}
          layers={layers}
          loading={session == null}>
          <div className="bg-graydark-800 w-full">
            <VideoFilmstripWithPlayback />
            <TrackletsAnnotation />
            <ActionSegmentsAnnotation />
          </div>
        </VideoEditor>
      </div>
    </>
  );
}
