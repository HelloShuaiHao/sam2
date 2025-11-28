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
import PrimaryCTAButton from '@/common/components/button/PrimaryCTAButton';
import useVideo from '@/common/components/video/editor/useVideo';
import {actionSegmentsAtom, sessionAtom} from '@/demo/atoms';
import {useAtomValue, useSetAtom} from 'jotai';
import {useCallback, useState} from 'react';

export default function TrackActionSegmentsButton() {
  const video = useVideo();
  const actionSegments = useAtomValue(actionSegmentsAtom);
  const [isTracking, setIsTracking] = useState(false);
  const [trackingProgress, setTrackingProgress] = useState(0);
  const setActionSegments = useSetAtom(actionSegmentsAtom);
  const setSession = useSetAtom(sessionAtom);

  // 计算有标注点的物体总数
  const totalObjectsToTrack = actionSegments.reduce((total, segment) => {
    const objectsWithPoints = segment.objects.filter(obj =>
      obj.points.some(framePoints => framePoints && framePoints.length > 0),
    );
    return total + objectsWithPoints.length;
  }, 0);

  const handleTrackAllSegments = useCallback(async () => {
    if (!video || isTracking) {
      return;
    }

    if (totalObjectsToTrack === 0) {
      alert('请先为动作片段中的物体添加标注点');
      return;
    }

    setIsTracking(true);
    setTrackingProgress(0);

    try {
      let completedObjects = 0;

      // 遍历所有动作片段
      for (const segment of actionSegments) {
        console.log(`[TrackActionSegments] 开始追踪片段: ${segment.name}`);

        // 遍历片段中的所有物体
        for (const obj of segment.objects) {
          // 检查物体是否有标注点
          const hasPoints = obj.points.some(
            framePoints => framePoints && framePoints.length > 0,
          );

          if (!hasPoints) {
            console.warn(
              `[TrackActionSegments] 物体 ${obj.name} 没有标注点，跳过`,
            );
            continue;
          }

          // 更新物体状态为正在追踪
          setActionSegments(segments =>
            segments.map(seg =>
              seg.id === segment.id
                ? {
                    ...seg,
                    objects: seg.objects.map(o =>
                      o.id === obj.id
                        ? {...o, isTracking: true, trackingProgress: 0}
                        : o,
                    ),
                  }
                : seg,
            ),
          );

          console.log(
            `[TrackActionSegments] 追踪物体: ${obj.name} (ID: ${obj.id}) 在片段 [${segment.frameStart}-${segment.frameEnd}]`,
          );

          try {
            // 调用片段范围内的追踪
            await video.trackObjectInSegment(
              obj.id,
              segment.id,
              segment.frameStart,
              segment.frameEnd,
            );

            // 追踪完成，更新进度
            completedObjects++;
            setTrackingProgress(completedObjects / totalObjectsToTrack);

            // 更新物体状态
            setActionSegments(segments =>
              segments.map(seg =>
                seg.id === segment.id
                  ? {
                      ...seg,
                      objects: seg.objects.map(o =>
                        o.id === obj.id
                          ? {...o, isTracking: false, trackingProgress: 1}
                          : o,
                      ),
                    }
                  : seg,
              ),
            );

            console.log(
              `[TrackActionSegments] 完成追踪物体: ${obj.name} (${completedObjects}/${totalObjectsToTrack})`,
            );
          } catch (error) {
            console.error(
              `[TrackActionSegments] 追踪物体 ${obj.name} 失败:`,
              error,
            );

            // 追踪失败，重置状态
            setActionSegments(segments =>
              segments.map(seg =>
                seg.id === segment.id
                  ? {
                      ...seg,
                      objects: seg.objects.map(o =>
                        o.id === obj.id
                          ? {...o, isTracking: false, trackingProgress: 0}
                          : o,
                      ),
                    }
                  : seg,
              ),
            );
          }
        }
      }

      console.log(
        `[TrackActionSegments] 全部追踪完成: ${completedObjects}/${totalObjectsToTrack} 个物体`,
      );

      // 标记session已运行过追踪
      setSession(previousSession =>
        previousSession == null
          ? previousSession
          : {...previousSession, ranPropagation: true},
      );
    } finally {
      setIsTracking(false);
      setTrackingProgress(0);
    }
  }, [
    video,
    isTracking,
    actionSegments,
    totalObjectsToTrack,
    setActionSegments,
    setSession,
  ]);

  // 如果没有片段或没有物体，禁用按钮
  const isDisabled = actionSegments.length === 0 || totalObjectsToTrack === 0;

  return (
    <PrimaryCTAButton
      disabled={isDisabled}
      onClick={handleTrackAllSegments}>
      {isTracking
        ? `取消追踪`
        : `开始追踪`}
    </PrimaryCTAButton>
  );
}
