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

import {
  actionSegmentsAtom,
  activeActionSegmentIdAtom,
  annotationModeAtom,
  isSelectingTimeRangeAtom,
  tempTimeRangeAtom,
} from '@/demo/atoms';
import stylex from '@stylexjs/stylex';
import {useAtom, useAtomValue, useSetAtom} from 'jotai';
import {useCallback, useRef, useState} from 'react';
import useVideo from '../editor/useVideo';

const styles = stylex.create({
  container: {
    position: 'relative',
    width: '100%',
    height: '40px',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: '4px',
    cursor: 'crosshair',
    userSelect: 'none',
  },
  segmentBar: {
    position: 'absolute',
    height: '100%',
    backgroundColor: 'rgba(59, 130, 246, 0.4)', // blue-500 with opacity
    borderLeft: '2px solid rgb(59, 130, 246)',
    borderRight: '2px solid rgb(59, 130, 246)',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
    ':hover': {
      backgroundColor: 'rgba(59, 130, 246, 0.6)',
    },
  },
  segmentBarActive: {
    backgroundColor: 'rgba(34, 197, 94, 0.5)', // green-500
    borderColor: 'rgb(34, 197, 94)',
    ':hover': {
      backgroundColor: 'rgba(34, 197, 94, 0.7)',
    },
  },
  tempSelection: {
    position: 'absolute',
    height: '100%',
    backgroundColor: 'rgba(59, 130, 246, 0.3)',
    border: '1px dashed rgb(59, 130, 246)',
    pointerEvents: 'none',
  },
  handle: {
    position: 'absolute',
    width: '8px',
    height: '100%',
    cursor: 'ew-resize',
    ':hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.3)',
    },
  },
  handleLeft: {
    left: '-4px',
  },
  handleRight: {
    right: '-4px',
  },
  label: {
    position: 'absolute',
    top: '-20px',
    left: '0',
    fontSize: '11px',
    color: '#fff',
    whiteSpace: 'nowrap',
    pointerEvents: 'none',
  },
});

type DragState = {
  type: 'create' | 'move' | 'resize-left' | 'resize-right';
  segmentId?: string;
  startX: number;
  startFrameStart?: number;
  startFrameEnd?: number;
};

export default function ActionSegmentTimeline() {
  const video = useVideo();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);

  const annotationMode = useAtomValue(annotationModeAtom);
  const [actionSegments, setActionSegments] = useAtom(actionSegmentsAtom);
  const [activeSegmentId, setActiveSegmentId] = useAtom(
    activeActionSegmentIdAtom,
  );
  const setTempTimeRange = useSetAtom(tempTimeRangeAtom);
  const setIsSelecting = useSetAtom(isSelectingTimeRangeAtom);

  // 像素位置转帧索引
  const pixelToFrame = useCallback(
    (x: number): number => {
      if (!containerRef.current || !video) return 0;
      const rect = containerRef.current.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, x / rect.width));
      return Math.floor(ratio * video.numberOfFrames);
    },
    [video],
  );

  // 帧索引转像素位置
  const frameToPixel = useCallback(
    (frame: number): number => {
      if (!containerRef.current || !video) return 0;
      const rect = containerRef.current.getBoundingClientRect();
      return (frame / video.numberOfFrames) * rect.width;
    },
    [video],
  );

  // 处理鼠标按下事件
  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (annotationMode !== 'action' || !containerRef.current) return;

      console.log('[ActionSegmentTimeline] 鼠标按下, video:', video, 'numberOfFrames:', video?.numberOfFrames);

      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const frame = pixelToFrame(x);

      console.log('[ActionSegmentTimeline] 起始位置:', {x, frame, rectWidth: rect.width});

      // 创建新的时间段
      setDragState({
        type: 'create',
        startX: x,
      });
      setTempTimeRange({start: frame, end: frame});
      setIsSelecting(true);

      // 设置指针捕获
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    [annotationMode, pixelToFrame, setTempTimeRange, setIsSelecting, video],
  );

  // 处理鼠标移动事件
  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragState || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const currentFrame = pixelToFrame(x);

      if (dragState.type === 'create') {
        const startFrame = pixelToFrame(dragState.startX);
        setTempTimeRange({
          start: Math.min(startFrame, currentFrame),
          end: Math.max(startFrame, currentFrame),
        });
      } else if (dragState.type === 'move' && dragState.segmentId) {
        // 移动整个片段
        const segment = actionSegments.find(s => s.id === dragState.segmentId);
        if (segment && dragState.startFrameStart !== undefined) {
          const deltaFrame = currentFrame - pixelToFrame(dragState.startX);
          const duration = segment.frameEnd - segment.frameStart;
          let newStart = dragState.startFrameStart + deltaFrame;
          let newEnd = newStart + duration;

          // 边界检查
          if (newStart < 0) {
            newStart = 0;
            newEnd = duration;
          }
          if (video && newEnd >= video.numberOfFrames) {
            newEnd = video.numberOfFrames - 1;
            newStart = newEnd - duration;
          }

          setActionSegments(segments =>
            segments.map(s =>
              s.id === dragState.segmentId
                ? {...s, frameStart: newStart, frameEnd: newEnd}
                : s,
            ),
          );
        }
      } else if (
        dragState.type === 'resize-left' &&
        dragState.segmentId &&
        dragState.startFrameEnd !== undefined
      ) {
        // 调整左边界
        const newStart = Math.min(currentFrame, dragState.startFrameEnd - 1);
        setActionSegments(segments =>
          segments.map(s =>
            s.id === dragState.segmentId ? {...s, frameStart: newStart} : s,
          ),
        );
      } else if (
        dragState.type === 'resize-right' &&
        dragState.segmentId &&
        dragState.startFrameStart !== undefined
      ) {
        // 调整右边界
        const newEnd = Math.max(currentFrame, dragState.startFrameStart + 1);
        setActionSegments(segments =>
          segments.map(s =>
            s.id === dragState.segmentId ? {...s, frameEnd: newEnd} : s,
          ),
        );
      }
    },
    [
      dragState,
      pixelToFrame,
      setTempTimeRange,
      actionSegments,
      setActionSegments,
      video,
    ],
  );

  // 处理鼠标释放事件
  const handlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragState) return;

      console.log('[ActionSegmentTimeline] 鼠标释放, dragState:', dragState);

      if (dragState.type === 'create' && containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const endFrame = pixelToFrame(x);
        const startFrame = pixelToFrame(dragState.startX);

        console.log('[ActionSegmentTimeline] 结束位置:', {
          x,
          endFrame,
          startFrame,
          diff: Math.abs(endFrame - startFrame),
        });

        // 只有当选择的区间大于等于1帧时才创建
        if (Math.abs(endFrame - startFrame) >= 1) {
          const newSegment = {
            id: `segment-${Date.now()}`,
            name: '未命名动作',
            frameStart: Math.min(startFrame, endFrame),
            frameEnd: Math.max(startFrame, endFrame),
            objects: [],
            createdAt: Date.now(),
          };
          console.log('[ActionSegmentTimeline] 创建新片段:', newSegment);
          setActionSegments([...actionSegments, newSegment]);
          setActiveSegmentId(newSegment.id);
        } else {
          console.log('[ActionSegmentTimeline] 区间太小，不创建片段');
        }

        setTempTimeRange(null);
        setIsSelecting(false);
      }

      setDragState(null);
      (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    },
    [
      dragState,
      pixelToFrame,
      actionSegments,
      setActionSegments,
      setActiveSegmentId,
      setTempTimeRange,
      setIsSelecting,
    ],
  );

  // 处理点击片段
  const handleSegmentClick = useCallback(
    (segmentId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setActiveSegmentId(segmentId);
    },
    [setActiveSegmentId],
  );

  // 处理拖动片段
  const handleSegmentPointerDown = useCallback(
    (segmentId: string, e: React.PointerEvent) => {
      e.stopPropagation();
      const segment = actionSegments.find(s => s.id === segmentId);
      if (!segment || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;

      setDragState({
        type: 'move',
        segmentId,
        startX: x,
        startFrameStart: segment.frameStart,
        startFrameEnd: segment.frameEnd,
      });

      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    [actionSegments],
  );

  // 处理调整边界手柄
  const handleResizePointerDown = useCallback(
    (segmentId: string, side: 'left' | 'right', e: React.PointerEvent) => {
      e.stopPropagation();
      const segment = actionSegments.find(s => s.id === segmentId);
      if (!segment || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;

      setDragState({
        type: side === 'left' ? 'resize-left' : 'resize-right',
        segmentId,
        startX: x,
        startFrameStart: segment.frameStart,
        startFrameEnd: segment.frameEnd,
      });

      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    [actionSegments],
  );

  // 计算时间显示（秒）
  const getTimeLabel = useCallback(
    (frameStart: number, frameEnd: number): string => {
      if (!video) return '';
      const fps = video.fps || 30;
      const startSec = (frameStart / fps).toFixed(1);
      const endSec = (frameEnd / fps).toFixed(1);
      const duration = ((frameEnd - frameStart) / fps).toFixed(1);
      return `${startSec}s - ${endSec}s (${duration}s)`;
    },
    [video],
  );

  // 只在动作模式下显示
  if (annotationMode !== 'action') {
    return null;
  }

  console.log('[ActionSegmentTimeline] 组件正在渲染');
  console.log('[ActionSegmentTimeline] - video 对象:', video);
  console.log('[ActionSegmentTimeline] - video.numberOfFrames:', video?.numberOfFrames);
  console.log('[ActionSegmentTimeline] - annotationMode:', annotationMode);
  console.log('[ActionSegmentTimeline] - actionSegments.length:', actionSegments.length);

  return (
    <div
      ref={containerRef}
      {...stylex.props(styles.container)}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onClick={() => console.log('[ActionSegmentTimeline] div 被点击')}
      onMouseDown={() => console.log('[ActionSegmentTimeline] onMouseDown 触发')}
      onMouseUp={() => console.log('[ActionSegmentTimeline] onMouseUp 触发')}
      style={{
        // 临时添加明显的边框用于调试
        border: '3px solid red',
        backgroundColor: 'rgba(255, 0, 0, 0.2)',
        minHeight: '60px', // 确保有足够高度
        zIndex: 999, // 确保在最上层
        position: 'relative',
      }}>
      <div style={{
        color: 'yellow',
        fontSize: '14px',
        padding: '5px',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        pointerEvents: 'none', // 确保这个文字不会阻止事件
      }}>
        🎬 动作片段时间轴 - 拖动创建时间段 (video.numberOfFrames: {video?.numberOfFrames})
      </div>
      {/* 渲染已存在的片段 */}
      {actionSegments.map(segment => {
        const left = frameToPixel(segment.frameStart);
        const width = frameToPixel(segment.frameEnd - segment.frameStart);
        const isActive = segment.id === activeSegmentId;

        return (
          <div
            key={segment.id}
            {...stylex.props(
              styles.segmentBar,
              isActive && styles.segmentBarActive,
            )}
            style={{
              left: `${left}px`,
              width: `${width}px`,
            }}
            onClick={e => handleSegmentClick(segment.id, e)}
            onPointerDown={e => handleSegmentPointerDown(segment.id, e)}>
            {/* 时间标签 */}
            <div {...stylex.props(styles.label)}>
              {segment.name} {getTimeLabel(segment.frameStart, segment.frameEnd)}
            </div>

            {/* 左侧调整手柄 */}
            <div
              {...stylex.props(styles.handle, styles.handleLeft)}
              onPointerDown={e => handleResizePointerDown(segment.id, 'left', e)}
            />

            {/* 右侧调整手柄 */}
            <div
              {...stylex.props(styles.handle, styles.handleRight)}
              onPointerDown={e =>
                handleResizePointerDown(segment.id, 'right', e)
              }
            />
          </div>
        );
      })}

      {/* 渲染临时选择区域 */}
      {dragState?.type === 'create' && containerRef.current && (
        <div
          {...stylex.props(styles.tempSelection)}
          style={{
            left: `${Math.min(dragState.startX, frameToPixel(pixelToFrame(dragState.startX)))}px`,
            width: `${Math.abs(frameToPixel(pixelToFrame(dragState.startX)) - dragState.startX)}px`,
          }}
        />
      )}
    </div>
  );
}
