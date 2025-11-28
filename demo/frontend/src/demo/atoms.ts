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
  defaultMessageMap,
  MessagesEventMap,
} from '@/common/components/snackbar/DemoMessagesSnackbarUtils';
import {Effects} from '@/common/components/video/effects/Effects';
import {
  DemoEffect,
  highlightEffects,
} from '@/common/components/video/effects/EffectUtils';
import {
  BaseTracklet,
  SegmentationPoint,
  StreamingState,
} from '@/common/tracker/Tracker';
import type {DataArray} from '@/jscocotools/mask';
import {atom} from 'jotai';

export type VideoData = {
  path: string;
  posterPath: string | null | undefined;
  url: string;
  posterUrl: string;
  width: number;
  height: number;
};

export const frameIndexAtom = atom<number>(0);

export const inputVideoAtom = atom<VideoData | null>(null);

// #####################
// SESSION
// #####################

export type Session = {
  id: string;
  ranPropagation: boolean;
};

export const sessionAtom = atom<Session | null>(null);

// #####################
// STREAMING/PLAYBACK
// #####################

export const isVideoLoadingAtom = atom<boolean>(false);

export const streamingStateAtom = atom<StreamingState>('none');

export const isPlayingAtom = atom<boolean>(false);

export const isStreamingAtom = atom<boolean>(false);

// #####################
// OBJECTS
// #####################

export type TrackletMask = {
  mask: DataArray;
  isEmpty: boolean;
};

export type TrackletObject = {
  id: number;
  color: string;
  thumbnail: string | null;
  points: SegmentationPoint[][];
  masks: TrackletMask[];
  isInitialized: boolean;
};

const MAX_NUMBER_TRACKLET_OBJECTS = 3;

export const activeTrackletObjectIdAtom = atom<number | null>(0);

export const activeTrackletObjectAtom = atom<BaseTracklet | null>(get => {
  const objectId = get(activeTrackletObjectIdAtom);
  const tracklets = get(trackletObjectsAtom);
  return tracklets.find(obj => obj.id === objectId) ?? null;
});

export const trackletObjectsAtom = atom<BaseTracklet[]>([]);

// 派生 atom：过滤掉属于动作片段的物体，只返回全局物体
export const globalTrackletObjectsAtom = atom<BaseTracklet[]>(get => {
  const allTracklets = get(trackletObjectsAtom);
  const actionSegments = get(actionSegmentsAtom);

  // 收集所有动作片段内的物体 ID
  const actionObjectIds = new Set(
    actionSegments.flatMap(seg => seg.objects.map(obj => obj.id)),
  );

  // 只保留不属于任何动作片段的物体
  return allTracklets.filter(t => !actionObjectIds.has(t.id));
});

// Custom names for tracklets (tracklet ID -> custom name)
export const trackletNamesAtom = atom<Record<number, string>>({});

export const maxTrackletObjectIdAtom = atom<number>(get => {
  const tracklets = get(globalTrackletObjectsAtom);
  return tracklets.reduce((prev, curr) => Math.max(prev, curr.id), 0);
});

export const isTrackletObjectLimitReachedAtom = atom<boolean>(
  get => get(globalTrackletObjectsAtom).length >= MAX_NUMBER_TRACKLET_OBJECTS,
);

export const areTrackletObjectsInitializedAtom = atom<boolean>(get =>
  get(globalTrackletObjectsAtom).every(obj => obj.isInitialized),
);

export const isFirstClickMadeAtom = atom(get => {
  const tracklets = get(globalTrackletObjectsAtom);
  return tracklets.some(tracklet => tracklet.points.length > 0);
});

export const pointsAtom = atom<SegmentationPoint[]>(get => {
  const frameIndex = get(frameIndexAtom);
  const activeTracklet = get(activeTrackletObjectAtom);
  return activeTracklet?.points[frameIndex] ?? [];
});

export const labelTypeAtom = atom<'positive' | 'negative'>('positive');

export const isAddObjectEnabledAtom = atom<boolean>(get => {
  const session = get(sessionAtom);
  const trackletsInitialized = get(areTrackletObjectsInitializedAtom);
  const isObjectLimitReached = get(isTrackletObjectLimitReachedAtom);
  return (
    session?.ranPropagation === false &&
    trackletsInitialized &&
    !isObjectLimitReached
  );
});

export const codeEditorOpenedAtom = atom<boolean>(false);

export const tutorialVideoEnabledAtom = atom<boolean>(true);

// #####################
// Effects
// #####################

type EffectConfig = {
  name: keyof Effects;
  variant: number;
  numVariants: number;
};

export const activeBackgroundEffectAtom = atom<EffectConfig>({
  name: 'Original',
  variant: 0,
  numVariants: 0,
});

export const activeHighlightEffectAtom = atom<EffectConfig>({
  name: 'Overlay',
  variant: 0,
  numVariants: 0,
});

export const activeHighlightEffectGroupAtom =
  atom<DemoEffect[]>(highlightEffects);

// #####################
// Toolbar
// #####################

export const toolbarTabIndex = atom<number>(0);

// #####################
// Messages snackbar
// #####################

export const messageMapAtom = atom<MessagesEventMap>(defaultMessageMap);

// #####################
// Upload state
// #####################

export const uploadingStateAtom = atom<'default' | 'uploading' | 'error'>(
  'default',
);

// #####################
// ACTION SEGMENTS (动作片段)
// #####################

export type ActionSegmentObject = {
  id: number;
  name: string; // 物体名称，如"剪刀"
  color: string;
  points: SegmentationPoint[][]; // 仅在片段时间范围内的帧
  masks: TrackletMask[];
  isTracking: boolean; // 是否正在追踪
  trackingProgress: number; // 追踪进度 0-1
  thumbnail: string | null; // 物体缩略图
};

export type ActionSegment = {
  id: string; // UUID
  name: string; // 动作名称，如"剪断胆囊管"
  frameStart: number; // 起始帧索引
  frameEnd: number; // 结束帧索引
  objects: ActionSegmentObject[]; // 该片段内标注的物体
  createdAt: number; // 创建时间戳
};

// 所有动作片段
export const actionSegmentsAtom = atom<ActionSegment[]>([]);

// 当前选中/正在编辑的动作片段ID
export const activeActionSegmentIdAtom = atom<string | null>(null);

// 当前活动的动作片段（派生atom）
export const activeActionSegmentAtom = atom<ActionSegment | null>(get => {
  const segmentId = get(activeActionSegmentIdAtom);
  const segments = get(actionSegmentsAtom);
  return segments.find(seg => seg.id === segmentId) ?? null;
});

// 标注模式：'object' = 全局物体标注，'action' = 动作片段标注
export const annotationModeAtom = atom<'object' | 'action'>('object');

// 动作片段内当前选中的物体ID
export const activeActionObjectIdAtom = atom<number | null>(null);

// 是否正在添加动作片段内的物体（控制InteractionLayer的显示）
export const isAddingActionObjectAtom = atom<boolean>(false);

// 是否正在选择时间段（拖动时间轴状态）
export const isSelectingTimeRangeAtom = atom<boolean>(false);

// 临时时间段选择（拖动过程中）
export const tempTimeRangeAtom = atom<{start: number; end: number} | null>(
  null,
);

// 当前选中的动作片段物体ID
export const activeActionSegmentObjectIdAtom = atom<number | null>(null);

// 当前活动的动作片段物体（派生atom）
export const activeActionSegmentObjectAtom = atom<ActionSegmentObject | null>(
  get => {
    const objectId = get(activeActionSegmentObjectIdAtom);
    const activeSegment = get(activeActionSegmentAtom);
    return (
      activeSegment?.objects.find(obj => obj.id === objectId) ?? null
    );
  },
);

// 派生 atom：获取当前片段内的所有物体
export const currentSegmentObjectsAtom = atom<ActionSegmentObject[]>(get => {
  const activeSegment = get(activeActionSegmentAtom);
  return activeSegment?.objects ?? [];
});

// 派生 atom：获取当前选中动作物体在当前帧的points
export const activeActionObjectPointsAtom = atom<SegmentationPoint[]>(get => {
  const activeObject = get(activeActionSegmentObjectAtom);
  const frameIndex = get(frameIndexAtom);
  if (!activeObject) {
    return [];
  }
  return activeObject.points[frameIndex] ?? [];
});

// 动作片段导出配置
export type ActionExportConfig = {
  segmentIds: string[]; // 要导出的片段ID列表，空数组表示导出全部
  targetFps: number;
  includeVisualizations: boolean;
};

export const actionExportConfigAtom = atom<ActionExportConfig>({
  segmentIds: [],
  targetFps: 30,
  includeVisualizations: true,
});
