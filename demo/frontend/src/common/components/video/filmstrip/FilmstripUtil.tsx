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
import {ActionSegment} from '@/demo/atoms';
import {CanvasForm, CanvasSpace, Font, Group, Pt, Triangle} from 'pts';
import SelectedFrameHelper from './SelectedFrameHelper';
import {PADDING_BOTTOM, PADDING_TOP} from './VideoFilmstrip';

export function getPointerPosition(
  event: React.PointerEvent<HTMLCanvasElement>,
) {
  const rect = event.currentTarget.getBoundingClientRect();
  return new Pt(event.clientX - rect.left, event.clientY - rect.top);
}

export function drawFilmstrip(
  filmstrip: ImageBitmap | null,
  space: CanvasSpace | undefined,
  form: CanvasForm | undefined,
) {
  if (filmstrip == null || space == undefined || form?.ctx == undefined) {
    return;
  }

  const ratio =
    filmstrip.width / (filmstrip.height + PADDING_TOP + PADDING_BOTTOM);

  form.image(
    [
      [0, PADDING_TOP],
      [space.size.x, space.size.x / ratio],
    ],
    filmstrip,
  );
}

export function getTimeFromFrame(frame: number, fps: number): string {
  const seconds = Math.floor(frame / fps);
  const frameRemaining = frame - fps * seconds;
  return `${seconds}:${frameRemaining.toFixed().toString().padStart(2, '0')}`;
}

export function drawMarker(
  space: CanvasSpace | undefined,
  form: CanvasForm | undefined,
  selectedFrameHelper: SelectedFrameHelper,
  pointerPosition: Pt | null,
  scanLabel: string | false,
  fps: number,
) {
  if (space == undefined || form?.ctx == undefined) {
    return;
  }

  const marker = Group.fromArray([
    [0, PADDING_TOP],
    [0, space.height - PADDING_BOTTOM],
  ]);

  const currentMarker = marker
    .clone()
    .add(Math.max(5, selectedFrameHelper.position), 0);

  const getTextPosition = (label: string, marker: Group) => {
    const textWidth = form.ctx.measureText(label).width;
    return marker[0]
      .$subtract(textWidth / 2, 0)
      .$min(space.width - textWidth, PADDING_TOP - 10)
      .$max(textWidth / 2 - 2, 0);
  };

  // draw current marker
  form
    .strokeOnly('#00000066', 5)
    .line(currentMarker)
    .strokeOnly('#fff', 1)
    .line(currentMarker)
    .fill('#000')
    .polygon(
      Triangle.fromCenter(currentMarker[0].$add(0, 10), 5).rotate2D(Math.PI),
    );

  // draw text
  const frameLabel = getTimeFromFrame(selectedFrameHelper.index, fps);
  form
    .font(new Font(12, 'monospace'))
    .fillOnly('#fff')
    .text(getTextPosition(frameLabel, currentMarker), frameLabel);

  // draw scanning ghost marker
  if (
    selectedFrameHelper.isScanning &&
    pointerPosition != null &&
    scanLabel != false
  ) {
    const scanMarker = marker.clone().add(pointerPosition.x, 0);
    form.strokeOnly('#ffffff66', 5).line(scanMarker);

    form
      .font(new Font(12, 'monospace'))
      .fillOnly('#8595A4')
      .text(getTextPosition(scanLabel, scanMarker), scanLabel);
  }
}

/**
 * 绘制动作片段高亮区域（iMovie 风格）
 */
export function drawActionSegments(
  space: CanvasSpace | undefined,
  form: CanvasForm | undefined,
  actionSegments: ActionSegment[],
  activeSegmentId: string | null,
  tempTimeRange: {start: number; end: number} | null,
  totalFrames: number,
) {
  if (space == undefined || form?.ctx == undefined || totalFrames === 0) {
    return;
  }

  const frameToX = (frame: number): number => {
    return (frame / totalFrames) * space.width;
  };

  const handleWidth = 4; // 手柄宽度
  const handleExtension = 8; // 手柄伸出的长度
  const cornerRadius = 3; // 圆角半径

  /**
   * 绘制花括号样式的手柄
   */
  const drawBracketHandle = (
    x: number,
    isLeft: boolean,
    color: string,
    isActive: boolean,
  ) => {
    const top = PADDING_TOP;
    const bottom = space.height - PADDING_BOTTOM;
    const height = bottom - top;
    const direction = isLeft ? -1 : 1;

    const ctx = form.ctx;
    if (!ctx) return;

    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = isActive ? 3 : 2;

    // 绘制垂直线
    ctx.fillRect(x - handleWidth / 2, top, handleWidth, height);

    // 绘制顶部和底部的弯曲
    ctx.beginPath();
    // 顶部弯曲
    ctx.moveTo(x, top);
    ctx.lineTo(x + direction * handleExtension, top);
    ctx.lineTo(x + direction * handleExtension, top + cornerRadius * 2);
    ctx.quadraticCurveTo(
      x + direction * handleExtension,
      top + cornerRadius,
      x + direction * (handleExtension - cornerRadius),
      top + cornerRadius,
    );

    // 底部弯曲
    ctx.moveTo(x, bottom);
    ctx.lineTo(x + direction * handleExtension, bottom);
    ctx.lineTo(x + direction * handleExtension, bottom - cornerRadius * 2);
    ctx.quadraticCurveTo(
      x + direction * handleExtension,
      bottom - cornerRadius,
      x + direction * (handleExtension - cornerRadius),
      bottom - cornerRadius,
    );

    ctx.stroke();
    ctx.restore();
  };

  /**
   * 绘制顶部标签背景和边框
   */
  const drawSegmentTopBar = (
    startX: number,
    endX: number,
    color: string,
    isActive: boolean,
  ) => {
    const ctx = form.ctx;
    if (!ctx) return;

    const topBarHeight = PADDING_TOP - 5;
    const width = endX - startX;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = isActive ? 2 : 1;

    // 顶部细线连接两个手柄
    ctx.beginPath();
    ctx.moveTo(startX, PADDING_TOP);
    ctx.lineTo(endX, PADDING_TOP);
    ctx.stroke();

    // 可选：绘制顶部标签区域的细边框
    if (topBarHeight > 0) {
      ctx.strokeStyle = color + '40'; // 半透明
      ctx.strokeRect(startX, 0, width, topBarHeight);
    }

    ctx.restore();
  };

  // 绘制临时选择区域
  if (tempTimeRange != null) {
    const startX = frameToX(tempTimeRange.start);
    const endX = frameToX(tempTimeRange.end);
    const color = '#3B82F6'; // blue-500

    drawBracketHandle(startX, true, color, false);
    drawBracketHandle(endX, false, color, false);
    drawSegmentTopBar(startX, endX, color, false);

    // 轻微的背景高亮（可选，非常淡）
    form
      .fillOnly('rgba(59, 130, 246, 0.08)')
      .rect([[startX, PADDING_TOP], [endX - startX, space.height - PADDING_TOP - PADDING_BOTTOM]]);
  }

  // 绘制已创建的时间段
  actionSegments.forEach(segment => {
    const startX = frameToX(segment.frameStart);
    const endX = frameToX(segment.frameEnd);
    const isActive = segment.id === activeSegmentId;
    const color = isActive ? '#22C55E' : '#3B82F6'; // green-500 : blue-500

    drawBracketHandle(startX, true, color, isActive);
    drawBracketHandle(endX, false, color, isActive);
    drawSegmentTopBar(startX, endX, color, isActive);

    // 非常淡的背景高亮
    form
      .fillOnly(isActive ? 'rgba(34, 197, 94, 0.1)' : 'rgba(59, 130, 246, 0.06)')
      .rect([[startX, PADDING_TOP], [endX - startX, space.height - PADDING_TOP - PADDING_BOTTOM]]);
  });
}
