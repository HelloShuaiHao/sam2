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
  ActionSegment,
  actionSegmentsAtom,
  activeActionSegmentIdAtom,
  annotationModeAtom,
  frameIndexAtom,
  isAddingActionObjectAtom,
  activeActionObjectIdAtom,
} from '@/demo/atoms';
import stylex from '@stylexjs/stylex';
import {useAtom, useAtomValue, useSetAtom} from 'jotai';
import {useCallback, useState} from 'react';
import useVideo from '@/common/components/video/editor/useVideo';

const styles = stylex.create({
  panel: {
    width: '280px',
    height: '100%',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    borderRadius: '8px',
    padding: '16px',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    overflowY: 'auto',
  },
  header: {
    fontSize: '16px',
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: '8px',
  },
  segmentList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  segmentItem: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '6px',
    padding: '12px',
    cursor: 'pointer',
    border: '1px solid transparent',
    transition: 'all 0.2s',
    ':hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.1)',
    },
  },
  segmentItemActive: {
    backgroundColor: 'rgba(34, 197, 94, 0.2)',
    borderColor: 'rgb(34, 197, 94)',
  },
  segmentHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  segmentName: {
    fontSize: '14px',
    fontWeight: '600',
    color: '#fff',
    flex: 1,
  },
  segmentTime: {
    fontSize: '11px',
    color: 'rgba(255, 255, 255, 0.6)',
    marginBottom: '8px',
  },
  objectList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    marginTop: '8px',
    paddingLeft: '12px',
  },
  objectItem: {
    fontSize: '12px',
    color: 'rgba(255, 255, 255, 0.8)',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  objectColor: {
    width: '12px',
    height: '12px',
    borderRadius: '2px',
  },
  buttonGroup: {
    display: 'flex',
    gap: '4px',
  },
  button: {
    padding: '4px 8px',
    fontSize: '11px',
    backgroundColor: 'rgba(59, 130, 246, 0.6)',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    ':hover': {
      backgroundColor: 'rgba(59, 130, 246, 0.8)',
    },
  },
  buttonDanger: {
    backgroundColor: 'rgba(239, 68, 68, 0.6)',
    ':hover': {
      backgroundColor: 'rgba(239, 68, 68, 0.8)',
    },
  },
  addButton: {
    width: '100%',
    padding: '10px',
    backgroundColor: 'rgba(34, 197, 94, 0.6)',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '600',
    ':hover': {
      backgroundColor: 'rgba(34, 197, 94, 0.8)',
    },
  },
  input: {
    width: '100%',
    padding: '6px 8px',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    borderRadius: '4px',
    color: '#fff',
    fontSize: '13px',
    '::placeholder': {
      color: 'rgba(255, 255, 255, 0.4)',
    },
  },
  emptyState: {
    textAlign: 'center',
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: '13px',
    padding: '20px',
  },
  addObjectButton: {
    padding: '6px 12px',
    fontSize: '12px',
    backgroundColor: 'rgba(59, 130, 246, 0.4)',
    color: '#fff',
    border: '1px dashed rgba(59, 130, 246, 0.6)',
    borderRadius: '4px',
    cursor: 'pointer',
    marginTop: '8px',
    width: '100%',
    ':hover': {
      backgroundColor: 'rgba(59, 130, 246, 0.6)',
    },
  },
});

export default function ActionSegmentPanel() {
  const video = useVideo();
  const annotationMode = useAtomValue(annotationModeAtom);
  const [actionSegments, setActionSegments] = useAtom(actionSegmentsAtom);
  const [activeSegmentId, setActiveSegmentId] = useAtom(
    activeActionSegmentIdAtom,
  );
  const setFrameIndex = useSetAtom(frameIndexAtom);
  const [isAddingActionObject, setIsAddingActionObject] = useAtom(
    isAddingActionObjectAtom,
  );
  const setActiveActionObjectId = useSetAtom(activeActionObjectIdAtom);

  const [editingSegmentId, setEditingSegmentId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState<string>('');

  // 格式化时间显示 - 所有 Hooks 必须在条件 return 之前调用
  const formatTime = useCallback(
    (frame: number): string => {
      if (!video) return '0.0s';
      const fps = video.fps || 30;
      return `${(frame / fps).toFixed(1)}s`;
    },
    [video],
  );

  // 计算时长
  const formatDuration = useCallback(
    (frameStart: number, frameEnd: number): string => {
      if (!video) return '';
      const fps = video.fps || 30;
      const duration = (frameEnd - frameStart) / fps;
      return `${formatTime(frameStart)} - ${formatTime(frameEnd)} (${duration.toFixed(1)}s)`;
    },
    [video, formatTime],
  );

  // 选中片段
  const handleSelectSegment = useCallback(
    (segmentId: string) => {
      setActiveSegmentId(segmentId);
      const segment = actionSegments.find(s => s.id === segmentId);
      if (segment && video) {
        // 跳转到片段起始帧
        video.frame = segment.frameStart;
        setFrameIndex(segment.frameStart);
      }
    },
    [actionSegments, setActiveSegmentId, setFrameIndex, video],
  );

  // 删除片段
  const handleDeleteSegment = useCallback(
    (segmentId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      if (confirm('确定要删除这个动作片段吗？')) {
        setActionSegments(segments => segments.filter(s => s.id !== segmentId));
        if (activeSegmentId === segmentId) {
          setActiveSegmentId(null);
        }
      }
    },
    [setActionSegments, activeSegmentId, setActiveSegmentId],
  );

  // 开始编辑名称
  const handleStartEdit = useCallback(
    (segment: ActionSegment, e: React.MouseEvent) => {
      e.stopPropagation();
      setEditingSegmentId(segment.id);
      setEditingName(segment.name);
    },
    [],
  );

  // 保存名称
  const handleSaveName = useCallback(
    (segmentId: string) => {
      if (editingName.trim()) {
        setActionSegments(segments =>
          segments.map(s =>
            s.id === segmentId ? {...s, name: editingName.trim()} : s,
          ),
        );
      }
      setEditingSegmentId(null);
      setEditingName('');
    },
    [editingName, setActionSegments],
  );

  // 取消编辑
  const handleCancelEdit = useCallback(() => {
    setEditingSegmentId(null);
    setEditingName('');
  }, []);

  // 处理回车键保存
  const handleKeyDown = useCallback(
    (segmentId: string, e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        handleSaveName(segmentId);
      } else if (e.key === 'Escape') {
        handleCancelEdit();
      }
    },
    [handleSaveName, handleCancelEdit],
  );

  // 开始添加物体
  const handleStartAddObject = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      // 清除当前选中的物体ID，确保点击时创建新物体
      setActiveActionObjectId(null);
      setIsAddingActionObject(true);
    },
    [setActiveActionObjectId, setIsAddingActionObject],
  );

  // 取消添加物体
  const handleCancelAddObject = useCallback(() => {
    setIsAddingActionObject(false);
    setActiveActionObjectId(null);
  }, [setIsAddingActionObject, setActiveActionObjectId]);

  // 只在动作模式下显示 - 条件判断必须在所有 Hooks 之后
  if (annotationMode !== 'action') {
    return null;
  }

  return (
    <div {...stylex.props(styles.panel)}>
      <div {...stylex.props(styles.header)}>动作片段</div>

      {actionSegments.length === 0 ? (
        <div {...stylex.props(styles.emptyState)}>
          在时间轴上拖动鼠标选择时间段
          <br />
          开始创建动作片段
        </div>
      ) : (
        <div {...stylex.props(styles.segmentList)}>
          {actionSegments.map(segment => {
            const isActive = segment.id === activeSegmentId;
            const isEditing = editingSegmentId === segment.id;

            return (
              <div
                key={segment.id}
                {...stylex.props(
                  styles.segmentItem,
                  isActive && styles.segmentItemActive,
                )}
                onClick={() => handleSelectSegment(segment.id)}>
                <div {...stylex.props(styles.segmentHeader)}>
                  {isEditing ? (
                    <input
                      {...stylex.props(styles.input)}
                      type="text"
                      value={editingName}
                      onChange={e => setEditingName(e.target.value)}
                      onKeyDown={e => handleKeyDown(segment.id, e)}
                      onBlur={() => handleSaveName(segment.id)}
                      autoFocus
                      onClick={e => e.stopPropagation()}
                    />
                  ) : (
                    <div {...stylex.props(styles.segmentName)}>
                      {segment.name}
                    </div>
                  )}
                </div>

                <div {...stylex.props(styles.segmentTime)}>
                  {formatDuration(segment.frameStart, segment.frameEnd)}
                </div>

                {!isEditing && (
                  <div {...stylex.props(styles.buttonGroup)}>
                    <button
                      {...stylex.props(styles.button)}
                      onClick={e => handleStartEdit(segment, e)}>
                      编辑
                    </button>
                    <button
                      {...stylex.props(styles.button, styles.buttonDanger)}
                      onClick={e => handleDeleteSegment(segment.id, e)}>
                      删除
                    </button>
                  </div>
                )}

                {/* 物体列表 */}
                {segment.objects.length > 0 && (
                  <div {...stylex.props(styles.objectList)}>
                    {segment.objects.map(obj => (
                      <div key={obj.id} {...stylex.props(styles.objectItem)}>
                        <div
                          {...stylex.props(styles.objectColor)}
                          style={{backgroundColor: obj.color}}
                        />
                        <span>{obj.name}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* 添加物体按钮 */}
                {isActive && (
                  <>
                    {!isAddingActionObject ? (
                      <button
                        {...stylex.props(styles.addObjectButton)}
                        onClick={handleStartAddObject}>
                        + 添加物体
                      </button>
                    ) : (
                      <div
                        style={{
                          marginTop: '8px',
                          padding: '8px',
                          backgroundColor: 'rgba(59, 130, 246, 0.2)',
                          borderRadius: '4px',
                          border: '1px solid rgba(59, 130, 246, 0.6)',
                        }}>
                        <div
                          style={{
                            color: 'rgb(59, 130, 246)',
                            fontSize: '12px',
                            marginBottom: '8px',
                          }}>
                          ✓ 点击视频画面添加物体
                        </div>
                        <button
                          {...stylex.props(styles.button, styles.buttonDanger)}
                          onClick={e => {
                            e.stopPropagation();
                            handleCancelAddObject();
                          }}
                          style={{width: '100%'}}>
                          取消
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
