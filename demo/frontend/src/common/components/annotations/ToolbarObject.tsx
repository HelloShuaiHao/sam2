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
import ObjectActions from '@/common/components/annotations/ObjectActions';
import ObjectPlaceholder from '@/common/components/annotations/ObjectPlaceholder';
import ObjectThumbnail from '@/common/components/annotations/ObjectThumbnail';
import ToolbarObjectContainer from '@/common/components/annotations/ToolbarObjectContainer';
import useVideo from '@/common/components/video/editor/useVideo';
import {BaseTracklet} from '@/common/tracker/Tracker';
import emptyFunction from '@/common/utils/emptyFunction';
import {activeTrackletObjectIdAtom, trackletNamesAtom} from '@/demo/atoms';
import {useSetAtom, useAtom} from 'jotai';
import {useState} from 'react';

type Props = {
  label: string;
  tracklet: BaseTracklet;
  isActive: boolean;
  isMobile?: boolean;
  onClick?: () => void;
  onThumbnailClick?: () => void;
};

export default function ToolbarObject({
  label,
  tracklet,
  isActive,
  isMobile = false,
  onClick,
  onThumbnailClick = emptyFunction,
}: Props) {
  const video = useVideo();
  const setActiveTrackletId = useSetAtom(activeTrackletObjectIdAtom);
  const [trackletNames, setTrackletNames] = useAtom(trackletNamesAtom);
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState('');

  async function handleCancelNewObject() {
    try {
      await video?.deleteTracklet(tracklet.id);
    } catch (error) {
      reportError(error);
    } finally {
      setActiveTrackletId(null);
    }
  }

  function handleLabelClick(e: React.MouseEvent) {
    if (!tracklet.isInitialized) return;
    e.stopPropagation();
    setEditValue(label);
    setIsEditing(true);
  }

  function handleLabelSave() {
    if (editValue.trim()) {
      setTrackletNames({
        ...trackletNames,
        [tracklet.id]: editValue.trim(),
      });
    } else {
      // If empty, remove custom name
      const newNames = {...trackletNames};
      delete newNames[tracklet.id];
      setTrackletNames(newNames);
    }
    setIsEditing(false);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') {
      handleLabelSave();
    } else if (e.key === 'Escape') {
      setIsEditing(false);
    }
  }

  if (!tracklet.isInitialized) {
    return (
      <ToolbarObjectContainer
        alignItems="center"
        isActive={isActive}
        title="New object"
        subtitle="No object is currently selected. Click an object in the video."
        thumbnail={<ObjectPlaceholder showPlus={false} />}
        isMobile={isMobile}
        onClick={onClick}
        onCancel={handleCancelNewObject}
      />
    );
  }

  return (
    <ToolbarObjectContainer
      isActive={isActive}
      onClick={onClick}
      title={
        isEditing ? (
          <input
            type="text"
            value={editValue}
            onChange={e => setEditValue(e.target.value)}
            onBlur={handleLabelSave}
            onKeyDown={handleKeyDown}
            autoFocus
            className="bg-transparent border-b border-white outline-none text-white w-full"
            onClick={e => e.stopPropagation()}
          />
        ) : (
          <span
            onClick={handleLabelClick}
            className="cursor-pointer hover:opacity-80"
            title="Click to edit name"
          >
            {label}
          </span>
        )
      }
      subtitle=""
      thumbnail={
        <ObjectThumbnail
          thumbnail={tracklet.thumbnail}
          color={tracklet.color}
          onClick={onThumbnailClick}
        />
      }
      isMobile={isMobile}>
      <ObjectActions objectId={tracklet.id} active={isActive} />
    </ToolbarObjectContainer>
  );
}
