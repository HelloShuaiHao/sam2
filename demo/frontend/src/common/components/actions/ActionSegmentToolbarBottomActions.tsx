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
import TrackActionSegmentsButton from '@/common/components/actions/TrackActionSegmentsButton';
import ToolbarBottomActionsWrapper from '@/common/components/toolbar/ToolbarBottomActionsWrapper';
import {actionSegmentsAtom} from '@/demo/atoms';
import {useAtomValue} from 'jotai';

export default function ActionSegmentToolbarBottomActions() {
  const actionSegments = useAtomValue(actionSegmentsAtom);

  // 检查是否有任何片段包含有标注点的物体
  const hasAnnotatedObjects = actionSegments.some(segment =>
    segment.objects.some(obj =>
      obj.points.some(framePoints => framePoints && framePoints.length > 0),
    ),
  );

  return (
    <ToolbarBottomActionsWrapper>
      {hasAnnotatedObjects && <TrackActionSegmentsButton />}
    </ToolbarBottomActionsWrapper>
  );
}
