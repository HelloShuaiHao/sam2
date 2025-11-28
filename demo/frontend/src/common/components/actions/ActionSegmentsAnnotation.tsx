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
import ActionSegmentSwimlane from '@/common/components/actions/ActionSegmentSwimlane';
import useVideo from '@/common/components/video/editor/useVideo';
import {
  ActionSegmentObject,
  actionSegmentsAtom,
  annotationModeAtom,
} from '@/demo/atoms';
import {m, spacing} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';
import {useAtomValue} from 'jotai';

const styles = stylex.create({
  container: {
    marginTop: m[3],
    minHeight: 75,
    paddingHorizontal: spacing[4],
    '@media screen and (max-width: 768px)': {
      minHeight: 25,
    },
  },
});

export default function ActionSegmentsAnnotation() {
  const video = useVideo();
  const annotationMode = useAtomValue(annotationModeAtom);
  const actionSegments = useAtomValue(actionSegmentsAtom);

  function handleSelectFrame(_object: ActionSegmentObject, index: number) {
    if (video !== null) {
      video.frame = index;
    }
  }

  // Only show in action mode
  if (annotationMode !== 'action') {
    return null;
  }

  return (
    <div {...stylex.props(styles.container)}>
      {actionSegments.map(segment =>
        segment.objects.map(obj => (
          <ActionSegmentSwimlane
            key={`${segment.id}-${obj.id}`}
            object={obj}
            segment={segment}
            onSelectFrame={handleSelectFrame}
          />
        )),
      )}
    </div>
  );
}
