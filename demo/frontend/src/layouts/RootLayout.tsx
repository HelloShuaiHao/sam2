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
import LoadingStateScreen from '@/common/loading/LoadingStateScreen';
import useSettingsContext from '@/settings/useSettingsContext';
import {Cog6ToothIcon} from '@heroicons/react/24/outline';
import stylex from '@stylexjs/stylex';
import {Suspense} from 'react';
import {Button, Indicator} from 'react-daisyui';
import {Outlet, Link, useLocation} from 'react-router-dom';

const styles = stylex.create({
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    maxHeight: '100vh',
    backgroundColor: '#000',
  },
  content: {
    position: 'relative',
    flex: '1 1 0%',
    display: 'flex',
    flexDirection: 'column',
    overflowX: 'auto',
    overflowY: {
      default: 'auto',
      '@media screen and (max-width: 768px)': 'auto',
    },
  },
  debugActions: {
    display: 'flex',
    flexDirection: 'column',
    position: 'fixed',
    top: 100,
    right: 0,
    backgroundColor: 'white',
    borderRadius: 3,
  },
});

export default function RootLayout() {
  const {openModal, hasChanged} = useSettingsContext();
  const location = useLocation();

  return (
    <div {...stylex.props(styles.container)}>
      {/* Navigation Bar */}
      <nav style={{
        display: 'flex',
        gap: '16px',
        padding: '12px',
        backgroundColor: '#1a1a1a',
        borderBottom: '1px solid #333'
      }}>
        <Link
          to="/"
          style={{
            padding: '8px 16px',
            color: location.pathname === '/' ? '#fff' : '#aaa',
            backgroundColor: location.pathname === '/' ? '#2563eb' : 'transparent',
            textDecoration: 'none',
            borderRadius: '4px',
            fontSize: '14px',
            fontWeight: 500
          }}
        >
          Video Annotation
        </Link>
        <Link
          to="/training"
          style={{
            padding: '8px 16px',
            color: location.pathname === '/training' ? '#fff' : '#aaa',
            backgroundColor: location.pathname === '/training' ? '#2563eb' : 'transparent',
            textDecoration: 'none',
            borderRadius: '4px',
            fontSize: '14px',
            fontWeight: 500
          }}
        >
          Training Workflow
        </Link>
        <Link
          to="/experiments"
          style={{
            padding: '8px 16px',
            color: location.pathname === '/experiments' ? '#fff' : '#aaa',
            backgroundColor: location.pathname === '/experiments' ? '#2563eb' : 'transparent',
            textDecoration: 'none',
            borderRadius: '4px',
            fontSize: '14px',
            fontWeight: 500
          }}
        >
          Experiments
        </Link>
      </nav>

      <div {...stylex.props(styles.content)}>
        <Suspense
          fallback={
            <LoadingStateScreen
              title="Loading demo..."
              description="This may take a few moments, you're almost there!"
            />
          }>
          <Outlet />
        </Suspense>
      </div>
      <div {...stylex.props(styles.debugActions)}>
        <Indicator>
          {hasChanged && (
            <Indicator.Item
              className="badge badge-primary scale-50"
              horizontal="start"
              vertical="top"
            />
          )}
          <Button
            color="ghost"
            onClick={openModal}
            shape="circle"
            size="xs"
            startIcon={<Cog6ToothIcon className="w-4 h-4" />}
            title="Bugnub"
          />
        </Indicator>
      </div>
    </div>
  );
}
