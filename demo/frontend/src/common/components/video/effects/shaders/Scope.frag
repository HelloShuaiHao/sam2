#version 300 es
// Copyright (c) Meta Platforms, Inc. and affiliates.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

precision mediump float;

in vec2 vTexCoord;

uniform sampler2D uSampler;
uniform vec2 uSize;
uniform int uNumMasks;
uniform bool uFillColor;
uniform bool uLight;
uniform bool uTransparency;
uniform sampler2D uMaskTexture0;
uniform sampler2D uMaskTexture1;
uniform sampler2D uMaskTexture2;
uniform sampler2D uMaskTexture3;
uniform sampler2D uMaskTexture4;
uniform sampler2D uMaskTexture5;

uniform vec4 uMaskColor0;
uniform vec4 uMaskColor1;
uniform vec4 uMaskColor2;
uniform vec4 uMaskColor3;
uniform vec4 uMaskColor4;
uniform vec4 uMaskColor5;

uniform vec4 bbox0;
uniform vec4 bbox1;
uniform vec4 bbox2;
uniform vec4 bbox3;
uniform vec4 bbox4;
uniform vec4 bbox5;

out vec4 fragColor;

void main() {
  vec4 color = texture(uSampler, vTexCoord);
  float aspectRatio = uSize.y / uSize.x;
  float radiusThreshold = 0.8f;
  float tickness = 0.085f;

  vec4 mask1 = vec4(0.0f);
  vec4 mask2 = vec4(0.0f);
  vec4 mask3 = vec4(0.0f);
  vec4 mask4 = vec4(0.0f);
  vec4 mask5 = vec4(0.0f);
  vec4 mask6 = vec4(0.0f);
  vec4 color1 = uMaskColor0 / 255.0;
  vec4 color2 = uMaskColor1 / 255.0;
  vec4 color3 = uMaskColor2 / 255.0;
  vec4 color4 = uMaskColor3 / 255.0;
  vec4 color5 = uMaskColor4 / 255.0;
  vec4 color6 = uMaskColor5 / 255.0;
  vec4 scopedColor = vec4(0.0f);

  bool scoped = false;
  vec4 whiteVariation = uTransparency ? vec4(0.0,0.0,0.0,1.0) : vec4(1.0);

  if(uNumMasks > 0) {
    mask1 = texture(uMaskTexture0, vec2(vTexCoord.y, vTexCoord.x));

    vec2 center1 = (bbox0.xy + bbox0.zw) * 0.5f;
    float radiusX1 = abs(bbox0.y - bbox0.w) * 0.5f;
    float radiusY1 = radiusX1 / aspectRatio;

    float distX1 = (vTexCoord.x - center1.x) / radiusX1;
    float distY1 = (vTexCoord.y - center1.y) / radiusY1;
    float dist1 = sqrt(pow(distX1, 2.0f) + pow(distY1, 2.0f));
   
    if(uFillColor) {
      if(dist1 >= radiusThreshold - tickness && dist1 <= radiusThreshold) {
        scoped = true;
        scopedColor = uLight ? whiteVariation : color1;
      }
    } else if(dist1 <= radiusThreshold) {
      scoped = true;
      scopedColor = uLight ? whiteVariation : color1;
    }
  }
  if(uNumMasks > 1) {
    mask2 = texture(uMaskTexture1, vec2(vTexCoord.y, vTexCoord.x));

    vec2 center2 = (bbox1.xy + bbox1.zw) * 0.5f;
    float radiusX2 = abs(bbox1.y - bbox1.w) * 0.5f;
    float radiusY2 = radiusX2 / aspectRatio;

    float distX2 = (vTexCoord.x - center2.x) / radiusX2;
    float distY2 = (vTexCoord.y - center2.y) / radiusY2;
    float dist2 = sqrt(pow(distX2, 2.0f) + pow(distY2, 2.0f));

    if(uFillColor) {
      if(dist2 >= radiusThreshold - tickness && dist2 <= radiusThreshold) {
        scoped = true;
        scopedColor = uLight ? whiteVariation : color2;
      }
    } else if(dist2 <= radiusThreshold) {
      scoped = true;
      scopedColor = uLight ? whiteVariation : color2;
    }
  }
  if(uNumMasks > 2) {
    mask3 = texture(uMaskTexture2, vec2(vTexCoord.y, vTexCoord.x));

    vec2 center3 = (bbox2.xy + bbox2.zw) * 0.5f;
    float radiusX3 = abs(bbox2.y - bbox2.w) * 0.5f;
    float radiusY3 = radiusX3 / aspectRatio;

    float distX3 = (vTexCoord.x - center3.x) / radiusX3;
    float distY3 = (vTexCoord.y - center3.y) / radiusY3;
    float dist3 = sqrt(pow(distX3, 2.0f) + pow(distY3, 2.0f));

    if(uFillColor) {
      if(dist3 >= radiusThreshold - tickness && dist3 <= radiusThreshold) {
        scoped = true;
        scopedColor = uLight ? whiteVariation : color3;
      }
    } else if(dist3 <= radiusThreshold) {
      scoped = true;
      scopedColor = uLight ? whiteVariation : color3;
    }
  }
  if(uNumMasks > 3) {
    mask4 = texture(uMaskTexture3, vec2(vTexCoord.y, vTexCoord.x));

    vec2 center4 = (bbox3.xy + bbox3.zw) * 0.5f;
    float radiusX4 = abs(bbox3.y - bbox3.w) * 0.5f;
    float radiusY4 = radiusX4 / aspectRatio;

    float distX4 = (vTexCoord.x - center4.x) / radiusX4;
    float distY4 = (vTexCoord.y - center4.y) / radiusY4;
    float dist4 = sqrt(pow(distX4, 2.0f) + pow(distY4, 2.0f));

    if(uFillColor) {
      if(dist4 >= radiusThreshold - tickness && dist4 <= radiusThreshold) {
        scoped = true;
        scopedColor = uLight ? whiteVariation : color4;
      }
    } else if(dist4 <= radiusThreshold) {
      scoped = true;
      scopedColor = uLight ? whiteVariation : color4;
    }
  }
  if(uNumMasks > 4) {
    mask5 = texture(uMaskTexture4, vec2(vTexCoord.y, vTexCoord.x));

    vec2 center5 = (bbox4.xy + bbox4.zw) * 0.5f;
    float radiusX5 = abs(bbox4.y - bbox4.w) * 0.5f;
    float radiusY5 = radiusX5 / aspectRatio;

    float distX5 = (vTexCoord.x - center5.x) / radiusX5;
    float distY5 = (vTexCoord.y - center5.y) / radiusY5;
    float dist5 = sqrt(pow(distX5, 2.0f) + pow(distY5, 2.0f));

    if(uFillColor) {
      if(dist5 >= radiusThreshold - tickness && dist5 <= radiusThreshold) {
        scoped = true;
        scopedColor = uLight ? whiteVariation : color5;
      }
    } else if(dist5 <= radiusThreshold) {
      scoped = true;
      scopedColor = uLight ? whiteVariation : color5;
    }
  }
  if(uNumMasks > 5) {
    mask6 = texture(uMaskTexture5, vec2(vTexCoord.y, vTexCoord.x));

    vec2 center6 = (bbox5.xy + bbox5.zw) * 0.5f;
    float radiusX6 = abs(bbox5.y - bbox5.w) * 0.5f;
    float radiusY6 = radiusX6 / aspectRatio;

    float distX6 = (vTexCoord.x - center6.x) / radiusX6;
    float distY6 = (vTexCoord.y - center6.y) / radiusY6;
    float dist6 = sqrt(pow(distX6, 2.0f) + pow(distY6, 2.0f));

    if(uFillColor) {
      if(dist6 >= radiusThreshold - tickness && dist6 <= radiusThreshold) {
        scoped = true;
        scopedColor = uLight ? whiteVariation : color6;
      }
    } else if(dist6 <= radiusThreshold) {
      scoped = true;
      scopedColor = uLight ? whiteVariation : color6;
    }
  }

  bool overlap = (mask1.r > 0.0f || mask2.r > 0.0f || mask3.r > 0.0f || mask4.r > 0.0f || mask5.r > 0.0f || mask6.r > 0.0f);

  if(scoped) {
    fragColor = overlap ? color : scopedColor;
    fragColor.a = uTransparency ? fragColor.a : 1.0;
  } else {
    fragColor = overlap ? color : vec4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}
