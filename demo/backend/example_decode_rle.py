#!/usr/bin/env python3
"""
Example: 如何从导出的 annotations.json 还原 mask

这个脚本演示如何使用 RLE 数据还原二值 mask
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def decode_rle(rle_counts: str, size: list) -> np.ndarray:
    """
    从 RLE 字符串还原 mask

    Args:
        rle_counts: RLE 编码字符串，如 "25985 3 252 6 249 8..."
        size: [height, width]

    Returns:
        Binary mask (H, W) with values 0 or 1
    """
    # 解析 run lengths
    run_lengths = list(map(int, rle_counts.split()))

    # 创建扁平化的 mask
    flat_mask = np.zeros(size[0] * size[1], dtype=np.uint8)

    current_idx = 0
    current_val = 0  # COCO RLE 从 0 开始

    for run_length in run_lengths:
        flat_mask[current_idx:current_idx + run_length] = current_val
        current_idx += run_length
        current_val = 1 - current_val  # 在 0 和 1 之间切换

    # 重塑为 2D，使用 Fortran 顺序（列优先）
    mask = flat_mask.reshape(size, order='F')

    return mask


def visualize_mask_from_json(json_path: str, frame_idx: int = 0):
    """
    从 annotations.json 读取并可视化 mask

    Args:
        json_path: annotations.json 文件路径
        frame_idx: 要可视化的帧索引
    """
    # 读取 JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 获取视频尺寸
    video_height = data['video']['height']
    video_width = data['video']['width']

    # 获取指定帧的 annotations
    frame_annotations = data['annotations'][frame_idx]

    print(f"📊 Frame {frame_annotations['frame_index']} @ {frame_annotations['timestamp_sec']:.3f}s")
    print(f"   Found {len(frame_annotations['objects'])} objects")

    # 创建图像
    fig, axes = plt.subplots(1, len(frame_annotations['objects']) + 1,
                             figsize=(5 * (len(frame_annotations['objects']) + 1), 5))

    if len(frame_annotations['objects']) == 0:
        axes = [axes]

    # 合并所有 masks
    combined_mask = np.zeros((video_height, video_width), dtype=np.uint8)

    for idx, obj in enumerate(frame_annotations['objects']):
        # 还原 mask
        mask = decode_rle(obj['mask_rle'], [video_height, video_width])

        print(f"\n🎯 Object {obj['object_id']}: {obj['label']}")
        print(f"   - BBox: {obj['bbox']} (x, y, w, h)")
        print(f"   - Area: {obj['area']} pixels")
        print(f"   - Confidence: {obj['confidence']}")
        print(f"   - Mask shape: {mask.shape}")
        print(f"   - Mask sum (验证): {mask.sum()} (should equal area: {obj['area']})")

        # 绘制单个 mask
        if len(frame_annotations['objects']) > 0:
            ax = axes[idx] if len(frame_annotations['objects']) > 1 else axes[0]
        else:
            ax = axes[0]

        ax.imshow(mask, cmap='gray')
        ax.set_title(f"{obj['label']}\nArea: {obj['area']}")
        ax.axis('off')

        # 添加到合并 mask
        combined_mask = np.maximum(combined_mask, mask * (idx + 1))

    # 显示合并的 mask
    if len(frame_annotations['objects']) > 0:
        ax = axes[-1]
        ax.imshow(combined_mask, cmap='tab10')
        ax.set_title('All Objects Combined')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('decoded_masks.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: decoded_masks.png")
    plt.show()


def verify_your_example():
    """验证你提供的 RLE 数据"""
    # 你的示例数据
    rle_counts = "25985 3 252 6 249 8 247 10 245 11 244 13 243 13 243 13 242 15 241 15 241 15 241 15 241 16 240 16 240 16 240 16 240 15 241 15 241 15 242 14 242 14 242 13 244 12 244 11 246 10 248 7 250 5 254 1 32635"
    bbox = [101, 123, 28, 16]
    area = 333

    # 假设视频尺寸（你需要从 JSON 中获取）
    # 从 RLE 数据推断：25985 + 3 + 252 + ... ≈ 256*256 = 65536
    size = [256, 256]  # [height, width]

    # 解码
    mask = decode_rle(rle_counts, size)

    print("🧪 验证你的 RLE 数据:")
    print(f"   - Mask shape: {mask.shape}")
    print(f"   - Declared area: {area}")
    print(f"   - Actual area (sum): {mask.sum()}")
    print(f"   - Match: {'✅ YES' if mask.sum() == area else '❌ NO'}")

    # 验证 bbox
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    y_min, y_max = row_indices[0], row_indices[-1]
    x_min, x_max = col_indices[0], col_indices[-1]
    calculated_bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]

    print(f"   - Declared bbox: {bbox}")
    print(f"   - Calculated bbox: {calculated_bbox}")
    print(f"   - Match: {'✅ YES' if bbox == calculated_bbox else '❌ NO'}")

    # 可视化
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    plt.title(f'Decoded Mask\nArea: {mask.sum()} pixels')

    # 绘制 bbox
    from matplotlib.patches import Rectangle
    rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                     linewidth=2, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('your_example_mask.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化已保存到: your_example_mask.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("RLE Mask 解码示例")
    print("=" * 60)

    # 验证你提供的示例
    verify_your_example()

    print("\n" + "=" * 60)
    print("要从实际的 annotations.json 解码:")
    print("  python example_decode_rle.py /path/to/annotations.json")
    print("=" * 60)

    # 如果提供了 JSON 文件路径
    import sys
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        if Path(json_path).exists():
            visualize_mask_from_json(json_path, frame_idx=0)
        else:
            print(f"❌ File not found: {json_path}")
