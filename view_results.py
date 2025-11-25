#!/usr/bin/env python3
"""
查看 GEAR-2025 训练结果的脚本
从 TensorBoard 日志中提取评估指标
"""

import os
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("需要安装 tensorboard: pip install tensorboard")
    sys.exit(1)


def view_results(log_dir):
    """从 TensorBoard 日志目录读取并显示结果"""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"错误: 日志目录不存在: {log_dir}")
        return
    
    print(f"正在读取日志: {log_dir}")
    print("=" * 80)
    
    # 加载事件文件
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    
    # 获取所有标量指标
    scalar_tags = ea.Tags().get('scalars', [])
    
    if not scalar_tags:
        print("未找到任何指标数据")
        return
    
    print(f"\n找到 {len(scalar_tags)} 个指标\n")
    
    # 分类显示指标
    train_metrics = []
    test_metrics = []
    other_metrics = []
    
    for tag in scalar_tags:
        if tag.startswith('train') or 'Train' in tag:
            train_metrics.append(tag)
        elif tag.startswith('Test') or tag.startswith('test'):
            test_metrics.append(tag)
        else:
            other_metrics.append(tag)
    
    # 显示训练指标
    if train_metrics:
        print("=" * 80)
        print("训练指标 (Training Metrics)")
        print("=" * 80)
        for tag in sorted(train_metrics):
            events = ea.Scalars(tag)
            if events:
                latest = events[-1]
                print(f"  {tag:40s} = {latest.value:.6f} (step: {latest.step}, epoch: {latest.wall_time:.0f})")
    
    # 直接关注测试指标
    if test_metrics:
        print("\n" + "=" * 80)
        print("测试指标 (Test Metrics)")
        print("=" * 80)
        for tag in sorted(test_metrics):
            events = ea.Scalars(tag)
            if events:
                latest = events[-1]
                best_event = max(events, key=lambda x: x.value)
                print(f"  {tag:40s} = {latest.value:.6f} (最新, step: {latest.step})")
                print(f"  {'最佳值':40s} = {best_event.value:.6f} (step: {best_event.step})")
    
    # 显示其他指标
    if other_metrics:
        print("\n" + "=" * 80)
        print("其他指标 (Other Metrics)")
        print("=" * 80)
        for tag in sorted(other_metrics):
            events = ea.Scalars(tag)
            if events:
                latest = events[-1]
                print(f"  {tag:40s} = {latest.value:.6f} (step: {latest.step})")
    
    # 显示检查点信息
    checkpoint_dir = log_path / "checkpoints"
    if checkpoint_dir.exists():
        print("\n" + "=" * 80)
        print("检查点文件 (Checkpoints)")
        print("=" * 80)
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if checkpoints:
            for ckpt in sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True):
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                print(f"  {ckpt.name} ({size_mb:.2f} MB)")
        else:
            print("  未找到检查点文件")
    
    print("\n" + "=" * 80)
    print("提示: 使用 TensorBoard 查看详细训练曲线:")
    print(f"  tensorboard --logdir {log_path.parent}")
    print("=" * 80)


if __name__ == "__main__":
    # 默认查看 retail/full 的结果
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        # 查找最新的日志目录
        base_log_dir = Path(__file__).parent / "logs" / "retail" / "full" / "lightning_logs"
        if base_log_dir.exists():
            versions = sorted([d for d in base_log_dir.iterdir() if d.is_dir()], 
                            key=lambda x: x.stat().st_mtime, reverse=True)
            if versions:
                log_dir = str(versions[0])
            else:
                log_dir = str(base_log_dir / "version_0")
        else:
            log_dir = "logs/retail/full/lightning_logs/version_0"
    
    view_results(log_dir)

