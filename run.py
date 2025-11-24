
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import Callback
from datetime import datetime
import time
import torch
from src.model import RecModel
from src.datamodule import RecDataModule


class TrainingTimeCallback(Callback):
    """记录训练时间并输出训练结果总结的回调"""
    
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None
        self.trainer = None
        
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*80}")
        print(f"训练开始时间: {start_datetime}")
        print(f"{'='*80}\n")
        self.trainer = trainer
        
    def on_train_end(self, trainer, pl_module):
        self.end_time = time.time()
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_time = self.end_time - self.start_time
        
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print(f"\n{'='*80}")
        print(f"训练结束时间: {end_datetime}")
        print(f"总训练时间: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_time:.2f} 秒)")
        print(f"{'='*80}\n")
        
        # 输出训练结果总结
        self._print_training_summary(trainer, pl_module)
        
    def _print_training_summary(self, trainer, pl_module):
        """打印训练结果总结"""
        print(f"\n{'='*80}")
        print("训练结果总结")
        print(f"{'='*80}")
        
        # 获取训练日志
        logged_metrics = trainer.logged_metrics if hasattr(trainer, 'logged_metrics') else {}
        callback_metrics = trainer.callback_metrics if hasattr(trainer, 'callback_metrics') else {}
        
        # 合并所有指标
        all_metrics = {**logged_metrics, **callback_metrics}
        
        # 提取并分类指标
        train_metrics = {}
        val_metrics = {}
        
        for key, value in all_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value
            if 'train' in key.lower() or 'Train' in key:
                train_metrics[key] = value
            elif 'val' in key.lower() or 'Val' in key:
                val_metrics[key] = value
        
        # 打印训练指标
        if train_metrics:
            print("\n训练指标:")
            print("-" * 80)
            for key, value in sorted(train_metrics.items()):
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        
        # 打印验证指标
        if val_metrics:
            print("\n验证指标:")
            print("-" * 80)
            # 按指标类型分组显示：HR@K, NDCG@K, MRR
            hr_metrics = {k: v for k, v in val_metrics.items() if 'HR@' in k or 'hr@' in k.lower()}
            ndcg_metrics = {k: v for k, v in val_metrics.items() if 'NDCG@' in k or 'ndcg@' in k.lower()}
            mrr_metrics = {k: v for k, v in val_metrics.items() if 'MRR' in k or 'mrr' in k.lower()}
            other_metrics = {k: v for k, v in val_metrics.items() 
                           if 'HR@' not in k and 'hr@' not in k.lower() 
                           and 'NDCG@' not in k and 'ndcg@' not in k.lower()
                           and 'MRR' not in k and 'mrr' not in k.lower()}
            
            if hr_metrics:
                print("  HR@K:")
                for key, value in sorted(hr_metrics.items()):
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.6f}")
                    else:
                        print(f"    {key}: {value}")
            
            if ndcg_metrics:
                print("  NDCG@K:")
                for key, value in sorted(ndcg_metrics.items()):
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.6f}")
                    else:
                        print(f"    {key}: {value}")
            
            if mrr_metrics:
                print("  MRR:")
                for key, value in sorted(mrr_metrics.items()):
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.6f}")
                    else:
                        print(f"    {key}: {value}")
            
            if other_metrics:
                for key, value in sorted(other_metrics.items()):
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value}")
        
        # 打印最佳模型信息
        best_model_path = None
        best_model_score = None
        
        # 尝试从ModelCheckpoint回调中获取最佳模型信息
        for callback in trainer.callbacks:
            if hasattr(callback, 'best_model_path') and hasattr(callback, 'best_model_score'):
                best_model_path = callback.best_model_path
                best_model_score = callback.best_model_score
                break
        
        # 如果没找到，尝试旧版本的checkpoint_callback属性
        if not best_model_path and hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            best_model_path = getattr(trainer.checkpoint_callback, 'best_model_path', None)
            best_model_score = getattr(trainer.checkpoint_callback, 'best_model_score', None)
        
        if best_model_path:
            print(f"\n最佳模型:")
            print(f"  路径: {best_model_path}")
            if best_model_score is not None:
                if isinstance(best_model_score, torch.Tensor):
                    best_model_score = best_model_score.item()
                print(f"  最佳分数: {best_model_score:.6f}")
        
        print(f"\n{'='*80}\n")


def cli_main():
    # 创建时间记录回调
    time_callback = TrainingTimeCallback()
    
    # 创建CLI，通过trainer_defaults添加回调
    # 注意：如果配置文件中已有callbacks，它们会被合并
    cli = LightningCLI(
        RecModel, 
        RecDataModule,
        trainer_defaults={
            'callbacks': [time_callback]
        }
    )

if __name__ == '__main__':
    cli_main()