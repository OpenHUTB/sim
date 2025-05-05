import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_results(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def parse_results(records):
    train_logs, val_logs = [], []
    for entry in records:
        if 'train' in entry:
            train_logs.append({
                'epoch': entry['meta']['epoch'],
                'train_loss': entry['train']['loss'],
                'train_top1': entry['train']['top1'],
                'train_top5': entry['train']['top5'],
                'lr': entry['train']['learning_rate']
            })
        if 'val' in entry:
            val_logs.append({
                'epoch': entry['meta']['epoch'],
                'val_loss': entry['val']['loss'],
                'val_top1': entry['val']['top1'],
                'val_top5': entry['val']['top5']
            })
    return pd.DataFrame(train_logs), pd.DataFrame(val_logs)

def find_best_metrics(val_df, train_df):
    """ 找到验证集表现最佳的参数 """
    # 合并验证集和训练集数据（按最近的epoch匹配）
    merged = pd.merge_asof(
        val_df.sort_values('epoch'),
        train_df[['epoch', 'train_loss', 'train_top1', 'train_top5', 'lr']].sort_values('epoch'),
        on='epoch',
        direction='nearest'
    )
    
    # 找到Top1准确率最高的记录
    best_idx = merged['val_top1'].idxmax()
    best = merged.loc[best_idx]
    
    return {
        'epoch': best['epoch'],
        'val_top1': best['val_top1'],
        'val_top5': best['val_top5'],
        'val_loss': best['val_loss'],
        'train_top1': best['train_top1'],
        'train_top5': best['train_top5'],
        'train_loss': best['train_loss'],
        'learning_rate': best['lr']
    }

def plot_metrics(train_df, val_df, save_path=None):
    # 强制设置字体参数
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'axes.unicode_minus': False
    })
    
    plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(train_df['epoch'], train_df['train_loss'], label='Train')
    plt.plot(val_df['epoch'], val_df['val_loss'], label='Validation')
    plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('Loss Curve')
    plt.legend(), plt.grid(True)

    # Top1 Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_df['epoch'], train_df['train_top1'], label='Train')
    plt.plot(val_df['epoch'], val_df['val_top1'], label='Validation')
    plt.xlabel('Epoch'), plt.ylabel('Accuracy'), plt.title('Top1 Accuracy')
    plt.legend(), plt.grid(True)

    # Top5 Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(train_df['epoch'], train_df['train_top5'], label='Train')
    plt.plot(val_df['epoch'], val_df['val_top5'], label='Validation')
    plt.xlabel('Epoch'), plt.ylabel('Accuracy'), plt.title('Top5 Accuracy')
    plt.legend(), plt.grid(True)

    # Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot(train_df['epoch'], train_df['lr'], label='LR')
    plt.xlabel('Epoch'), plt.ylabel('Learning Rate'), plt.title('Learning Rate Schedule')
    plt.legend(), plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'training_metrics3.png'), 
                   dpi=300, 
                   bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 清理缓存
    os.system('rm -rf ~/.cache/matplotlib')

    # 自动创建result目录并保存结果
    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    pkl_path = os.path.join(result_dir, "results.pkl")
    
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    
    records = load_results(pkl_path)
    train_df, val_df = parse_results(records)
    
    # 数据清洗
    train_df = train_df.sort_values('epoch').drop_duplicates(subset=['epoch'])
    val_df = val_df.sort_values('epoch').drop_duplicates(subset=['epoch'])
    
    # 绘制图表
    plot_metrics(train_df, val_df, save_path=output_dir)
    
    # 打印最佳参数
    best_metrics = find_best_metrics(val_df, train_df)
    print("\n=== 最佳验证表现 ===")
    print(f"Epoch:           {best_metrics['epoch']:.2f}")
    print(f"验证集 Top1准确率: {best_metrics['val_top1']*100:.2f}%")
    print(f"验证集 Top5准确率: {best_metrics['val_top5']*100:.2f}%")
    print(f"验证集损失值:      {best_metrics['val_loss']:.4f}")
    print(f"训练集 Top1准确率: {best_metrics['train_top1']*100:.2f}%")
    print(f"训练集 Top5准确率: {best_metrics['train_top5']*100:.2f}%")
    print(f"训练集损失值:      {best_metrics['train_loss']:.4f}")
    print(f"学习率:           {best_metrics['learning_rate']:.6f}")

    