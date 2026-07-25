# 数据集预览工具 - 图片和表格形式展示

import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)


class DatasetPreview:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def load_and_preview_fashion_mnist(self):
        """加载并预览Fashion-MNIST数据集"""
        print("=== Fashion-MNIST数据集预览 ===\n")

        # 加载数据集
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=self.transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=self.transform
        )

        # 基本统计信息
        self._print_basic_info(train_dataset, test_dataset)

        # 1. 显示样本图片
        self._display_sample_images(train_dataset)

        # 2. 类别分布可视化
        self._plot_class_distribution(train_dataset, test_dataset)

        # 3. 创建模拟交互数据
        interaction_data = self._create_sample_interaction_data(train_dataset)

        # 4. 交互数据预览
        self._preview_interaction_data(interaction_data)

        # 5. 用户行为分析
        self._analyze_user_behavior(interaction_data)

        # 6. 物品流行度分析
        self._analyze_item_popularity(interaction_data)

        return train_dataset, test_dataset, interaction_data

    def _print_basic_info(self, train_dataset, test_dataset):
        """打印基本统计信息"""
        print("📊 数据集基本信息:")
        print(f"训练集大小: {len(train_dataset):,}")
        print(f"测试集大小: {len(test_dataset):,}")
        print(f"总样本数: {len(train_dataset) + len(test_dataset):,}")
        print(f"图像尺寸: {train_dataset[0][0].shape}")
        print(f"类别数量: {len(train_dataset.classes)}")
        print("类别名称:", train_dataset.classes)
        print()

        # 显示图像统计信息
        sample_image, sample_label = train_dataset[0]
        print(f"图像统计信息:")
        print(f"  像素值范围: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
        print(f"  图像形状: {sample_image.shape}")
        print(f"  数据类型: {sample_image.dtype}")
        print()

    def _display_sample_images(self, dataset, num_samples=20):
        """显示样本图片"""
        print("🖼️ 显示样本图片...")

        # 创建子图
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.ravel()

        # 从每个类别中随机选择样本
        class_indices = {}
        for i, (_, label) in enumerate(dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)

        selected_indices = []
        for label in class_indices:
            if class_indices[label]:
                selected_indices.append(np.random.choice(class_indices[label]))

        # 补充随机样本
        while len(selected_indices) < num_samples:
            idx = np.random.randint(len(dataset))
            if idx not in selected_indices:
                selected_indices.append(idx)

        selected_indices = selected_indices[:num_samples]

        for i, idx in enumerate(selected_indices):
            image, label = dataset[idx]

            # 转换为numpy并反归一化
            img = image.squeeze().numpy()
            img = (img * 0.5) + 0.5  # 反归一化

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'{dataset.classes[label]}\n(标签: {label})', fontsize=9)
            axes[i].axis('off')

        plt.suptitle('Fashion-MNIST 样本图片', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # 显示单个大图
        self._display_large_sample(dataset)

    def _display_large_sample(self, dataset, num_large=5):
        """显示大尺寸样本图片"""
        print("显示大尺寸样本图片...")

        fig, axes = plt.subplots(1, num_large, figsize=(15, 3))
        if num_large == 1:
            axes = [axes]

        for i in range(num_large):
            idx = np.random.randint(len(dataset))
            image, label = dataset[idx]

            img = image.squeeze().numpy()
            img = (img * 0.5) + 0.5

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'{dataset.classes[label]}\n(标签: {label})', fontsize=12)
            axes[i].axis('off')

        plt.suptitle('大尺寸样本展示', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _plot_class_distribution(self, train_dataset, test_dataset):
        """绘制类别分布图"""
        print("📈 分析类别分布...")

        # 统计训练集和测试集的类别分布
        train_labels = [label for _, label in train_dataset]
        test_labels = [label for _, label in test_dataset]

        train_counts = pd.Series(train_labels).value_counts().sort_index()
        test_counts = pd.Series(test_labels).value_counts().sort_index()

        # 创建子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # 1. 训练集类别分布
        bars1 = ax1.bar(range(len(train_counts)), train_counts.values,
                        color=plt.cm.Set3(np.arange(len(train_counts))))
        ax1.set_title('训练集类别分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('类别')
        ax1.set_ylabel('样本数量')
        ax1.set_xticks(range(len(train_counts)))
        ax1.set_xticklabels([train_dataset.classes[i] for i in train_counts.index],
                            rotation=45, ha='right')

        # 在柱子上添加数值
        for bar, count in zip(bars1, train_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 100,
                     f'{count}', ha='center', va='bottom', fontsize=9)

        # 2. 测试集类别分布
        bars2 = ax2.bar(range(len(test_counts)), test_counts.values,
                        color=plt.cm.Set3(np.arange(len(test_counts))))
        ax2.set_title('测试集类别分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('类别')
        ax2.set_ylabel('样本数量')
        ax2.set_xticks(range(len(test_counts)))
        ax2.set_xticklabels([test_dataset.classes[i] for i in test_counts.index],
                            rotation=45, ha='right')

        for bar, count in zip(bars2, test_counts.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 20,
                     f'{count}', ha='center', va='bottom', fontsize=9)

        # 3. 训练集和测试集比例对比
        categories = [train_dataset.classes[i] for i in range(len(train_dataset.classes))]
        train_percentages = [train_counts.get(i, 0) / len(train_dataset) * 100 for i in range(10)]
        test_percentages = [test_counts.get(i, 0) / len(test_dataset) * 100 for i in range(10)]

        x = np.arange(len(categories))
        width = 0.35

        bars3_1 = ax3.bar(x - width / 2, train_percentages, width, label='训练集', alpha=0.7)
        bars3_2 = ax3.bar(x + width / 2, test_percentages, width, label='测试集', alpha=0.7)

        ax3.set_title('训练集 vs 测试集类别比例', fontsize=14, fontweight='bold')
        ax3.set_xlabel('类别')
        ax3.set_ylabel('百分比 (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印统计信息
        print("\n类别分布统计:")
        df_dist = pd.DataFrame({
            '类别': categories,
            '训练集数量': [train_counts.get(i, 0) for i in range(10)],
            '训练集比例%': [f"{train_counts.get(i, 0) / len(train_dataset) * 100:.1f}" for i in range(10)],
            '测试集数量': [test_counts.get(i, 0) for i in range(10)],
            '测试集比例%': [f"{test_counts.get(i, 0) / len(test_dataset) * 100:.1f}" for i in range(10)]
        })
        print(df_dist.to_string(index=False))
        print()

    def _create_sample_interaction_data(self, dataset, n_users=1000, n_interactions=5000):
        """创建模拟的用户-物品交互数据"""
        print("📋 创建模拟交互数据...")

        n_items = min(2000, len(dataset))

        interactions = []
        for i in range(n_interactions):
            user_id = np.random.randint(0, n_users)
            item_id = np.random.randint(0, n_items)
            _, item_label = dataset[item_id]

            # 基于用户偏好和物品特征的评分
            user_preference = np.random.randn(10)
            preference_match = np.exp(-0.5 * np.linalg.norm(user_preference - np.eye(10)[item_label]))

            # 生成评分 (1-5分)
            base_rating = 1 + 4 * preference_match
            rating = max(1, min(5, base_rating + np.random.normal(0, 0.3)))

            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'item_label': item_label,
                'item_name': dataset.classes[item_label],
                'rating': rating,
                'timestamp': np.random.randint(1000000000, 2000000000)
            })

        data = pd.DataFrame(interactions)

        # 添加分类标签
        data['rating_category'] = pd.cut(data['rating'],
                                         bins=[0, 2, 3, 4, 5.1],
                                         labels=['差评(1-2)', '中评(2-3)', '好评(3-4)', '强烈推荐(4-5)'])

        data['is_positive'] = (data['rating'] >= 4).astype(int)

        print(f"创建了 {len(data)} 条交互记录")
        print(f"用户数量: {data['user_id'].nunique()}")
        print(f"物品数量: {data['item_id'].nunique()}")
        print(f"平均评分: {data['rating'].mean():.2f}")
        print(f"正样本比例: {data['is_positive'].mean():.3f}")
        print()

        return data

    def _preview_interaction_data(self, data):
        """预览交互数据"""
        print("📊 交互数据预览:")

        # 显示数据表格前10行
        print("数据表前10行:")
        display_data = data.head(10).copy()
        display_data['rating'] = display_data['rating'].round(2)
        print(display_data[['user_id', 'item_id', 'item_name', 'rating', 'rating_category']].to_string(index=False))
        print()

        # 数据基本信息
        print("数据基本信息:")
        print(f"总记录数: {len(data):,}")
        print(f"用户数: {data['user_id'].nunique():,}")
        print(f"物品数: {data['item_id'].nunique():,}")
        print(f"稀疏度: {len(data) / (data['user_id'].nunique() * data['item_id'].nunique()) * 100:.4f}%")
        print()

        # 评分分布
        self._plot_rating_distribution(data)

        # 时间分布（模拟）
        self._plot_temporal_distribution(data)

    def _plot_rating_distribution(self, data):
        """绘制评分分布图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 评分分布直方图
        rating_counts = data['rating'].value_counts().sort_index()
        bars = ax1.bar(rating_counts.index, rating_counts.values,
                       color=plt.cm.viridis(np.linspace(0, 1, len(rating_counts))))
        ax1.set_title('评分分布直方图', fontsize=14, fontweight='bold')
        ax1.set_xlabel('评分')
        ax1.set_ylabel('数量')
        ax1.grid(True, alpha=0.3)

        for bar, count in zip(bars, rating_counts.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f'{count}', ha='center', va='bottom', fontsize=10)

        # 2. 评分类别饼图
        rating_cat_counts = data['rating_category'].value_counts()
        colors = plt.cm.Pastel1(np.arange(len(rating_cat_counts)))
        wedges, texts, autotexts = ax2.pie(rating_cat_counts.values, labels=rating_cat_counts.index,
                                           autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('评分类别分布', fontsize=14, fontweight='bold')

        # 3. 评分箱线图
        rating_by_category = [data[data['rating_category'] == cat]['rating'] for cat in rating_cat_counts.index]
        box_plot = ax3.boxplot(rating_by_category, labels=rating_cat_counts.index, patch_artist=True)
        ax3.set_title('各评分类别分布箱线图', fontsize=14, fontweight='bold')
        ax3.set_ylabel('评分')
        ax3.grid(True, alpha=0.3)

        # 设置箱线图颜色
        colors = plt.cm.Set3(np.arange(len(rating_cat_counts)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        # 4. 累计评分分布
        sorted_ratings = np.sort(data['rating'])
        cumulative = np.arange(1, len(sorted_ratings) + 1) / len(sorted_ratings)
        ax4.plot(sorted_ratings, cumulative, linewidth=2)
        ax4.set_title('评分累计分布函数', fontsize=14, fontweight='bold')
        ax4.set_xlabel('评分')
        ax4.set_ylabel('累计比例')
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(sorted_ratings, cumulative, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('评分分布分析', fontsize=16, fontweight='bold', y=1.02)
        plt.show()

        # 打印评分统计
        print("评分统计信息:")
        rating_stats = data['rating'].describe()
        print(rating_stats)
        print()

    def _plot_temporal_distribution(self, data):
        """绘制时间分布图（模拟）"""
        # 将时间戳转换为日期
        data['date'] = pd.to_datetime(data['timestamp'], unit='s')
        data['hour'] = data['date'].dt.hour
        data['day_of_week'] = data['date'].dt.dayofweek

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 1. 小时分布
        hour_counts = data['hour'].value_counts().sort_index()
        ax1.bar(hour_counts.index, hour_counts.values, color='skyblue', alpha=0.7)
        ax1.set_title('24小时交互分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('小时')
        ax1.set_ylabel('交互数量')
        ax1.grid(True, alpha=0.3)

        # 2. 星期分布
        day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        day_counts = data['day_of_week'].value_counts().sort_index()
        ax2.bar(day_names, day_counts.values, color='lightcoral', alpha=0.7)
        ax2.set_title('每周交互分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('星期')
        ax2.set_ylabel('交互数量')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _analyze_user_behavior(self, data):
        """分析用户行为"""
        print("👤 用户行为分析...")

        user_stats = data.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'item_id': 'nunique',
            'is_positive': 'sum'
        }).round(3)

        user_stats.columns = ['交互次数', '平均评分', '评分标准差', '交互物品数', '正样本数']
        user_stats['正样本比例'] = (user_stats['正样本数'] / user_stats['交互次数']).round(3)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 用户交互次数分布
        interaction_counts = user_stats['交互次数']
        ax1.hist(interaction_counts, bins=30, color='lightblue', alpha=0.7, edgecolor='black')
        ax1.set_title('用户交互次数分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('交互次数')
        ax1.set_ylabel('用户数量')
        ax1.grid(True, alpha=0.3)

        # 2. 用户平均评分分布
        avg_ratings = user_stats['平均评分']
        ax2.hist(avg_ratings.dropna(), bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_title('用户平均评分分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('平均评分')
        ax2.set_ylabel('用户数量')
        ax2.grid(True, alpha=0.3)

        # 3. 用户活跃度 vs 平均评分
        ax3.scatter(interaction_counts, avg_ratings, alpha=0.6, color='purple')
        ax3.set_title('用户活跃度 vs 平均评分', fontsize=14, fontweight='bold')
        ax3.set_xlabel('交互次数')
        ax3.set_ylabel('平均评分')
        ax3.grid(True, alpha=0.3)

        # 4. 用户正样本比例分布
        positive_ratios = user_stats['正样本比例']
        ax4.hist(positive_ratios.dropna(), bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax4.set_title('用户正样本比例分布', fontsize=14, fontweight='bold')
        ax4.set_xlabel('正样本比例')
        ax4.set_ylabel('用户数量')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印用户行为统计
        print("用户行为统计:")
        print(user_stats.describe().round(3))
        print()

    def _analyze_item_popularity(self, data):
        """分析物品流行度"""
        print("📦 物品流行度分析...")

        item_stats = data.groupby(['item_id', 'item_name']).agg({
            'rating': ['count', 'mean', 'std'],
            'user_id': 'nunique',
            'is_positive': 'sum'
        }).round(3)

        item_stats.columns = ['被评分次数', '平均评分', '评分标准差', '评分用户数', '正样本数']
        item_stats['正样本比例'] = (item_stats['正样本数'] / item_stats['被评分次数']).round(3)
        item_stats = item_stats.sort_values('被评分次数', ascending=False)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 物品流行度分布（被评分次数）
        popularity = item_stats['被评分次数']
        ax1.hist(popularity, bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
        ax1.set_title('物品流行度分布（被评分次数）', fontsize=14, fontweight='bold')
        ax1.set_xlabel('被评分次数')
        ax1.set_ylabel('物品数量')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # 2. 物品平均评分分布
        item_avg_ratings = item_stats['平均评分']
        ax2.hist(item_avg_ratings.dropna(), bins=30, color='lightseagreen', alpha=0.7, edgecolor='black')
        ax2.set_title('物品平均评分分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('平均评分')
        ax2.set_ylabel('物品数量')
        ax2.grid(True, alpha=0.3)

        # 3. 流行度 vs 平均评分
        ax3.scatter(popularity, item_avg_ratings, alpha=0.6, color='teal')
        ax3.set_title('物品流行度 vs 平均评分', fontsize=14, fontweight='bold')
        ax3.set_xlabel('被评分次数')
        ax3.set_ylabel('平均评分')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)

        # 4. 类别流行度
        category_popularity = data.groupby('item_name')['item_id'].count().sort_values(ascending=False)
        bars = ax4.bar(range(len(category_popularity)), category_popularity.values,
                       color=plt.cm.tab10(np.arange(len(category_popularity))))
        ax4.set_title('各类别物品流行度', fontsize=14, fontweight='bold')
        ax4.set_xlabel('物品类别')
        ax4.set_ylabel('被评分次数')
        ax4.set_xticks(range(len(category_popularity)))
        ax4.set_xticklabels(category_popularity.index, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印最受欢迎的物品
        print("最受欢迎的10个物品:")
        top_items = item_stats.head(10)
        print(top_items[['被评分次数', '平均评分', '正样本比例']].round(3))
        print()

        # 打印类别统计
        print("各类别统计:")
        category_stats = data.groupby('item_name').agg({
            'item_id': 'nunique',
            'rating': ['mean', 'count']
        }).round(3)
        category_stats.columns = ['物品数量', '平均评分', '总评分次数']
        category_stats = category_stats.sort_values('总评分次数', ascending=False)
        print(category_stats)
        print()


def main():
    """主函数"""
    print("=== 数据集预览工具 ===\n")

    preview = DatasetPreview()
    train_dataset, test_dataset, interaction_data = preview.load_and_preview_fashion_mnist()

    print("✅ 数据集预览完成！")
    print("\n数据集可用于以下任务:")
    print("1. 图像分类 - 10个服装类别")
    print("2. 推荐系统 - 用户-物品交互数据")
    print("3. 计算机视觉 - 图像特征学习")
    print("4. 图神经网络 - 用户-物品关系建模")

    return train_dataset, test_dataset, interaction_data


if __name__ == "__main__":
    train_dataset, test_dataset, interaction_data = main()