import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import urllib.request
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class BasicDataset:  # 定义数据集处理类，负责加载、预处理Amazon-Book数据集
    def __init__(self, path='./data/amazon-book', split='train', max_interactions=500000):  # 构造函数，初始化数据集参数
        self.path = path  # 数据存储目录路径，默认为 './data/amazon-book'
        self.split = split  # 数据集类型：'train' 表示训练集，'test' 表示测试集
        self.interactions = []  # 存储所有用户-物品交互对的列表，格式为 [(user, item), ...]
        self.num_users = 0  # 用户总数
        self.num_items = 0  # 物品（图书）总数
        self.test_dict = {} if split == 'test' else None  # 测试集专用：存储每个用户的真实喜好物品列表 {user: [item1, item2, ...]}

        os.makedirs(self.path, exist_ok=True)  # 创建数据目录，如果已存在则不报错
        train_file = os.path.join(self.path, 'train.txt')  # 构造训练集文件路径
        test_file = os.path.join(self.path, 'test.txt')  # 构造测试集文件路径

        if not os.path.exists(train_file):  # 如果本地没有训练集文件
            print("Downloading train.txt for Amazon-book...")  # 打印下载提示信息
            urllib.request.urlretrieve(  # 从指定URL下载训练集文件
                'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/amazon-book/train.txt',
                train_file  # 保存到本地路径
            )
        if not os.path.exists(test_file):  # 如果本地没有测试集文件
            print("Downloading test.txt for Amazon-book...")  # 打印下载提示信息
            urllib.request.urlretrieve(  # 从指定URL下载测试集文件
                'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/amazon-book/test.txt',
                test_file  # 保存到本地路径
            )

        self._load_data(max_interactions)  # 调用私有方法加载指定数量的交互数据
        if split == 'test':  # 如果是测试集
            self._load_test_dict()  # 额外加载测试集的真实标签字典

    def _load_data(self, max_interactions):  # 私有方法：从txt文件加载交互数据
        data_file = os.path.join(self.path, f'{self.split}.txt')  # 根据split构造数据文件路径
        users = set()  # 用于收集所有出现过的用户ID
        items = set()  # 用于收集所有出现过的物品ID
        count = 0  # 已加载的交互记录计数器
        with open(data_file, 'r') as f:  # 打开数据文件进行读取
            for line in f:  # 逐行读取
                if not line.strip():  # 跳过空行
                    continue
                parts = list(map(int, line.strip().split()))  # 将行内容分割并转为整数列表
                if len(parts) < 2:  # 如果格式不正确（至少需要 user + 1个item）
                    continue
                user = parts[0]  # 第一列为用户ID
                user_items = parts[1:]  # 剩余列为该用户交互过的物品ID列表
                for item in user_items:  # 遍历该用户的所有交互物品
                    if count >= max_interactions:  # 如果已达到最大加载数量限制
                        break  # 停止加载
                    self.interactions.append((user, item))  # 添加交互对到列表
                    users.add(user)  # 记录用户ID
                    items.add(item)  # 记录物品ID
                    count += 1  # 计数器加1
                if count >= max_interactions:  # 再次检查是否达到上限
                    break

        self.num_users = max(users) + 1 if users else 0  # 计算用户总数（ID从0开始，故+1）
        self.num_items = max(items) + 1 if items else 0  # 计算物品总数
        self.interactions = np.array(self.interactions, dtype=np.int32)  # 转为NumPy数组，便于后续索引
        print(f"加载数据 {self.split} 数据: {len(self.interactions)} 交互记录, "  # 打印加载统计信息
              f"{self.num_users} 用户, {self.num_items} 物品")

    def _load_test_dict(self):  # 私有方法：仅测试集使用，加载每个用户的真实喜好物品列表
        self.test_dict = {}  # 初始化空字典
        with open(os.path.join(self.path, 'test.txt'), 'r') as f:  # 打开测试集文件
            for line in f:  # 逐行读取
                if not line.strip():  # 跳过空行
                    continue
                parts = list(map(int, line.strip().split()))  # 转为整数列表
                if len(parts) < 2:  # 格式检查
                    continue
                user = parts[0]  # 用户ID
                self.test_dict[user] = parts[1:]  # 存储该用户的真实喜好物品列表

    def __len__(self):  # 魔术方法：返回交互记录数量，便于len(dataset)调用
        return len(self.interactions)

    def get_edge_index(self):  # 核心方法：构建LightGCN所需的无向二分图边索引
        users = self.interactions[:, 0]  # 提取所有交互中的用户ID列
        items = self.interactions[:, 1] + self.num_users  # 提取物品ID并偏移到用户ID之后（形成二分图）
        edge_index = np.stack([np.concatenate([users, items]),  # 源节点：先所有用户→物品，再所有物品→用户
                               np.concatenate([items, users])])  # 目标节点：形成双向边
        return torch.LongTensor(edge_index)  # 转为PyTorch LongTensor，shape: [2, 2*边数]

    def get_norm_factors(self):  # 核心方法：计算LightGCN对称归一化所需的 D^{-0.5} 因子
        total_nodes = self.num_users + self.num_items  # 总节点数（用户 + 物品）
        degrees = np.zeros(total_nodes)  # 初始化度数组
        for u, i in self.interactions:  # 遍历所有交互
            degrees[u] += 1  # 用户节点度 +1
            degrees[i + self.num_users] += 1  # 对应物品节点度 +1
        norm = np.power(degrees, -0.5)  # 计算每个节点的 D^{-0.5}
        norm[np.isinf(norm)] = 0.0  # 处理孤立节点（度为0），避免无穷大
        return torch.FloatTensor(norm)  # 转为FloatTensor，shape: [总节点数]

    def sample(self, batch_size):  # BPR负采样方法：为训练提供批量三元组 (u, i+, i-)
        idx = np.random.randint(0, len(self.interactions), batch_size)  # 随机采样batch_size个正样本索引
        users = torch.LongTensor(self.interactions[idx, 0])  # 对应用户
        pos_items = torch.LongTensor(self.interactions[idx, 1])  # 对应正样本物品
        neg_items = torch.randint(0, self.num_items, (batch_size,))  # 为每个正样本随机采样一个负样本物品（均匀分布）
        return users, pos_items, neg_items  # 返回三个tensor


class LightGCN(nn.Module):  # 定义LightGCN模型类，继承torch.nn.Module
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=3):  # 构造函数
        super().__init__()  # 调用父类初始化
        self.num_users = num_users  # 用户数量
        self.num_items = num_items  # 物品数量
        self.embed_dim = embed_dim  # 嵌入向量维度（默认64）
        self.num_layers = num_layers  # 图传播层数（默认3层）

        self.embedding_user = nn.Embedding(num_users, embed_dim)  # 用户初始嵌入层（可学习参数）
        self.embedding_item = nn.Embedding(num_items, embed_dim)  # 物品初始嵌入层（可学习参数）
        nn.init.normal_(self.embedding_user.weight, std=0.1)  # 正态初始化用户嵌入
        nn.init.normal_(self.embedding_item.weight, std=0.1)  # 正态初始化物品嵌入

    def forward(self, edge_index, norm):  # 前向传播函数：核心计算逻辑
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)  # 拼接初始用户和物品嵌入，得到第0层嵌入
        embs = [all_emb]  # 列表收集所有层（包括第0层）的嵌入，用于最终平均

        for _ in range(self.num_layers):  # 进行K层轻量传播
            new_emb = torch.zeros_like(all_emb)  # 初始化当前层嵌入为零
            src_emb = all_emb[edge_index[0]]  # 获取所有边的源节点嵌入
            dst_norm = norm[edge_index[1]]  # 获取目标节点的归一化因子 D^{-0.5}
            src_norm = norm[edge_index[0]]  # 获取源节点的归一化因子 D^{-0.5}
            new_emb.index_add_(0, edge_index[1],  # 向目标节点累加：实现对称归一化聚合
                               src_emb * src_norm.view(-1, 1) * dst_norm.view(-1, 1))  # 逐元素相乘后累加
            embs.append(new_emb)  # 收集当前层嵌入
            all_emb = new_emb  # 更新为下一层的输入

        embs = torch.stack(embs, dim=1)  # 堆叠所有层嵌入，shape: [总节点数, K+1, embed_dim]
        final_emb = embs.mean(dim=1)  # 对所有层取平均，得到最终嵌入（LightGCN关键创新）
        user_emb, item_emb = torch.split(final_emb, [self.num_users, self.num_items])  # 分离用户和物品最终嵌入
        return user_emb, item_emb  # 返回最终嵌入

    def bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb):  # BPR损失函数
        u_emb = user_emb[users]  # 取出当前batch的用户嵌入
        pos_i_emb = item_emb[pos_items]  # 取出正样本物品嵌入
        neg_i_emb = item_emb[neg_items]  # 取出负样本物品嵌入

        pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)  # 计算正样本内积分数
        neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)  # 计算负样本内积分数

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))  # BPR成对排序损失，加1e-8防止log(0)
        return loss  # 返回平均损失


def train_model(model, dataset, args):  # 训练主函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 使用Adam优化器，仅优化初始嵌入参数
    model.train()  # 设置模型为训练模式

    edge_index = dataset.get_edge_index().to(args.device)  # 将边索引移到指定设备（CPU/GPU）
    norm = dataset.get_norm_factors().to(args.device)  # 将归一化因子移到指定设备

    losses = []  # 存储每个epoch的平均损失
    steps_per_epoch = max(1, len(dataset) // args.batch_size)  # 计算每个epoch的batch数量

    for epoch in range(args.epochs):  # 遍历所有epoch
        total_loss = 0.0  # 本epoch累计损失
        for _ in range(steps_per_epoch):  # 遍历本epoch所有batch
            users, pos, neg = dataset.sample(args.batch_size)  # BPR负采样得到batch数据
            users, pos, neg = users.to(args.device), pos.to(args.device), neg.to(args.device)  # 数据移到设备

            user_emb, item_emb = model(edge_index, norm)  # 前向传播：全图计算最终嵌入
            loss = model.bpr_loss(users, pos, neg, user_emb, item_emb)  # 计算BPR损失

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新

            total_loss += loss.item()  # 累计损失值

        avg_loss = total_loss / steps_per_epoch  # 计算本epoch平均损失
        losses.append(avg_loss)  # 记录到列表
        print(f'Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f}')  # 打印训练进度

    plt.figure(figsize=(8, 5))  # 创建绘图窗口
    plt.plot(range(1, args.epochs + 1), losses)  # 绘制损失曲线
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('BPR Loss')  # y轴标签
    plt.title('Training Loss Curve (Amazon-book Reduced)')  # 图标题
    plt.grid(True)  # 显示网格
    plt.savefig('loss_curve.png')  # 保存图像文件
    plt.show()  # 显示图像

    torch.save(model.state_dict(), 'lightgcn_amazon_model.pth')  # 保存模型权重参数
    print("已将模型保存为 'lightgcn_amazon_model.pth'")  # 打印保存成功提示

    return model, edge_index, norm  # 返回训练好的模型和图结构


def evaluate(model, test_dataset, edge_index, norm, args, K=20):  # 评估函数：计算Recall@20
    model.eval()  # 设置模型为评估模式（关闭dropout等）
    with torch.no_grad():  # 关闭梯度计算，节省内存
        user_emb, item_emb = model(edge_index, norm)  # 获取所有节点的最终嵌入

        recalls = []  # 存储每个用户的Recall值
        for user, gt_items in test_dataset.test_dict.items():  # 遍历测试集所有用户
            if len(gt_items) == 0 or user >= model.num_users:  # 跳过无效用户
                continue
            ratings = torch.matmul(user_emb[user], item_emb.t())  # 计算该用户对所有物品的评分（内积）
            _, topk = torch.topk(ratings, K + len(gt_items))  # 取Top-K+真实物品数（避免遗漏）
            topk_items = topk.cpu().numpy()  # 转为numpy数组

            hits = len(set(topk_items[:K]) & set(gt_items))  # 计算前K个推荐中命中真实喜好的数量
            recalls.append(hits / len(gt_items))  # 计算该用户的Recall

        print(f'Recall@{K}: {np.mean(recalls):.4f}|{K}本推荐中命中用户真实喜欢的概率 ')  # 打印平均Recall


def generate_knowledge_graph_with_neighbors(user, topk_items, gt_items=None,
                                            user_emb=None, item_emb=None,
                                            train_dataset=None, edge_index=None, norm=None,
                                            model=None, K=20, num_neighbors=5):
    G = nx.Graph()

    target_user_label = f'User_{user} (Target)'
    G.add_node(target_user_label, type='user', color='blue', size=1200)

    # 1. 添加推荐图书（绿色）
    for rank, item in enumerate(topk_items):
        label = f'Book_{item} (Rank {rank + 1})'
        G.add_node(label, type='item', color='green', size=900)
        G.add_edge(target_user_label, label, relation='recommended', style='solid', width=2.0, color='green')

    # 2. 添加Ground Truth图书（红色）
    if gt_items:
        for item in set(gt_items):  # 去重
            label = f'Book_{item} (GT)'
            if label not in G:
                G.add_node(label, type='item', color='red', size=900)
            G.add_edge(target_user_label, label, relation='ground_truth', style='dashed', width=2.5, color='red')

    # 3. 找出与目标用户嵌入最相似的 num_neighbors 个其他用户（排除自己）
    if user_emb is not None:
        target_emb = user_emb[user].unsqueeze(0)  # [1, dim]
        # 计算与所有其他用户的余弦相似度
        all_user_emb = user_emb
        sims = torch.nn.functional.cosine_similarity(target_emb, all_user_emb, dim=1)
        sims[user] = -1  # 排除自己
        top_sim_values, top_sim_users = torch.topk(sims, min(num_neighbors, len(sims) - 1))

        print(f"为用户 {user} 找到 {len(top_sim_users)} 个最相似的邻居用户（相似度前{num_neighbors}）")

        # 4. 添加这些相似用户（橙色）和他们喜欢的图书
        for sim_val, sim_user in zip(top_sim_values, top_sim_users):
            sim_user = sim_user.item()
            sim_val = sim_val.item()
            sim_user_label = f'User_{sim_user} (sim {sim_val:.3f})'
            G.add_node(sim_user_label, type='similar_user', color='orange', size=1000)
            G.add_edge(target_user_label, sim_user_label,
                       relation=f'similar user ({sim_val:.3f})',
                       style='solid', width=2.0, color='orange')

            # 获取这个相似用户在训练集中喜欢的图书（从train_dataset.interactions）
            user_items = train_dataset.interactions[train_dataset.interactions[:, 0] == sim_user][:, 1]
            for item in user_items[:10]:  # 每个邻居用户显示10本书
                item_label = f'Book_{item}'
                if item_label not in G:
                    G.add_node(item_label, type='item', color='lightblue', size=700)
                G.add_edge(sim_user_label, item_label, relation='liked by neighbor', style='dotted', width=1.5,
                           color='gray')

                # 如果这个书也在推荐列表中，高亮连接
                if item in topk_items:
                    rank = topk_items.index(item) + 1
                    G.add_edge(target_user_label, item_label,
                               relation='recommended via neighbor', style='solid', width=4.0, color='purple')

    # 5. 添加推荐图书之间的高阶相似性（紫色点线，同之前）
    if item_emb is not None and len(topk_items) > 1:
        topk_tensor = torch.LongTensor(topk_items).to(item_emb.device)
        topk_emb = item_emb[topk_tensor]
        sim_matrix = torch.nn.functional.cosine_similarity(topk_emb.unsqueeze(1), topk_emb.unsqueeze(0), dim=-1)
        threshold = 0.3
        added = 0
        for i in range(len(topk_items)):
            for j in range(i + 1, len(topk_items)):
                if sim_matrix[i][j] > threshold:
                    src = f'Book_{topk_items[i]} (Rank {i + 1})'
                    dst = f'Book_{topk_items[j]} (Rank {j + 1})'
                    G.add_edge(src, dst, relation=f'high-order sim ({sim_matrix[i][j]:.2f})',
                               style='dotted', width=1.8, color='purple')
                    added += 1
        print(f"添加了 {added} 条推荐图书间的高阶相似边")

    plt.figure(figsize=(24, 18))

    pos = nx.spring_layout(G, k=1.0, iterations=100, seed=42)

    # 节点颜色映射
    color_map = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', '')
        if node_type == 'user':
            color_map.append('blue')
        elif node_type == 'similar_user':
            color_map.append('orange')
        elif 'GT' in node:
            color_map.append('red')
        elif 'Rank' in node:
            color_map.append('green')
        else:
            color_map.append('lightblue')

    node_sizes = [G.nodes[n].get('size', 700) for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # 不同类型边分开画
    edges_by_style = {}
    for u, v, d in G.edges(data=True):
        style = d.get('style', 'solid')
        color = d.get('color', 'black')
        width = d.get('width', 1.0)
        key = (style, color, width)
        if key not in edges_by_style:
            edges_by_style[key] = []
        edges_by_style[key].append((u, v))

    for (style, color, width), edgelist in edges_by_style.items():
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, style=style, edge_color=color, width=width)

    edge_labels = {}
    for u, v, d in G.edges(data=True):
        rel = d['relation']
        if 'similar user' in rel or 'high-order sim' in rel:
            edge_labels[(u, v)] = rel.split('(')[0] if '(' in rel else rel

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title(f'GNN Recommendation Knowledge Graph (Target User {user})\n'
              '蓝色: 目标用户 | 橙色: 相似邻居用户 | 绿色: Top推荐图书 | 红色: 真实喜好 | 浅蓝: 邻居用户喜欢的书',
              fontsize=22, pad=40)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(f'gnn_full_kg_user_{user}.png', dpi=300, bbox_inches='tight')
    plt.show()

def user_interface(model, edge_index, norm, test_dict, train_dataset, args, K=20):  # 交互式推荐界面函数
    print(f"输入用户ID (0 ~ {model.num_users - 1}), 获取Top-{K}推荐书籍ID。输入 'exit' 退出。")  # 打印使用提示

    while True:  # 进入无限循环等待用户输入
        inp = input("\n用户ID: ").strip()  # 读取用户输入并去除首尾空格
        if inp.lower() == 'exit':  # 如果输入exit
            break  # 退出循环
        try:
            user = int(inp)  # 尝试转为整数
            if 0 <= user < model.num_users:  # 检查用户ID是否在有效范围内
                model.eval()  # 设置为评估模式
                with torch.no_grad():  # 关闭梯度
                    user_emb, item_emb = model(edge_index, norm)  # 计算最终嵌入
                    ratings = torch.matmul(user_emb[user], item_emb.t())  # 计算评分
                    _, topk = torch.topk(ratings, K)  # 取Top-K物品
                    topk_items = topk.cpu().numpy().tolist()  # 转为Python列表
                    print(f"用户 {user} 的 Top-{K} 推荐书籍ID: {topk_items}")  # 打印推荐结果

                gt_items = test_dict.get(user, [])  # 获取该用户的真实喜好（测试集）
                generate_knowledge_graph_with_neighbors(  # 调用知识图谱生成函数（不注释）
                    user=user,
                    topk_items=topk_items,
                    gt_items=gt_items,
                    user_emb=user_emb,
                    item_emb=item_emb,
                    train_dataset=train_dataset,
                    edge_index=edge_index,
                    norm=norm,
                    model=model,
                    K=K,
                    num_neighbors=8
                    )
            else:
                print("用户ID超出范围")  # ID无效提示
        except ValueError:  # 输入非数字
            print("请输入有效数字或 'exit'")  # 错误提示


def main_menu(args):  # 主菜单函数：程序入口逻辑
    model_path = 'lightgcn_amazon_model.pth'  # 模型保存路径

    print("\n基于GNN的图书推荐系统")  # 打印系统标题
    print("1. 训练新模型")  # 菜单选项1
    print("2. 加载现有模型并检测结果")  # 菜单选项2
    print("3. 退出")  # 菜单选项3

    choice = input("请选择 (1/2/3): ").strip()  # 读取用户选择

    train_dataset = BasicDataset(path='./data/amazon-book', split='train', max_interactions=100000)  # 加载训练集（限制10万条）
    test_dataset = BasicDataset(path='./data/amazon-book', split='test',max_interactions=50000)  # 加载测试集（限制5万条）

    model = LightGCN(train_dataset.num_users, train_dataset.num_items, args.embed_dim, args.num_layers).to(args.device)  # 创建模型实例并移到设备

    if choice == '1':  # 选择训练新模型
        print("开始训练新模型...")
        model, edge_index, norm = train_model(model, train_dataset, args)  # 调用训练函数
    elif choice == '2':  # 选择加载已有模型
        if os.path.exists(model_path):  # 检查模型文件是否存在
            print("加载现有模型...")
            model.load_state_dict(torch.load(model_path, map_location=args.device))  # 加载权重
            edge_index = train_dataset.get_edge_index().to(args.device)  # 重新计算图结构
            norm = train_dataset.get_norm_factors().to(args.device)
        else:
            print("未找到模型文件，请先训练模型。")
            return  # 退出函数
    elif choice == '3':  # 选择退出
        return  # 直接返回
    else:
        print("无效选择")  # 非法输入
        return

    print("\n检测模型性能...")  # 打印评估提示
    evaluate(model, test_dataset, edge_index, norm, args)  # 调用评估函数

    user_interface(model, edge_index, norm, test_dataset.test_dict, train_dataset, args)  # 进入交互推荐模式


if __name__ == '__main__':  # 程序入口判断
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--batch_size', type=int, default=4096)  # 添加batch_size参数
    parser.add_argument('--lr', type=float, default=0.001)  # 学习率
    parser.add_argument('--embed_dim', type=int, default=64)  # 嵌入维度
    parser.add_argument('--epochs', type=int, default=100)  # 训练轮数
    parser.add_argument('--num_layers', type=int, default=3)  # 传播层数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择
    args = parser.parse_args()  # 解析命令行参数

    main_menu(args)  # 调用主菜单启动程序
