# GNN_Book_Recommendation
A GNN-based book recommendation system models users, books, and their interactions as a graph, leveraging graph neural networks to capture complex, high-order relationships for more accurate and personalized recommendations.
## Dataset

This project utilizes the **Amazon Book Dataset**. The raw data consists of the following files:

- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

---

## Dataset Preview

Before diving into model training, it is highly recommended to get familiar with the data distribution and content.  
I have written a utility script for this purpose:
You can run `datasetpreview.py` to quickly **preview the dataset details**, including data volume, label mapping, and sample structures.

---

## Model Architecture

After becoming familiar with the dataset, you can start training your own recommendation engine.  

This project adopts **LightGCN**, a lightweight yet powerful variant of Graph Neural Networks (GNN). By removing unnecessary feature transformations and non-linearities, LightGCN efficiently captures high-order collaborative filtering signals between users and books.

---

## Results & Visualization

The trained recommendation model successfully generates personalized suggestions, providing **the top 10 books with the highest confidence scores** for every user.

To make the results intuitive, I have applied visual analytics to the recommendation outputs. You can check out the following visualization materials included in the repository:

## About Graph Neural Networks (GNN)
## 1. Graph Construction (Data Modeling)

The foundation of the system is how we define the graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.

| Element | Description |
| :--- | :--- |
| **Nodes ($\mathcal{V}$)** | Two main types: **User nodes** ($U$) and **Book nodes** ($B$). Optionally add Knowledge Graph entities (Author, Publisher, Genre). |
| **Edges ($\mathcal{E}$)** | Interaction behaviors: `rate`, `purchase`, `add_to_cart`, `like`, `review`. |
| **Node Features** | User: age, location; Book: title/blurb embeddings (BERT), price, avg. rating. |
| **Edge Weights** | Implicit (0/1) or explicit (rating score, confidence level). |

Typically modeled as a **Bipartite Graph**, where edges only exist between $U$ and $B$.

---

## 2. GNN Architecture Design

### 2.1 Core Idea: Message Passing
GNNs learn node representations by iteratively aggregating feature information from neighboring nodes.

$$
\mathbf{h}_v^{(l+1)} = \sigma \left( \mathbf{W}^{(l)} \cdot \text{AGG}\left(\{\mathbf{h}_u^{(l)} : u \in \mathcal{N}(v)\}\right) \right)
$$

### 2.2 Recommended Backbones
For recommendation scenarios, simpler and more efficient models usually outperform complex ones:

1.  **LightGCN (Highly Recommended)**
    *   Removes feature transformation matrices and non-linear activation functions.
    *   Purely uses neighborhood aggregation to refine embeddings—lightweight and state-of-the-art for collaborative filtering.
2.  **NGCF (Neural Graph Collaborative Filtering)**
    *   Exploits high-order connectivities in user-item bipartite graphs explicitly.
3.  **GraphSAGE / GAT**
    *   Useful when incorporating rich side information (e.g., book abstracts) and needing attention mechanisms to weigh neighbor importance.

---

## 3. Training & Prediction Pipeline

### Step 1: Embedding Propagation
Stack $K$ layers of graph convolution to capture $K$-order connectivity (e.g., "Users who read this also read that").

### Step 2: Representation Fusion
Combine embeddings from all layers (usually via weighted sum or concatenation):

$$
\mathbf{e}_u = \frac{1}{K+1} \sum_{k=0}^{K} \mathbf{e}_u^{(k)}
$$

### Step 3: Interaction Prediction
Calculate the matching score between user $u$ and book $i$:

$$
\hat{y}_{ui} = \mathbf{e}_u^\top \mathbf{e}_i
$$

### Step 4: Optimization Objective
Use **BPR (Bayesian Personalized Ranking)** loss to ensure the predicted score of a positively interacted book is higher than an unobserved one:

$$
\mathcal{L}_{BPR} = -\sum_{(u,i,j) \in \mathcal{O}} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda\|\Theta\|^2
$$

---
