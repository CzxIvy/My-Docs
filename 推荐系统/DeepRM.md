## DeepRM

![../../_images/deepfm_architecture.png](D:\files\研一\推荐系统\assets\deepfm_architecture.png)

DeepFM 是目前推荐系统中最经典的 **End-to-End** 模型之一。它最美妙的设计在于：**FM 部分（负责低阶特征组合）和 DNN 部分（负责高阶特征组合）共享同一个 Embedding 层**，从而实现了无需任何人工特征工程即可同时学习低阶和高阶交互。

我将通过一个完整的**代码架构示例**，配合**数据流跟踪（Step-by-Step Trace）**来为您解析 DeepFM。

------

### 1. DeepFM 核心代码架构 (TensorFlow/Keras 风格)

这段代码展示了 DeepFM 的逻辑骨架。为了清晰，我省略了部分繁琐的定义，着重展示数据流向。

Python

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class DeepFM(Model):
    def __init__(self, num_features, embedding_size, dnn_units=[64, 32]):
        super(DeepFM, self).__init__()
        self.embedding_size = embedding_size
        self.num_features = num_features
        
        # --- 1. 共享 Embedding 层 (核心) ---
        # 用于二阶交叉(FM) 和 高阶交叉(DNN)
        self.embedding_layer = layers.Embedding(num_features, embedding_size)
        
        # --- 2. FM 组件 ---
        # 2.1 一阶线性项 (Linear Part): 每个特征对应一个标量权重 w
        self.linear_layer = layers.Embedding(num_features, 1)
        # 2.2 二阶交叉项: 不需要额外参数，直接复用 embedding_layer 计算
        
        # --- 3. DNN 组件 ---
        self.dnn_layers = [layers.Dense(u, activation='relu') for u in dnn_units]
        self.dnn_output_layer = layers.Dense(1) # 输出标量
        
        # --- 4. 最终输出 ---
        self.final_activation = layers.Activation('sigmoid')

    def call(self, inputs):
        # inputs: [Batch_Size, Field_Num] (稀疏特征的索引)
        
        # === Part 1: 共享 Embedding 查表 ===
        # embeddings: [B, F, D]
        embeddings = self.embedding_layer(inputs)
        
        # === Part 2: FM 组件计算 ===
        # A. 一阶项 (Linear): [B, F, 1] -> Sum -> [B, 1]
        linear_part = tf.reduce_sum(self.linear_layer(inputs), axis=1)
        
        # B. 二阶项 (Second Order): 使用 FM 的核心优化公式
        # Sum_Square: (∑v)^2
        sum_square = tf.square(tf.reduce_sum(embeddings, axis=1)) 
        # Square_Sum: ∑(v^2)
        square_sum = tf.reduce_sum(tf.square(embeddings), axis=1)
        # 0.5 * ((∑v)^2 - ∑v^2) -> Sum over embedding dim -> [B, 1]
        second_order_part = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1, keepdims=True)
        
        # === Part 3: DNN 组件计算 ===
        # Flatten: [B, F, D] -> [B, F*D]
        dnn_input = layers.Flatten()(embeddings)
        
        dnn_out = dnn_input
        for layer in self.dnn_layers:
            dnn_out = layer(dnn_out)
        # DNN Output: [B, 1]
        dnn_logit = self.dnn_output_layer(dnn_out)
        
        # === Part 4: 结果融合 ===
        # Logits = Linear + FM_Interaction + DNN
        total_logit = linear_part + second_order_part + dnn_logit
        output = self.final_activation(total_logit)
        
        return output
```

------

### 2. 步骤跟踪 (Step-by-Step Trace)

假设我们有以下配置：

- **Batch Size ($B$)**: 2 (两条样本)
- **Field Num ($F$)**: 3 (例如：User_ID, Item_ID, City)
- **Embedding Size ($D$)**: 4
- **DNN 结构**: [8, 1]

#### 输入数据

输入 `inputs` 是离散特征的索引 ID。

- 维度: `[2, 3]`
- 数值示例: `[[10, 5, 2], [3, 5, 8]]`

------

#### 步骤 1: 共享 Embedding 查表 (The Shared Base)

DeepFM 的精髓：FM 和 DNN 共用这一步的结果。

- **操作**: 根据索引查表。
- **计算**: `embeddings = embedding_layer(inputs)`
- **维度变化**: `[2, 3]` $\to$ **`[2, 3, 4]`**
- **物理含义**: 每个样本的 3 个特征都被映射成了 4 维的稠密向量。

------

#### 步骤 2: FM 组件计算 (Wide Part)

FM 负责捕获低阶特征（一阶）和特征间的共现关系（二阶）。

**A. 一阶项 (Linear Term)**

- **操作**: 查 `linear_layer` 表（每个特征对应一个标量 $w$），然后求和。
- **维度变化**: `[2, 3]` $\to$ `[2, 3, 1]` (查表) $\to$ `tf.reduce_sum` $\to$ **`[2, 1]`**
- **结果**: `linear_part` (包含两个样本各自的线性得分)。

**B. 二阶项 (Interaction Term)**

- **输入**: 使用步骤 1 得到的 `embeddings` (`[2, 3, 4]`)。
- **计算 (核心公式)**: $0.5 \sum (\sum v)^2 - \sum v^2$
  1. **Sum**: 沿 Feature 轴(axis=1)求和 $\to$ `[2, 1, 4]`
  2. **Square**: 元素平方 $\to$ `[2, 1, 4]`
  3. **Minus**: 相减 $\to$ `[2, 1, 4]`
  4. **Reduce Sum**: 沿 Embedding 轴(axis=2)求和 $\to$ **`[2, 1]`**
- **结果**: `second_order_part` (包含两个样本的二阶交叉得分)。

------

#### 步骤 3: DNN 组件计算 (Deep Part)

DNN 负责捕获高阶、非线性的特征组合。

- **输入**: 依然是步骤 1 得到的 `embeddings` (`[2, 3, 4]`)。
- **操作 1: Flatten (展平)**
  - 将每个样本的所有特征向量拼成一个长向量。
  - 维度变化: `[2, 3, 4]` $\to$ **`[2, 12]`** (因为 $3 \times 4 = 12$)。
- **操作 2: Dense Layer 1 (Hidden Layer)**
  - 假设隐藏单元数为 8。
  - 计算: `ReLU(Wx + b)`
  - 维度变化: `[2, 12]` $\to$ **`[2, 8]`**。
- **操作 3: Output Layer**
  - 映射到标量输出。
  - 维度变化: `[2, 8]` $\to$ **`[2, 1]`**。
- **结果**: `dnn_logit`。

------

#### 步骤 4: 结果融合 (Fusion)

将三部分的 Logits 直接相加。

- 公式:

  

  $$\text{Total} = \underbrace{\text{Linear}}_{\text{Linear}} + \underbrace{\text{FM}}_{\text{2nd-Order}} + \underbrace{\text{DNN}}_{\text{High-Order}}$$

- 计算:

  linear_part ([2, 1]) + second_order_part ([2, 1]) + dnn_logit ([2, 1])

- **维度**: **`[2, 1]`**

- 最终激活:

  Sigmoid(Total) $\to$ 输出概率值 (0~1)。

------

### DeepFM 为什么强？(总结)

通过上面的步骤跟踪，我们可以清晰地看到 DeepFM 的优势：

1. **没有信息的浪费**: 同一个输入（Sparse Features），经过同一个 Embedding 层，产生的信息既被用来计算**显式的二阶共现**（FM 侧），也被用来计算**隐式的高阶模式**（DNN 侧）。
2. **端到端训练**: Embedding 层的参数 $\mathbf{V}$ 会同时接收来自 FM 侧（要求 $\mathbf{v}_i$ 能很好地表达二阶关系）和 DNN 侧（要求 $\mathbf{v}_i$ 能很好地支持高阶推理）的梯度更新。这使得特征表示非常准确。
3. **广度与深度的结合**:
   - **FM** 保证了记忆能力（Memorization），对于像“啤酒与尿布”这种强关联规则，FM 能直接通过内积捕获。
   - **DNN** 保证了泛化能力（Generalization），对于从未出现过的特征组合，DNN 可以通过抽象的非线性变换进行推断。