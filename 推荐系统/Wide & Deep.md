## Wide & Deep

![img](D:\files\研一\推荐系统\assets\wide_and_deep-1765804777898-3.png)

### Wide 部分

Wide部分本质上是一个广义线性模型，比如逻辑回归。它的优势在于结构简单、可解释更强，并且能高效地“记忆”那些显而易见的关联规则。其数学表达形式如下：
$$
y=w^Tx+b
$$
其中，y是预测值，$$w^T$$是模型权重，$$x$$是特征向量，b是偏置项。

Wide部分的关键在于其输入的特征向量。它不仅包含原始特征，更重要的是包含了大量**人工设计的交叉特征（Cross-product Features）**。交叉特征可以将多个独立的特征组合成一个新的特征，用于捕捉特定的共现模式。例如，在应用商店的推荐场景中，我们可以创建一个交叉特征`AND(installed_app=photo_editor, impression_app=filter_pack)`，它代表用户已经安装了“照片编辑器”应用，并且现在看到了“滤镜包”应用的推荐。

通过这种方式，Wide部分能够直接、快速地学习到“照片编辑器用户对滤镜包应用有更高的安装意愿”这类强关联规则，正是“记忆能力”的直接体现。

#### 核心代码

```python
# 遍历所有需要交叉的特征对
for i in range(len(cross_feature_columns)):
    for j in range(i + 1, len(cross_feature_columns)):
        fc_i = cross_feature_columns[i]
        fc_j = cross_feature_columns[j]

        # 获取两个特征的输入
        feat_i = input_layer_dict[fc_i.name]  # [B, 1]
        feat_j = input_layer_dict[fc_j.name]  # [B, 1]

        # 为每个特征对创建独立的权重表
        cross_vocab_size = fc_i.vocab_size * fc_j.vocab_size
        cross_embedding = Embedding(
            input_dim=cross_vocab_size,
            output_dim=1,  # 标量权重，直接记住这对特征的影响
            name=f"cross_{fc_i.name}_{fc_j.name}"
        )

        # 将特征对组合成单一索引并查找权重
        combined_index = feat_i * fc_j.vocab_size + feat_j
        cross_weight = cross_embedding(combined_index)  # 查表得到这对特征的权重
        cross_weights.append(cross_weight)

# 所有交叉特征权重相加
cross_logits = tf.add_n(cross_weights)
```

这段代码的设计体现了Wide部分的本质：为每个特征组合分配一个独立的权重，通过查表操作直接“记住”历史数据中的共现模式。

#### 代码示例与步骤跟踪

假设：有三个特征需要交叉：**性别 (Gender)**、**商品类别 (Category)** 和 **城市 (City)**。

前提定义：

```python
# 假设 cross_feature_columns 结构如下：
cross_feature_columns = [
    FeatureConfig(name="Gender", vocab_size=3),     # 索引 0: GENDER
    FeatureConfig(name="Category", vocab_size=100), # 索引 1: CATEGORY
    FeatureConfig(name="City", vocab_size=500)      # 索引 2: CITY
]

# 假设输入张量 (Batch_Size=2)
input_layer_dict = {
    "Gender": tf.constant([[1], [0]]),   # 样本1: 男(1), 样本2: 女(0)
    "Category": tf.constant([[5], [99]]), # 样本1: Cat_5, 样本2: Cat_99
    "City": tf.constant([[10], [250]])    # 样本1: City_10, 样本2: City_250
}

cross_weights = [] # 用于收集所有交叉项权重的列表
```

##### 步骤 1: 遍历所有需要交叉的特征对

| **(i, j)** | **循环处理的特征对** |
| ---------- | :------------------: |
| **(0, 1)** |  Gender & Category   |
| **(0, 2)** |    Gender & City     |
| **(1, 2)** |   Category & City    |

我们将以 **(0, 1) Gender & Category** 为例进行详细跟踪。

##### 步骤 2: 获取特征配置和输入张量 (i=0, j=1)

```python
		fc_i = cross_feature_columns[0]  # Gender (vocab_size=3)
        fc_j = cross_feature_columns[1]  # Category (vocab_size=100)

        # 获取两个特征的输入
        feat_i = input_layer_dict[fc_i.name]  # [B, 1]
        feat_j = input_layer_dict[fc_j.name]  # [B, 1]
```

示例值：

- $fc\_i.name = \text{"Gender"}$, $fc\_i.vocab\_size = 3$
- $fc\_j.name = \text{"Category"}$, $fc\_j.vocab\_size = 100$
- $\text{feat}_i = \begin{bmatrix} [1] \\ [0] \end{bmatrix}$ (Gender)
- $\text{feat}_j = \begin{bmatrix} [5] \\ [99] \end{bmatrix}$ (Category)

##### 步骤 3: 创建交叉特征权重表 (Embedding)

```python
		# cross_vocab_size = 3 * 100 = 300
        cross_vocab_size = fc_i.vocab_size * fc_j.vocab_size
        
        cross_embedding = Embedding(
            input_dim=cross_vocab_size,  # 300
            output_dim=1,
            name=f"cross_Gender_Category"
        )
```

**示例值：**

- 内存中创建了一个包含 300 个可训练参数的查找表 `cross_Gender_Category`。

##### 步骤 4: 组合索引并查找权重

```python
		# combined_index = feat_i * fc_j.vocab_size + feat_j
        combined_index = feat_i * 100 + feat_j
        
        cross_weight = cross_embedding(combined_index)  # 查表得到这对特征的权重
        cross_weights.append(cross_weight)
```

**示例值：**

1. combined_index 计算:

   

   $$\begin{bmatrix} [1] \\ [0] \end{bmatrix} \times 100 + \begin{bmatrix} [5] \\ [99] \end{bmatrix} = \begin{bmatrix} [100] \\ [0] \end{bmatrix} + \begin{bmatrix} [5] \\ [99] \end{bmatrix} = \begin{bmatrix} [105] \\ [99] \end{bmatrix}$$

   

   combined_index 张量: $\begin{bmatrix} [105] \\ [99] \end{bmatrix}$

2. **`cross_embedding` 查找:**

   - 样本 1: 查找索引 105，得到权重 $\mathbf{w}_{105}$ (e.g., $+0.5$)
   - 样本 2: 查找索引 99，得到权重 $\mathbf{w}_{99}$ (e.g., $-0.2$)

   `cross_weight_GC` 张量: $\begin{bmatrix} +0.5 \\ -0.2 \end{bmatrix}$

3. cross_weights 列表:

   cross_weights 现在包含: $\left[ \begin{bmatrix} +0.5 \\ -0.2 \end{bmatrix} \right]$ (Gender & Category 的贡献)

##### 步骤 5: 循环继续 (略)

- **(0, 2) Gender & City:** 计算 $K = \text{Gender} \times 500 + \text{City}$，查表得到 $\text{cross\_weight}_{\text{GCity}}$ (e.g., $\begin{bmatrix} +0.1 \\ -0.15 \end{bmatrix}$)
- **(1, 2) Category & City:** 计算 $K = \text{Category} \times 500 + \text{City}$，查表得到 $\text{cross\_weight}_{\text{CCity}}$ (e.g., $\begin{bmatrix} +0.05 \\ +0.0 \end{bmatrix}$)

cross_weights 列表现在包含三个张量：
$$
\text{cross\_weights} = \left[ \begin{bmatrix} +0.5 \\ -0.2 \end{bmatrix}, \begin{bmatrix} +0.1 \\ -0.15 \end{bmatrix}, \begin{bmatrix} +0.05 \\ +0.0 \end{bmatrix} \right]
$$

##### 步骤 6: 汇总所有交叉特征权重

```python
# 所有交叉特征权重相加
cross_logits = tf.add_n(cross_weights)
```

**示例值：**

$$\text{cross\_logits} = \begin{bmatrix} +0.5 \\ -0.2 \end{bmatrix} + \begin{bmatrix} +0.1 \\ -0.15 \end{bmatrix} + \begin{bmatrix} +0.05 \\ +0.0 \end{bmatrix} = \begin{bmatrix} 0.5 + 0.1 + 0.05 \\ -0.2 - 0.15 + 0.0 \end{bmatrix} = \begin{bmatrix} +0.65 \\ -0.35 \end{bmatrix}$$

**最终输出：**

- $\text{cross\_logits}$ 张量 $\begin{bmatrix} +0.65 \\ -0.35 \end{bmatrix}$
  - +0.65 是样本 1（男/Cat_5/City_10）的所有二阶交叉项的总贡献。
  - -0.35 是样本 2（女/Cat_99/City_250）的所有二阶交叉项的总贡献。

这段代码高效地为 Wide 模型计算了所有二阶特征交叉项的线性贡献。

### Deep 部分

Deep部分是一个标准的前馈神经网络（DNN），它负责模型的“泛化能力”。与Wide部分依赖人工特征工程不同，Deep部分可以自动学习特征之间的高阶、非线性关系。

它的工作流程如下：首先，对于那些高维稀疏的类别特征（如用户ID、物品ID），通过一个**嵌入层（Embedding Layer）**将它们映射为低维、稠密的向量。这些嵌入向量能够捕捉到特征的潜在语义信息，是实现泛化的基础。例如，《流浪地球》和《三体》的电影ID在嵌入空间中的距离，可能会比《流浪地球》和《熊出没》更近。

随后，这些嵌入向量与其他数值特征拼接在一起，被送入多层神经网络中进行前向传播：
$$
a^{(l+1)}=f(W^{(l)}a^{(l)}+b{(l)})
$$
其中，$$a^{(l+1)}$$是第层的激活值，$$W^{(l)}$$和$$b{(l)}$$是该层的权重和偏置，$$f$$是激活函数（如ReLU）。通过逐层抽象，DNN能够发掘出数据中隐藏的复杂模式，从而对未曾见过的特征组合也能做出合理的预测。

#### 核心代码

Deep部分的实现分为两个关键步骤：首先将类别特征映射为稠密向量，然后通过多层神经网络学习高阶特征交互：

```python
# 1. 特征嵌入：将稀疏的类别特征转换为稠密向量
group_feature_dict = {}
for group_name, _ in group_embedding_feature_dict.items():
    group_feature_dict[group_name] = concat_group_embedding(
        group_embedding_feature_dict, group_name, axis=1, flatten=True
    )  # B x (N * D) - 拼接所有特征的嵌入向量

# 2. 深度神经网络：逐层学习特征的非线性组合
deep_logits = []
for group_name, group_feature in group_feature_dict.items():
    # 构建多层神经网络
    deep_out = DNNs(
        units=dnn_units,  # 例如 [64, 32]
        activation="relu",  # ReLU激活函数
        dropout_rate=dnn_dropout_rate
    )(group_feature)

    # 输出层：将深度特征映射为预测分数
    deep_logit = tf.keras.layers.Dense(1, activation=None)(deep_out)
    deep_logits.append(deep_logit)
```

这种设计使得模型能够自动学习特征的语义表示，例如将“物品A”相关的特征映射到向量空间的相近位置，从而实现对未见过的特征组合的泛化预测。

#### 代码示例与步骤跟踪

Deep 部分的核心思想是：

1. **特征嵌入 (Embedding):** 将高维、稀疏的离散特征（如用户ID、商品ID）转换为低维、稠密的连续向量（Embedding Vector）。
2. **特征拼接 (Concatenation):** 将同一组（Group）内的所有特征的 Embedding 向量拼接在一起。
3. **多层感知机 (DNN):** 将拼接后的向量输入到多层全连接网络中，学习特征的深层组合。

前提假设：

| **配置项**               | **示例值**                                    | **说明**                |
| :----------------------- | :-------------------------------------------- | :---------------------- |
| **Batch Size ($B$)**     | 2                                             | 两个样本                |
| **特征分组**             | `User` 组, `Item` 组                          | 两个主要的特征组        |
| **Embedding 维度 ($D$)** | 8                                             | 每个特征嵌入为 8 维向量 |
| **DNN 结构**             | `dnn_units = [64, 32]`                        | 两个隐藏层              |
| **输入特征**             | 假设 `User` 组有两个特征，`Item` 组有三个特征 |                         |

**输入张量示例:**

假设 `group_embedding_feature_dict` 存储了所有特征的嵌入向量。

| **特征组** | **特征**    | **嵌入向量形状** | **示例 (每个 Batch)**          |
| ---------- | ----------- | ---------------- | ------------------------------ |
| **User**   | UserID      | $[B, 1, 8]$      | $\mathbf{E}_{\text{User\_ID}}$ |
|            | User_City   | $[B, 1, 8]$      | $\mathbf{E}_{\text{City}}$     |
| **Item**   | ItemID      | $[B, 1, 8]$      | $\mathbf{E}_{\text{Item\_ID}}$ |
|            | Item_Cate   | $[B, 1, 8]$      | $\mathbf{E}_{\text{Cate}}$     |
|            | Item_Seller | $[B, 1, 8]$      | $\mathbf{E}_{\text{Seller}}$   |

##### 步骤 1: 特征嵌入与拼接 (Embedding & Concatenation)

```python
# 1. 特征嵌入：将稀疏的类别特征转换为稠密向量
group_feature_dict = {}
for group_name, _ in group_embedding_feature_dict.items():
    group_feature_dict[group_name] = concat_group_embedding(
        group_embedding_feature_dict, group_name, axis=1, flatten=True
    )  # B x (N * D) - 拼接所有特征的嵌入向量
```

**目的：** 将同一组内的所有特征的嵌入向量拼接成一个长向量，作为 DNN 的输入。

**以 `User` 组为例：**

1. 输入特征的嵌入向量列表:
   $$
   \text{User\_Embeddings} = [\mathbf{E}_{\text{User\_ID}}, \mathbf{E}_{\text{City}}]
   $$
   每个向量形状为 $[B, 1, 8]$。

2. **执行 `concat_group_embedding`：**

   - **Concatenation ($\text{axis}=1$):** 将 $N=2$ 个 $[B, 1, 8]$ 的张量在第二维（特征维度）上拼接。得到形状 $[B, 2, 8]$。
   - **Flatten:** 将 $[B, 2, 8]$ 展平为 $[B, 2 \times 8] = [B, 16]$。

3. 结果：
   $$
   \text{group\_feature\_dict}[\text{"User"}] = \text{User Feature Vector (UFV)}
   $$
   形状: $[B, 16]$

**以 `Item` 组为例：**

1. 输入特征的嵌入向量列表:
   $$
   \text{Item\_Embeddings} = [\mathbf{E}_{\text{Item\_ID}}, \mathbf{E}_{\text{Cate}}, \mathbf{E}_{\text{Seller}}]
   $$

2. **执行 `concat_group_embedding`：**

   - **Concatenation:** 将 $N=3$ 个 $[B, 1, 8]$ 的张量拼接，得到形状 $[B, 3, 8]$。
   - **Flatten:** 展平为 $[B, 3 \times 8] = [B, 24]$。

3. 结果：
   $$
   \text{group\_feature\_dict}[\text{"Item"}] = \text{Item Feature Vector (IFV)}
   $$
   形状: $[B, 24]$

##### 步骤 2: 深度神经网络计算 (DNN Calculation)

```python
# 2. 深度神经网络：逐层学习特征的非线性组合
deep_logits = []
for group_name, group_feature in group_feature_dict.items():
    # 构建多层神经网络 (DNNs 是一个预定义的 Keras/TensorFlow 模型)
    deep_out = DNNs(
        units=[64, 32],
        activation="relu",
        dropout_rate=0.1
    )(group_feature)

    # 输出层：将深度特征映射为预测分数
    deep_logit = tf.keras.layers.Dense(1, activation=None)(deep_out)
    deep_logits.append(deep_logit)
```

**目的：** 为每个特征组训练一个独立的 DNN，学习该组内特征的非线性交互，并将结果映射为一个 Logit 预测分数。

###### 2.1 循环处理 `User` 组 ($\text{UFV}$, Shape: $[B, 16]$)

2.1.1 DNN 隐藏层计算

- **输入:** $\text{UFV}$ (Shape: $[B, 16]$)

- 第 1 层 (units=64):
  $$
  \mathbf{h}_1 = \text{ReLU}(\text{UFV} \cdot \mathbf{W}_1 + \mathbf{b}_1)
  $$
  $\mathbf{h}_1$ 形状: $[B, 64]$

- 第 2 层 (units=32):
  $$
  \mathbf{h}_2 = \text{ReLU}(\mathbf{h}_1 \cdot \mathbf{W}_2 + \mathbf{b}_2)
  $$
  $\mathbf{h}_2$ 形状: $[B, 32]$

- `deep_out` (User): $[B, 32]$

2.1.2 输出层计算

- 全连接层 (Dense(1)):

  

  $$\text{deep\_logit}_{\text{User}} = \mathbf{h}_2 \cdot \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}}$$

  

  $\text{deep\_logit}_{\text{User}}$ 形状: $[B, 1]$ (例如 $\begin{bmatrix} +0.1 \\ -0.05 \end{bmatrix}$)

2.1.3 存储结果

- `deep_logits` 列表: $\left[ \begin{bmatrix} +0.1 \\ -0.05 \end{bmatrix} \right]$

###### 2.2 循环处理 `Item` 组 ($\text{IFV}$, Shape: $[B, 24]$)

2.2.1 DNN 隐藏层计算

- **输入:** $\text{IFV}$ (Shape: $[B, 24]$)
- **DNN 结构:** 同样是 $[64, 32]$，但使用的是**另一套独立**的权重 $\mathbf{W}'$ 和 $\mathbf{b}'$。
- `deep_out` (Item): $[B, 32]$

2.2.2 输出层计算

- 全连接层 (Dense(1)):

  

  $$\text{deep\_logit}_{\text{Item}} = \mathbf{h}'_2 \cdot \mathbf{W}'_{\text{out}} + \mathbf{b}'_{\text{out}}$$

  

  $\text{deep\_logit}_{\text{Item}}$ 形状: $[B, 1]$ (例如 $\begin{bmatrix} +0.4 \\ +0.9 \end{bmatrix}$)

2.2.3 存储结果

- `deep_logits` 列表: $\left[ \begin{bmatrix} +0.1 \\ -0.05 \end{bmatrix}, \begin{bmatrix} +0.4 \\ +0.9 \end{bmatrix} \right]$

### 两者结合

Wide & Deep模型通过联合训练，将两部分的输出结合起来进行最终的预测。其预测概率如下：
$$
P(Y=1|\mathbf{x})=\sigma(\mathbf{w}_{wide}^T[\mathbf{x},\phi(\mathbf{x})]+\mathbf{w}_{deep}^Ta^{(lf)}+b)
$$
在这里，$$sigma$$是Sigmoid函数，$$[\mathbf{x},\phi(\mathbf{x})]$$代表Wide部分的输入（包含原始特征和交叉特征），$$a^{(lf)}$$是Deep部分最后一层的输出向量，$$\mathbf{w}_{wide}^T$$，$$\mathbf{w}_{deep}$$和$$b$$是最终预测层的权重和偏置。模型的梯度在反向传播时会同时更新Wide和Deep两部分的所有参数。

一个值得注意的工程细节是，由于两部分处理的特征类型不同，它们通常会采用不同的优化器。

- **Wide部分**的输入特征非常稀疏，常使用带L1正则化的FTRL ([Ferreira and Soares, 2025](https://datawhalechina.github.io/fun-rec/chapter_references/references.html#id46)) 等优化器。L1正则化可以产生稀疏的权重，相当于自动进行特征选择，让模型只“记住”重要的规则。
- **Deep部分**的参数是稠密的，更适合使用像AdaGrad ([Duchi *et al.*, 2011](https://datawhalechina.github.io/fun-rec/chapter_references/references.html#id69)) 或Adam ([Kingma and Ba, 2014](https://datawhalechina.github.io/fun-rec/chapter_references/references.html#id70)) 这样的优化器。

#### 核心代码

联合训练的核心是将Wide和Deep两部分的输出进行融合：

```python
# Wide部分：线性特征 + 交叉特征
linear_logit = get_linear_logits(input_layer_dict, feature_columns)
cross_logit = get_cross_logits(input_layer_dict, feature_columns)

# Deep部分：多个特征组的深度网络输出
deep_logits = []
for group_name, group_feature in group_feature_dict.items():
    deep_out = DNNs(units=dnn_units, activation="relu", dropout_rate=dnn_dropout_rate)(
        group_feature
    )
    deep_logit = tf.keras.layers.Dense(1, activation=None)(deep_out)
    deep_logits.append(deep_logit)

# 联合训练：将Wide和Deep的输出相加
wide_deep_logits = add_tensor_func(deep_logits + [linear_logit, cross_logit])

# 最终预测：通过sigmoid函数输出点击概率
output = tf.keras.layers.Dense(1, activation="sigmoid")(wide_deep_logits)
```

Wide & Deep模型的意义不只是提供了一个新的网络结构，更重要的是给出了一个思路：怎么把记忆能力和泛化能力结合起来。该模型不仅成为了许多推荐业务的基线模型，更为后续精排模型的发展提供了重要的参考。
