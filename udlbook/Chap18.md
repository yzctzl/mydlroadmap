好的，请看我的翻译和解答。

***

# 第十八章
# 扩散模型

第十五章介绍了生成对抗模型，这类模型能生成看似合理的样本，但没有在数据上定义一个概率分布。第十六章讨论了归一化流。这类模型确实定义了这样的一个概率分布，但必须对网络施加架构上的约束；每一层都必须是可逆的，并且其雅可比矩阵的行列式必须易于计算。第十七章介绍了变分自编码器，它同样具有坚实的概率基础，但其似然函数的计算是难解的，必须用一个下界来近似。

本章将介绍**扩散模型 (diffusion models)**。与归一化流一样，它们是概率模型，定义了从潜变量到观测数据的非线性映射，其中两者具有相同的维度。与变分自编码器一样，它们使用基于一个将数据映射**到**潜变量的编码器的下界来近似数据似然。然而，在扩散模型中，这个编码器是**预先确定的**；目标是学习一个作为该过程逆过程的解码器，并可以用它来生成样本。扩散模型易于训练，并且能生成质量非常高的样本，其逼真度超过了GANs所能生成的样本。在阅读本章之前，读者应熟悉变分自编码器（第十七章）。

## 18.1 概述

一个扩散模型由一个**编码器 (encoder)** 和一个**解码器 (decoder)** 组成。编码器接收一个数据样本 $\mathbf{x}$，并通过一系列中间潜变量 $\mathbf{z}_1, \dots, \mathbf{z}_T$ 对其进行映射。解码器则逆转这个过程；它从 $\mathbf{z}_T$ 开始，通过 $\mathbf{z}_{T-1}, \dots, \mathbf{z}_1$ 反向映射，直到最终（重新）创建一个数据点 $\mathbf{x}$。在编码器和解码器中，映射都是**随机的 (stochastic)** 而非确定性的。

编码器是**预先指定的**；它通过将输入与白噪声样本逐渐混合（图18.1）。只要有足够的步骤，最终潜变量的条件分布 $q(\mathbf{z}_T|\mathbf{x})$ 和边缘分布 $q(\mathbf{z}_T)$ 都会变成标准正态分布。由于这个过程是预先指定的，所有学习到的参数都在解码器中。

在解码器中，一系列网络被训练来在每对相邻的潜变量 $\mathbf{z}_t$ 和 $\mathbf{z}_{t-1}$ 之间进行反向映射。损失函数鼓励每个网络都去逆转相应的编码器步骤。其结果是噪声从表示中被逐渐移除，直到一个看似逼真的数据样本保留下来。为了生成一个新的数据样本 $\mathbf{x}$，我们从 $q(\mathbf{z}_T)$ 中抽取一个样本，并将其通过解码器。

在18.2节，我们详细考虑编码器。它的性质并不显而易见，但对于学习算法至关重要。在18.3节，我们讨论解码器。18.4节推导了训练算法，18.5节对其进行了重构以使其更实用。18.6节讨论了实现细节，包括如何使生成以文本提示为条件。

---

> **图 18.1 扩散模型**
> 编码器（前向或扩散过程）将输入 $\mathbf{x}$ 通过一系列潜变量 $\mathbf{z}_1, \dots, \mathbf{z}_T$ 进行映射。这个过程是预先指定的，并逐渐将数据与噪声混合，直到只剩下噪声。解码器（逆向过程）是学习得到的，它将数据通过潜变量反向传递，在每个阶段去除噪声。训练后，通过采样噪声向量 $\mathbf{z}_T$ 并将其通过解码器来生成新样本。

---

## 18.2 编码器 (前向过程)

**扩散 (diffusion)** 或**前向过程 (forward process)**¹（图18.2）通过一系列与 $\mathbf{x}$ 大小相同的中间变量 $\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_T$ 来映射一个数据样本 $\mathbf{x}$，其规则如下：

$$
\begin{aligned}
\mathbf{z}_1 &= \sqrt{1-\beta_1}\mathbf{x} + \sqrt{\beta_1}\boldsymbol{\epsilon}_1 \\
\mathbf{z}_t &= \sqrt{1-\beta_t}\mathbf{z}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_t \quad \forall t \in 2, \dots, T,
\end{aligned}
\tag{18.1}
$$

其中 $\boldsymbol{\epsilon}_t$ 是从标准正态分布中抽取的噪声。第一项衰减了数据以及到目前为止添加的任何噪声，第二项则添加了更多噪声。超参数 $\beta_t \in$ 决定了噪声混合的速度，并被统称为**噪声方案 (noise schedule)**。前向过程也可以等效地写成：

$$
\begin{aligned}
q(\mathbf{z}_1|\mathbf{x}) &= \text{Norm}_{\mathbf{z}_1}[\sqrt{1-\beta_1}\mathbf{x}, \beta_1\mathbf{I}] \\
q(\mathbf{z}_t|\mathbf{z}_{t-1}) &= \text{Norm}_{\mathbf{z}_t}[\sqrt{1-\beta_t}\mathbf{z}_{t-1}, \beta_t\mathbf{I}] \quad \forall t \in \{2, \dots, T\}.
\end{aligned}
\tag{18.2}
$$

这是一个**马尔可夫链 (Markov chain)**，因为 $\mathbf{z}_t$ 的概率完全由前一个变量 $\mathbf{z}_{t-1}$ 的值决定。只要有足够的步骤 $T$，原始数据的全部痕迹都会被移除，而 $q(\mathbf{z}_T|\mathbf{x}) = q(\mathbf{z}_T)$ 会变成一个标准正态分布。² 给定输入 $\mathbf{x}$，所有潜变量 $\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_T$ 的联合分布是：

$$
q(\mathbf{z}_{1\dots T}|\mathbf{x}) = q(\mathbf{z}_1|\mathbf{x}) \prod_{t=2}^T q(\mathbf{z}_t|\mathbf{z}_{t-1}).
\tag{18.3}
$$

> ¹注意，这与归一化流的命名法相反，在归一化流中，逆向映射是从数据移动到潜变量，而前向映射是反向移动。

> ²我们使用 $q(\mathbf{z}_t|\mathbf{z}_{t-1})$ 而不是 $\text{Pr}(\mathbf{z}_t|\mathbf{z}_{t-1})$ 以匹配前一章中VAE编码器的符号。

---

> **图 18.2 前向过程**
> a) 我们考虑一维数据 $\mathbf{x}$，其中 $T=100$ 个潜变量 $\mathbf{z}_1, \dots, \mathbf{z}_{100}$ 并且在所有步骤中 $\beta=0.03$。三个 $\mathbf{x}$ 的值（灰色、青色和橙色）被初始化（顶行）。这些值通过 $\mathbf{z}_1, \dots, \mathbf{z}_{100}$ 传播。在每一步，变量的值通过乘以 $\sqrt{1-\beta}$ 来衰减，并加上均值为零、方差为 $\beta$ 的噪声（方程18.1）。因此，这三个例子嘈杂地通过这些变量传播，并有向零移动的趋势。b) 条件概率 $\text{Pr}(\mathbf{z}_1|\mathbf{x})$ 和 $\text{Pr}(\mathbf{z}_t|\mathbf{z}_{t-1})$ 是正态分布，其均值比当前点略微更接近零，并具有固定的方差 $\beta_t$（方程18.2）。

> **图 18.3 扩散核**
> a) 点 $\mathbf{x}^*=2.0$ 使用方程18.1通过潜变量传播（显示了五条路径，为灰色）。扩散核 $q(\mathbf{z}_t|\mathbf{x}^*)$ 是给定我们从 $\mathbf{x}^*$ 开始时，变量 $\mathbf{z}_t$ 上的概率分布。它可以以封闭形式计算，并且是一个正态分布，其均值向零移动，方差随 $t$ 增加而增加。热图显示了每个变量的 $q(\mathbf{z}_t|\mathbf{x}^*)$。青色线显示了距离均值 $\pm 2$ 个标准差的范围。b) 扩散核 $q(\mathbf{z}_t|\mathbf{x}^*)$ 在 $t=20, 40, 80$ 时被明确地显示出来。在实践中，扩散核允许我们采样一个与给定 $\mathbf{x}^*$ 对应的潜变量 $\mathbf{z}_t$，而无需计算中间变量 $\mathbf{z}_1, \dots, \mathbf{z}_{t-1}$。当 $t$ 变得非常大时，扩散核变成一个标准正态分布。

> **图 18.4 边缘分布**
> a) 给定一个初始密度 $\text{Pr}(\mathbf{x})$（顶行），扩散过程在通过潜变量 $\mathbf{z}_t$ 时逐渐模糊分布，并将其移向一个标准正态分布。每个后续的热图水平线代表一个边缘分布 $q(\mathbf{z}_t)$。b) 顶图显示了初始分布 $\text{Pr}(\mathbf{x})$。另外两个图分别显示了边缘分布 $q(\mathbf{z}_{20})$ 和 $q(\mathbf{z}_{60})$。

---

### 18.2.1 扩散核 $q(\mathbf{z}_t|\mathbf{x})$

为了训练解码器来逆转这个过程，我们在时间 $t$ 对同一个样本 $\mathbf{x}$ 使用多个样本 $\mathbf{z}_t$。然而，当 $t$ 很大时，使用方程18.1顺序地生成这些样本是耗时的。幸运的是，对于 $q(\mathbf{z}_t|\mathbf{x})$ 有一个封闭形式的表达式，它允许我们直接抽取给定初始数据点 $\mathbf{x}$ 的样本 $\mathbf{z}_t$，而无需计算中间变量 $\mathbf{z}_1, \dots, \mathbf{z}_{t-1}$。这被称为**扩散核 (diffusion kernel)**（图18.3）。
为了推导 $q(\mathbf{z}_t|\mathbf{x})$ 的表达式，考虑前向过程的前两个步骤：

$$
\begin{aligned}
\mathbf{z}_1 &= \sqrt{1-\beta_1}\cdot\mathbf{x} + \sqrt{\beta_1}\cdot\boldsymbol{\epsilon}_1 \\
\mathbf{z}_2 &= \sqrt{1-\beta_2}\cdot\mathbf{z}_1 + \sqrt{\beta_2}\cdot\boldsymbol{\epsilon}_2.
\end{aligned}
\tag{18.4}
$$

将第一个方程代入第二个，我们得到：

$$
\mathbf{z}_2 = \sqrt{1-\beta_2}(\sqrt{1-\beta_1}\cdot\mathbf{x} + \sqrt{\beta_1}\cdot\boldsymbol{\epsilon}_1) + \sqrt{\beta_2}\cdot\boldsymbol{\epsilon}_2 \\
= \sqrt{(1-\beta_2)(1-\beta_1)}\mathbf{x} + \sqrt{1-\beta_2}\sqrt{\beta_1}\boldsymbol{\epsilon}_1 + \sqrt{\beta_2}\boldsymbol{\epsilon}_2.
\tag{18.5}
$$

最后两项是来自均值为零、方差分别为 $(1-\beta_2)\beta_1$ 和 $\beta_2$ 的正态分布的独立样本。这个和的均值为零，其方差是各分量方差之和（见问题18.2），所以：

$$
\mathbf{z}_2 = \sqrt{(1-\beta_2)(1-\beta_1)}\cdot\mathbf{x} + \sqrt{1-(1-\beta_2)(1-\beta_1)}\cdot\boldsymbol{\epsilon},
\tag{18.6}
$$

其中 $\boldsymbol{\epsilon}$ 也是从标准正态分布中抽取的样本。
如果我们通过将这个方程代入 $\mathbf{z}_3$ 的表达式来继续这个过程，依此类推，我们可以证明：

$$
\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\mathbf{x} + \sqrt{1-\bar{\alpha}_t}\cdot\boldsymbol{\epsilon},
\tag{18.7}
$$

其中 $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$。我们可以等效地用概率形式写出：

$$
q(\mathbf{z}_t|\mathbf{x}) = \text{Norm}_{\mathbf{z}_t}[\sqrt{\bar{\alpha}_t}\mathbf{x}, (1-\bar{\alpha}_t)\mathbf{I}].
\tag{18.8}
$$

对于任何起始数据点 $\mathbf{x}$，变量 $\mathbf{z}_t$ 都是正态分布的，具有已知的均值和方差。因此，如果我们不关心通过中间变量 $\mathbf{z}_1, \dots, \mathbf{z}_{t-1}$ 的演化历史，就很容易从 $q(\mathbf{z}_t|\mathbf{x})$ 生成样本。

### 18.2.2 边缘分布 $q(\mathbf{z}_t)$

边缘分布 $q(\mathbf{z}_t)$ 是在给定可能的起始点 $\mathbf{x}$ 的分布和每个起始点的可能扩散路径的情况下，观测到 $\mathbf{z}_t$ 值
的概率（图18.4）。它可以通
过考虑联合分布 $q(\mathbf{x}, \mathbf{z}_{1\dots t})$ 并对除 $\mathbf{z}_t$ 之外的所有变量进行边缘化来计算：

$$
q(\mathbf{z}_t) = \iint q(\mathbf{z}_{1\dots t}, \mathbf{x}) d\mathbf{z}_{1\dots t-1} d\mathbf{x} = \iint q(\mathbf{z}_{1\dots t}|\mathbf{x})\text{Pr}(\mathbf{x}) d\mathbf{z}_{1\dots t-1} d\mathbf{x},
\tag{18.9}
$$

其中 $q(\mathbf{z}_{1\dots t}|\mathbf{x})$ 在方程18.3中定义。
然而，由于我们现在有一个表达式用于“跳过”中间变量的扩散核 $q(\mathbf{z}_t|\mathbf{x})$，我们可以等效地写成：

$$
q(\mathbf{z}_t) = \int q(\mathbf{z}_t|\mathbf{x})\text{Pr}(\mathbf{x})d\mathbf{x}.
\tag{18.10}
$$

因此，如果我们从数据分布 $\text{Pr}(\mathbf{x})$ 中重复采样，并在每个样本上叠加扩散核 $q(\mathbf{z}_t|\mathbf{x})$，结果就是边缘分布 $q(\mathbf{z}_t)$（图18.4）。然而，由于我们不知道原始数据分布 $\text{Pr}(\mathbf{x})$，边缘分布无法以封闭形式写出。

### 18.2.3 条件分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$

我们将条件概率 $q(\mathbf{z}_t|\mathbf{z}_{t-1})$ 定义为混合过程（方程18.2）。为了逆转这个过程，我们应用贝叶斯规则：

$$
q(\mathbf{z}_{t-1}|\mathbf{z}_t) = \frac{q(\mathbf{z}_t|\mathbf{z}_{t-1})q(\mathbf{z}_{t-1})}{q(\mathbf{z}_t)}.
\tag{18.11}
$$

这是难解的，因为我们无法计算边缘分布 $q(\mathbf{z}_{t-1})$。
对于这个简单的一维例子，可以用数值方法评估 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$（图18.5）。通常，它们的形式是复杂的，但在许多情况下，它们可以很好地被一个正态分布近似。这很重要，因为当我们构建解码器时，我们将使用一个正态分布来近似逆过程。

### 18.2.4 条件扩散分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})$

还有一个与编码器相关的最终分布需要考虑。我们上面注意到，我们无法找到条件分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$，因为我们不知道边缘分布 $q(\mathbf{z}_{t-1})$。然而，如果我们知道起始变量 $\mathbf{x}$，那么我们**确实**知道前一个时间的分布 $q(\mathbf{z}_{t-1}|\mathbf{x})$。这正是扩散核（图18.3），并且是正态分布的。

因此，可以以封闭形式计算**条件扩散分布** $q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})$（图18.6）。这个分布被用来训练解码器。它是当我们知道当前潜变量 $\mathbf{z}_t$ 和训练数据样本 $\mathbf{x}$（当然，我们在训练时是知道的）时，$\mathbf{z}_{t-1}$ 的分布。为了计算 $q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})$ 的表达式，我们从贝叶斯规则开始：

$$
q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) = \frac{q(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{x})q(\mathbf{z}_{t-1}|\mathbf{x})}{q(\mathbf{z}_t|\mathbf{x})} \propto q(\mathbf{z}_t|\mathbf{z}_{t-1})q(\mathbf{z}_{t-1}|\mathbf{x}) \\
\propto \text{Norm}_{\mathbf{z}_t}[\sqrt{1-\beta_t}\mathbf{z}_{t-1}, \beta_t\mathbf{I}] \text{Norm}_{\mathbf{z}_{t-1}}[\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}, (1-\bar{\alpha}_{t-1})\mathbf{I}] \\
\propto \text{Norm}_{\mathbf{z}_{t-1}}\left[\frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_t, \frac{\beta_t}{1-\beta_t}\mathbf{I}\right] \text{Norm}_{\mathbf{z}_{t-1}}[\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}, (1-\bar{\alpha}_{t-1})\mathbf{I}]
\tag{18.12}
$$

在第一行和第二行之间，我们使用了扩散过程是马尔可夫的，关于 $\mathbf{z}_t$ 的所有信息都被 $\mathbf{z}_{t-1}$ 捕获了。在第三行和第四行之间，我们使用了**高斯变量变换恒等式 (Gaussian change of variables identity)**：参考：Appendix C.3.4 Gaussian change of variables

$$
\text{Norm}_{\mathbf{v}}[\mathbf{Aw}, \mathbf{B}] \propto \text{Norm}_{\mathbf{w}}[(\mathbf{A}^T\mathbf{B}^{-1}\mathbf{A})^{-1}\mathbf{A}^T\mathbf{B}^{-1}\mathbf{v}, (\mathbf{A}^T\mathbf{B}^{-1}\mathbf{A})^{-1}],
\tag{18.13}
$$

以 $\mathbf{z}_{t-1}$ 的形式重写第一个分布。然后我们使用第二个高斯恒等式：

$$
\text{Norm}_{\mathbf{w}}[\mathbf{a}, \mathbf{A}] \text{Norm}_{\mathbf{w}}[\mathbf{b}, \mathbf{B}] \propto \text{Norm}_{\mathbf{w}}[(\mathbf{A}^{-1}+\mathbf{B}^{-1})^{-1}(\mathbf{A}^{-1}\mathbf{a}+\mathbf{B}^{-1}\mathbf{b}), (\mathbf{A}^{-1}+\mathbf{B}^{-1})^{-1}],
\tag{18.14}
$$

来组合 $\mathbf{z}_{t-1}$ 中的两个正态分布，这给出了：

$$
q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) = \text{Norm}_{\mathbf{z}_{t-1}}\left[\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x} + \frac{\sqrt{1-\beta_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{z}_t, \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{I}\right].
\tag{18.15}
$$

请注意，方程18.12, 18.13, 和 18.14 中的比例常数必须相互抵消，因为最终结果已经是一个正确归一化的概率分布。

---

> **图 18.5 条件分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$**
> a) 边缘密度 $q(\mathbf{z}_t)$，高亮显示了三个点 $\mathbf{z}_t^*$。b) 概率 $q(\mathbf{z}_{t-1}|\mathbf{z}_t^*)$（青色曲线）通过贝叶斯规则计算，并且与 $q(\mathbf{z}_t^*|\mathbf{z}_{t-1})q(\mathbf{z}_{t-1})$ 成正比。通常，它不是正态分布的（顶图），尽管通常正态分布是一个很好的近似（底两图）。第一个似然项 $q(\mathbf{z}_t^*|\mathbf{z}_{t-1})$ 在 $\mathbf{z}_{t-1}$ 中是正态的（方程18.2），其均值比 $\mathbf{z}_t^*$ 离零点稍远（棕色曲线）。第二项是边缘密度 $q(\mathbf{z}_{t-1})$（灰色曲线）。

> **图 18.6 条件分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})$**
> a) $\mathbf{x}^*=-2.1$ 的扩散核，高亮显示了三个点 $\mathbf{z}_t^*$。b) 概率 $q(\mathbf{z}_{t-1}|\mathbf{z}_t^*, \mathbf{x}^*)$ 通过贝叶斯规则计算，并且与 $q(\mathbf{z}_t^*|\mathbf{z}_{t-1})q(\mathbf{z}_{t-1}|\mathbf{x}^*)$ 成正比。这是正态分布的，可以以封闭形式计算。第一个似然项 $q(\mathbf{z}_t^*|\mathbf{z}_{t-1})$ 在 $\mathbf{z}_{t-1}$ 中是正态的（方程18.2），其均值比 $\mathbf{z}_t^*$ 离零点稍远（棕色曲线）。第二项是扩散核 $q(\mathbf{z}_{t-1}|\mathbf{x}^*)$（灰色曲线）。

---

## 18.3 解码器模型 (逆向过程)

当我们学习一个扩散模型时，我们学习的是**逆向过程 (reverse process)**。换句话说，我们学习一系列从潜变量 $\mathbf{z}_T$ 到 $\mathbf{z}_{T-1}$，从 $\mathbf{z}_{T-1}$ 到 $\mathbf{z}_{T-2}$，依此类推，直到我们到达数据 $\mathbf{x}$ 的概率映射。扩散过程的真实逆向分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$ 是依赖于数据分布 $\text{Pr}(\mathbf{x})$ 的复杂多峰分布（图18.5）。我们用正态分布来近似它们：

$$
\begin{aligned}
\text{Pr}(\mathbf{z}_T) &= \text{Norm}_{\mathbf{z}_T}[\mathbf{0}, \mathbf{I}] \\
\text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t) &= \text{Norm}_{\mathbf{z}_{t-1}}[\mathbf{f}_t[\mathbf{z}_t, \boldsymbol{\phi}_t], \sigma_t^2\mathbf{I}] \\
\text{Pr}(\mathbf{x}|\mathbf{z}_1, \boldsymbol{\phi}_1) &= \text{Norm}_{\mathbf{x}}[\mathbf{f}_1[\mathbf{z}_1, \boldsymbol{\phi}_1], \sigma_1^2\mathbf{I}],
\end{aligned}
\tag{18.16}
$$

其中 $\mathbf{f}_t[\mathbf{z}_t, \boldsymbol{\phi}_t]$ 是一个计算从 $\mathbf{z}_t$ 到前一个潜变量 $\mathbf{z}_{t-1}$ 的估计映射中正态分布均值的神经网络。项 $\{\sigma_t^2\}$ 是预先确定的。如果扩散过程中的超参数 $\beta_t$ 接近于零（并且时间步数 $T$ 很大），那么这个正态近似将是合理的。
我们使用**祖先采样**从 $\text{Pr}(\mathbf{x})$ 生成新样本。我们从 $\text{Pr}(\mathbf{z}_T)$ 中抽取 $\mathbf{z}_T$。然后我们从 $\text{Pr}(\mathbf{z}_{T-1}|\mathbf{z}_T, \boldsymbol{\phi}_T)$ 中采样 $\mathbf{z}_{T-1}$，从 $\text{Pr}(\mathbf{z}_{T-2}|\mathbf{z}_{T-1}, \boldsymbol{\phi}_{T-1})$ 中采样 $\mathbf{z}_{T-2}$，依此类推，直到我们最终从 $\text{Pr}(\mathbf{x}|\mathbf{z}_1, \boldsymbol{\phi}_1)$ 生成 $\mathbf{x}$。

## 18.4 训练

观测变量 $\mathbf{x}$ 和潜变量 $\{\mathbf{z}_t\}$ 的联合分布是：

$$
\text{Pr}(\mathbf{x}, \mathbf{z}_{1\dots T}|\boldsymbol{\phi}_{1\dots T}) = \text{Pr}(\mathbf{x}|\mathbf{z}_1, \boldsymbol{\phi}_1) \left[\prod_{t=2}^T \text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t)\right] \cdot \text{Pr}(\mathbf{z}_T).
\tag{18.17}
$$

观测数据 $\text{Pr}(\mathbf{x}|\boldsymbol{\phi}_{1\dots T})$ 的似然是通过对潜变量进行边缘化找到的：参考：Appendix C.1.2 Marginalization

$$
\text{Pr}(\mathbf{x}|\boldsymbol{\phi}_{1\dots T}) = \int \text{Pr}(\mathbf{x}, \mathbf{z}_{1\dots T}|\boldsymbol{\phi}_{1\dots T}) d\mathbf{z}_{1\dots T}.
\tag{18.18}
$$

为了训练模型，我们针对参数 $\boldsymbol{\phi}$，最大化训练数据 $\{\mathbf{x}_i\}$ 的对数似然：

$$
\hat{\boldsymbol{\phi}}_{1\dots T} = \underset{\boldsymbol{\phi}_{1\dots T}}{\mathrm{argmax}} \left[ \sum_{i=1}^I \log[\text{Pr}(\mathbf{x}_i|\boldsymbol{\phi}_{1\dots T})] \right].
\tag{18.19}
$$

我们无法直接最大化这个，因为方程18.18中的边缘化是难解的。因此，我们使用詹森不等式来定义一个似然的下界，并针对这个下界优化参数 $\boldsymbol{\phi}_{1\dots T}$，这与我们为VAE所做的一样（见17.3.1节）。

### 18.4.1 证据下界 (ELBO)

为了推导下界，我们将对数似然乘以并除以编码器分布 $q(\mathbf{z}_{1\dots T}|\mathbf{x})$，并应用詹森不等式（见17.3.2节）：

$$
\log[\text{Pr}(\mathbf{x}|\boldsymbol{\phi}_{1\dots T})] = \log\left[\int \text{Pr}(\mathbf{x}, \mathbf{z}_{1\dots T}|\boldsymbol{\phi}_{1\dots T})d\mathbf{z}_{1\dots T}\right] \\
= \log\left[\int q(\mathbf{z}_{1\dots T}|\mathbf{x}) \frac{\text{Pr}(\mathbf{x}, \mathbf{z}_{1\dots T}|\boldsymbol{\phi}_{1\dots T})}{q(\mathbf{z}_{1\dots T}|\mathbf{x})} d\mathbf{z}_{1\dots T}\right] \\
\ge \int q(\mathbf{z}_{1\dots T}|\mathbf{x}) \log\left[\frac{\text{Pr}(\mathbf{x}, \mathbf{z}_{1\dots T}|\boldsymbol{\phi}_{1\dots T})}{q(\mathbf{z}_{1\dots T}|\mathbf{x})}\right] d\mathbf{z}_{1\dots T}.
\tag{18.20}
$$

这给了我们证据下界（ELBO）：

$$
\text{ELBO}[\boldsymbol{\phi}_{1\dots T}] = \int q(\mathbf{z}_{1\dots T}|\mathbf{x}) \log\left[\frac{\text{Pr}(\mathbf{x}, \mathbf{z}_{1\dots T}|\boldsymbol{\phi}_{1\dots T})}{q(\mathbf{z}_{1\dots T}|\mathbf{x})}\right] d\mathbf{z}_{1\dots T}.
\tag{18.21}
$$

在VAE中，编码器 $q(\mathbf{z}|\mathbf{x})$ 近似于潜变量的后验分布以使下界紧致，解码器则最大化这个下界（图17.10）。在扩散模型中，解码器必须做所有的工作，因为编码器没有参数。它通过 (i) 改变其参数使得静态编码器确实近似于后验 $\text{Pr}(\mathbf{z}_{1\dots T}|\mathbf{x}, \boldsymbol{\phi}_{1\dots T})$，以及 (ii) 优化其自身参数以关于该下界（见图17.6）来使下界更紧。

### 18.4.2 简化 ELBO

我们现在将ELBO中的对数项操纵成我们将优化的最终形式。我们首先代入方程18.17和18.3中分子和分母的定义：

$$
\log\left[\frac{\text{Pr}(\mathbf{x}, \mathbf{z}_{1\dots T}|\boldsymbol{\phi}_{1\dots T})}{q(\mathbf{z}_{1\dots T}|\mathbf{x})}\right] = \log\left[\frac{\text{Pr}(\mathbf{x}|\mathbf{z}_1, \boldsymbol{\phi}_1) \prod_{t=2}^T \text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t) \cdot \text{Pr}(\mathbf{z}_T)}{q(\mathbf{z}_1|\mathbf{x})\prod_{t=2}^T q(\mathbf{z}_t|\mathbf{z}_{t-1})}\right] \\
= \log\left[\frac{\text{Pr}(\mathbf{x}|\mathbf{z}_1, \boldsymbol{\phi}_1)}{q(\mathbf{z}_1|\mathbf{x})}\right] + \log\left[\frac{\prod_{t=2}^T \text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t)}{\prod_{t=2}^T q(\mathbf{z}_t|\mathbf{z}_{t-1})}\right] + \log[\text{Pr}(\mathbf{z}_T)].
\tag{18.22}
$$

然后我们展开第二项的分母：

$$
q(\mathbf{z}_t|\mathbf{z}_{t-1}) = q(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{x}) = \frac{q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})q(\mathbf{z}_t|\mathbf{x})}{q(\mathbf{z}_{t-1}|\mathbf{x})},
\tag{18.23}
$$

其中第一个等式成立是因为关于变量 $\mathbf{z}_t$ 的所有信息都包含在 $\mathbf{z}_{t-1}$ 中，所以额外以数据 $\mathbf{x}$ 为条件是无关紧要的。第二个等式是贝叶斯规则的直接应用。
将这个结果代入给出：

$$
\log\left[\frac{\text{Pr}(\mathbf{x}, \mathbf{z}_{1\dots T}|\boldsymbol{\phi}_{1\dots T})}{q(\mathbf{z}_{1\dots T}|\mathbf{x})}\right] \\
= \log\left[\frac{\text{Pr}(\mathbf{x}|\mathbf{z}_1, \boldsymbol{\phi}_1)}{q(\mathbf{z}_1|\mathbf{x})}\right] + \log\left[\frac{\prod_{t=2}^T \text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t) \cdot q(\mathbf{z}_{t-1}|\mathbf{x})}{\prod_{t=2}^T q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})q(\mathbf{z}_t|\mathbf{x})}\right] + \log[\text{Pr}(\mathbf{z}_T)] \\
= \log[\text{Pr}(\mathbf{x}|\mathbf{z}_1, \boldsymbol{\phi}_1)] + \log\left[\frac{\prod_{t=2}^T \text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t)}{\prod_{t=2}^T q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})}\right] + \log\left[\frac{\text{Pr}(\mathbf{z}_T)}{q(\mathbf{z}_T|\mathbf{x})}\right] \\
\approx \log[\text{Pr}(\mathbf{x}|\mathbf{z}_1, \boldsymbol{\phi}_1)] + \sum_{t=2}^T \log\left[\frac{\text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t)}{q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})}\right],
\tag{18.24}
$$

在第二行和第三行之间，除了两个项外，比率 $q(\mathbf{z}_{t-1}|\mathbf{x})/q(\mathbf{z}_t|\mathbf{x})$ 的乘积中的所有项都相互抵消，只剩下 $q(\mathbf{z}_1|\mathbf{x})$ 和 $q(\mathbf{z}_T|\mathbf{x})$。第三行中的最后一项近似为 $\log=0$，因为前向过程的结果 $q(\mathbf{z}_T|\mathbf{x})$ 是一个标准正态分布，因此等于先验 $\text{Pr}(\mathbf{z}_T)$。

简化的ELBO因此是：

$$
\text{ELBO}[\boldsymbol{\phi}_{1\dots T}] \approx \int q(\mathbf{z}_{1\dots T}|\mathbf{x}) \left[ \log[\text{Pr}(\mathbf{x}|\mathbf{z}_1, \boldsymbol{\phi}_1)] + \sum_{t=2}^T \log\left[\frac{\text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t)}{q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})}\right] \right] d\mathbf{z}_{1\dots T} \\
= \mathbb{E}_{q(\mathbf{z}_1|\mathbf{x})}[\log[\text{Pr}(\mathbf{x}|\mathbf{z}_1, \boldsymbol{\phi}_1)]] - \sum_{t=2}^T \mathbb{E}_{q(\mathbf{z}_t|\mathbf{x})}[D_{KL}[q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) || \text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t)]],
\tag{18.25}
$$

在第二行和第三行之间，我们对 $q(\mathbf{z}_{1\dots T}|\mathbf{x})$ 中的无关变量进行了边缘化，并使用了KL散度的定义（见问题18.7）。

### 18.4.3 分析 ELBO

ELBO中的第一个概率项在方程18.16中定义：

$$
\text{Pr}(\mathbf{x}|\mathbf{z}_1, \boldsymbol{\phi}_1) = \text{Norm}_\mathbf{x}[\mathbf{f}_1[\mathbf{z}_1, \boldsymbol{\phi}_1], \sigma_1^2\mathbf{I}],
\tag{18.26}
$$

并且等价于VAE中的重构项。如果模型预测与观测数据匹配，ELBO会更大。与VAE一样，我们将使用蒙特卡洛估计来近似这个量的对数的期望（见方程17.22-17.23），其中我们用从 $q(\mathbf{z}_1|\mathbf{x})$ 中抽取的样本来估计期望。
ELBO中的KL散度项衡量了 $\text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t)$ 和 $q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})$ 之间的距离，这两个分布分别在方程18.16和18.15中定义：

$$
\begin{aligned}
\text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t) &= \text{Norm}_{\mathbf{z}_{t-1}}[\mathbf{f}_t[\mathbf{z}_t, \boldsymbol{\phi}_t], \sigma_t^2\mathbf{I}] \\
q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) &= \text{Norm}_{\mathbf{z}_{t-1}}\left[\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x} + \frac{\sqrt{1-\beta_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{z}_t, \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{I}\right].
\end{aligned}
\tag{18.27}
$$

两个正态分布之间的KL散度有一个封闭形式的表达式。此外，这个表达式中的许多项不依赖于 $\boldsymbol{\phi}$（见问题18.8），表达式简化为均值之间的平方差加上一个常数 $C$：

$$
D_{KL}[q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) || \text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t)] = \frac{1}{2\sigma_t^2}\left|\left| \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x} + \frac{\sqrt{1-\beta_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{z}_t - \mathbf{f}_t[\mathbf{z}_t, \boldsymbol{\phi}_t] \right|\right|^2 + C.
\tag{18.28}
$$

### 18.4.4 扩散损失函数

为了拟合模型，我们针对参数 $\boldsymbol{\phi}_{1\dots T}$ 最大化ELBO。我们通过乘以-1并用样本近似期望，将其重塑为一个最小化问题，得到损失函数：

$$
L[\boldsymbol{\phi}_{1\dots T}] = \sum_{i=1}^I \left( -\log[\text{Norm}_{\mathbf{x}_i}[\mathbf{f}_1[\mathbf{z}_{i1}, \boldsymbol{\phi}_1], \sigma_1^2\mathbf{I}]] \\
+ \sum_{t=2}^T \frac{1}{2\sigma_t^2} \left|\left| \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_i + \frac{\sqrt{1-\beta_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{z}_{it} - \mathbf{f}_t[\mathbf{z}_{it}, \boldsymbol{\phi}_t] \right|\right|^2 \right),
\tag{18.29}
$$

其中 $\mathbf{x}_i$ 是第 $i$ 个数据点，$\mathbf{z}_{it}$ 是在扩散步骤 $t$ 的相关潜变量。

## 18.5 损失函数的重参数化

尽管可以使用方程18.29中的损失函数，但已发现扩散模型在不同的参数化下工作得更好；损失函数被修改，使得模型旨在预测与原始数据样本混合以创建当前变量的噪声。18.5.1节讨论了对目标（方程18.29第二行中的前两项）的重参数化，18.5.2节讨论了对网络（方程18.29第二行中的最后一项）的重参数化。

### 18.5.1 目标的重参数化

原始的扩散更新由下式给出：

$$
\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\cdot\mathbf{x} + \sqrt{1-\bar{\alpha}_t}\cdot\boldsymbol{\epsilon}.
\tag{18.30}
$$

由此可知，方程18.28中的数据项 $\mathbf{x}$ 可以表示为扩散后的图像减去添加的噪声：

$$
\mathbf{x} = \frac{1}{\sqrt{\bar{\alpha}_t}}\mathbf{z}_t - \frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}}\cdot\boldsymbol{\epsilon}.
\tag{18.31}
$$

将此代入方程18.29中的目标项，得到：

$$
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x} + \frac{(1-\bar{\alpha}_{t-1})\sqrt{1-\beta_t}}{1-\bar{\alpha}_t}\mathbf{z}_t \\
= \frac{(1-\bar{\alpha}_{t-1})\sqrt{1-\beta_t}}{1-\bar{\alpha}_t}\mathbf{z}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\left(\frac{1}{\sqrt{\bar{\alpha}_t}}\mathbf{z}_t - \frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}}\boldsymbol{\epsilon}\right) \\
= \left( \frac{(1-\bar{\alpha}_{t-1})\sqrt{1-\beta_t}}{1-\bar{\alpha}_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(1-\bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} \right) \mathbf{z}_t - \frac{\beta_t}{\sqrt{\bar{\alpha}_t}\sqrt{1-\bar{\alpha}_t}\sqrt{1-\beta_t}}\boldsymbol{\epsilon},
\tag{18.32}
$$

我们利用了 $\sqrt{\bar{\alpha}_t}/\sqrt{\bar{\alpha}_{t-1}}=\sqrt{1-\beta_t}$ 这个事实。进一步简化，我们得到：

$$
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x} + \frac{(1-\bar{\alpha}_{t-1})\sqrt{1-\beta_t}}{1-\bar{\alpha}_t}\mathbf{z}_t \\
= \left( \frac{(1-\bar{\alpha}_{t-1})\sqrt{1-\beta_t}}{1-\bar{\alpha}_t} + \frac{\beta_t}{(1-\bar{\alpha}_t)\sqrt{1-\beta_t}} \right) \mathbf{z}_t - \dots\boldsymbol{\epsilon} \\
= \left( \frac{(1-\bar{\alpha}_{t-1})(1-\beta_t)+\beta_t}{(1-\bar{\alpha}_t)\sqrt{1-\beta_t}} \right) \mathbf{z}_t - \dots\boldsymbol{\epsilon} \\
= \left( \frac{1-\bar{\alpha}_t}{(1-\bar{\alpha}_t)\sqrt{1-\beta_t}} \right) \mathbf{z}_t - \dots\boldsymbol{\epsilon} \\
= \frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{1-\beta_t}}\boldsymbol{\epsilon},
\tag{18.33}
$$

在第二行和第三行之间，我们将第一项的分子和分母乘以 $\sqrt{1-\beta_t}$，展开各项，并简化了第一项的分子。
将此代回损失函数（方程18.29），我们有：

$$
L[\boldsymbol{\phi}_{1\dots T}] = \sum_{i=1}^I \left(-\log[\text{Norm}_{\mathbf{x}_i}[\mathbf{f}_1[\mathbf{z}_{i1}, \boldsymbol{\phi}_1], \sigma_1^2\mathbf{I}]] \\
+\sum_{t=2}^T \frac{1}{2\sigma_t^2} \left|\left| \frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_{it} - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{1-\beta_t}}\boldsymbol{\epsilon}_{it} - \mathbf{f}_t[\mathbf{z}_{it}, \boldsymbol{\phi}_t] \right|\right|^2\right).
\tag{18.34}
$$

### 18.5.2 网络的重参数化

现在我们将模型 $\hat{\mathbf{z}}_{t-1} = \mathbf{f}_t[\mathbf{z}_t, \boldsymbol{\phi}_t]$ 替换为一个新模型 $\hat{\boldsymbol{\epsilon}} = \mathbf{g}_t[\mathbf{z}_t, \boldsymbol{\phi}_t]$，该模型预测与 $\mathbf{x}$ 混合以创建 $\mathbf{z}_t$ 的噪声 $\boldsymbol{\epsilon}$：

$$
\mathbf{f}_t[\mathbf{z}_t, \boldsymbol{\phi}_t] = \frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{1-\beta_t}}\mathbf{g}_t[\mathbf{z}_t, \boldsymbol{\phi}_t].
\tag{18.35}
$$

将新模型代入方程18.34，得到准则：

$$
L[\boldsymbol{\phi}_{1\dots T}] = \sum_{i=1}^I \left(-\log[\text{Norm}_{\mathbf{x}_i}[\mathbf{f}_1[\mathbf{z}_{i1}, \boldsymbol{\phi}_1], \sigma_1^2\mathbf{I}]] + \sum_{t=2}^T \frac{\beta_t^2}{(1-\bar{\alpha}_t)(1-\beta_t)2\sigma_t^2} ||\mathbf{g}_t[\mathbf{z}_{it}, \boldsymbol{\phi}_t] - \boldsymbol{\epsilon}_{it}||^2\right).
\tag{18.36}
$$

对数正态项可以写成一个最小二乘损失加上一个常数 $C_i$（5.3.1节）：

$$
L[\boldsymbol{\phi}_{1\dots T}] = \sum_{i=1}^I \left( \frac{1}{2\sigma_1^2}||\mathbf{x}_i - \mathbf{f}_1[\mathbf{z}_{i1}, \boldsymbol{\phi}_1]||^2 + \sum_{t=2}^T \frac{\beta_t^2}{(1-\bar{\alpha}_t)(1-\beta_t)2\sigma_t^2} ||\mathbf{g}_t[\mathbf{z}_{it}, \boldsymbol{\phi}_t] - \boldsymbol{\epsilon}_{it}||^2 \right) + C_i.
\tag{18.37}
$$

分别从方程18.31和18.35中代入 $\mathbf{x}$ 和 $\mathbf{f}_1[\mathbf{z}_1, \boldsymbol{\phi}_1]$ 的定义，第一项简化为：

$$
\frac{1}{2\sigma_1^2}||\mathbf{x}_i - \mathbf{f}_1[\mathbf{z}_{i1}, \boldsymbol{\phi}_1]||^2 = \frac{1}{2\sigma_1^2}\left|\left| \frac{\beta_1}{\sqrt{1-\bar{\alpha}_1}\sqrt{1-\beta_1}}\mathbf{g}_1[\mathbf{z}_{i1}, \boldsymbol{\phi}_1] - \frac{\beta_1}{\sqrt{1-\bar{\alpha}_1}\sqrt{1-\beta_1}}\boldsymbol{\epsilon}_{i1} \right|\right|^2.
\tag{18.38}
$$

将此加回到最终的损失函数中，得到：

$$
L[\boldsymbol{\phi}_{1\dots T}] = \sum_{i=1}^I \sum_{t=1}^T \frac{\beta_t^2}{(1-\bar{\alpha}_t)(1-\beta_t)2\sigma_t^2} ||\mathbf{g}_t[\mathbf{z}_{it}, \boldsymbol{\phi}_t] - \boldsymbol{\epsilon}_{it}||^2,
\tag{18.39}
$$

我们忽略了加法常数 $C_i$。
在实践中，缩放因子（在每个时间步可能不同）被忽略，给出一个更简单的公式：

$$
L[\boldsymbol{\phi}_{1\dots T}] = \sum_{i=1}^I \sum_{t=1}^T ||\mathbf{g}_t[\mathbf{z}_{it}, \boldsymbol{\phi}_t] - \boldsymbol{\epsilon}_{it}||^2 = \sum_{i=1}^I \sum_{t=1}^T ||\mathbf{g}_t[\sqrt{\bar{\alpha}_t}\mathbf{x}_i + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_{it}, \boldsymbol{\phi}_t] - \boldsymbol{\epsilon}_{it}||^2,
\tag{18.40}
$$

其中我们在第二行中使用了扩散核（方程18.30）重写了 $\mathbf{z}_t$。

## 18.6 实现

这导致了用于训练模型（算法18.1）和采样（算法18.2）的直接算法。训练算法的优点是 (i) 实现简单，并且 (ii) 自然地增强了数据集；我们可以在每个时间步使用不同的噪声实例 $\boldsymbol{\epsilon}$，根据需要多次重用每个原始数据点 $\mathbf{x}_i$。采样算法的缺点是它需要串行处理许多神经网络 $\mathbf{g}_t[\mathbf{z}_t, \boldsymbol{\phi}_t]$，因此耗时。

**算法 18.1: 扩散模型训练**
**输入**: 训练数据 $\mathbf{x}$
**输出**: 模型参数 $\boldsymbol{\phi}_t$
**repeat**
  **for** $i \in \mathcal{B}$ **do**
    $t \sim \text{Uniform}[1, \dots, T]$  // 批次中每个训练样本索引
    $\boldsymbol{\epsilon} \sim \text{Norm}[\mathbf{0}, \mathbf{I}]$      // 采样随机时间步
    $l_i = ||\mathbf{g}_t[\sqrt{\bar{\alpha}_t}\mathbf{x}_i + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \boldsymbol{\phi}_t] - \boldsymbol{\epsilon}||^2$ // 采样噪声
  **end for**
  累积批次的损失并执行梯度步骤
**until** converged

**算法 18.2: 采样**
**输入**: 模型, $\mathbf{g}_t[\cdot, \boldsymbol{\phi}_t]$
**输出**: 样本, $\mathbf{x}$
$\mathbf{z}_T \sim \text{Norm}_{\mathbf{z}}[\mathbf{0}, \mathbf{I}]$ // 采样最后一个潜变量
**for** $t=T, \dots, 2$ **do**
  $\hat{\mathbf{z}}_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{1-\beta_t}}\mathbf{g}_t[\mathbf{z}_t, \boldsymbol{\phi}_t]$ // 预测前一个潜变量
  $\boldsymbol{\epsilon} \sim \text{Norm}[\mathbf{0}, \mathbf{I}]$ // 抽取新的噪声向量
  $\mathbf{z}_{t-1} = \hat{\mathbf{z}}_{t-1} + \sigma_t\boldsymbol{\epsilon}$ // 向前一个潜变量添加噪声
**end for**
$\mathbf{x} = \frac{1}{\sqrt{1-\beta_1}}\mathbf{z}_1 - \frac{\beta_1}{\sqrt{1-\bar{\alpha}_1}\sqrt{1-\beta_1}}\mathbf{g}_1[\mathbf{z}_1, \boldsymbol{\phi}_1]$ // 从$\mathbf{z}_1$生成样本，不加噪声

### 18.6.1 应用于图像

扩散模型在建模图像数据方面非常成功。在这里，我们需要构建能够接收一个噪声图像并预测在每个步骤中添加的噪声的模型。对于这种图像到图像的映射，明显的架构选择是U-Net（图11.10）。然而，可能会有非常大量的扩散步骤，训练和存储多个U-Net是低效的。解决方案是训练一个单一的U-Net，它也接收一个表示时间步的预定向量作为输入（图18.9）。在实践中，这个向量被调整大小以匹配U-Net每个阶段的通道数，并用于在每个空间位置上对表示进行偏移和/或缩放。
当超参数 $\beta_t$ 接近于零时，条件概率 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$ 变得更接近正态分布，匹配解码器分布 $\text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t)$ 的形式，因此需要大量的时间步。然而，这使得采样缓慢。我们可能需要通过U-Net模型运行 $T=1000$ 步才能生成好的图像。

### 18.6.2 提高生成速度

损失函数（方程18.40）要求扩散核具有 $q(\mathbf{z}_t|\mathbf{x}) = \text{Norm}[\sqrt{\bar{\alpha}_t}\mathbf{x}, \sqrt{1-\bar{\alpha}_t}\cdot\mathbf{I}]$ 的形式。对于任何与此关系兼容的前向过程，相同的损失函数将是有效的，并且存在一个此类兼容过程的族。这些都由相同的损失函数优化，但具有不同的前向过程规则和不同的相应规则，关于如何使用估计的噪声 $\mathbf{g}_t[\mathbf{z}_t, \boldsymbol{\phi}_t]$ 来从 $\mathbf{z}_t$ 预测 $\mathbf{z}_{t-1}$ 的逆过程（图18.10）。
在这个族中，有**去噪扩散隐式模型 (denoising diffusion implicit models)**，它们在从 $\mathbf{x}$ 到 $\mathbf{z}_1$ 的第一步之后不再是随机的，以及**加速采样模型 (accelerated sampling models)**，其中前向过程仅在时间步的子序列上定义。这允许一个跳过时间步的逆过程，因此使采样更有效率；当不再是随机过程时，可以用50个时间步创建好的样本。这比以前快得多，但仍然比大多数其他生成模型慢。

### 18.6.3 条件生成

如果数据有关联的标签 $\mathbf{c}$，可以利用这些来控制生成。有时这可以改善GANs中的生成结果，我们可能也期望在扩散模型中是这样；如果你有一些关于图像包含什么的信息，去噪一个图像会更容易。扩散模型中条件合成的一种方法是**分类器引导 (classifier guidance)**。这修改了从 $\mathbf{z}_t$ 到 $\mathbf{z}_{t-1}$ 的去噪更新，以考虑类别信息 $\mathbf{c}$。在实践中，这意味着向最终的更新步骤（算法18.2）添加一个额外的项：

$$
\hat{\mathbf{z}}_{t-1} = \hat{\mathbf{z}}_{t-1} + \sigma_t^2 \frac{\partial \log[\text{Pr}(c|\mathbf{z}_t)]}{\partial \mathbf{z}_t} + \sigma_t \boldsymbol{\epsilon}.
\tag{18.41}
$$

新项取决于基于潜变量 $\mathbf{z}_t$ 的分类器 $\text{Pr}(c|\mathbf{z}_t)$ 的梯度。这将特征从U-Net的下采样部分映射到类别 $\mathbf{c}$。像U-Net一样，它通常在所有时间步之间共享，并接收时间作为输入。从 $\mathbf{z}_t$ 到 $\mathbf{z}_{t-1}$ 的更新现在使类别 $\mathbf{c}$ 更有可能。
**无分类器引导 (Classifier-free guidance)** 避免了学习一个单独的分类器 $\text{Pr}(c|\mathbf{z}_t)$，而是将类别信息并入主模型 $\mathbf{g}_t[\mathbf{z}_t, \boldsymbol{\phi}_t, \mathbf{c}]$。在实践中，这通常采取的形式是，基于 $\mathbf{c}$ 添加一个嵌入到U-Net的层中，方式与添加时间步的方式类似（见图18.9）。这个模型是在条件和无条件目标上联合训练的，通过随机地丢弃类别信息。因此，它可以在测试时生成无条件或条件的数据样本，或两者的任何加权组合。这带来了一个令人惊讶的优势；如果条件信息被过分加权，模型倾向于产生非常高质量但略微刻板的样本。这在某种程度上类似于在GANs中使用截断（图15.10）。

---

> **图 18.7 拟合模型**
> a) 可以通过从标准正态分布 $\text{Pr}(\mathbf{z}_T)$（底行）采样，然后从 $\text{Pr}(\mathbf{z}_{T-1}|\mathbf{z}_T) = \text{Norm}_{\mathbf{z}_{T-1}}[\mathbf{f}_T[\mathbf{z}_T, \boldsymbol{\phi}_T], \sigma_T^2\mathbf{I}]$ 采样 $\mathbf{z}_{T-1}$，依此类推，直到我们到达 $\mathbf{x}$（显示了五条路径）来生成单个样本。估计的边缘密度（热图）是这些样本的聚合，并且与真实的边缘密度（图18.4）相似。b) 估计的分布 $\text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t)$（棕色曲线）是来自图18.5的扩散模型 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$（青色曲线）的真实后验的一个合理近似。估计和真实模型的边缘分布 $\text{Pr}(\mathbf{z}_t)$ 和 $q(\mathbf{z}_t)$（分别为深蓝色和灰色曲线）也相似。

> **图 18.8 拟合模型结果**
> 青色和棕色曲线是原始和估计的密度，并分别对应于图18.4和18.7的顶行。垂直条是从模型生成的分类样本，通过从 $\text{Pr}(\mathbf{z}_T)$ 采样并通过变量 $\mathbf{z}_{T-1}, \mathbf{z}_{T-2}, \dots$ 向后传播，如图18.7中的五条路径所示。

> **图 18.9 用作图像扩散模型的U-Net**
> 网络旨在预测添加到图像中的噪声。它由一个减小尺度并增加通道数的编码器和一个增加尺度并减少通道数的解码器组成。编码器表示被拼接到它们在解码器中的伙伴上。相邻表示之间的连接由残差块和周期性的全局自注意力组成，其中每个空间位置与每个其他空间位置相互作用。一个单一的网络用于所有时间步，通过将一个正弦时间嵌入（图12.5）通过一个浅层神经网络，并将结果添加到U-Net的每个阶段的每个空间位置的通道中。

> **图 18.10 与同一模型兼容的不同扩散过程**
> a) 在真实边缘分布上叠加的重参数化模型的五条采样轨迹。顶行代表 $\text{Pr}(\mathbf{x})$，后续行代表 $q(\mathbf{x}_t)$。b) 从重参数化模型生成的样本直方图，与真实密度曲线 $\text{Pr}(\mathbf{x})$ 一起绘制。训练好的同一模型与一系列扩散模型（以及相反方向的相应更新）兼容，包括去噪扩散隐式模型（DDIM），它是确定性的，并且在每一步都不添加噪声。c) 来自DDIM的五条轨迹。d) 来自DDIM的样本直方图。同一模型也与加速扩散模型兼容，后者跳过推理步骤以提高采样速度。e) 来自加速模型的五条轨迹。f) 来自加速模型的样本直方图。

> **图 18.11 基于文本提示的级联条件生成**
> a) 一个由一系列U-Net组成的扩散模型被用来生成一个 $64 \times 64$ 的图像。b) 这个生成是以一个由语言模型计算的句子嵌入为条件的。c) 一个更高分辨率的 $256 \times 256$ 图像被生成，并以较小的图像和文本编码为条件。d) 这被重复以创建一个 $1024 \times 1024$ 的图像。e) 最终的图像序列。改编自 Saharia et al. (2022b)。

---

### 18.6.4 提高生成质量

与其他生成模型一样，最高质量的结果来自于应用技巧和基础模型扩展的组合。首先，有人指出，估计逆过程的方差 $\sigma_t^2$ 和均值（即，图18.7中棕色正态分布的宽度）也有帮助。这尤其在用较少步骤采样时改善了结果。其次，可以修改前向过程中的噪声方案，使 $\beta_t$ 在每一步都变化，这也可以改善结果。
第三，为了生成高分辨率图像，使用了一系列级联的扩散模型。第一个创建一个低分辨率图像（可能由类别信息引导）。随后的扩散模型生成逐渐更高分辨率的图像。它们通过调整这个大小并将其附加到组成U-Net的层，以及任何其他类别信息，来以前一个低分辨率图像为条件（图18.11）。

结合所有这些技术，可以生成非常高质量的图像。图18.12显示了以ImageNet类别为条件的模型生成的图像示例。同一个模型能学会生成如此多样的类别，尤其令人印象深刻。图18.13显示了由一个模型生成的图像，该模型被训练以像BERT这样的语言模型编码的文本标题为条件，这些标题以与时间步相同的方式插入到模型中（图18.9和18.11）。这产生了与标题一致的非常逼真的图像。由于扩散模型本质上是随机的，因此可以生成多个以相同标题为条件的图像。

## 18.7 总结

扩散模型通过将当前表示与随机噪声重复混合，将数据样本通过一系列潜变量进行映射。经过足够的步骤，表示变得与白噪声无法区分。由于这些步骤很小，每个步骤的逆向去噪过程可以用一个正态分布来近似，并由一个深度学习模型来预测。损失函数基于证据下界（ELBO），并最终形成一个简单的最小二乘公式。
对于图像生成，每个去噪步骤都使用一个U-Net来实现，因此采样比其他生成模型慢。为了提高生成速度，可以将扩散模型更改为确定性公式，这样用较少步骤的采样效果很好。已经提出了几种方法来以类别信息、图像和文本信息为条件进行生成。结合这些方法可以产生令人印象深刻的文本到图像的合成结果。

---

> **图 18.12 使用分类器引导的条件生成**
> 以不同ImageNet类别为条件的图像样本。同一个模型能产生高度多样的图像类别的高质量样本。改编自 Dhariwal & Nichol (2021)。

> **图 18.13 使用文本提示的条件生成**
> 在一个级联生成框架中，以一个由大型语言模型编码的文本提示为条件合成的图像。随机模型可以产生许多与提示兼容的不同图像。该模型可以计数对象并将文本并入图像。改编自 Saharia et al. (2022b)。

---

### 注释

去噪扩散模型由 Sohl-Dickstein et al. (2015) 引入，早期基于分数匹配的相关工作由 Song & Ermon (2019) 完成。Ho et al. (2020) 生成了与GANs具有竞争力的图像样本，并引发了对该领域的兴趣浪潮。本章中的大部分阐述，包括原始公式和重参数化，都源自这篇论文。Dhariwal & Nichol (2021) 提高了这些结果的质量，并首次表明扩散模型的图像在弗雷歇初始距离方面在数量上优于GAN模型。在撰写本文时，条件图像合成的最先进结果已由 Karras et al. (2022) 实现。关于去噪扩散模型的综述可以在 Croitoru et al. (2022), Cao et al. (2022), Luo (2022), 和 Yang et al. (2022) 中找到。

> **图像应用**：扩散模型的应用包括文本到图像的生成 (Nichol et al., 2022; Ramesh et al., 2022; Saharia et al., 2022b)、图像到图像的任务，如着色、修复、裁剪和恢复 (Saharia et al., 2022a)、超分辨率 (Saharia et al., 2022c)、图像编辑 (Hertz et al., 2022; Meng et al., 2021)、移除对抗性扰动 (Nie et al., 2022)、语义分割 (Baranchuk et al., 2022) 和医学成像 (Song et al., 2021b; Chung & Ye, 2022; Chung et al., 2022; Peng et al., 2022; Xie & Li, 2022; Luo et al., 2022)，其中扩散模型有时用作先验。
> 
**不同数据类型**：扩散模型也被应用于视频数据 (Ho et al., 2022b; Harvey et al., 2022; Yang et al., 2022; Höppe et al., 2022; Voleti et al., 2022) 用于生成、过去和未来帧的预测以及插值。它们已被用于3D形状生成 (Zhou et al., 2021; Luo & Hu, 2021)，最近引入了一种技术，仅使用2D文本到图像的扩散模型来生成3D模型 (Poole et al., 2023)。Austin et al. (2021) 和 Hoogeboom et al. (2021) 研究了离散数据的扩散模型。Kong et al. (2021) 和 Chen et al. (2021d) 将扩散模型应用于音频数据。

**去噪的替代方案**：本章中的扩散模型将噪声与数据混合，并构建一个模型来逐渐去噪结果。然而，使用噪声来降级图像不是必需的。Rissanen et al. (2022) 设计了一种逐渐模糊图像的方法，而 Bansal et al. (2022) 表明，同样的想法适用于一大类不必是随机的退化。这些包括掩蔽、变形、模糊和像素化。

**与其他生成模型的比较**：扩散模型合成的图像质量高于其他生成模型，并且训练简单。它们可以被看作是层次化VAE的特例 (Vahdat & Kautz, 2020; Sønderby et al., 2016b)，其中编码器是固定的，潜空间与数据大小相同。它们是概率性的，但在基本形式下，它们只能计算数据点似然的一个下界。然而，Kingma et al. (2021) 表明，这个下界在来自归一化流和自回归模型的测试数据上，优于精确的对数似然。扩散模型的似然可以通过转换为常微分方程 (Song et al., 2021c) 或通过训练一个带有基于扩散的准则的连续归一化流模型 (Lipman et al., 2022) 来计算。扩散模型的主要缺点是它们很慢，并且潜空间没有语义解释。

**提高质量**：已经提出了许多技术来提高图像质量。这些包括18.5节中描述的网络重参数化和后续项的等权重 (Ho et al., 2020)。Choi et al. (2022) 随后研究了损失函数中各项的不同加权。
Kingma et al. (2021) 通过学习去噪权重 $\beta_t$ 来提高了模型的测试对数似然。相反，Nichol & Dhariwal (2021) 通过在每个时间步学习去噪估计的单独方差 $\sigma^2$ 以及均值来提高性能。Bao et al. (2022) 展示了如何在训练模型后学习方差。
Ho et al. (2022a) 开发了级联方法来产生非常高分辨率的图像（图18.11）。为了防止低分辨率图像中的伪影传播到更高分辨率，他们引入了**噪声条件增强**；在这里，低分辨率条件图像通过在每个训练步骤添加噪声来降级。这减少了训练期间对低分辨率图像确切细节的依赖。在推理期间也这样做，其中通过在不同值上扫描来选择最佳噪声水平。

**提高速度**：扩散模型的主要缺点之一是它们训练和采样的时间很长。**稳定扩散 (Stable diffusion)** (Rombach et al., 2022) 使用传统的自编码器将原始数据投影到一个较小的潜空间，然后在这个较小的空间中运行扩散过程。这具有为扩散过程减少训练数据维度和允许描述其他数据类型（文本、图等）的优点。Vahdat et al. (2021) 应用了类似的方法。
Song et al. (2021a) 表明，整个扩散过程家族都与训练目标兼容。这些过程中的大多数是非马尔可夫的（即，扩散步骤不仅依赖于前一步的结果）。这些模型之一是**去噪扩散隐式模型 (denoising diffusion implicit model, DDIM)**，其中更新不是随机的（图18.10c）。这个模型适合采取更大的步骤（图18.10e）而不会引入大的误差。它有效地将模型转换为一个常微分方程（ODE），其中轨迹具有低曲率，并允许应用高效的数值方法来求解ODE。
Song et al. (2021c) 建议将底层的随机微分方程转换为具有与原始过程相同边缘分布的概率流ODE。Vahdat et al. (2021), Xiao et al. (2022b), 和 Karras et al. (2022) 都利用求解ODE的技术来加速合成。Karras et al. (2022) 确定了用于采样的最佳时间离散化，并评估了不同的采样器计划。这些和其他改进的结果是合成所需步骤的显著下降。
采样之所以慢，是因为需要许多小的扩散步骤来确保后验分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$ 接近高斯分布（图18.5），因此解码器中的高斯分布是合适的。如果我们使用一个在每个去噪步骤描述更复杂分布的模型，那么我们一开始就可以使用更少的扩散步骤。为此，Xiao et al. (2022b) 研究了使用条件GAN模型，Gao et al. (2021) 研究了使用条件能量基模型。尽管这些模型无法描述原始数据分布，但它们足以预测（更简单的）逆向扩散步骤。
Salimans & Ho (2022) 将去噪过程的相邻步骤提炼成一个单一的步骤以加速合成。Dockhorn et al. (2022) 向扩散过程引入了动量。这使得轨迹更平滑，因此更适合粗略采样。

**条件生成**：Dhariwal & Nichol (2021) 引入了分类器引导，其中一个分类器学习识别在每个步骤中正在合成的对象的类别，并用此来偏置去噪更新朝向该类别。这效果很好，但训练一个单独的分类器是昂贵的。**无分类器引导 (Classifier-free guidance)** (Ho & Salimans, 2022) 通过在类似于dropout的过程中，在一定比例的时间内丢弃类别信息，来同时训练条件和无条件去噪模型。这种技术允许控制条件和无条件组件的相对贡献。过分加权条件组件会导致模型产生更典型和逼真的样本。
对图像进行条件的标准技术是将（调整大小后的）图像附加到U-Net的不同层。例如，这在超分辨率的级联生成过程中被使用 (Ho et al., 2022a)。Choi et al. (2021) 提供了一种在无条件扩散模型中以图像为条件的方法，通过将潜变量与条件图像的潜变量进行匹配。对文本进行条件的标准技术是将文本嵌入线性变换到与U-Net层相同的大小，然后以与引入时间嵌入相同的方式将其添加到表示中（图18.9）。
现有的扩散模型也可以通过使用一个称为**控制网络 (control network)** 的神经网络结构，进行微调以在边缘图、关节位置、分割、深度图等上进行条件化 (Zhang & Agrawala, 2023)。

**文本到图像**：在扩散模型之前，最先进的文本到图像系统是基于 Transformer 的（例如，Ramesh et al., 2021）。GLIDE (Nichol et al., 2022) 和 Dall-E 2 (Ramesh et al., 2022) 都以来自CLIP模型 (Radford et al., 2021) 的嵌入为条件，该模型为文本和图像数据生成联合嵌入。Imagen (Saharia et al., 2022b) 表明，来自大型语言模型的文本嵌入可以产生更好的结果（见图18.13）。同一作者引入了一个基准 (DrawBench)，旨在评估模型呈现颜色、对象数量、空间关系和其他特征的能力。Feng et al. (2022) 开发了一个中文文本到图像的模型。

**与其他模型的联系**：本章将扩散模型描述为层次化变分自编码器，因为这种方法与本书的其他部分联系最紧密。然而，扩散模型也与随机微分方程（考虑图18.5中的路径）和分数匹配 (Song & Ermon, 2019, 2020) 有密切的联系。Song et al. (2021c) 提出了一个基于随机微分方程的框架，它包含了去噪和分数匹配的解释。扩散模型也与归一化流有密切的联系 (Zhang & Chen, 2021)。Yang et al. (2022) 提供了扩散模型与其他生成方法之间关系的概述。

---

***

### 习题

**问题 18.1** 证明如果 $\text{Cov}[\mathbf{x}_{t-1}]=\mathbf{I}$ 并且我们使用更新 $\mathbf{x}_t = \sqrt{1-\beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_t$，那么 $\text{Cov}[\mathbf{x}_t]=\mathbf{I}$，所以方差保持不变。

**思路与解答：**

这是问题17.5的应用。
令 $\mathbf{A} = \sqrt{1-\beta_t}\mathbf{I}$ 和 $\mathbf{b} = \sqrt{\beta_t}\boldsymbol{\epsilon}_t$。
这里 $\mathbf{x}_{t-1}$ 和 $\boldsymbol{\epsilon}_t$ 是独立的随机变量。
$\text{Cov}[\mathbf{x}_t] = \text{Cov}[\sqrt{1-\beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_t]$
利用独立变量和的协方差等于协方差之和：
$= \text{Cov}[\sqrt{1-\beta_t}\mathbf{x}_{t-1}] + \text{Cov}[\sqrt{\beta_t}\boldsymbol{\epsilon}_t]$
利用 $\text{Cov}[c\mathbf{X}] = c^2\text{Cov}[\mathbf{X}]$：
$= (1-\beta_t)\text{Cov}[\mathbf{x}_{t-1}] + \beta_t\text{Cov}[\boldsymbol{\epsilon}_t]$
已知 $\text{Cov}[\mathbf{x}_{t-1}]=\mathbf{I}$ 且 $\boldsymbol{\epsilon}_t$ 是标准正态噪声，所以 $\text{Cov}[\boldsymbol{\epsilon}_t]=\mathbf{I}$。
$= (1-\beta_t)\mathbf{I} + \beta_t\mathbf{I} = (1-\beta_t+\beta_t)\mathbf{I} = \mathbf{I}$。
方差保持为单位矩阵 $\mathbf{I}$。

**问题 18.2** 考虑变量 $z = a\cdot\epsilon_1 + b\cdot\epsilon_2$，其中 $\epsilon_1, \epsilon_2$ 是独立的标准正态分布。证明 $\mathbb{E}[z]=0$ 且 $\text{Var}[z] = a^2+b^2$。

**思路与解答：**

1.  **期望**:
    $\mathbb{E}[z] = \mathbb{E}[a\epsilon_1 + b\epsilon_2] = a\mathbb{E}[\epsilon_1] + b\mathbb{E}[\epsilon_2]$
    因为 $\mathbb{E}[\epsilon_1]=\mathbb{E}[\epsilon_2]=0$，所以 $\mathbb{E}[z] = a\cdot 0 + b\cdot 0 = 0$。

2.  **方差**:
    $\text{Var}[z] = \text{Var}[a\epsilon_1 + b\epsilon_2]$
    因为 $\epsilon_1, \epsilon_2$ 独立，所以 $\text{Var}(X+Y)=\text{Var}(X)+\text{Var}(Y)$。
    $= \text{Var}[a\epsilon_1] + \text{Var}[b\epsilon_2]$
    利用 $\text{Var}(cX) = c^2\text{Var}(X)$：
    $= a^2\text{Var}[\epsilon_1] + b^2\text{Var}[\epsilon_2]$
    因为 $\text{Var}[\epsilon_1]=\text{Var}[\epsilon_2]=1$，所以 $\text{Var}[z] = a^2\cdot 1 + b^2\cdot 1 = a^2+b^2$。
    因此，我们可以等价地计算 $z = \sqrt{a^2+b^2}\cdot\epsilon$，其中 $\epsilon$ 也是标准正态分布。

**问题 18.3** 继续方程18.5的过程，证明 $\mathbf{z}_3 = \sqrt{(1-\beta_3)(1-\beta_2)(1-\beta_1)}\cdot\mathbf{x} + \sqrt{1-(1-\beta_3)(1-\beta_2)(1-\beta_1)}\cdot\boldsymbol{\epsilon}'$。

**思路与解答：**

1.  我们有 $\mathbf{z}_2 = \sqrt{\bar{\alpha}_2}\mathbf{x} + \sqrt{1-\bar{\alpha}_2}\boldsymbol{\epsilon}$，其中 $\bar{\alpha}_2 = (1-\beta_1)(1-\beta_2)$。
2.  $\mathbf{z}_3 = \sqrt{1-\beta_3}\mathbf{z}_2 + \sqrt{\beta_3}\boldsymbol{\epsilon}_3$
3.  代入 $\mathbf{z}_2$:
    $\mathbf{z}_3 = \sqrt{1-\beta_3}(\sqrt{\bar{\alpha}_2}\mathbf{x} + \sqrt{1-\bar{\alpha}_2}\boldsymbol{\epsilon}) + \sqrt{\beta_3}\boldsymbol{\epsilon}_3$
    $= \sqrt{(1-\beta_3)\bar{\alpha}_2}\mathbf{x} + \sqrt{1-\beta_3}\sqrt{1-\bar{\alpha}_2}\boldsymbol{\epsilon} + \sqrt{\beta_3}\boldsymbol{\epsilon}_3$
4.  注意到 $\bar{\alpha}_3 = (1-\beta_3)\bar{\alpha}_2$。
5.  后面的噪声项是两个独立的零均值高斯噪声的和。其方差是各自方差之和：
    $\text{Var} = (1-\beta_3)(1-\bar{\alpha}_2) + \beta_3 = 1 - \beta_3 - \bar{\alpha}_2 + \beta_3\bar{\alpha}_2 + \beta_3 = 1 - \bar{\alpha}_2(1-\beta_3) = 1 - \bar{\alpha}_3$。
6.  因此，噪声项可以合并为一个单一的高斯噪声 $\sqrt{1-\bar{\alpha}_3}\boldsymbol{\epsilon}'$。
7.  最终得到 $\mathbf{z}_3 = \sqrt{\bar{\alpha}_3}\mathbf{x} + \sqrt{1-\bar{\alpha}_3}\boldsymbol{\epsilon}'$。

**问题 18.4*** 证明关系：$\text{Norm}_{\mathbf{v}}[\mathbf{Aw}, \mathbf{B}] \propto \text{Norm}_{\mathbf{w}}[(\mathbf{A}^T\mathbf{B}^{-1}\mathbf{A})^{-1}\mathbf{A}^T\mathbf{B}^{-1}\mathbf{v}, (\mathbf{A}^T\mathbf{B}^{-1}\mathbf{A})^{-1}]$。

**思路与解答：**

这是高斯分布的线性变换性质。
1.  左边是关于 $\mathbf{v}$ 的分布，均值为 $\mathbf{Aw}$，协方差为 $\mathbf{B}$。
    $\text{Pr}(\mathbf{v}|\mathbf{w}) \propto \exp\left(-\frac{1}{2}(\mathbf{v}-\mathbf{Aw})^T \mathbf{B}^{-1} (\mathbf{v}-\mathbf{Aw})\right)$。
2.  我们想把它看作是关于 $\mathbf{w}$ 的函数（作为 $\mathbf{w}$ 的似然）。
3.  展开指数项: $\mathbf{v}^T\mathbf{B}^{-1}\mathbf{v} - 2\mathbf{v}^T\mathbf{B}^{-1}\mathbf{A}\mathbf{w} + \mathbf{w}^T\mathbf{A}^T\mathbf{B}^{-1}\mathbf{A}\mathbf{w}$。
4.  这是一个关于 $\mathbf{w}$ 的二次型。我们可以通过配方法将其写成一个关于 $\mathbf{w}$ 的高斯分布形式。
    一个高斯分布 $\text{Norm}(\boldsymbol{\mu}, \mathbf{\Sigma})$ 的指数项是 $-\frac{1}{2}(\mathbf{w}^T\mathbf{\Sigma}^{-1}\mathbf{w} - 2\mathbf{w}^T\mathbf{\Sigma}^{-1}\boldsymbol{\mu} + \dots)$。
5.  对比系数，我们得到:
    *   协方差的逆: $\mathbf{\Sigma}^{-1} = \mathbf{A}^T\mathbf{B}^{-1}\mathbf{A} \implies \mathbf{\Sigma} = (\mathbf{A}^T\mathbf{B}^{-1}\mathbf{A})^{-1}$
    *   均值项: $\mathbf{\Sigma}^{-1}\boldsymbol{\mu} = \mathbf{A}^T\mathbf{B}^{-1}\mathbf{v} \implies \boldsymbol{\mu} = \mathbf{\Sigma}\mathbf{A}^T\mathbf{B}^{-1}\mathbf{v} = (\mathbf{A}^T\mathbf{B}^{-1}\mathbf{A})^{-1}\mathbf{A}^T\mathbf{B}^{-1}\mathbf{v}$
6.  这与右边的分布参数完全吻合。

**问题 18.5*** 证明关系：$\text{Norm}_{\mathbf{x}}[\mathbf{a}, \mathbf{A}]\text{Norm}_{\mathbf{x}}[\mathbf{b}, \mathbf{B}] \propto \text{Norm}_{\mathbf{x}}[(\mathbf{A}^{-1}+\mathbf{B}^{-1})^{-1}(\mathbf{A}^{-1}\mathbf{a}+\mathbf{B}^{-1}\mathbf{b}), (\mathbf{A}^{-1}+\mathbf{B}^{-1})^{-1}]$。

**思路与解答：**

这是两个高斯分布乘积的性质。
1.  取两个高斯PDF的对数，然后相加：
    $\log(\text{Pr}_1 \cdot \text{Pr}_2) = -\frac{1}{2}(\mathbf{x}-\mathbf{a})^T \mathbf{A}^{-1}(\mathbf{x}-\mathbf{a}) - \frac{1}{2}(\mathbf{x}-\mathbf{b})^T \mathbf{B}^{-1}(\mathbf{x}-\mathbf{b}) + C$
2.  展开关于 $\mathbf{x}$ 的二次项和一次项：
    *   二次项: $-\frac{1}{2}\mathbf{x}^T(\mathbf{A}^{-1}+\mathbf{B}^{-1})\mathbf{x}$
    *   一次项: $\mathbf{x}^T(\mathbf{A}^{-1}\mathbf{a}+\mathbf{B}^{-1}\mathbf{b})$
3.  同样，通过与标准高斯指数形式 $-\frac{1}{2}(\mathbf{x}^T\mathbf{\Sigma}^{-1}\mathbf{x} - 2\mathbf{x}^T\mathbf{\Sigma}^{-1}\boldsymbol{\mu} + \dots)$ 对比：
    *   新协方差的逆: $\mathbf{\Sigma}^{-1} = \mathbf{A}^{-1}+\mathbf{B}^{-1} \implies \mathbf{\Sigma} = (\mathbf{A}^{-1}+\mathbf{B}^{-1})^{-1}$
    *   新均值项: $\mathbf{\Sigma}^{-1}\boldsymbol{\mu} = \mathbf{A}^{-1}\mathbf{a}+\mathbf{B}^{-1}\mathbf{b} \implies \boldsymbol{\mu} = \mathbf{\Sigma}(\mathbf{A}^{-1}\mathbf{a}+\mathbf{B}^{-1}\mathbf{b})$
4.  这与右边的分布参数吻合。

**问题 18.6*** 推导方程18.15。

**思路与解答：**

这个问题结合了18.4和18.5的结论。
我们有两个关于 $\mathbf{z}_{t-1}$ 的高斯分布相乘：
1.  $\text{Norm}_{\mathbf{z}_{t-1}}\left[\frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_t, \frac{\beta_t}{1-\beta_t}\mathbf{I}\right]$
2.  $\text{Norm}_{\mathbf{z}_{t-1}}[\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}, (1-\bar{\alpha}_{t-1})\mathbf{I}]$

应用问题18.5的公式，其中：
*   $\mathbf{a} = \frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_t$, $\mathbf{A} = \frac{\beta_t}{1-\beta_t}\mathbf{I}$
*   $\mathbf{b} = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}$, $\mathbf{B} = (1-\bar{\alpha}_{t-1})\mathbf{I}$
新均值 $\boldsymbol{\mu}'$ 和新协方差 $\mathbf{\Sigma}'$：
*   $\mathbf{\Sigma}'^{-1} = \frac{1-\beta_t}{\beta_t}\mathbf{I} + \frac{1}{1-\bar{\alpha}_{t-1}}\mathbf{I} = \frac{(1-\beta_t)(1-\bar{\alpha}_{t-1})+\beta_t}{\beta_t(1-\bar{\alpha}_{t-1})}\mathbf{I} = \frac{1-\bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}\mathbf{I}$
*   $\boldsymbol{\mu}' = \mathbf{\Sigma}'(\frac{1-\beta_t}{\beta_t} \frac{\mathbf{z}_t}{\sqrt{1-\beta_t}} + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\mathbf{x}) = \dots$
经过代数化简，可以得到方程18.15的结果。

**问题 18.7*** 从方程18.25的第二行推导出第三行。
**思路与解答:**
第二行是：$\int q(\mathbf{z}_{1\dots T}|\mathbf{x}) \left[ \dots \right] d\mathbf{z}_{1\dots T}$
利用期望的定义 $\mathbb{E}_{q(z)}[f(z)] = \int q(z)f(z)dz$。
*   第一项: $\int q(\mathbf{z}_{1\dots T}|\mathbf{x}) \log[\text{Pr}(\mathbf{x}|\mathbf{z}_1)] d\mathbf{z}_{1\dots T}$。
    $\log[\text{Pr}(\mathbf{x}|\mathbf{z}_1)]$ 只与 $\mathbf{z}_1$ 有关。我们可以对其他变量 $\mathbf{z}_{2\dots T}$ 积分掉，得到边缘分布 $q(\mathbf{z}_1|\mathbf{x})$。
    所以该项为 $\int q(\mathbf{z}_1|\mathbf{x}) \log[\text{Pr}(\mathbf{x}|\mathbf{z}_1)] d\mathbf{z}_1 = \mathbb{E}_{q(\mathbf{z}_1|\mathbf{x})}[\log[\text{Pr}(\mathbf{x}|\mathbf{z}_1)]]$。
*   第二项: $\int q(\mathbf{z}_{1\dots T}|\mathbf{x}) \sum_t \log[\dots] d\mathbf{z}_{1\dots T}$
    $\log$ 项只与 $\mathbf{z}_t, \mathbf{z}_{t-1}$ 有关。我们可以对其他变量积分掉，得到边缘分布 $q(\mathbf{z}_t, \mathbf{z}_{t-1}|\mathbf{x})$。
    该项为 $\sum_t \int q(\mathbf{z}_t, \mathbf{z}_{t-1}|\mathbf{x}) \log[\dots] d\mathbf{z}_t d\mathbf{z}_{t-1}$
    $= \sum_t \mathbb{E}_{q(\mathbf{z}_t, \mathbf{z}_{t-1}|\mathbf{x})}[\log[\dots]]$
    $= \sum_t \mathbb{E}_{q(\mathbf{z}_t|\mathbf{x})} [ \mathbb{E}_{q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})} [\log[\dots]] ]$
    $= \sum_t \mathbb{E}_{q(\mathbf{z}_t|\mathbf{x})} [-D_{KL}[q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) || \text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t)]]$
这与第三行匹配。

**问题 18.8*** KL散度... 代入方程18.27的定义，并证明只有方程18.28的第一项依赖于参数 $\phi$。

**思路与解答：**

两个高斯分布 $\mathcal{N}(\mathbf{a}, \mathbf{A})$ 和 $\mathcal{N}(\mathbf{b}, \mathbf{B})$ 之间的KL散度公式是给定的。
*   $\mathbf{a}$ 是 $q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})$ 的均值, $\mathbf{A}$ 是其协方差。
*   $\mathbf{b} = \mathbf{f}_t[\mathbf{z}_t, \boldsymbol{\phi}_t]$ 是 $\text{Pr}(\mathbf{z}_{t-1}|\mathbf{z}_t, \boldsymbol{\phi}_t)$ 的均值, $\mathbf{B} = \sigma_t^2\mathbf{I}$ 是其协方差。
*   只有均值 $\mathbf{b}$ 依赖于参数 $\boldsymbol{\phi}_t$。
*   查看KL散度公式的各项：
    *   $\text{tr}[\mathbf{B}^{-1}\mathbf{A}]$: 不依赖 $\boldsymbol{\phi}_t$。
    *   $-d$: 不依赖 $\boldsymbol{\phi}_t$。
    *   $(\mathbf{a}-\mathbf{b})^T\mathbf{B}^{-1}(\mathbf{a}-\mathbf{b})$: 包含 $\mathbf{b}$，因此**依赖** $\boldsymbol{\phi}_t$。
    *   $\log(\det(\mathbf{B})/\det(\mathbf{A}))$: 不依赖 $\boldsymbol{\phi}_t$。
*   所以，只有 $(\mathbf{a}-\mathbf{b})^T\mathbf{B}^{-1}(\mathbf{a}-\mathbf{b})$ 项依赖于 $\boldsymbol{\phi}_t$。
    代入 $\mathbf{B}=\sigma_t^2\mathbf{I}$，得到 $\frac{1}{\sigma_t^2}(\mathbf{a}-\mathbf{b})^T(\mathbf{a}-\mathbf{b}) = \frac{1}{\sigma_t^2}||\mathbf{a}-\mathbf{b}||^2$。
    这与方程18.28的形式一致（除了一个常数因子 $1/2$）。

**问题 18.9*** 证明 $\sqrt{\bar{\alpha}_t/\bar{\alpha}_{t-1}} = \sqrt{1-\beta_t}$。
**思路与解答:**
$\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$
$\bar{\alpha}_{t-1} = \prod_{s=1}^{t-1} (1-\beta_s)$
$\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} = \frac{\prod_{s=1}^t (1-\beta_s)}{\prod_{s=1}^{t-1} (1-\beta_s)} = 1-\beta_t$
两边开方即可。

**问题 18.10*** 证明方程18.33。
**思路与解答:**
这是一个代数化简。从方程18.32的第三行开始，我们已经证明了目标均值为：
$\frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_t - \frac{\beta_t}{\sqrt{\bar{\alpha}_t}\sqrt{1-\bar{\alpha}_t}\sqrt{1-\beta_t}}\boldsymbol{\epsilon}$
我们需要证明这个形式与方程18.33最后一行的形式是等价的，这需要进一步的代数操作。

**问题 18.11*** 证明方程18.38。
**思路与解答:**
第一项是 $\frac{1}{2\sigma_1^2}||\mathbf{x}_i - \mathbf{f}_1[\mathbf{z}_{i1}, \boldsymbol{\phi}_1]||^2$
代入 $\mathbf{x}_i = \frac{1}{\sqrt{\bar{\alpha}_1}}\mathbf{z}_{i1} - \frac{\sqrt{1-\bar{\alpha}_1}}{\sqrt{\bar{\alpha}_1}}\boldsymbol{\epsilon}_{i1}$ 和 $\mathbf{f}_1[\dots]$ 的表达式，然后化简。这是一个纯粹的代数练习。

**问题 18.12** 无分类器引导... 我们还讨论了减少变异并产生更刻板输出的方法。这些方法是什么？你认为我们应该以这种方式限制生成模型的输出是不可避免的吗？

**思路与解答：**

1.  **其他模型中的方法**:
    *   **Transformer解码器 (GPT)**: 使用 **Top-k 采样**或**核心采样 (Nucleus Sampling)** 来避免从概率长尾中采样，使输出更连贯。
    *   **GANs**: 使用**截断技巧 (Truncation Trick)**，即只从潜空间的高概率区域（靠近均值）采样。
    *   **GLOW (归一化流)**: 从一个被提升到正幂的基础密度中采样，这同样偏爱中心的样本。

2.  **是否不可避免**:
    这是一个开放性的哲学问题，没有唯一答案。
    *   **赞成方 (是不可避免的)**:
        *   **质量 vs. 多样性权衡**: 生成模型的目标是在“逼真度”和“多样性”之间找到平衡。完全不受限制的采样可能会产生大量无意义或低质量的样本（对应概率分布的尾部）。限制输出是一种实用的方法，可以牺牲一些罕见的多样性来换取更高的平均质量。
        *   **应用需求**: 在许多应用中（如内容创作、艺术生成），用户需要的是高质量、可控的输出，而不是随机的、不可预测的结果。刻板化（或风格化）是满足这些需求的一种方式。
    *   **反对方 (不是不可避免的)**:
        *   **模型能力的体现**: 一个完美的生成模型应该能够准确地建模整个数据分布，包括其尾部。需要通过截断等技巧来“美化”输出，恰恰说明了当前模型在捕捉分布的复杂性方面还存在缺陷。
        *   **创造力与新颖性**: 真正的创造力往往来自于“长尾”和“意外组合”。过度限制输出可能会扼杀模型的创造潜力，使其只能生成“安全”但“无聊”的内容。
    *   **我的观点**: 在当前阶段，这是**实用主义下的必然选择**。由于模型和训练数据的不完美，完全自由的生成往往质量不可控。限制输出是一种有效的“后处理”手段。然而，终极目标应该是构建一个足够强大的模型，使其在无需外部限制的情况下，也能自然地生成高质量且多样化的内容。