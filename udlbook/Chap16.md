好的，请看我的翻译和解答。

***

# 第十六章
# 归一化流

第十五章介绍了生成对抗网络（GANs）。这类生成模型通过将一个潜变量传递给深度网络来创造新的样本。GANs的训练原则是让生成的样本与真实数据无法区分。然而，它们并没有在数据样本上定义一个概率分布。因此，评估一个新样本属于同一数据集的概率并非易事。

本章我们描述**归一化流 (normalizing flows)**。它们通过一个深度网络，将一个简单的分布变换成一个更复杂的分布，从而学习一个概率模型。归一化流既可以从这个分布中采样，也可以评估新样本的概率。然而，它们需要特殊的架构：每一层都必须是**可逆的 (invertible)**。换句话说，它必须能够双向地变换数据。

## 16.1 一维示例

归一化流是概率生成模型：它们为训练数据拟合一个概率分布（图14.2b）。考虑对一个一维分布 $\text{Pr}(x)$ 进行建模。归一化流从一个潜变量 $z$ 上的简单且易于处理的**基础分布 (base distribution)** $\text{Pr}(z)$ 开始，并应用一个函数 $x = f[z, \boldsymbol{\phi}]$，其中的参数 $\boldsymbol{\phi}$ 被选择以使得 $\text{Pr}(x)$ 具有我们期望的分布（图16.1）。生成一个新样本 $x^*$ 很简单；我们从基础密度中抽取 $z^*$，然后通过函数传递它，得到 $x^* = f[z^*, \boldsymbol{\phi}]$。

### 16.1.1 测量概率

测量数据点 $x$ 的概率则更具挑战性。考虑将一个函数 $f[z, \boldsymbol{\phi}]$ 应用于具有已知密度 $\text{Pr}(z)$ 的随机变量 $z$。概率密度在被函数拉伸的区域会减小，在被压缩的区域会增加，以确保新分布下的总面积保持为一。一个函数 $f[z, \boldsymbol{\phi}]$ 拉伸或压缩其输入的程度取决于其**梯度 (gradient)** 的大小。如果输入的微小变化导致输出发生较大变化，它就拉伸了函数。如果输入的微小变化导致输出发生较小变化，它就压缩了函数（图16.2）。

更准确地说，变换后分布下数据 $x$ 的概率是：

$$
\text{Pr}(x|\boldsymbol{\phi}) = \left| \frac{\partial f[z, \boldsymbol{\phi}]}{\partial z} \right|^{-1} \cdot \text{Pr}(z),
\tag{16.1}
$$

其中 $z = f^{-1}[x, \boldsymbol{\phi}]$ 是创建 $x$ 的潜变量。$\text{Pr}(z)$ 项是这个潜变量在基础密度下的原始概率。这个概率会根据函数导数的大小进行调整。如果导数大于一，概率就减小。如果小于一，概率就增加。

---

> **图 16.1 变换概率分布**
> a) 基础密度是一个定义在潜变量 $z$ 上的标准正态分布。b) 这个变量通过一个函数 $x = f[z, \boldsymbol{\phi}]$ 变换到一个新的变量 $x$，这个新变量 c) 有一个新的分布。为了从这个模型中采样，我们从基础密度中抽取值 $z$（图(a)中的绿色和棕色箭头是两个例子）。我们如(b)中虚线箭头所示，将它们通过函数 $f[z, \boldsymbol{\phi}]$，以生成 $x$ 的值，这些值在(c)中用箭头标示。

> **图 16.2 变换分布**
> 基础密度（青色，底部）通过一个函数（蓝色曲线，右上）来创建模型密度（橙色，左侧）。考虑将基础密度划分为等间隔（灰色竖线）。相邻线之间的概率质量在变换后必须保持不变。青色阴影区域通过函数的一个梯度大于一的部分，因此这个区域被拉伸了。因此，橙色阴影区域的高度必须更低，以保持与青色阴影区域相同的面积。在其他地方（例如，$z=-2$），梯度小于一，模型密度相对于基础密度增加。

> **图 16.3 逆向映射（归一化方向）**
> 如果函数是可逆的，那么就有可能将模型密度变换回原始的基础密度。一个点 $x$ 在模型密度下的概率，部分取决于其在基础密度下等价点 $z$ 的概率（见公式16.1）。

---

### 16.1.2 前向与逆向映射

为了从分布中抽取样本，我们需要前向映射 $x = f[z, \boldsymbol{\phi}]$，但为了测量似然，我们需要计算逆映射 $z = f^{-1}[x, \boldsymbol{\phi}]$。因此，我们需要明智地选择 $f[z, \boldsymbol{\phi}]$ 使其**可逆**。

前向映射有时被称为**生成方向 (generative direction)**。基础密度通常被选为标准正态分布。因此，逆向映射被称为**归一化方向 (normalizing direction)**，因为它将 $x$ 上的复杂分布转换回 $z$ 上的正态分布（图16.3）。

### 16.1.3 学习

为了学习分布，我们寻找能最大化训练数据 $\{\mathbf{x}_i\}_{i=1}^I$ 的似然的参数 $\hat{\boldsymbol{\phi}}$，或者等价地，最小化负对数似然：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\mathrm{argmax}} \left[ \prod_{i=1}^I \text{Pr}(\mathbf{x}_i|\boldsymbol{\phi}) \right] \\
= \underset{\boldsymbol{\phi}}{\mathrm{argmin}} \left[ -\sum_{i=1}^I \log[\text{Pr}(\mathbf{x}_i|\boldsymbol{\phi})] \right] \\
= \underset{\boldsymbol{\phi}}{\mathrm{argmin}} \left[ \sum_{i=1}^I -\log\left[ \left| \frac{\partial f[z_i, \boldsymbol{\phi}]}{\partial z_i} \right|^{-1} \right] - \log[\text{Pr}(z_i)] \right] \\
= \underset{\boldsymbol{\phi}}{\mathrm{argmin}} \left[ \sum_{i=1}^I \log\left[ \left| \frac{\partial f[z_i, \boldsymbol{\phi}]}{\partial z_i} \right| \right] - \log[\text{Pr}(z_i)] \right],
\tag{16.2}
$$

这里我们在第一行假设了数据是独立同分布的，并在第三行使用了来自方程16.1的似然定义。

## 16.2 一般情况

上一节通过变换一个更简单的基础密度 $\text{Pr}(z)$ 来建立了一个模拟概率分布 $\text{Pr}(x)$ 的简单一维示例。我们现在将其扩展到多变量分布 $\text{Pr}(\mathbf{x})$ 和 $\text{Pr}(\mathbf{z})$，并增加变换由深度神经网络定义的复杂性。

考虑将一个函数 $\mathbf{x} = \mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]$ 应用于一个具有基础密度 $\text{Pr}(\mathbf{z})$ 的随机变量 $\mathbf{z} \in \mathbb{R}^D$，其中 $\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]$ 是一个深度网络。得到的变量 $\mathbf{x} \in \mathbb{R}^D$ 有一个新的分布。一个新样本 $\mathbf{x}^*$ 可以通过 (i) 从基础密度中抽取一个样本 $\mathbf{z}^*$ 并 (ii) 将其通过神经网络传递，使得 $\mathbf{x}^* = \mathbf{f}[\mathbf{z}^*, \boldsymbol{\phi}]$ 来从这个分布中抽取。

与方程16.1类比，一个样本在该分布下的似然是：

$$
\text{Pr}(\mathbf{x}|\boldsymbol{\phi}) = \left| \frac{\partial \mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]}{\partial \mathbf{z}} \right|^{-1} \cdot \text{Pr}(\mathbf{z}),
\tag{16.3}
$$

其中 $\mathbf{z} = \mathbf{f}^{-1}[\mathbf{x}, \boldsymbol{\phi}]$ 是创建 $\mathbf{x}$ 的潜变量 $\mathbf{z}$。第一项是 $D \times D$ **雅可比矩阵 (Jacobian matrix)** $\partial \mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]/\partial \mathbf{z}$ 的**行列式 (determinant)** 的逆，该矩阵在位置 $(i, j)$ 包含元素 $\partial f_j[\mathbf{z}, \boldsymbol{\phi}]/\partial z_i$。正如绝对导数在一维函数应用时衡量了一个点上的面积变化，绝对行列式在多变量函数中衡量了一个点上的体积变化。第二项是潜变量在基础密度下的概率。参考：Appendix B.3.8 Determinant, Appendix B.5 Jacobian

---

> **图 16.4 深度神经网络的前向与逆向映射**
> 基础密度（左）通过网络层 $\mathbf{f}_1[\cdot, \boldsymbol{\phi}_1], \mathbf{f}_2[\cdot, \boldsymbol{\phi}_2], \dots$ 逐渐变换以创建模型密度。每一层都是可逆的，我们可以等效地认为层的逆将模型密度逐渐变换（或“流向”）回基础密度。

---

### 16.2.1 使用深度神经网络的前向映射

在实践中，前向映射 $\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]$ 通常由一个神经网络定义，该网络由一系列带有参数 $\boldsymbol{\phi}_k$ 的层 $\mathbf{f}_k[\cdot, \boldsymbol{\phi}_k]$ 复合而成：

$$
\mathbf{x} = \mathbf{f}[\mathbf{z}, \boldsymbol{\phi}] = \mathbf{f}_K[\mathbf{f}_{K-1}[\dots \mathbf{f}_2[\mathbf{f}_1[\mathbf{z}, \boldsymbol{\phi}_1], \boldsymbol{\phi}_2], \dots \boldsymbol{\phi}_{K-1}], \boldsymbol{\phi}_K].
\tag{16.4}
$$

逆映射（归一化方向）由每一层 $\mathbf{f}_k^{-1}[\cdot, \boldsymbol{\phi}_k]$ 的逆以相反顺序复合定义：

$$
\mathbf{z} = \mathbf{f}^{-1}[\mathbf{x}, \boldsymbol{\phi}] = \mathbf{f}_1^{-1}[\mathbf{f}_2^{-1}[\dots \mathbf{f}_{K-1}^{-1}[\mathbf{f}_K^{-1}[\mathbf{x}, \boldsymbol{\phi}_K], \boldsymbol{\phi}_{K-1}], \dots \boldsymbol{\phi}_2], \boldsymbol{\phi}_1].
\tag{16.5}
$$

基础密度 $\text{Pr}(\mathbf{z})$ 通常被定义为多元标准正态分布（即，均值为零，协方差为单位矩阵）。因此，每个后续逆层的效果是逐渐将数据密度移动或“**流 (flow)**”向这个正态分布（图16.4）。这引出了“**归一化流 (normalizing flows)**”的名称。

前向映射的雅可比矩阵可以表示为：

$$
\frac{\partial \mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]}{\partial \mathbf{z}} = \frac{\partial \mathbf{f}_1[\mathbf{z}, \boldsymbol{\phi}_1]}{\partial \mathbf{z}} \cdot \frac{\partial \mathbf{f}_2[\mathbf{f}_1, \boldsymbol{\phi}_2]}{\partial \mathbf{f}_1} \cdots \frac{\partial \mathbf{f}_{K-1}[\mathbf{f}_{K-2}, \boldsymbol{\phi}_{K-1}]}{\partial \mathbf{f}_{K-2}} \cdot \frac{\partial \mathbf{f}_K[\mathbf{f}_{K-1}, \boldsymbol{\phi}_K]}{\partial \mathbf{f}_{K-1}},
\tag{16.6}
$$

这里我们重载了符号，用 $\mathbf{f}_k$ 表示函数 $\mathbf{f}_k[\cdot, \boldsymbol{\phi}_k]$ 的输出。这个雅可比矩阵的绝对行列式可以通过取各个绝对行列式的乘积来计算：

$$
\left| \frac{\partial \mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]}{\partial \mathbf{z}} \right| = \left| \frac{\partial \mathbf{f}_1[\mathbf{z}, \boldsymbol{\phi}_1]}{\partial \mathbf{z}} \right| \cdot \left| \frac{\partial \mathbf{f}_2[\mathbf{f}_1, \boldsymbol{\phi}_2]}{\partial \mathbf{f}_1} \right| \cdots \left| \frac{\partial \mathbf{f}_{K-1}[\mathbf{f}_{K-2}, \boldsymbol{\phi}_{K-1}]}{\partial \mathbf{f}_{K-2}} \right| \cdot \left| \frac{\partial \mathbf{f}_K[\mathbf{f}_{K-1}, \boldsymbol{\phi}_K]}{\partial \mathbf{f}_{K-1}} \right|.
\tag{16.7}
$$

逆映射的雅可比矩阵的绝对行列式可以通过对公式16.5应用同样的规则来找到。它是前向映射中绝对行列式的倒数。

我们使用负对数似然准则，通过一个包含 $I$ 个训练样本的数据集 $\{\mathbf{x}_i\}$ 来训练归一化流：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\mathrm{argmax}} \left[ \prod_{i=1}^I \text{Pr}(\mathbf{z}_i) \cdot \left| \frac{\partial \mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}]}{\partial \mathbf{z}_i} \right|^{-1} \right] \\
= \underset{\boldsymbol{\phi}}{\mathrm{argmin}} \left[ \sum_{i=1}^I \log\left[ \left| \frac{\partial \mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}]}{\partial \mathbf{z}_i} \right| \right] - \log[\text{Pr}(\mathbf{z}_i)] \right],
\tag{16.8}
$$

其中 $\mathbf{z}_i = \mathbf{f}^{-1}[\mathbf{x}_i, \boldsymbol{\phi}]$，$\text{Pr}(\mathbf{z}_i)$ 是在基础分布下测量的，绝对行列式 $|\partial \mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}] / \partial \mathbf{z}_i|$ 由方程16.7给出。参考：Problem 16.3

### 16.2.2 对网络层的要求

归一化流的理论是直接的。然而，为了使其在实践中可行，我们需要具有四个属性的神经网络层 $\mathbf{f}_k$。

1.  **表达能力**: 网络层的集合必须足够**富有表现力 (expressive)**，以将一个多元标准正态分布映射到一个任意的密度。
2.  **可逆性**: 网络层必须是**可逆的**；每一层都必须定义从任何输入点到输出点的唯一的**一对一映射 (a bijection)**。如果多个输入被映射到同一个输出，逆映射将是不明确的。
3.  **高效求逆**: 必须能够**高效地计算**每一层的逆。我们需要在每次评估似然时都这样做。这在训练期间会重复发生，因此必须有一个封闭形式的解或一个快速的求逆算法。
4.  **高效行列式计算**: 对于前向或逆向映射，也必须能够**高效地评估雅可比矩阵的行列式**。

## 16.3 可逆网络层

我们现在描述用于这些模型中的不同的可逆网络层或**流 (flows)**。我们从线性和逐元素流开始。它们很容易求逆，并且可以计算它们的雅可比矩阵的行列式，但它们都不足以描述基础密度的任意变换。然而，它们构成了**耦合流 (coupling flows)**、**自回归流 (autoregressive flows)** 和**残差流 (residual flows)** 的构建块，所有这些都更具表现力。

### 16.3.1 线性流

线性流的形式为 $\mathbf{f}[\mathbf{h}] = \boldsymbol{\beta} + \mathbf{\Omega}\mathbf{h}$。如果矩阵 $\mathbf{\Omega}$ 是可逆的，则线性变换是可逆的。对于 $\mathbf{\Omega} \in \mathbb{R}^{D \times D}$，求逆的计算量是 $O(D^3)$。雅可比矩阵的行列式就是 $\mathbf{\Omega}$ 的行列式，也可以在 $O(D^3)$ 时间内计算。这意味着随着维度 $D$ 的增加，线性流的计算成本会变得很高。参考：Appendix A Big O notation, Appendix B.4 Matrix types, Problem 16.4

如果矩阵 $\mathbf{\Omega}$ 采用特殊形式，那么求逆和行列式的计算会变得更有效率，但变换的通用性会降低。例如，对角矩阵只需要 $O(D)$ 的计算来进行求逆和行列式计算，但 $\mathbf{h}$ 的元素之间没有相互作用。正交矩阵也更容易求逆，它们的行列式是固定的，但它们不允许对各个维度进行缩放。三角矩阵更实用；它们可以使用一个称为**回代 (back-substitution)** 的过程在 $O(D^2)$ 时间内求逆，其行列式就是对角线值的乘积。

一种使线性流既通用、求逆高效，又能高效计算雅可比矩阵的方法是直接根据其**LU分解**来参数化它。换句话说，我们使用：

$$
\mathbf{\Omega} = \mathbf{PL}(\mathbf{U}+\mathbf{D}),
\tag{16.9}
$$

其中 $\mathbf{P}$ 是一个预定的置换矩阵，$\mathbf{L}$ 是一个下三角矩阵，$\mathbf{U}$ 是一个对角线上为零的上三角矩阵，$\mathbf{D}$ 是一个提供那些缺失的对角元素的对角矩阵。这可以在 $O(D^2)$ 时间内求逆，对数行列式就是 $\mathbf{L}$ 和 $\mathbf{D}$ 对角线上绝对值的对数和。参考：Problems 16.5-16.6

不幸的是，线性流的表达能力不足。当一个线性函数 $\mathbf{f}[\mathbf{h}] = \boldsymbol{\beta} + \mathbf{\Omega}\mathbf{h}$ 应用于正态分布的输入 $\text{Norm}_{\mathbf{h}}[\boldsymbol{\mu}, \mathbf{\Sigma}]$ 时，结果也是正态分布的，其均值和方差分别为 $\boldsymbol{\beta} + \mathbf{\Omega}\boldsymbol{\mu}$ 和 $\mathbf{\Omega}\mathbf{\Sigma}\mathbf{\Omega}^T$。因此，仅使用线性流是不可能将一个正态分布映射到一个任意密度的。

### 16.3.2 逐元素流

由于线性流的表达能力不足，我们必须转向非线性流。其中最简单的是**逐元素流 (elementwise flows)**，它对输入的每个元素应用一个带参数 $\boldsymbol{\phi}$ 的逐点非线性函数 $f[\cdot, \boldsymbol{\phi}]$，使得：

$$
\mathbf{f}[\mathbf{h}] = [f[h_1, \boldsymbol{\phi}], f[h_2, \boldsymbol{\phi}], \dots, f[h_D, \boldsymbol{\phi}]]^T.
\tag{16.10}
$$

雅可比矩阵 $\partial\mathbf{f}[\mathbf{h}]/\partial\mathbf{h}$ 是对角的，因为第 $d$ 个输入到 $\mathbf{f}[\mathbf{h}]$ 只影响第 $d$ 个输出。其行列式是对角线上条目的乘积，所以：

$$
\left| \frac{\partial \mathbf{f}[\mathbf{h}]}{\partial \mathbf{h}} \right| = \prod_{d=1}^D \left| \frac{\partial f[h_d, \boldsymbol{\phi}]}{\partial h_d} \right|.
\tag{16.11}
$$

函数 $f[\cdot, \boldsymbol{\phi}]$ 可以是一个固定的可逆非线性函数，比如leaky ReLU（图3.13），在这种情况下没有参数，或者它可以是任何参数化的可逆一对一映射。一个简单的例子是具有 $K$ 个区域的分段线性函数（图16.5），它将 $$ 映射到 $$，定义为：参考：Problem 16.7

$$
f[h, \boldsymbol{\phi}] = \left(\sum_{k=1}^{b-1} \phi_k\right) + (hK-b+1)\phi_b,
\tag{16.12}
$$

其中参数 $\phi_1, \phi_2, \dots, \phi_K$ 是正的且总和为1，而 $b=\lfloor Kh \rfloor+1$ 是包含 $h$ 的区间的索引。第一项是所有前面区间的总和，第二项表示 $h$ 在当前区间中所占的比例。这个函数很容易求逆，并且它的梯度几乎处处都可以计算。有很多类似的方案可以创建平滑函数，通常使用样条函数，其参数确保了函数是单调的，因此是可逆的。参考：Problems 16.8-16.9

逐元素流是非线性的，但不混合输入维度，所以它们无法在变量之间创建相关性。当与线性流（它确实混合维度）交替使用时，可以建模更复杂的变换。然而，在实践中，逐元素流被用作更复杂的层（如耦合流）的组成部分。

---

> **图 16.5 分段线性映射**
> 一个可逆的分段线性映射 $h' = f[h, \boldsymbol{\phi}]$ 可以通过将输入域 $h \in$ 分为 $K$ 个大小相等的区域（这里 $K=5$）来创建。每个区域都有一个由参数 $\phi_k$ 决定的斜率。a) 如果这些参数是正的且总和为一，那么 b) 函数将是可逆的，并将映射到输出域 $h' \in$。

---

### 16.3.3 耦合流

**耦合流 (Coupling flows)** 将输入 $\mathbf{h}$ 分为两部分，使得 $\mathbf{h} = [\mathbf{h}_1^T, \mathbf{h}_2^T]^T$，并定义流 $\mathbf{f}[\mathbf{h}, \boldsymbol{\phi}]$ 为：

$$
\begin{aligned}
\mathbf{h}'_1 &= \mathbf{h}_1 \\
\mathbf{h}'_2 &= g[\mathbf{h}_2, \boldsymbol{\phi}[\mathbf{h}_1]].
\end{aligned}
\tag{16.13}
$$

这里 $g[\cdot, \boldsymbol{\phi}]$ 是一个带有参数 $\boldsymbol{\phi}[\mathbf{h}_1]$ 的逐元素流（或其他可逆层），而这些参数本身是输入 $\mathbf{h}_1$ 的非线性函数（图16.6）。函数 $\boldsymbol{\phi}[\cdot]$ 通常是某种神经网络，并且不必是可逆的。原始变量可以恢复为：

$$
\begin{aligned}
\mathbf{h}_1 &= \mathbf{h}'_1 \\
\mathbf{h}_2 &= g^{-1}[\mathbf{h}'_2, \boldsymbol{\phi}[\mathbf{h}_1]].
\end{aligned}
\tag{16.14}
$$

如果函数 $g[\cdot, \boldsymbol{\phi}]$ 是一个逐元素流，雅可比矩阵将是下三角矩阵，左上角是单位矩阵，右下角是逐元素变换的导数。其行列式是这些对角值的乘积。

逆和雅可比矩阵可以高效地计算，但这种方法只以一种依赖于前半部分参数的方式变换后半部分参数。为了进行更一般的变换，$\mathbf{h}$ 的元素在层与层之间使用**置换矩阵 (permutation matrices)** 进行随机打乱，因此每个变量最终都会被每个其他变量变换。在实践中，这些置换矩阵很难学习。因此，它们被随机初始化然后冻结。对于像图像这样的结构化数据，通道被分为两半 $\mathbf{h}_1$ 和 $\mathbf{h}_2$，并使用 $1 \times 1$ 卷积在层之间进行置换。参考：Appendix B.4.4 Permutation matrix

---

> **图 16.6 耦合流**
> a) 输入（橙色向量）被分为 $\mathbf{h}_1$ 和 $\mathbf{h}_2$。输出的第一部分 $\mathbf{h}'_1$（青色向量）是 $\mathbf{h}_1$ 的一个副本。输出 $\mathbf{h}'_2$ 是通过对 $\mathbf{h}_2$ 应用一个可逆变换 $g[\cdot, \boldsymbol{\phi}]$ 创建的，其中的参数 $\boldsymbol{\phi}$ 本身是 $\mathbf{h}_1$ 的一个（不必要可逆的）函数。b) 在逆映射中，$\mathbf{h}_1 = \mathbf{h}'_1$。这允许我们计算参数 $\boldsymbol{\phi}[\mathbf{h}_1]$，然后应用逆 $g^{-1}[\mathbf{h}'_2, \boldsymbol{\phi}]$ 来恢复 $\mathbf{h}_2$。

---

### 16.3.4 自回归流

**自回归流 (Autoregressive flows)** 是耦合流的一种泛化，它将每个输入维度视为一个单独的“块”（图16.7）。它们基于输入 $\mathbf{h}$ 的前 $d-1$ 个维度来计算输出 $\mathbf{h}'$ 的第 $d$ 个维度：

$$
h'_d = g[h_d, \boldsymbol{\phi}[\mathbf{h}_{1:d-1}]].
\tag{16.15}
$$

函数 $g[\cdot, \cdot]$ 被称为**变换器 (transformer)**¹，参数 $\boldsymbol{\phi}, \boldsymbol{\phi}[\mathbf{h}_1], \boldsymbol{\phi}[\mathbf{h}_1, \mathbf{h}_2], \dots$ 被称为**调节器 (conditioners)**。与耦合流一样，变换器 $g[\cdot, \boldsymbol{\phi}]$ 必须是可逆的，但调节器 $\boldsymbol{\phi}[\cdot]$ 可以是任何形式，并且通常是神经网络。如果变换器和调节器足够灵活，自回归流是**万能逼近器 (universal approximators)**，因为它们可以表示任何概率分布。

可以使用一个带有适当掩码的网络来并行计算输出 $\mathbf{h}'$ 的所有条目，以便位置 $d$ 的参数 $\boldsymbol{\phi}$ 只依赖于之前的位置。这被称为**掩码自回归流 (masked autoregressive flow)**。其原理与掩码自注意力非常相似（12.7.2节）；关联输入与先前输出的连接被修剪。

反转变换的效率较低。考虑前向映射：

$$
\begin{aligned}
h'_1 &= g[h_1, \boldsymbol{\phi}] \\
h'_2 &= g[h_2, \boldsymbol{\phi}[\mathbf{h}_1]] \\
h'_3 &= g[h_3, \boldsymbol{\phi}[\mathbf{h}_{1:2}]] \\
h'_4 &= g[h_4, \boldsymbol{\phi}[\mathbf{h}_{1:3}]].
\end{aligned}
\tag{16.16}
$$

这必须使用与耦合流类似的原理顺序地反转：

$$
\begin{aligned}
h_1 &= g^{-1}[h'_1, \boldsymbol{\phi}] \\
h_2 &= g^{-1}[h'_2, \boldsymbol{\phi}[\mathbf{h}_1]] \\
h_3 &= g^{-1}[h'_3, \boldsymbol{\phi}[\mathbf{h}_{1:2}]] \\
h_4 &= g^{-1}[h'_4, \boldsymbol{\phi}[\mathbf{h}_{1:3}]].
\end{aligned}
\tag{16.17}
$$

这不能并行完成，因为 $h_d$ 的计算依赖于 $\mathbf{h}_{1:d-1}$（即，到目前为止的部分结果）。因此，当输入很大时，求逆是耗时的。参考：Notebook 16.2 Autoregressive flows

> ¹这与第12章中讨论的 Transformer 层无关。

---

> **图 16.7 自回归流**
> 输入 $\mathbf{h}$（橙色列）和输出 $\mathbf{h}'$（青色列）被分割成它们的分量维度（这里是四个维度）。a) 输出 $h'_1$ 是输入 $h_1$ 的一个可逆变换。输出 $h'_2$ 是输入 $h_2$ 的一个可逆函数，其中参数依赖于 $h_1$。输出 $h'_3$ 是输入 $h_3$ 的一个可逆函数，其中参数依赖于先前的输入 $h_1$ 和 $h_2$，依此类推。没有一个输出依赖于另一个，所以它们可以并行计算。b) 自回归流的逆使用与耦合流类似的方法计算。然而，请注意，要计算 $h_2$，我们必须已经知道 $h_1$；要计算 $h_3$，我们必须已经知道 $h_1$ 和 $h_2$，依此类推。因此，逆不能并行计算。

---

### 16.3.5 逆自回归流

掩码自回归流是在归一化（逆）方向上定义的。这是为了高效地评估似然，从而学习模型所必需的。然而，采样需要前向方向，其中每个变量必须在每一层顺序计算，这是缓慢的。如果我们使用一个自回归流进行前向（生成）变换，那么采样是高效的，但计算似然（和训练）是缓慢的。这被称为**逆自回归流 (inverse autoregressive flow)**。

一个允许快速学习和也快速（但近似）采样的技巧是，构建一个掩码自回归流来学习分布（教师），然后用它来训练一个可以从中高效采样的逆自回归流（学生）。这需要一种不同的归一化流的公式，它从另一个函数而不是一组样本中学习（见16.5.3节）。

### 16.3.6 残差流：iRevNet

**残差流 (Residual flows)** 的灵感来自残差网络。它们将输入分为两部分 $\mathbf{h} = [\mathbf{h}_1^T, \mathbf{h}_2^T]^T$（与耦合流一样）并定义输出为：

$$
\begin{aligned}
\mathbf{h}'_1 &= \mathbf{h}_1 + \mathbf{f}_1[\mathbf{h}_2, \boldsymbol{\phi}_1] \\
\mathbf{h}'_2 &= \mathbf{h}_2 + \mathbf{f}_2[\mathbf{h}'_1, \boldsymbol{\phi}_2],
\end{aligned}
\tag{16.18}
$$

其中 $\mathbf{f}_1[\cdot, \boldsymbol{\phi}_1]$ 和 $\mathbf{f}_2[\cdot, \boldsymbol{\phi}_2]$ 是两个不必要可逆的函数（图16.8）。逆可以通过颠倒计算顺序来计算：

$$
\begin{aligned}
\mathbf{h}_2 &= \mathbf{h}'_2 - \mathbf{f}_2[\mathbf{h}'_1, \boldsymbol{\phi}_2] \\
\mathbf{h}_1 &= \mathbf{h}'_1 - \mathbf{f}_1[\mathbf{h}_2, \boldsymbol{\phi}_1].
\end{aligned}
\tag{16.19}
$$

与耦合流一样，分割成块限制了可以表示的变换族。因此，输入在层之间被置换，以便变量可以以任意方式混合。

这个公式可以很容易地求逆，但对于一般的函数 $\mathbf{f}_1[\cdot, \boldsymbol{\phi}_1]$ 和 $\mathbf{f}_2[\cdot, \boldsymbol{\phi}_2]$，没有有效的方法来计算雅可比矩阵。这个公式有时被用来在训练残差网络时节省内存；因为网络是可逆的，所以在前向传播中存储每一层的激活值是不必要的。参考：Problem 16.10

---

> **图 16.8 残差流**
> a) 一个可逆函数是通过将输入分割成 $\mathbf{h}_1$ 和 $\mathbf{h}_2$ 并创建两个残差层来计算的。在第一个中，$\mathbf{h}_2$ 被处理，$\mathbf{h}_1$ 被添加。在第二个中，结果被处理，$\mathbf{h}_2$ 被添加。b) 在反向机制中，函数以相反的顺序计算，加法操作变为减法。

---

### 16.3.7 残差流和收缩映射：iResNet

利用残差网络的另一种方法是利用**巴拿赫不动点定理 (Banach fixed point theorem)** 或**收缩映射定理 (contraction mapping theorem)**，该定理指出每个收缩映射都有一个不动点。一个收缩映射 $\mathbf{f}[\cdot]$ 具有以下性质：

$$
\text{dist}[\mathbf{f}[\mathbf{z}'], \mathbf{f}[\mathbf{z}]] < \beta \cdot \text{dist}[\mathbf{z}', \mathbf{z}] \quad \forall \mathbf{z}, \mathbf{z}',
\tag{16.20}
$$

其中 $\text{dist}[\cdot, \cdot]$ 是一个距离函数，且 $0 < \beta < 1$。当一个具有此属性的函数被迭代（即，输出被重复地作为输入反馈）时，结果会收敛到一个不动点，即 $\mathbf{f}[\mathbf{z}] = \mathbf{z}$（图16.9）。要理解这一点，可以考虑将函数应用于不动点和当前位置；不动点保持不变，但两者之间的距离必须变小，因此当前位置必须更接近不动点。参考：Notebook 16.3 Contraction mappings

这个定理可以用来反转形式为以下的方程：

$$
\mathbf{y} = \mathbf{z} + \mathbf{f}[\mathbf{z}]
\tag{16.21}
$$

如果 $\mathbf{f}[\mathbf{z}]$ 是一个收缩映射。换句话说，它可以用来找到映射到给定值 $\mathbf{y}^*$ 的 $\mathbf{z}^*$。这可以通过从任何点 $\mathbf{z}_0$ 开始并迭代 $\mathbf{z}_{k+1} = \mathbf{y}^* - \mathbf{f}[\mathbf{z}_k]$ 来完成。这在 $\mathbf{z} + \mathbf{f}[\mathbf{z}] = \mathbf{y}^*$ 处有一个不动点（图16.9b）。

同样的原理可以用来反转形式为 $\mathbf{h}' = \mathbf{h} + \mathbf{f}[\mathbf{h}, \boldsymbol{\phi}]$ 的残差网络层，如果我们确保 $\mathbf{f}[\mathbf{h}, \boldsymbol{\phi}]$ 是一个收缩映射。在实践中，这意味着**利普希茨常数 (Lipschitz constant)** 必须小于1。假设激活函数的斜率不大于1，这等价于确保每个权重矩阵 $\mathbf{\Omega}$ 的最大**奇异值 (singular value)** 小于1。一种粗略的方法是确保权重 $\mathbf{\Omega}$ 的绝对量级通过裁剪来保持较小。参考：Appendix B.1.1 Lipschitz constant, Appendix B.3.7 Singular values

雅可比行列式不容易计算，但它的对数可以通过一系列技巧来近似：

$$
\log\left[\left| \mathbf{I} + \frac{\partial\mathbf{f}[\mathbf{h}, \boldsymbol{\phi}]}{\partial\mathbf{h}} \right|\right] = \text{trace}\left[\log\left[\mathbf{I} + \frac{\partial\mathbf{f}[\mathbf{h}, \boldsymbol{\phi}]}{\partial\mathbf{h}}\right]\right] \\
= \sum_{k=1}^\infty \frac{(-1)^{k-1}}{k} \text{trace}\left[\left(\frac{\partial\mathbf{f}[\mathbf{h}, \boldsymbol{\phi}]}{\partial\mathbf{h}}\right)^k\right],
\tag{16.22}
$$

其中我们在第一行使用了恒等式 $\log[|\mathbf{A}|] = \text{trace}[\log[\mathbf{A}]]$，并在第二行中将其展开为幂级数。

即使我们截断这个级数，计算组成项的**迹 (trace)** 在计算上仍然是昂贵的。因此，我们使用**哈钦森迹估计器 (Hutchinson's trace estimator)** 来近似它。考虑一个均值为 $\mathbf{0}$、方差为 $\mathbf{I}$ 的正态随机变量 $\boldsymbol{\epsilon}$。矩阵 $\mathbf{A}$ 的迹可以估计为：

$$
\text{trace}[\mathbf{A}] = \text{trace}[\mathbf{A}\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T]] = \text{trace}[\mathbb{E}[\mathbf{A}\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T]] = \mathbb{E}[\text{trace}[\mathbf{A}\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T]] = \mathbb{E}[\text{trace}[\boldsymbol{\epsilon}^T\mathbf{A}\boldsymbol{\epsilon}]] = \mathbb{E}[\boldsymbol{\epsilon}^T\mathbf{A}\boldsymbol{\epsilon}],
\tag{16.23}
$$

其中第一行为真，因为 $\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T] = \mathbf{I}$。第二行源于期望算子的性质。第三行来自迹算子的线性。第四行是由于迹对循环置换的不变性。最后一行为真，因为第四行中的参数现在是一个标量。我们通过从 $\text{Pr}(\boldsymbol{\epsilon})$ 中抽取样本 $\boldsymbol{\epsilon}_i$ 来估计迹：

$$
\text{trace}[\mathbf{A}] = \mathbb{E}[\boldsymbol{\epsilon}^T\mathbf{A}\boldsymbol{\epsilon}] \approx \frac{1}{I}\sum_{i=1}^I \boldsymbol{\epsilon}_i^T\mathbf{A}\boldsymbol{\epsilon}_i.
\tag{16.24}
$$

通过这种方式，我们可以近似泰勒展开（方程16.22）的幂的迹，并评估对数概率。

---

> **图 16.9 收缩映射**
> 如果一个函数的绝对斜率处处小于1，迭代该函数会收敛到一个不动点 $\mathbf{f}[\mathbf{z}] = \mathbf{z}$。a) 从 $\mathbf{z}_0$ 开始，我们评估 $\mathbf{z}_1 = \mathbf{f}[\mathbf{z}_0]$。然后我们将 $\mathbf{z}_1$ 反馈回函数并迭代。最终，该过程收敛到点 $\mathbf{f}[\mathbf{z}]=\mathbf{z}$（即，函数与虚线对角恒等线相交的地方）。b) 这可以用来反转形式为 $\mathbf{y} = \mathbf{z} + \mathbf{f}[\mathbf{z}]$ 的方程，对于一个值 $\mathbf{y}^*$，通过注意到 $\mathbf{y}^* - \mathbf{f}[\mathbf{z}]$ 的不动点（橙色线与虚线恒等线相交的地方）与 $\mathbf{y}^* = \mathbf{z} + \mathbf{f}[\mathbf{z}]$ 的位置相同。

---

## 16.4 多尺度流

在归一化流中，潜空间 $\mathbf{z}$ 必须与数据空间 $\mathbf{x}$ 的大小相同，但我们知道自然数据集通常可以用更少的底层变量来描述。在某个点上，我们必须引入所有这些变量，但将它们全部通过整个网络是低效的。这引出了**多尺度流 (multi-scale flows)** 的想法（图16.10）。

在生成方向上，多尺度流将潜向量划分为 $\mathbf{z} = [\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_N]$。第一部分 $\mathbf{z}_1$ 由一系列与 $\mathbf{z}_1$ 维度相同的可逆层处理，直到某个点，$\mathbf{z}_2$ 被附加并与第一部分结合。这个过程一直持续到网络的大小与数据 $\mathbf{x}$ 相同。在归一化方向上，网络从 $\mathbf{x}$ 的全维度开始，但当它到达添加 $\mathbf{z}_n$ 的点时，这一部分将根据基础分布进行评估。

---

> **图 16.10 多尺度流**
> 在归一化流中，潜空间 $\mathbf{z}$ 必须与模型密度的大小相同。然而，它可以被划分为几个组件，这些组件可以在不同的层逐渐引入。这使得密度估计和采样都更快。对于逆过程，黑色箭头被反转，每个块的最后一部分跳过剩余的处理。例如，$\mathbf{f}_3^{-1}[\cdot, \boldsymbol{\phi}_3]$ 只在钱三个块上操作，第四个块变成 $\mathbf{z}_4$ 并根据基础密度进行评估。

---

## 16.5 应用

我们现在描述归一化流的三个应用。首先，我们考虑建模概率密度。其次，我们考虑用于合成图像的GLOW模型。最后，我们讨论使用归一化流来近似其他分布。

### 16.5.1 建模密度

在本书讨论的四种生成模型中，归一化流是唯一能够计算新样本精确对数似然的模型。生成对抗网络不是概率性的，变分自编码器和扩散模型都只能返回似然的一个下界。² 图16.11描绘了使用i-ResNet在两个玩具问题中估计的概率分布。密度估计的一个应用是**异常检测 (anomaly detection)**；一个干净数据集的数据分布是使用归一化流模型描述的。低概率的新样本被标记为异常值。然而，必须谨慎，因为可能存在不属于典型集合的高概率异常值（见图8.13）。

### 16.5.2 合成

**生成流 (Generative flows)** 或 **GLOW**，是一个可以创建高保真图像（图16.12）并使用了本章中许多思想的归一化流模型。在归一化方向上理解它最容易。GLOW从一个包含RGB图像的 $256 \times 256 \times 3$ 张量开始。它使用耦合层，其中通道被划分为两半。第二半在每个空间位置都受到一个不同的仿射变换，其中仿射变换的参数是由在另一半通道上运行的二维卷积神经网络计算的。耦合层与 $1 \times 1$ 卷积交替进行，参数化为LU分解，以混合通道。

分辨率会周期性地减半，通过将每个 $2 \times 2$ 的块组合成一个具有四倍通道数的位置。GLOW是一个多尺度流，一些通道被周期性地移除，成为潜向量 $\mathbf{z}$ 的一部分。图像是离散的（由于RGB值的量化），因此向输入添加噪声以防止训练似然无限增长。这被称为**去量化 (dequantization)**。

为了采样更逼真的图像，GLOW模型从一个被提升到正幂的基础密度中采样。这会选择更靠近密度中心的样本，而不是尾部的样本。这类似于GANs中的截断技巧（图15.10）。值得注意的是，这些样本不如GANs或扩散模型的样本好。这是否是由于与可逆层相关的根本限制，还是仅仅因为在这个目标上的研究投入较少，尚不清楚。

图16.13展示了一个使用GLOW进行插值的例子。两个潜向量是通过在归一化方向上变换两个真实图像来计算的。这些潜向量之间的中间点是通过线性插值计算的，然后这些点使用网络在生成方向上被投射回图像空间。结果是一组在两个真实图像之间进行逼真插值的图像。

---

> **图 16.11 建模密度**
> a) 玩具二维数据样本。b) 使用i-ResNet建模的密度。c-d) 第二个例子。改编自 Behrmann et al. (2019)。

> **图 16.12 从在CelebA HQ数据集（Karras et al., 2018）上训练的GLOW中采样的样本。**
> 样本质量尚可，尽管GANs和扩散模型能产生更优的结果。改编自 Kingma & Dhariwal (2018)。

> **图 16.13 使用GLOW模型进行插值**
> 左侧和右侧的图像是真实的人。中间的图像是通过将真实图像投影到潜空间，进行插值，然后将插值点投影回图像空间来计算的。改编自 Kingma & Dhariwal (2018)。

---

### 16.5.3 近似其他密度模型

归一化流也可以学习生成近似一个现有密度的样本，该密度易于评估但难以采样。在这种情况下，我们称归一化流 $\text{Pr}(\mathbf{x}|\boldsymbol{\phi})$ 为**学生 (student)**，目标密度 $q(\mathbf{x})$ 为**教师 (teacher)**。

为了取得进展，我们从学生模型生成样本 $\mathbf{x}_i = \mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}]$。由于我们自己生成了这些样本，我们知道它们对应的潜变量 $\mathbf{z}_i$，并且我们可以在不求逆的情况下计算它们在学生模型中的似然。因此，我们可以使用像掩码自回归流这样求逆缓慢的模型。我们定义一个基于**反向KL散度 (reverse KL divergence)** 的损失函数，该函数鼓励学生和教师的似然相同，并用它来训练学生模型（图16.14）：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\mathrm{argmin}}\ \text{KL}\left[\frac{1}{I}\sum_{i=1}^I \delta[\mathbf{x} - \mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}]] \left|\right| q(\mathbf{x})\right].
\tag{16.25}
$$

这种方法与归一化流的典型用法形成对比，后者使用最大似然来构建一个来自未知分布的样本 $\mathbf{x}_i$ 的概率模型 $\text{Pr}(\mathbf{x}_i, \boldsymbol{\phi})$，这依赖于前向KL散度中的交叉熵项（5.7节）：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\mathrm{argmin}}\ \text{KL}\left[\frac{1}{I}\sum_{i=1}^I \delta[\mathbf{x} - \mathbf{x}_i] \left|\right| \text{Pr}(\mathbf{x}_i, \boldsymbol{\phi})\right].
\tag{16.26}
$$

归一化流可以使用这个技巧来建模VAEs中的后验分布（见第17章）。

## 16.6 总结

归一化流变换一个基础分布（通常是正态分布）来创建一个新的密度。它们的优点是既可以精确地评估样本的似然，又可以生成新的样本。然而，它们的架构约束是每一层都必须是可逆的；我们需要前向变换来生成样本，需要后向变换来评估似然。

同样重要的是，雅可比矩阵可以被高效地估计以评估似然；这必须重复地进行以学习密度。然而，即使雅可比矩阵不能被高效地估计，可逆层本身仍然是有用的；它们将训练一个 $K$ 层网络的内存需求从 $O[K]$ 减少到 $O$。

本章回顾了可逆网络层或流。我们考虑了线性和逐元素流，它们简单但表达能力不足。然后我们描述了更复杂的流，如耦合流、自回归流和残差流。最后，我们展示了如何使用归一化流来估计似然、生成和插值图像，以及近似其他分布。

---

> **图 16.14 近似密度模型**
> a) 训练数据。b) 通常，我们修改流模型参数以最小化从训练数据到流模型的KL散度。这等价于最大似然拟合（5.7节）。c) 或者，我们可以修改流参数 $\boldsymbol{\phi}$ 以最小化从流样本 $\mathbf{x}_i = \mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}]$ 到 d) 一个目标密度的KL散度。

---

### 注释

归一化流最早由 Rezende & Mohamed (2015) 引入，但其思想渊源可以追溯到 Tabak & Vanden-Eijnden (2010), Tabak & Turner (2013), 和 Rippel & Adams (2013) 的工作。归一化流的综述可以在 Kobyzev et al. (2020) 和 Papamakarios et al. (2021) 中找到。Kobyzev et al. (2020) 对许多归一化流方法进行了定量比较。他们得出结论，Flow++模型（一个带有新颖的逐元素变换和其他创新的耦合流）在当时表现最好。

**可逆网络层 (Invertible network layers)**：可逆层减少了反向传播算法的内存需求；前向传播中的激活值不再需要存储，因为它们可以在后向传播中重新计算。除了本章讨论的常规网络层和残差层 (Gomez et al., 2017; Jacobsen et al., 2018) 外，还为图神经网络 (Li et al., 2021a)、循环神经网络 (MacKay et al., 2018)、掩码卷积 (Song et al., 2019)、U-Nets (Brügger et al., 2019; Etmann et al., 2020) 和 Transformer (Mangalam et al., 2022) 开发了可逆层。

**径向流和平面流 (Radial and planar flows)**：最初的归一化流论文 (Rezende & Mohamed, 2015) 使用了平面流（沿某些维度收缩或扩展分布）和径向流（围绕某个点扩展或收缩）。这些流的逆不容易计算，但它们对于近似采样缓慢或似然只能评估到一个未知缩放因子的分布很有用（图16.14）。

**应用 (Applications)**：应用包括图像生成 (Ho et al., 2019; Kingma & Dhariwal, 2018)、噪声建模 (Abdelhamed et al., 2019)、视频生成 (Kumar et al., 2019b)、音频生成 (Esling et al., 2019; Kim et al., 2018; Prenger et al., 2019)、图生成 (Madhawa et al., 2019)、图像分类 (Kim et al., 2021; Mackowiak et al., 2021)、图像隐写 (Lu et al., 2021)、超分辨率 (Yu et al., 2020; Wolf et al., 2021; Liang et al., 2021)、风格迁移 (An et al., 2021)、运动风格迁移 (Wen et al., 2021)、3D形状建模 (Paschalidou et al., 2021)、压缩 (Zhang et al., 2021b)、sRGB到RAW图像转换 (Xing et al., 2021)、去噪 (Liu et al., 2021b)、异常检测 (Yu et al., 2021)、图像到图像翻译 (Ardizzone et al., 2020)、在不同分子干预下合成细胞显微镜图像 (Yang et al., 2021)，和光传输模拟 (Müller et al., 2019b)。对于使用图像数据的应用，学习前必须添加噪声，因为输入是量化的，因此是离散的（见 Theis et al., 2016）。
Rezende & Mohamed (2015) 使用归一化流来建模VAEs中的后验分布。Abdal et al. (2021) 使用归一化流来建模StyleGAN潜空间中属性的分布，然后使用这些分布来改变真实图像中的指定属性。Wolf et al. (2021) 使用归一化流来学习给定干净图像的噪声输入图像的条件图像，从而模拟可以用来训练去噪或超分辨率模型的噪声数据。
归一化流在物理学 (Kanwar et al., 2020; Köhler et al., 2020; Noé et al., 2019; Wirnsberger et al., 2020; Wong et al., 2020)、自然语言处理 (Tran et al., 2019; Ziegler & Rush, 2019; Zhou et al., 2019; He et al., 2018; Jin et al., 2019) 和强化学习 (Schroecker et al., 2019; Haarnoja et al., 2018a; Mazoure et al., 2020; Ward et al., 2019; Touati et al., 2020) 中也发现了广泛的用途。

**线性流 (Linear flows)**：对角线性流可以表示像BatchNorm (Dinh et al., 2016) 和ActNorm (Kingma & Dhariwal, 2018) 这样的归一化变换。Tomczak & Welling (2016) 研究了组合三角矩阵和使用由Householder变换参数化的正交变换。Kingma & Dhariwal (2018) 提出了16.5.2节中描述的LU参数化。Hoogeboom et al. (2019b) 提出了使用QR分解，这不需要预定的置换矩阵。卷积是深度学习中广泛使用的线性变换（图10.4），但它们的逆和行列式不直接。Kingma & Dhariwal (2018) 使用了 $1 \times 1$ 卷积，这实际上是在每个位置上分别应用的全线性变换。Zheng et al. (2017) 引入了ConvFlow，它被限制在一维卷积。Hoogeboom et al. (2019b) 为建模二维卷积提供了更通用的解决方案，要么通过堆叠掩码自回归卷积，要么通过在傅里叶域中操作。

**逐元素流和耦合函数 (Elementwise flows and coupling functions)**：逐元素流独立地使用相同的函数变换每个变量（但每个变量的参数不同）。相同的流可以用来形成耦合和自回归流中的耦合函数，在这种情况下，它们的参数取决于前面的变量。为了可逆，这些函数必须是单调的。
一个加性耦合函数 (Dinh et al., 2015) 只是向变量添加一个偏移量。仿射耦合函数缩放变量并添加一个偏移量，被Dinh et al. (2015), Dinh et al. (2016), Kingma & Dhariwal (2018), Kingma et al. (2016), 和 Papamakarios et al. (2017) 使用。Ziegler & Rush (2019) 提出了非线性平方流，它是一个带有五个参数的多项式的可逆比率。连续混合CDF (Ho et al., 2019) 应用一个基于K个logistics混合的累积密度函数（CDF）的单调变换，后由一个逆logistic sigmoid、缩放和偏移组成。
分段线性耦合函数（图16.5）由 Müller et al. (2019b) 开发。从那时起，基于三次样条 (Durkan et al., 2019a) 和有理二次样条 (Durkan et al., 2019b) 的系统被提出。Huang et al. (2018a) 引入了神经自回归流，其中函数由一个产生单调函数的神经网络表示。一个充分条件是权重都是正的，并且激活函数是单调的。训练一个权重受正约束的网络是困难的，因此这导致了非受约束的单调神经网络 (Wehenkel & Louppe, 2019)，它建模严格正函数，然后数值积分以得到单调函数。Jaini et al. (2019) 构建了可以以封闭形式积分的正函数，基于一个经典结果，即所有正的单变量多项式都是多项式的平方和。最后，Dinh et al. (2019) 研究了分段单调耦合函数。

**耦合流 (Coupling flows)**：Dinh et al. (2015) 引入了将维度分成两半的耦合流（图16.6）。Dinh et al. (2016) 引入了RealNVP，它通过取交替的像素或通道块来划分图像输入。Das et al. (2019) 提出了根据导数的大小来选择传播部分的特征。Dinh et al. (2016) 将多尺度流（其中维度是逐渐引入的）解释为参数对另一半数据没有依赖性的耦合流。Kruse et al. (2021) 引入了耦合流的层次化公式，其中每个分区被递归地分成两个。GLOW（图16.12-16.13）由 Kingma & Dhariwal (2018) 设计并使用耦合流，NICE (Dinh et al., 2015)、RealNVP (Dinh et al., 2016)、FloWaveNet (Kim et al., 2018)、WaveGLOW (Prenger et al., 2019) 和 Flow++ (Ho et al., 2019) 也是如此。

**自回归流 (Autoregressive flows)**：Kingma et al. (2016) 使用自回归模型进行归一化流。Germain et al. (2015) 开发了一种通用方法来掩蔽先前的变量。Papamakarios et al. (2017) 利用这一点在掩码自回归流中同时计算前向方向的所有输出。Kingma et al. (2016) 引入了逆自回归流。Parallel WaveNet (Van den Oord et al., 2018) 将WaveNet (Van den Oord et al., 2016a)（一种不同类型的音频生成模型）提炼成一个逆自回归流，以便采样可以很快（见图16.14c-d）。

**残差流 (Residual flows)**：残差流基于残差网络 (He et al., 2016a)。RevNets (Gomez et al., 2017) 和 iRevNets (Jacobsen et al., 2018) 将输入分为两个部分（图16.8），每个部分都通过一个残差网络。这些网络是可逆的，但雅可比矩阵的行列式不容易计算。残差连接可以被解释为常微分方程的离散化，这种观点导致了不同的可逆架构 (Chang et al., 2018, 2019a)。然而，这些网络的雅可比矩阵仍然不能高效地计算。Behrmann et al. (2019) 注意到，如果其利普希茨常数小于1，网络可以使用不动点迭代来求逆。这导致了iResNet，其中雅可比矩阵的对数行列式可以使用哈钦森的迹估计器来估计 (Hutchinson, 1989)。Chen et al. (2019) 通过使用俄罗斯轮盘赌估计器移除了方程16.22中幂级数截断引起的偏差。

**无穷小流 (Infinitesimal flows)**：如果残差网络可以被看作是常微分方程（ODE）的离散化，那么下一个逻辑步骤是直接用ODE来表示变量的变化。神经ODE由 Chen et al. (2018e) 探索，并利用了ODE中前向和后向传播的标准方法。不再需要雅可比矩阵来计算似然；这由一个不同的ODE表示，其中对数概率的变化与前向传播的导数的迹相关。Grathwohl et al. (2019) 使用哈钦森估计器来估计迹，并进一步简化了这一点。Finlay et al. (2020) 向损失函数添加了正则化项，使训练更容易，而 Dupont et al. (2019) 增强了表示，以允许神经ODE表示更广泛的微分同胚。Tzen & Raginsky (2019) 和 Peluchetti & Favaro (2020) 用随机微分方程取代了ODEs。

**普适性 (Universality)**：普适性指的是归一化流能够任意好地建模任何概率分布的能力。一些流（例如，平面的，逐元素的）不具有此属性。当耦合函数是神经单调网络 (Huang et al., 2018a)、基于单调多项式 (Jaini et al., 2020) 或基于样条 (Kobyzev et al., 2020) 时，自回归流可以被证明具有普适性。对于维度 $D$，一系列 $D$ 个耦合流可以形成一个自回归流。要理解为什么，请注意，划分为两部分 $\mathbf{h}_1$ 和 $\mathbf{h}_2$ 意味着在任何给定层 $\mathbf{h}_2$ 仅依赖于先前的变量（图16.6）。因此，如果我们每层将 $\mathbf{h}_1$ 的大小增加一个，我们就可以重现一个自回归流，结果是普适的。尚不清楚耦合流是否可以用少于 $D$ 层来达到普适性。然而，它们在实践中（例如，GLOW）效果很好，而不需要这种诱导的自回归结构。

**其他工作 (Other work)**：归一化流的研究活跃领域包括离散流 (Hoogeboom et al., 2019a; Tran et al., 2019)、非欧几里得流形上的归一化流 (Gemici et al., 2016; Wang & Wang, 2019) 和旨在创建对变换族不变的密度的等变流 (Köhler et al., 2020; Rezende et al., 2019) 的研究。

---

***

### 习题

**问题 16.1** 考虑使用函数 $x=f[z]=z^2$ 变换定义在 $z \in$ 上的均匀基础密度。求变换后分布 $\text{Pr}(x)$ 的表达式。

**思路与解答：**

1.  基础分布: $\text{Pr}(z) = 1$ for $z \in$。
2.  变换函数: $x = z^2$。
3.  逆函数: $z = f^{-1}(x) = \sqrt{x}$。
4.  导数: $\frac{dx}{dz} = \frac{df[z]}{dz} = 2z$。
5.  应用变量变换公式: $\text{Pr}(x) = \text{Pr}(z(x)) \left|\frac{dz}{dx}\right|$。
    或者使用本书公式 (16.1): $\text{Pr}(x) = \text{Pr}(z) \left|\frac{df[z]}{dz}\right|^{-1}$。
    $\text{Pr}(x) = 1 \cdot |2z|^{-1} = \frac{1}{2z}$。
6.  用 $x$ 表示: 将 $z=\sqrt{x}$ 代入，得到 $\text{Pr}(x) = \frac{1}{2\sqrt{x}}$。
7.  定义域: 当 $z \in$ 时，$x \in$。
    所以，$\text{Pr}(x) = \frac{1}{2\sqrt{x}}$ for $x \in$。

**问题 16.2*** 考虑变换一个标准正态分布... 求变换后分布 $\text{Pr}(x)$ 的表达式。

**思路与解答：**

1.  基础分布: $\text{Pr}(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$。
2.  变换函数: $x = f[z] = \frac{1}{1+e^{-z}}$ (Logistic Sigmoid 函数)。
3.  逆函数: 从 $x = \frac{1}{1+e^{-z}}$ 解出 $z$。
    $\frac{1}{x} = 1+e^{-z} \implies e^{-z} = \frac{1}{x}-1 = \frac{1-x}{x}$
    $-z = \log\left(\frac{1-x}{x}\right) \implies z = \log\left(\frac{x}{1-x}\right)$ (Logit 函数)。
4.  导数: $f'[z] = \frac{d}{dz} (1+e^{-z})^{-1} = -(1+e^{-z})^{-2}(-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}$。
    我们可以用 $x$ 来简化它: $f'[z] = \frac{1}{1+e^{-z}} \frac{e^{-z}}{1+e^{-z}} = x(1-x)$。
5.  应用公式 (16.1): $\text{Pr}(x) = \text{Pr}(z) |f'[z]|^{-1}$。
    $\text{Pr}(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left[\log\left(\frac{x}{1-x}\right)\right]^2\right) \cdot \frac{1}{x(1-x)}$。
6.  定义域: $z \in (-\infty, \infty)$ 映射到 $x \in (0,1)$。
    这个分布被称为**对数正态分布 (Logit-normal distribution)**。

**问题 16.3*** 写出逆映射 $z=f^{-1}[x, \phi]$ 的雅可比矩阵及其绝对行列式的表达式，形式类似于方程16.6和16.7。

**思路与解答：**

1.  **逆映射的雅可比矩阵**:
    我们有 $\mathbf{z} = \mathbf{f}_1^{-1}[\mathbf{f}_2^{-1}[\dots \mathbf{f}_K^{-1}[\mathbf{x}, \boldsymbol{\phi}_K] \dots]]$。
    应用链式法则:
    $\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{f}_1^{-1}}{\partial \mathbf{f}_2^{-1}} \cdot \frac{\partial \mathbf{f}_2^{-1}}{\partial \mathbf{f}_3^{-1}} \cdots \frac{\partial \mathbf{f}_K^{-1}}{\partial \mathbf{x}}$
    令 $\mathbf{y}_k = \mathbf{f}_{k+1}^{-1}[\dots]$，则 $\frac{\partial \mathbf{y}_{k-1}}{\partial \mathbf{y}_k} = \frac{\partial \mathbf{f}_k^{-1}}{\partial \mathbf{y}_k}$。
    因此:
    $$
    \frac{\partial \mathbf{f}^{-1}[\mathbf{x}, \boldsymbol{\phi}]}{\partial \mathbf{x}} = \frac{\partial \mathbf{f}_K^{-1}[\mathbf{x}, \boldsymbol{\phi}_K]}{\partial \mathbf{x}} \cdot \frac{\partial \mathbf{f}_{K-1}^{-1}[\mathbf{f}_K^{-1}, \boldsymbol{\phi}_{K-1}]}{\partial \mathbf{f}_K^{-1}} \cdots \frac{\partial \mathbf{f}_1^{-1}[\mathbf{f}_2^{-1}, \boldsymbol{\phi}_1]}{\partial \mathbf{f}_2^{-1}}
    $$
2.  **绝对行列式**:
    利用 $|AB| = |A||B|$ 的性质:
    $$
    \left| \frac{\partial \mathbf{f}^{-1}[\mathbf{x}, \boldsymbol{\phi}]}{\partial \mathbf{x}} \right| = \left| \frac{\partial \mathbf{f}_K^{-1}[\mathbf{x}, \boldsymbol{\phi}_K]}{\partial \mathbf{x}} \right| \cdot \left| \frac{\partial \mathbf{f}_{K-1}^{-1}[\mathbf{f}_K^{-1}, \boldsymbol{\phi}_{K-1}]}{\partial \mathbf{f}_K^{-1}} \right| \cdots \left| \frac{\partial \mathbf{f}_1^{-1}[\mathbf{f}_2^{-1}, \boldsymbol{\phi}_1]}{\partial \mathbf{f}_2^{-1}} \right|
    $$
    同时，根据反函数定理，$\left|\frac{\partial \mathbf{f}^{-1}(\mathbf{y})}{\partial \mathbf{y}}\right| = \left|\frac{\partial \mathbf{f}(\mathbf{x})}{\partial \mathbf{x}}\right|^{-1}$，所以上式等于方程16.7的倒数。

**问题 16.4** 手动计算下列矩阵的逆和行列式。

**思路与解答：**

*   **对于 $\mathbf{\Omega}_1$**: 这是一个对角矩阵。
    *   **逆**: 对角线上每个元素取倒数。
        $$
        \mathbf{\Omega}_1^{-1} = \begin{pmatrix} 1/2 & 0 & 0 & 0 \\ 0 & -1/5 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1/2 \end{pmatrix}
        $$
    *   **行列式**: 对角线上元素的乘积。
        $\det(\mathbf{\Omega}_1) = 2 \times (-5) \times 1 \times 2 = -20$。

*   **对于 $\mathbf{\Omega}_2$**: 这是一个下三角矩阵。
    *   **逆**: 逆矩阵也是下三角矩阵，可以通过高斯-若尔当消元法等方法求解。
    *   **行列式**: 对角线上元素的乘积。
        $\det(\mathbf{\Omega}_2) = 1 \times 4 \times 2 \times 1 = 8$。

**问题 16.5** 考虑一个均值为 $\boldsymbol{\mu}$、协方差为 $\mathbf{\Sigma}$ 的随机变量 $\mathbf{z}$，被变换为 $\mathbf{x} = \mathbf{Az} + \mathbf{b}$。证明 $\mathbf{x}$ 的期望值为 $\mathbf{A}\boldsymbol{\mu}+\mathbf{b}$，协方差为 $\mathbf{A}\mathbf{\Sigma}\mathbf{A}^T$。

**思路与解答：**

1.  **期望值**:
    $\mathbb{E}[\mathbf{x}] = \mathbb{E}[\mathbf{Az} + \mathbf{b}]$
    利用期望的线性性质 $\mathbb{E}[aX+b] = a\mathbb{E}[X]+b$:
    $\mathbb{E}[\mathbf{x}] = \mathbf{A}\mathbb{E}[\mathbf{z}] + \mathbf{b} = \mathbf{A}\boldsymbol{\mu} + \mathbf{b}$。

2.  **协方差**:
    $\text{Cov}(\mathbf{x}) = \mathbb{E}[(\mathbf{x}-\mathbb{E}[\mathbf{x}])(\mathbf{x}-\mathbb{E}[\mathbf{x}])^T]$
    $\mathbf{x}-\mathbb{E}[\mathbf{x}] = (\mathbf{Az}+\mathbf{b}) - (\mathbf{A}\boldsymbol{\mu}+\mathbf{b}) = \mathbf{A}(\mathbf{z}-\boldsymbol{\mu})$
    $\text{Cov}(\mathbf{x}) = \mathbb{E}[\mathbf{A}(\mathbf{z}-\boldsymbol{\mu})(\mathbf{A}(\mathbf{z}-\boldsymbol{\mu}))^T]$
    $= \mathbb{E}[\mathbf{A}(\mathbf{z}-\boldsymbol{\mu})(\mathbf{z}-\boldsymbol{\mu})^T \mathbf{A}^T]$
    利用期望的线性性质，将常数矩阵 $\mathbf{A}$ 和 $\mathbf{A}^T$ 提出:
    $= \mathbf{A} \mathbb{E}[(\mathbf{z}-\boldsymbol{\mu})(\mathbf{z}-\boldsymbol{\mu})^T] \mathbf{A}^T$
    其中 $\mathbb{E}[(\mathbf{z}-\boldsymbol{\mu})(\mathbf{z}-\boldsymbol{\mu})^T]$ 正是 $\mathbf{z}$ 的协方差 $\mathbf{\Sigma}$。
    因此, $\text{Cov}(\mathbf{x}) = \mathbf{A}\mathbf{\Sigma}\mathbf{A}^T$。

**问题 16.6*** 证明如果 $\mathbf{x} = \mathbf{Az}+\mathbf{b}$ 且 $\text{Pr}(\mathbf{z}) = \text{Norm}_\mathbf{z}[\boldsymbol{\mu}, \mathbf{\Sigma}]$，那么 $\text{Pr}(\mathbf{x}) = \text{Norm}_\mathbf{x}[\mathbf{A}\boldsymbol{\mu}+\mathbf{b}, \mathbf{A}\mathbf{\Sigma}\mathbf{A}^T]$。

**思路与解答：**

1.  从变量变换公式 $\text{Pr}(\mathbf{x}) = \text{Pr}(\mathbf{z}(\mathbf{x})) \left| \frac{\partial \mathbf{z}}{\partial \mathbf{x}} \right|$ 开始。
2.  $\mathbf{z} = \mathbf{A}^{-1}(\mathbf{x}-\mathbf{b})$。
3.  $\text{Pr}(\mathbf{z})$ 的概率密度函数为:
    $\frac{1}{\sqrt{(2\pi)^D \det(\mathbf{\Sigma})}} \exp\left(-\frac{1}{2}(\mathbf{z}-\boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{z}-\boldsymbol{\mu})\right)$
4.  雅可比行列式: $\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \mathbf{A}^{-1}$。所以 $\left|\frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right| = |\det(\mathbf{A}^{-1})| = \frac{1}{|\det(\mathbf{A})|}$。
5.  代入:
    $\text{Pr}(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^D \det(\mathbf{\Sigma})}} \exp\left(-\frac{1}{2}(\mathbf{A}^{-1}(\mathbf{x}-\mathbf{b})-\boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{A}^{-1}(\mathbf{x}-\mathbf{b})-\boldsymbol{\mu})\right) \frac{1}{|\det(\mathbf{A})|}$
6.  整理指数项:
    $(\mathbf{A}^{-1}(\mathbf{x}-(\mathbf{A}\boldsymbol{\mu}+\mathbf{b})))^T \mathbf{\Sigma}^{-1} (\mathbf{A}^{-1}(\mathbf{x}-(\mathbf{A}\boldsymbol{\mu}+\mathbf{b})))$
    $= (\mathbf{x}-\boldsymbol{\mu}_x)^T (\mathbf{A}^{-1})^T \mathbf{\Sigma}^{-1} \mathbf{A}^{-1} (\mathbf{x}-\boldsymbol{\mu}_x)$
    $= (\mathbf{x}-\boldsymbol{\mu}_x)^T (\mathbf{A}\mathbf{\Sigma}\mathbf{A}^T)^{-1} (\mathbf{x}-\boldsymbol{\mu}_x)$
7.  整理系数项:
    $\frac{1}{|\det(\mathbf{A})|\sqrt{\det(\mathbf{\Sigma})}} = \frac{1}{\sqrt{\det(\mathbf{A})^2\det(\mathbf{\Sigma})}} = \frac{1}{\sqrt{\det(\mathbf{A}\mathbf{\Sigma}\mathbf{A}^T)}}$
8.  组合起来，得到 $\text{Pr}(\mathbf{x}) = \text{Norm}_\mathbf{x}[\boldsymbol{\mu}_x, \mathbf{\Sigma}_x]$，其中 $\boldsymbol{\mu}_x = \mathbf{A}\boldsymbol{\mu}+\mathbf{b}$，$\mathbf{\Sigma}_x = \mathbf{A}\mathbf{\Sigma}\mathbf{A}^T$。

**问题 16.7** Leaky ReLU...写出它的逆函数和逆雅可比行列式绝对值的表达式。

**思路与解答：**

1.  **逆函数**:
    令 $y = \text{LReLU}(z)$。
    *   如果 $z < 0$, $y = 0.1z \implies z = 10y$。因为 $z<0$, 所以 $y<0$。
    *   如果 $z \ge 0$, $y = z$。因为 $z \ge 0$, 所以 $y \ge 0$。
    $$
    \text{LReLU}^{-1}[y] = \begin{cases} 10y & y < 0 \\ y & y \ge 0 \end{cases}
    $$
2.  **逆雅可比行列式绝对值**:
    对于逐元素变换 $\mathbf{x} = \mathbf{f}[\mathbf{z}]$，$\left|\frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right|^{-1} = \left|\prod_d \frac{df_d}{dz_d}\right|^{-1} = \prod_d \left|\frac{df_d}{dz_d}\right|^{-1}$。
    $\frac{d \text{LReLU}(z)}{dz} = \begin{cases} 0.1 & z < 0 \\ 1 & z \ge 0 \end{cases}$。
    所以 $\left|\frac{d \text{LReLU}(z_d)}{dz_d}\right|^{-1} = \begin{cases} 10 & z_d < 0 \\ 1 & z_d \ge 0 \end{cases}$。
    最终的行列式是所有维度上这些值的乘积:
    $$
    \left|\frac{\partial \mathbf{f}[\mathbf{z}]}{\partial \mathbf{z}}\right|^{-1} = \prod_{d=1}^D \begin{cases} 10 & z_d < 0 \\ 1 & z_d \ge 0 \end{cases} = 10^{\text{count}(z_d < 0)}
    $$

**问题 16.8** 考虑将分段线性函数...应用于输入...雅可比矩阵和它的行列式是什么？

**思路与解答：**

因为变换是逐元素应用的，所以...
1.  **雅可比矩阵**: 是一个**对角矩阵**。
    其对角线上的第 $d$ 个元素是 $\frac{\partial f[h_d, \boldsymbol{\phi}]}{\partial h_d}$。
    从方程16.12可知，导数等于 $h_d$ 所在区间的斜率 $\phi_{b_d} K$。
    所以，$\mathbf{J}_{dd} = K \phi_{b_d}$，其中 $b_d = \lfloor Kh_d \rfloor + 1$。
2.  **行列式**: 是对角线上元素的乘积。
    $$
    \det(\mathbf{J}) = \prod_{d=1}^D (K \phi_{b_d})
    $$

**问题 16.9*** 考虑构建一个基于...平方根函数的分段流...画出函数和它的逆函数。

**思路与解答：**
这是一个函数可视化问题，需要根据公式绘制。
*   $K=5$, bins are `[0, 0.2]`, `(0.2, 0.4]`, ...
*   **函数 $h' = f[h, \boldsymbol{\phi}]$**: 在每个区间内，它是一个平移和缩放的平方根函数。由于所有 $\phi_k$ 都是正的，函数是单调递增的，因此可逆。
*   **逆函数 $h = f^{-1}[h', \boldsymbol{\phi}]$**: 逆函数将是一个分段的**平方函数**。

**问题 16.10** 画出残差流...的雅可比矩阵结构。

**思路与解答：**

令 $\mathbf{h} = [\mathbf{h}_1, \mathbf{h}_2]$ 和 $\mathbf{h}'=[\mathbf{h}'_1, \mathbf{h}'_2]$。
$\mathbf{h}'_1 = \mathbf{h}_1 + \mathbf{f}_1[\mathbf{h}_2]$
$\mathbf{h}'_2 = \mathbf{h}_2 + \mathbf{f}_2[\mathbf{h}'_1] = \mathbf{h}_2 + \mathbf{f}_2[\mathbf{h}_1 + \mathbf{f}_1[\mathbf{h}_2]]$
雅可比矩阵 $\mathbf{J} = \begin{pmatrix} \frac{\partial \mathbf{h}'_1}{\partial \mathbf{h}_1} & \frac{\partial \mathbf{h}'_1}{\partial \mathbf{h}_2} \\ \frac{\partial \mathbf{h}'_2}{\partial \mathbf{h}_1} & \frac{\partial \mathbf{h}'_2}{\partial \mathbf{h}_2} \end{pmatrix}$。
*   $\frac{\partial \mathbf{h}'_1}{\partial \mathbf{h}_1} = \mathbf{I}$
*   $\frac{\partial \mathbf{h}'_1}{\partial \mathbf{h}_2} = \frac{\partial \mathbf{f}_1}{\partial \mathbf{h}_2}$
*   $\frac{\partial \mathbf{h}'_2}{\partial \mathbf{h}_1} = \frac{\partial \mathbf{f}_2}{\partial \mathbf{h}'_1}\frac{\partial \mathbf{h}'_1}{\partial \mathbf{h}_1} = \frac{\partial \mathbf{f}_2}{\partial \mathbf{h}'_1}$
*   $\frac{\partial \mathbf{h}'_2}{\partial \mathbf{h}_2} = \mathbf{I} + \frac{\partial \mathbf{f}_2}{\partial \mathbf{h}'_1}\frac{\partial \mathbf{h}'_1}{\partial \mathbf{h}_2} = \mathbf{I} + \frac{\partial \mathbf{f}_2}{\partial \mathbf{h}'_1} \frac{\partial \mathbf{f}_1}{\partial \mathbf{h}_2}$

1.  **(i) $\mathbf{f}_1, \mathbf{f}_2$ 是全连接网络**:
    所有子块 $\frac{\partial \mathbf{f}_1}{\partial \mathbf{h}_2}$ 等都是**稠密矩阵**。因此，整个雅可比矩阵 $\mathbf{J}$ 是一个**稠密矩阵**。
2.  **(ii) $\mathbf{f}_1, \mathbf{f}_2$ 是逐元素流**:
    所有子块 $\frac{\partial \mathbf{f}_1}{\partial \mathbf{h}_2}$ 等都是**对角矩阵**。因此，整个雅可比矩阵 $\mathbf{J}$ 具有块结构，但每个块都是对角的。

**问题 16.11*** 写出方程16.25中KL散度的表达式。为什么我们只能评估 $q(x)$ 到一个缩放因子 $\kappa$ 也没关系？网络需要可逆才能最小化这个损失函数吗？

**思路与解答：**

1.  **KL散度表达式**:
    令 $P_{student}(\mathbf{x}) = \frac{1}{I} \sum_i \delta[\mathbf{x} - \mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}]]$。
    $KL(P_{student} || q) = \int P_{student}(\mathbf{x}) \log\frac{P_{student}(\mathbf{x})}{q(\mathbf{x})} d\mathbf{x}$
    $= \frac{1}{I}\sum_i \int \delta[\mathbf{x} - \mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}]] \log\frac{P_{student}(\mathbf{x})}{q(\mathbf{x})} d\mathbf{x}$
    $= \frac{1}{I}\sum_i \log\frac{P_{student}(\mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}])}{q(\mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}])}$
    $P_{student}$ 是离散的，这个表达式有问题。
    正确的理解是，我们用从学生模型采样的点来**蒙特卡洛估计** KL散度。
    $KL(P_{student} || q) = \mathbb{E}_{\mathbf{x} \sim P_{student}} [\log P_{student}(\mathbf{x}) - \log q(\mathbf{x})]$
    $\approx \frac{1}{I} \sum_{i=1}^I (\log \text{Pr}_{student}(\mathbf{x}_i|\boldsymbol{\phi}) - \log q(\mathbf{x}_i))$
    其中 $\mathbf{x}_i = \mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}]$。
2.  **缩放因子 $\kappa$**:
    如果 $q(\mathbf{x}) = \kappa \tilde{q}(\mathbf{x})$，其中 $\tilde{q}$ 是我们能评估的。
    $\log q(\mathbf{x}) = \log\kappa + \log\tilde{q}(\mathbf{x})$。
    KL散度中的 $\log\kappa$ 是一个与参数 $\boldsymbol{\phi}$ 无关的常数，在求梯度时会消失。因此**没关系**。
3.  **网络是否需要可逆**:
    **不需要**。因为我们从学生模型**正向采样** $\mathbf{x}_i = \mathbf{f}[\mathbf{z}_i, \boldsymbol{\phi}]$，我们已经知道了每个 $\mathbf{x}_i$ 对应的 $\mathbf{z}_i$。我们可以直接计算 $\text{Pr}_{student}(\mathbf{x}_i|\boldsymbol{\phi}) = \text{Pr}(\mathbf{z}_i) |\det(\partial\mathbf{f}/\partial\mathbf{z})|^{-1}$，而**无需计算逆函数 $f^{-1}$**。这就是可以使用逆向慢的自回归流作为学生模型的原因。