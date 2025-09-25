# 附录 A
# 符号表示

本附录详细说明了本书中使用的符号。这主要遵循计算机科学中的标准惯例，但深度学习适用于许多不同领域，因此在此进行全面解释。此外，还有一些本书独有的符号惯例，包括函数的表示法以及参数和变量之间的系统性区分。

**标量、向量、矩阵和张量**
标量用小写或大写字母 $a, A, \alpha$ 表示。列向量（即一维数字数组）用小写粗体字母 $\mathbf{a}, \boldsymbol{\phi}$ 表示，行向量则为列向量的转置 $\mathbf{a}^T, \boldsymbol{\phi}^T$。矩阵和张量（即二维和N维数字数组）都用大写粗体字母 $\mathbf{B}, \mathbf{\Phi}$ 表示。

**变量和参数**
变量（通常是函数的输入和输出或中间计算）总是用罗马字母 $\mathbf{a}, \mathbf{b}, \mathbf{C}$ 表示。参数（函数或概率分布的内部参数）总是用希腊字母 $\alpha, \beta, \Gamma$ 表示。通用的、未指定的参数用 $\phi$ 表示。这一区别在全书中都保持，除了强化学习中的策略，根据通常的惯例用 $\pi$ 表示。

**集合**
集合用花括号表示，所以 $\{0, 1, 2\}$ 表示数字0, 1, 和 2。符号 $\{0, 1, 2, \dots\}$ 表示非负整数集。有时，我们想指定一个变量集，$\{x_i\}_{i=1}^I$ 表示 $I$ 个变量 $x_1, \dots, x_I$。当不必指定集合中有多少项时，这被缩短为 $\{x_i\}$。符号 $\{\mathbf{x}_i, y_i\}_{i=1}^I$ 表示 $I$ 对 $\mathbf{x}_i, y_i$ 的集合。命名集合的惯例是使用书法字体。值得注意的是，$\mathcal{B}_t$ 用来表示训练期间迭代 $t$ 时一个批次中的索引集。一个集合 $\mathcal{S}$ 中的元素数量用 $|\mathcal{S}|$ 表示。
集合 $\mathbb{R}$ 表示实数集。集合 $\mathbb{R}^+$ 表示非负实数集。符号 $\mathbb{R}^D$ 表示包含实数的 $D$ 维向量集。符号 $\mathbb{R}^{D_1 \times D_2}$ 表示维度为 $D_1 \times D_2$ 的矩阵集。符号 $\mathbb{R}^{D_1 \times D_2 \times D_3}$ 表示大小为 $D_1 \times D_2 \times D_3$ 的张量集，依此类推。
符号 $[a, b]$ 表示从 $a$ 到 $b$ 的实数，包括 $a$ 和 $b$ 本身。当方括号被圆括号替换时，表示不包括相邻的值。例如，集合 $(-\pi, \pi]$ 表示从 $-\pi$ 到 $\pi$ 的实数，但不包括 $-\pi$。
集合的成员资格用符号 $\in$ 表示，所以 $x \in \mathbb{R}^+$ 意味着变量 $x$ 是一个非负实数，而符号 $\mathbf{\Sigma} \in \mathbb{R}^{D \times D}$ 表示 $\mathbf{\Sigma}$ 是一个大小为 $D \times D$ 的矩阵。有时，我们想系统地处理一个集合的每个元素，符号 $\forall \{1, \dots, K\}$ 意味着“对于所有”从1到K的整数。

**函数**
函数以一个名称表示，后面跟着包含该函数参数的方括号。例如，$\log[x]$ 返回变量 $x$ 的对数。当函数返回一个向量时，它用粗体书写并以小写字母开头。例如，函数 $\mathbf{y} = \text{mlp}[\mathbf{x}, \boldsymbol{\phi}]$ 返回一个向量 $\mathbf{y}$，并有向量参数 $\mathbf{x}$ 和 $\boldsymbol{\phi}$。当一个函数返回一个矩阵或张量时，它用粗体书写并以大写字母开头。例如，函数 $\mathbf{Y} = \text{Sa}[\mathbf{X}, \boldsymbol{\phi}]$ 返回一个矩阵 $\mathbf{Y}$，并有参数 $\mathbf{X}$ 和 $\boldsymbol{\phi}$。当我们想故意模糊一个函数的参数时，我们使用点符号（例如，$\text{mlp}[\cdot, \boldsymbol{\phi}]$）。

**最小化和最大化**
一些特殊函数在全文中被反复使用：
*   函数 $\min_x[f[x]]$ 返回函数 $f[x]$ 在变量 $x$ 的所有可能值上的最小值。这个符号通常在不指定如何找到这个最小值的细节的情况下使用。
*   函数 $\text{argmin}_x[f[x]]$ 返回最小化 $f[x]$ 的 $x$ 的值，所以如果 $y = \text{argmin}_x[f[x]]$，那么 $\min_x[f[x]] = f[y]$。
*   函数 $\max_x[f[x]]$ 和 $\text{argmax}_x[f[x]]$ 对最大化函数执行等效操作。

**概率分布**
概率分布应该写成 $\text{Pr}(x=a)$，表示随机变量 $x$ 取值为 $a$。然而，这个符号很 cumbersome。因此，我们通常简化它，只写成 $\text{Pr}(x)$，其中 $x$ 根据方程的意义表示随机变量或它所取的值。给定 $x$ 时 $y$ 的条件概率写作 $\text{Pr}(y|x)$。$y$ 和 $x$ 的联合概率写作 $\text{Pr}(y, x)$。这两种形式可以结合，所以 $\text{Pr}(y|x, \boldsymbol{\phi})$ 表示给定我们知道 $x$ 和 $\boldsymbol{\phi}$ 时变量 $y$ 的概率。类似地，$\text{Pr}(y, x|\boldsymbol{\phi})$ 表示给定我们知道 $\boldsymbol{\phi}$ 时变量 $y$ 和 $x$ 的概率。当我们需要在同一个变量上有两个概率分布时，我们为第一个分布写 $\text{Pr}(x)$，为第二个写 $q(x)$。更多关于概率分布的信息可以在附录C中找到。

**渐进符号**
**渐进符号 (Asymptotic notation)** 被用来比较不同算法随着输入大小 $D$ 的增加所做的工作量。这可以通过多种方式完成，但本书只使用**大O符号 (big-O notation)**，它表示一个算法中计算增长的上界。如果存在一个常数 $c>0$ 和整数 $n_0$，使得对于所有 $n>n_0$，都有 $f[n] < c \cdot g[n]$，那么一个函数 $f[n]$ 就是 $O[g[n]]$。
这个符号提供了一个算法最坏情况运行时间的界限。例如，当我们说一个 $D \times D$ 矩阵的求逆是 $O[D^3]$ 时，我们的意思是，一旦 $D$ 足够大，计算将不会比某个常数乘以 $D^3$ 增长得更快。这给了我们一个关于对不同大小的矩阵求逆的可行性的概念。如果 $D=10^3$，那么求逆可能需要大约 $10^9$ 次操作。

**杂项**
数学方程中的一个小圆点旨在提高可读性，没有实际意义（或者只是意味着乘法）。例如，$a \cdot f[x]$ 与 $af[x]$ 相同，但更容易阅读。为了避免歧义，点积写成 $\mathbf{a}^T\mathbf{b}$（见附录B.3.4）。一个左箭头符号 $\leftarrow$ 表示赋值，所以 $x \leftarrow x+2$ 意味着我们将当前 $x$ 的值加上2。


# 附录 B
# 数学基础

本附录回顾了正文中使用的数学概念。

## B.1 函数

函数定义了从一个集合 $\mathcal{X}$（例如，实数集）到另一个集合 $\mathcal{Y}$ 的映射。
**单射 (injection)** 是一种一对一的函数，其中第一个集合中的每个元素都映射到第二个集合中一个唯一的位置（但第二个集合中可能有没有被映射到的元素）。
**满射 (surjection)** 是一种函数，其中第二个集合中的每个元素都接收到来自第一个集合的映射（但第一个集合中可能有多个元素映射到第二个集合中的同一个元素）。
**双射 (bijection)** 或**双射映射 (bijective mapping)** 是一种既是单射又是满射的函数。它在两个集合的所有成员之间提供了一一对应的关系。
**微分同胚 (diffeomorphism)** 是双射的一个特例，其中前向和反向映射都是可微的。

### B.1.1 利普希茨常数

如果对于所有的 $z_1, z_2$，函数 $f[z]$ 满足以下条件，则称其为**利普希茨连续 (Lipschitz continuous)**：

$$
||f[z_1] - f[z_2]|| \le \beta ||z_1 - z_2||,
\tag{B.1}
$$

其中 $\beta$ 被称为**利普希茨常数 (Lipschitz constant)**，它决定了函数相对于距离度量的最大梯度（即，函数可以变化得多快）。如果利普希茨常数小于1，则该函数是一个**收缩映射 (contraction mapping)**，我们可以使用巴拿赫不动点定理来找到任何点的逆（见图16.9）。

将两个利普希茨常数分别为 $\beta_1$ 和 $\beta_2$ 的函数进行复合，会创建一个新的利普希茨连续函数，其常数小于或等于 $\beta_1\beta_2$。将两个利普希茨常数分别为 $\beta_1$ 和 $\beta_2$ 的函数相加，会创建一个新的利普希茨连续函数，其常数小于或等于 $\beta_1+\beta_2$。对于一个线性变换 $\mathbf{f}[\mathbf{z}] = \mathbf{A}\mathbf{z}+\mathbf{b}$，相对于欧几里得距离度量的利普希茨常数是 $\mathbf{A}$ 的最大特征值。

### B.1.2 凸性

如果我们可以一个函数上任意两点之间画一条直线，并且这条线总是位于该函数**上方**，那么这个函数就是**凸函数 (convex)**。类似地，如果任意两点之间的直线总是位于函数**下方**，那么这个函数就是**凹函数 (concave)**。根据定义，凸（凹）函数最多只有一个最小值（最大值）。

如果我们可以一个 $\mathbb{R}^D$ 区域边界上的任意两点之间画一条直线，而这条线不会在别处与边界相交，那么这个区域就是**凸区域 (convex region)**。梯度下降保证能够找到定义在凸区域上的凸函数的全局最小值。

### B.1.3 特殊函数

以下函数在正文中使用：
*   **指数函数 (exponential function)** $y = \exp[x]$（图 B.1a）将一个实变量 $x \in \mathbb{R}$ 映射到一个非负数 $y \in \mathbb{R}^+$，即 $y = e^x$。
*   **对数函数 (logarithm)** $x = \log[y]$（图 B.1b）是指数函数的逆函数，它将一个非负数 $y \in \mathbb{R}^+$ 映射到一个实变量 $x \in \mathbb{R}$。请注意，本书中所有的对数都是自然对数（即，以 $e$ 为底）。
*   **伽马函数 (gamma function)** $\Gamma[x]$（图 B.1c）定义为：
    $$
    \Gamma[x] = \int_0^\infty t^{x-1}e^{-t}dt.
    \tag{B.2}
    $$
    它将阶乘函数扩展到连续值，因此对于 $x \in \{1, 2, \dots\}$，有 $\Gamma[x] = (x-1)!$。
*   **狄拉克δ函数 (Dirac delta function)** $\delta[z]$ 的总面积为1，且全部集中在位置 $z=0$ 处。一个包含 $N$ 个元素的数据集可以被看作是一个由 $N$ 个以每个数据点 $x_i$ 为中心的δ函数之和构成，并由 $1/N$ 缩放的概率分布。δ函数通常被画成一个箭头（例如，图5.12）。δ函数具有一个关键性质：
    $$
    \int f[x]\delta[x-x_0]dx = f[x_0].
    \tag{B.3}
    $$

### B.1.4 斯特林公式

**斯特林公式 (Stirling's formula)**（图 B.2）使用以下公式来近似阶乘函数（以及伽马函数）：

$$
x! \approx \sqrt{2\pi x}\left(\frac{x}{e}\right)^x.
\tag{B.4}
$$

<br>

**图 B.1 指数函数、对数函数和伽马函数**
a) 指数函数将一个实数映射到一个正数。它是一个凸函数。b) 对数函数是指数函数的逆函数，它将一个正数映射到一个实数。它是一个凹函数。c) 伽马函数是阶乘函数的连续扩展，因此对于 $x \in \{1, 2, \dots\}$，有 $\Gamma[x] = (x-1)!$。

**图 B.2 斯特林公式**
阶乘函数 $x!$ 可以由斯特林公式 $\text{Stir}[x]$ 近似，该公式对所有实数值都有定义。

<br>

## B.2 二项式系数

**二项式系数 (Binomial coefficients)** 写作 $\binom{n}{k}$，读作“n选k”。它们是正整数，代表从一个包含 $n$ 个物品的集合中，不放回地选择一个包含 $k$ 个物品的无序子集的方案数。二项式系数可以用简单的公式计算：

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}.
\tag{B.5}
$$

### B.2.1 自相关

一个连续函数 $f[z]$ 的**自相关 (autocorrelation)** $r[\tau]$ 定义为：

$$
r[\tau] = \int_{-\infty}^\infty f[t+\tau]f[t]dt,
\tag{B.6}
$$

其中 $\tau$ 是时间延迟。有时，这个值会通过 $r$ 进行归一化，使得时间延迟为零时的自相关为1。自相关函数是函数与其自身作为偏移（即时间延迟）的函数的**相关性**的度量。如果一个函数变化缓慢且可预测，那么随着时间延迟从零增加，自相关函数将缓慢减小。如果函数变化快速且不可预测，那么它将迅速减小到零。

## B.3 向量、矩阵和张量

在机器学习中，一个向量 $\mathbf{x} \in \mathbb{R}^D$ 是一个由 $D$ 个数字组成的一维数组，我们假设它们以列的形式组织。类似地，一个矩阵 $\mathbf{Y} \in \mathbb{R}^{D_1 \times D_2}$ 是一个具有 $D_1$ 行和 $D_2$ 列的二维数字数组。一个张量 $\mathbf{z} \in \mathbb{R}^{D_1 \times D_2 \times \dots \times D_N}$ 是一个N维的数字数组。令人困惑的是，在像PyTorch和TensorFlow这样的深度学习API中，这三种量都存储在被称为“张量”的对象中。

### B.3.1 转置

一个矩阵 $\mathbf{A} \in \mathbb{R}^{D_1 \times D_2}$ 的**转置 (transpose)** $\mathbf{A}^T \in \mathbb{R}^{D_2 \times D_1}$ 是通过围绕主对角线反射它形成的，因此第 $k$ 列成为第 $k$ 行，反之亦然。如果我们取一个矩阵乘积 $\mathbf{AB}$ 的转置，那么我们取原始矩阵的转置但颠倒顺序，使得：

$$
(\mathbf{AB})^T = \mathbf{B}^T \mathbf{A}^T.
\tag{B.7}
$$

一个列向量 $\mathbf{a}$ 的转置是一个行向量 $\mathbf{a}^T$，反之亦然。

### B.3.2 向量和矩阵的范数

对于一个向量 $\mathbf{z}$，**$\ell_p$ 范数 ($\ell_p$ norm)** 定义为：

$$
||\mathbf{z}||_p = \left(\sum_{d=1}^D |z_d|^p\right)^{1/p},
\tag{B.8}
$$

对于实数值 $p \ge 1$。当 $p=2$ 时，这返回向量的长度，这被称为**欧几里得范数 (Euclidean norm)**。这是深度学习中最常用的情况，通常指数 $p$ 会被省略，欧几里得范数就只写作 $||\mathbf{z}||$。当 $p=\infty$ 时，该运算符返回向量中绝对值的最大值。

范数可以以类似的方式对矩阵进行计算。例如，一个矩阵 $\mathbf{Z}$ 的**$\ell_2$ 范数**（被称为**弗罗贝尼乌斯范数 (Frobenius norm)**）计算为：

$$
||\mathbf{Z}||_F = \left(\sum_{i=1}^I \sum_{j=1}^J |z_{ij}|^2\right)^{1/2}.
\tag{B.9}
$$

### B.3.3 矩阵的乘积

两个矩阵 $\mathbf{A} \in \mathbb{R}^{D_1 \times D_2}$ 和 $\mathbf{B} \in \mathbb{R}^{D_2 \times D_3}$ 的乘积 $\mathbf{C} = \mathbf{AB}$ 是第三个矩阵 $\mathbf{C} \in \mathbb{R}^{D_1 \times D_3}$，其中：

$$
C_{ij} = \sum_{d=1}^{D_2} A_{id}B_{dj}.
\tag{B.10}
$$

### B.3.4 向量的点积

两个向量 $\mathbf{a} \in \mathbb{R}^D$ 和 $\mathbf{b} \in \mathbb{R}^D$ 的**点积 (dot product)** $\mathbf{a}^T\mathbf{b}$ 是一个标量，定义为：

$$
\mathbf{a}^T\mathbf{b} = \mathbf{b}^T\mathbf{a} = \sum_{d=1}^D a_d b_d.
\tag{B.11}
$$

可以证明，点积与第一个向量的欧几里得范数乘以第二个向量的欧几里得范数再乘以它们之间的夹角 $\theta$ 成正比：

$$
\mathbf{a}^T\mathbf{b} = ||\mathbf{a}|| \cdot ||\mathbf{b}|| \cos[\theta].
\tag{B.12}
$$

### B.3.5 逆

一个方阵 $\mathbf{A}$ 可能有也可能没有一个**逆 (inverse)** $\mathbf{A}^{-1}$，使得 $\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$。如果一个矩阵没有逆，它就被称为**奇异的 (singular)**。如果我们取一个矩阵乘积 $\mathbf{AB}$ 的逆，其中 $\mathbf{A}$ 和 $\mathbf{B}$ 都是方的且可逆的，那么我们可以等效地分别取每个矩阵的逆并颠倒乘法顺序。

$$
(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}.
\tag{B.13}
$$

通常，对一个 $D \times D$ 的矩阵求逆需要 $O(D^3)$ 次操作。然而，对于某些特殊类型的矩阵，包括对角矩阵、正交矩阵和三角矩阵，求逆更有效率（见B.4节）。

### B.3.6 子空间

考虑一个矩阵 $\mathbf{A} \in \mathbb{R}^{D_1 \times D_2}$。如果矩阵的列数 $D_2$ 少于行数 $D_1$（即，矩阵是“瘦长”的），乘积 $\mathbf{Ax}$ 无法达到 $D_1$ 维输出空间中的所有可能位置。这个乘积由 $\mathbf{A}$ 的 $D_2$ 列和 $\mathbf{x}$ 的 $D_2$ 个元素加权组成，并且只能达到由这些列张成的线性子空间。这被称为矩阵的**列空间 (column space)**。相反，对于一个“矮胖”的矩阵 $\mathbf{A}$，映射到零的输入空间部分（即，那些 $\mathbf{x}$ 使得 $\mathbf{Ax}=0$）被称为矩阵的**零空间 (nullspace)**。

### B.3.7 特征谱

如果我们将单位圆上的二维点集乘以一个 $2 \times 2$ 的矩阵 $\mathbf{A}$，它们会映射到一个椭圆（图 B.3）。这个椭圆的主轴和次轴的半径（即，最长和最短的方向）对应于矩阵的**奇异值 (singular values)** $\lambda_1$ 和 $\lambda_2$ 的大小。同样的想法也适用于更高的维度。一个 $D$ 维的球体被一个 $D \times D$ 的矩阵 $\mathbf{A}$ 映射到一个 $D$ 维的椭球体。这个椭球体的 $D$ 个主轴的半径决定了奇异值的大小。对于对称方阵，相同的信息由**特征值 (eigenvalues)** 捕获，这里它们与奇异值相同。

一个方阵的**谱范数 (spectral norm)** 是最大的绝对特征值。当矩阵应用于一个单位长度的向量时，它捕获了可能的最大幅度变化。因此，它告诉我们关于变换的利普希茨常数的信息。特征值的集合有时被称为**特征谱 (eigenspectrum)**，它告诉我们矩阵在所有方向上应用的缩放幅度。这个信息可以用**行列式 (determinant)** 和**迹 (trace)** 来总结。

### B.3.8 行列式和迹

每个方阵 $\mathbf{A}$ 都有一个与之关联的标量，称为**行列式 (determinant)**，记为 $|\mathbf{A}|$ 或 $\det[\mathbf{A}]$，它是特征值的乘积。因此，它与矩阵对不同输入应用的平均缩放有关。绝对行列式小的矩阵在相乘时倾向于减小向量的范数。绝对行列式大的矩阵则倾向于增加范数。如果一个矩阵是**奇异的**，行列式将为零，并且至少有一个空间方向在应用矩阵时被映射到原点。矩阵表达式的行列式遵循以下规则：

$$
\begin{aligned}
|\mathbf{A}^T| &= |\mathbf{A}| \\
|\mathbf{AB}| &= |\mathbf{A}||\mathbf{B}| \\
|\mathbf{A}^{-1}| &= 1/|\mathbf{A}|.
\end{aligned}
\tag{B.14}
$$

一个方阵的**迹 (trace)** 是对角线值的和（矩阵本身不必是对角的）或者是特征值的和。迹遵循以下规则：

$$
\begin{aligned}
\text{trace}[\mathbf{A}^T] &= \text{trace}[\mathbf{A}] \\
\text{trace}[\mathbf{AB}] &= \text{trace}[\mathbf{BA}] \\
\text{trace}[\mathbf{A}+\mathbf{B}] &= \text{trace}[\mathbf{A}] + \text{trace}[\mathbf{B}] \\
\text{trace}[\mathbf{ABC}] &= \text{trace}[\mathbf{BCA}] = \text{trace}[\mathbf{CAB}],
\end{aligned}
\tag{B.15}
$$

在最后一个关系中，迹仅对循环置换是不变的，因此通常 $\text{trace}[\mathbf{ABC}] \neq \text{trace}[\mathbf{BAC}]$。

## B.4 特殊类型的矩阵

计算一个方阵 $\mathbf{A} \in \mathbb{R}^{D \times D}$ 的逆的复杂度是 $O(D^3)$，计算行列式也是如此。然而，对于某些具有特殊性质的矩阵，这些计算可以更有效率。

### B.4.1 对角矩阵

**对角矩阵 (diagonal matrix)** 在主对角线之外的地方都是零。如果这些对角线元素都非零，逆矩阵也是一个对角矩阵，每个对角线元素 $d_{ii}$ 被替换为 $1/d_{ii}$。行列式是对角线上值的乘积。一个特例是**单位矩阵 (identity matrix)**，其对角线上是1。因此，它的逆也是单位矩阵，它的行列式是1。

### B.4.2 三角矩阵

**下三角矩阵 (lower triangular matrix)** 的所有非零值都在主对角线和/或其下方的位置。**上三角矩阵 (upper triangular matrix)** 的所有非零值都在主对角线和/或其上方的位置。在这两种情况下，矩阵都可以在 $O(D^2)$ 时间内求逆（见问题16.4），行列式就是对角线上值的乘积。

### B.4.3 正交矩阵

**正交矩阵 (Orthogonal matrices)** 代表围绕原点的旋转和反射，所以在图 B.3中，圆将被映射到另一个单位半径的圆，但被旋转并可能被反射。因此，特征值的幅度都必须为1，行列式必须是1或-1。一个正交矩阵的逆是它的转置，所以 $\mathbf{A}^{-1}=\mathbf{A}^T$。

### B.4.4 置换矩阵

**置换矩阵 (permutation matrix)** 在每行每列中只有一个非零条目，并且所有这些条目的值都为1。它是正交矩阵的一个特例，所以它的逆是它自己的转置，它的行列式总是 $\pm 1$。顾名思义，它具有置换向量条目的效果。例如：

$$
\begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix}
\begin{pmatrix} a \\ b \\ c \end{pmatrix} =
\begin{pmatrix} b \\ c \\ a \end{pmatrix}.
\tag{B.16}
$$

### B.4.5 线性代数

线性代数是研究线性函数的数学，其形式为：

$$
f[z_1, z_2, \dots, z_D] = \phi_1 z_1 + \phi_2 z_2 + \dots + \phi_D z_D,
\tag{B.17}
$$

其中 $\phi_1, \dots, \phi_D$ 是定义函数的参数。我们经常在右侧添加一个常数项 $\phi_0$。这在技术上是一个**仿射函数 (affine function)**，但在机器学习中通常被称为线性的。我们全书都采用这种约定。

### B.4.6 矩阵形式的线性方程

考虑一组线性函数：

$$
\begin{aligned}
y_1 &= \phi_{10} + \phi_{11}z_1 + \phi_{12}z_2 + \phi_{13}z_3 \\
y_2 &= \phi_{20} + \phi_{21}z_1 + \phi_{22}z_2 + \phi_{23}z_3 \\
y_3 &= \phi_{30} + \phi_{31}z_1 + \phi_{32}z_2 + \phi_{33}z_3.
\end{aligned}
\tag{B.18}
$$

这些可以写成矩阵形式：

$$
\begin{pmatrix} y_1 \\ y_2 \\ y_3 \end{pmatrix} =
\begin{pmatrix} \phi_{10} \\ \phi_{20} \\ \phi_{30} \end{pmatrix} +
\begin{pmatrix} \phi_{11} & \phi_{12} & \phi_{13} \\ \phi_{21} & \phi_{22} & \phi_{23} \\ \phi_{31} & \phi_{32} & \phi_{33} \end{pmatrix}
\begin{pmatrix} z_1 \\ z_2 \\ z_3 \end{pmatrix},
\tag{B.19}
$$

或者简写为 $\mathbf{y} = \boldsymbol{\phi}_0 + \mathbf{\Phi z}$，其中 $y_i = \phi_{i0} + \sum_{j=1}^3 \phi_{ij}z_j$。

## B.5 矩阵微积分

本书的大多数读者都会习惯于这样的想法，即如果我们有一个函数 $y=f[x]$，我们可以计算导数 $dy/dx$，它表示当我们对 $x$ 做一个小的改变时 $y$ 如何变化。这个想法可以扩展到将向量 $\mathbf{x}$ 映射到标量 $y$ 的函数 $y=f[\mathbf{x}]$，将向量 $\mathbf{x}$ 映射到向量 $\mathbf{y}$ 的函数 $\mathbf{y}=\mathbf{f}[\mathbf{x}]$，将矩阵 $\mathbf{X}$ 映射到向量 $\mathbf{y}$ 的函数 $\mathbf{y}=\mathbf{f}[\mathbf{X}]$ 等等。**矩阵微积分 (matrix calculus)** 的规则帮助我们计算这些量的导数。导数采取以下形式：
*   对于一个函数 $y=f[\mathbf{x}]$，其中 $y \in \mathbb{R}$ 且 $\mathbf{x} \in \mathbb{R}^D$，导数 $\partial y / \partial \mathbf{x}$ 也是一个 $D$ 维向量，其中第 $i$ 个元素计算为 $\partial y / \partial x_i$。
*   对于一个函数 $\mathbf{y}=\mathbf{f}[\mathbf{x}]$，其中 $\mathbf{y} \in \mathbb{R}^{D_y}$ 且 $\mathbf{x} \in \mathbb{R}^{D_x}$，导数 $\partial\mathbf{y}/\partial\mathbf{x}$ 是一个 $D_y \times D_x$ 的矩阵，其中元素 $(i, j)$ 包含导数 $\partial y_j/\partial x_i$。这被称为**雅可比矩阵 (Jacobian)**，在其他文献中有时写为 $\nabla_{\mathbf{x}}\mathbf{y}$。
*   对于一个函数 $\mathbf{y}=\mathbf{f}[\mathbf{X}]$，其中 $\mathbf{y} \in \mathbb{R}^{D_y}$ 且 $\mathbf{X} \in \mathbb{R}^{D_1 \times D_2}$，导数 $\partial\mathbf{y}/\partial\mathbf{X}$ 是一个包含导数 $\partial y_i/\partial X_{jk}$ 的三维张量。

通常这些矩阵和向量的导数在表面上与标量情况有相似的形式。例如，我们有：

$$
y = \mathbf{a}^T\mathbf{x} \quad \rightarrow \quad \frac{\partial y}{\partial\mathbf{x}} = \mathbf{a},
\tag{B.20}
$$

以及

$$
\mathbf{y} = \mathbf{Ax} \quad \rightarrow \quad \frac{\partial\mathbf{y}}{\partial\mathbf{x}} = \mathbf{A}^T.
\tag{B.21}
$$

# 附录 C
# 概率论

概率论对于深度学习至关重要。在监督学习中，深度网络隐式地依赖于损失函数的概率公式。在无监督学习中，生成模型旨在产生从与训练数据相同的概率分布中抽取的样本。强化学习发生在马尔可夫决策过程中，这些过程是根据概率分布定义的。本附录为机器学习中使用的概率论提供了入门介绍。

## C.1 随机变量和概率分布

**随机变量 (random variable)** $x$ 表示一个不确定的量。它可能是**离散的 (discrete)**（只取某些值，例如整数）或**连续的 (continuous)**（在连续统上取任何值，例如实数）。如果我们观察一个随机变量 $x$ 的多个实例，它会取不同的值，而取不同值的相对倾向由一个**概率分布 (probability distribution)** $\text{Pr}(x)$ 来描述。

对于一个离散变量，这个分布将一个概率 $\text{Pr}(x=k) \in$ 与每个可能的结果 $k$ 相关联，这些概率的总和为1。对于一个连续变量，有一个非负的**概率密度 (probability density)** $\text{Pr}(x=a) \ge 0$ 与域中每个值 $a$ 相关联，这个概率密度函数（PDF）在该域上的积分必须为1。对于任何点 $a$，这个密度可以大于1。从现在开始，我们假设随机变量是连续的。这些思想对于离散分布是完全相同的，只是用求和代替积分。

### C.1.1 联合概率

考虑我们有两个随机变量 $x$ 和 $y$ 的情况。**联合分布 (joint distribution)** $\text{Pr}(x, y)$ 告诉我们 $x$ 和 $y$ 取特定值组合的倾向（图 C.1a）。现在有一个非负的概率密度 $\text{Pr}(x=a, y=b)$ 与每对值 $x=a$ 和 $y=b$ 相关联，并且必须满足：

$$
\iint \text{Pr}(x, y) dx dy = 1.
\tag{C.1}
$$

这个想法可以扩展到两个以上的变量，所以 $x, y, z$ 的联合密度写作 $\text{Pr}(x, y, z)$。有时，我们将多个随机变量存储在一个向量 $\mathbf{x}$ 中，并将其联合密度写作 $\text{Pr}(\mathbf{x})$。扩展这个，我们可以将两个向量 $\mathbf{x}$ 和 $\mathbf{y}$ 中所有变量的联合密度写作 $\text{Pr}(\mathbf{x}, \mathbf{y})$。

<br>

**图 C.1 联合分布与边缘分布**
a) 联合分布 $\text{Pr}(x, y)$ 捕获了变量 $x$ 和 $y$ 取不同值组合的倾向。在这里，概率密度由色图表示，所以更亮的位置更可能出现。例如，组合 $x=6, y=6$ 比组合 $x=5, y=0$ 观察到的可能性要小得多。b) 变量 $x$ 的边缘分布 $\text{Pr}(x)$ 可以通过对 $y$ 积分来恢复。c) 变量 $y$ 的边缘分布 $\text{Pr}(y)$ 可以通过对 $x$ 积分来恢复。

<br>

### C.1.2 边缘化

如果我们知道两个变量上的联合分布 $\text{Pr}(x, y)$，我们可以通过对另一个变量积分来恢复**边缘分布 (marginal distributions)** $\text{Pr}(x)$ 和 $\text{Pr}(y)$（图 C.1b-c）：

$$
\begin{aligned}
\int \text{Pr}(x, y) \cdot dx &= \text{Pr}(y) \\
\int \text{Pr}(x, y) \cdot dy &= \text{Pr}(x).
\end{aligned}
\tag{C.2}
$$

这个过程被称为**边缘化 (marginalization)**，其解释是我们在计算一个变量的分布，而不管另一个变量取什么值。边缘化的思想可以扩展到更高的维度，所以如果我们有一个联合分布 $\text{Pr}(x, y, z)$，我们可以通过对 $y$ 积分来恢复联合分布 $\text{Pr}(x, z)$。

### C.1.3 条件概率和似然

**条件概率 (conditional probability)** $\text{Pr}(x|y)$ 是在假设我们知道 $y$ 的值的情况下，变量 $x$ 取某个特定值的概率。竖线读作英文单词“given”，所以 $\text{Pr}(x|y)$ 是给定 $y$ 时 $x$ 的概率。条件概率 $\text{Pr}(x|y)$ 可以通过在固定的 $y$ 值处对联合分布 $\text{Pr}(x, y)$ 进行切片来找到。然后这个切片被该值 $y$ 发生的概率（切片下的总面积）所除，以使条件分布的和为1（图 C.2）：

$$
\text{Pr}(x|y) = \frac{\text{Pr}(x, y)}{\text{Pr}(y)}.
\tag{C.3}
$$

类似地，

$$
\text{Pr}(y|x) = \frac{\text{Pr}(x, y)}{\text{Pr}(x)}.
\tag{C.4}
$$

当我们将条件概率 $\text{Pr}(x|y)$ 视为 $x$ 的函数时，它的和必须为1。当我们将相同的量 $\text{Pr}(x|y)$ 视为 $y$ 的函数时，它被称为给定 $y$ 时 $x$ 的**似然 (likelihood)**，并且它的和不必为1。

<br>

**图 C.2 条件分布**
a) 变量 $x$ 和 $y$ 的联合分布 $\text{Pr}(x, y)$。b) 给定 $y$ 取值为3.0时，变量 $x$ 的条件概率 $\text{Pr}(x|y=3.0)$，是通过取联合概率的水平“切片” $\text{Pr}(x, y=3.0)$（面板a中的顶部青色线），并将其除以该切片中的总面积 $\text{Pr}(y=3.0)$ 来找到的，这样它就形成了一个积分为1的有效概率分布。c) 联合概率 $\text{Pr}(x, y=-1.0)$ 是类似地使用在 $y=-1.0$ 处的切片找到的。

<br>

### C.1.4 贝叶斯法则

从方程C.3和C.4，我们得到联合概率 $\text{Pr}(x, y)$ 的两个表达式：

$$
\text{Pr}(x, y) = \text{Pr}(x|y)\text{Pr}(y) = \text{Pr}(y|x)\text{Pr}(x),
\tag{C.5}
$$

我们可以重新排列得到：

$$
\text{Pr}(x|y) = \frac{\text{Pr}(y|x)\text{Pr}(x)}{\text{Pr}(y)}.
\tag{C.6}
$$

这个表达式将给定 $y$ 时 $x$ 的条件概率 $\text{Pr}(x|y)$ 与给定 $x$ 时 $y$ 的条件概率 $\text{Pr}(y|x)$ 联系起来，被称为**贝叶斯法则 (Bayes' rule)**。
贝叶斯法则中的每一项都有一个名字。$\text{Pr}(y|x)$ 项是给定 $x$ 时 $y$ 的**似然**，$\text{Pr}(x)$ 项是 $x$ 的**先验概率 (prior probability)**。分母 $\text{Pr}(y)$ 被称为**证据 (evidence)**，左侧的 $\text{Pr}(x|y)$ 被称为给定 $y$ 时 $x$ 的**后验概率 (posterior probability)**。这个方程从先验 $\text{Pr}(x)$（我们在观测 $y$ 之前对 $x$ 的了解）映射到后验 $\text{Pr}(x|y)$（我们在观测 $y$ 之后对 $x$ 的了解）。

### C.1.5 独立性

如果随机变量 $y$ 的值没有告诉我们任何关于 $x$ 的信息，反之亦然，我们就说 $x$ 和 $y$ 是**独立的 (independent)**，我们可以写成 $\text{Pr}(x|y)=\text{Pr}(x)$ 和 $\text{Pr}(y|x)=\text{Pr}(y)$。由此可知，所有的条件分布 $\text{Pr}(y|x=\cdot)$ 都是相同的，条件分布 $\text{Pr}(x|y=\cdot)$ 也是如此。
从方程C.5中联合概率的第一个表达式开始，我们看到联合分布变成了边缘分布的乘积：

$$
\text{Pr}(x, y) = \text{Pr}(x|y)\text{Pr}(y) = \text{Pr}(x)\text{Pr}(y)
\tag{C.7}
$$

当变量是独立的时（图 C.3）。

<br>

**图 C.3 独立性**
a) 当两个变量 $x$ 和 $y$ 独立时，联合分布分解为边缘分布的乘积，所以 $\text{Pr}(x,y) = \text{Pr}(x)\text{Pr}(y)$。独立性意味着知道一个变量的值没有告诉我们任何关于另一个变量的信息。b-c) 因此，所有的条件分布 $\text{Pr}(x|y=\cdot)$ 都是相同的，并且等于边缘分布 $\text{Pr}(x)$。

<br>

## C.2 期望

考虑一个函数 $f[x]$ 和一个在 $x$ 上定义的概率分布 $\text{Pr}(x)$。一个随机变量 $x$ 的函数 $f[\cdot]$ 相对于概率分布 $\text{Pr}(x)$ 的**期望值 (expected value)** 定义为：

$$
\mathbb{E}_x[f[x]] = \int f[x]\text{Pr}(x)dx.
\tag{C.8}
$$

顾名思义，这是在考虑了看到不同 $x$ 值的概率之后，$f[x]$ 的期望或平均值。这个思想可以推广到多个随机变量的函数 $f[\cdot, \cdot]$：

$$
\mathbb{E}_{x,y}[f[x, y]] = \iint f[x, y]\text{Pr}(x, y)dxdy.
\tag{C.9}
$$

期望总是相对于一个或多个变量上的一个分布来取的。然而，当分布的选择是显而易见时，我们通常不明确地写出它，而是写成 $\mathbb{E}[f[x]]$ 而不是 $\mathbb{E}_x[f[x]]$。
如果我们从 $\text{Pr}(x)$ 中抽取大量的 $I$ 个样本 $\{x_i\}_{i=1}^I$，为每个样本计算 $f[x_i]$，然后取这些值的平均值，结果将近似于该函数的期望 $\mathbb{E}[f[x]]$：

$$
\mathbb{E}_x[f[x]] \approx \frac{1}{I}\sum_{i=1}^I f[x_i].
\tag{C.10}
$$

### C.2.1 操作期望的规则

有四个操作期望的规则：

$$
\begin{aligned}
\mathbb{E}[k] &= k \\
\mathbb{E}[k \cdot f[x]] &= k \cdot \mathbb{E}[f[x]] \\
\mathbb{E}[f[x] + g[x]] &= \mathbb{E}[f[x]] + \mathbb{E}[g[x]] \\
\mathbb{E}_{x,y}[f[x]g[y]] &= \mathbb{E}_x[f[x]] \cdot \mathbb{E}_y[g[y]] \quad \text{如果 x, y 独立,}
\end{aligned}
\tag{C.11}
$$

其中 $k$ 是任意常数。这些在下面为连续情况进行了证明。
**规则 1**: 一个常数值 $k$ 的期望 $\mathbb{E}[k]$ 就是 $k$。

$$
\mathbb{E}[k] = \int k \cdot \text{Pr}(x)dx = k \int \text{Pr}(x)dx = k.
\tag{C.12}
$$

**规则 2**: 一个常数 $k$ 乘以一个变量 $x$ 的函数的期望 $\mathbb{E}[k \cdot f[x]]$ 是 $k$ 乘以该函数的期望 $\mathbb{E}[f[x]]$。

$$
\mathbb{E}[k \cdot f[x]] = \int k \cdot f[x]\text{Pr}(x)dx = k \cdot \int f[x]\text{Pr}(x)dx = k \cdot \mathbb{E}[f[x]].
\tag{C.13}
$$

**规则 3**: 一系列项的和的期望 $\mathbb{E}[f[x]+g[x]]$ 是期望的和 $\mathbb{E}[f[x]]+\mathbb{E}[g[x]]$。

$$
\mathbb{E}[f[x]+g[x]] = \int(f[x]+g[x]) \cdot \text{Pr}(x)dx = \int (f[x] \cdot \text{Pr}(x) + g[x] \cdot \text{Pr}(x))dx \\
= \int f[x] \cdot \text{Pr}(x)dx + \int g[x] \cdot \text{Pr}(x)dx = \mathbb{E}[f[x]] + \mathbb{E}[g[x]].
\tag{C.14}
$$

**规则 4**: 如果 $x$ 和 $y$ 是独立的，一系列项的乘积的期望 $\mathbb{E}[f[x] \cdot g[y]]$ 是期望的乘积 $\mathbb{E}[f[x]] \cdot \mathbb{E}[g[y]]$。

$$
\mathbb{E}[f[x]g[y]] = \iint f[x] \cdot g[y]\text{Pr}(x, y)dxdy = \iint f[x] \cdot g[y]\text{Pr}(x)\text{Pr}(y)dxdy \\
= \int f[x] \cdot \text{Pr}(x)dx \int g[y] \cdot \text{Pr}(y)dy = \mathbb{E}[f[x]]\mathbb{E}[g[y]],
\tag{C.15}
$$

我们在这里的第一行和第二行之间使用了独立性的定义（方程C.7）。

这四个规则可以推广到多变量情况：

$$
\begin{aligned}
\mathbb{E}[\mathbf{A}] &= \mathbf{A} \\
\mathbb{E}[\mathbf{A} \cdot \mathbf{f}[\mathbf{x}]] &= \mathbf{A}\mathbb{E}[\mathbf{f}[\mathbf{x}]] \\
\mathbb{E}[\mathbf{f}[\mathbf{x}] + \mathbf{g}[\mathbf{x}]] &= \mathbb{E}[\mathbf{f}[\mathbf{x}]] + \mathbb{E}[\mathbf{g}[\mathbf{x}]] \\
\mathbb{E}_{\mathbf{x},\mathbf{y}}[\mathbf{f}[\mathbf{x}]^T \mathbf{g}[\mathbf{y}]] &= \mathbb{E}_\mathbf{x}[\mathbf{f}[\mathbf{x}]]^T \mathbb{E}_\mathbf{y}[\mathbf{g}[\mathbf{y}]] \quad \text{如果 x, y 独立,}
\end{aligned}
\tag{C.16}
$$

其中现在 $\mathbf{A}$ 是一个常数矩阵，$\mathbf{f}[\mathbf{x}]$ 是一个返回向量的向量 $\mathbf{x}$ 的函数，$\mathbf{g}[\mathbf{y}]$ 是一个也返回向量的向量 $\mathbf{y}$ 的函数。

### C.2.2 均值、方差和协方差

对于某些函数 $f[\cdot]$ 的选择，期望被赋予一个特殊的名字。这些量通常用来总结复杂分布的性质。例如，当 $f[x]=x$ 时，得到的期望 $\mathbb{E}[x]$ 被称为**均值 (mean)**，$\mu$。它是分布中心的一个度量。类似地，与均值的期望平方偏差 $\mathbb{E}[(x-\mu)^2]$ 被称为**方差 (variance)**，$\sigma^2$。这是分布离散程度的一个度量。**标准差 (standard deviation)** $\sigma$ 是方差的正平方根。它也衡量了分布的离散程度，但它的优点是与变量 $x$ 的单位相同。

顾名思义，两个变量 $x$ 和 $y$ 的**协方差 (covariance)** $\mathbb{E}[(x-\mu_x)(y-\mu_y)]$ 衡量了它们协同变化的程度。这里 $\mu_x$ 和 $\mu_y$ 分别代表变量 $x$ 和 $y$ 的均值。当两个变量的方差都很大，并且当 $y$ 增加时 $x$ 的值也倾向于增加时，协方差会很大。

如果两个变量是独立的，那么它们的协方差为零。然而，协方差为零并不意味着独立。例如，考虑一个概率均匀分布在以 $x, y$ 平面原点为中心的单位半径圆上的分布 $\text{Pr}(x, y)$。平均而言，当 $y$ 增加时 $x$ 没有增加或减少的趋势，反之亦然。然而，知道 $x=0$ 的值告诉我们 $y$ 有均等的机会取值为 $\pm 1$，所以变量不可能是独立的。

存储在列向量 $\mathbf{x} \in \mathbb{R}^D$ 中的多个随机变量的协方差可以用 $D \times D$ 的**协方差矩阵 (covariance matrix)** $\mathbb{E}[(\mathbf{x}-\boldsymbol{\mu}_x)(\mathbf{x}-\boldsymbol{\mu}_x)^T]$ 来表示，其中向量 $\boldsymbol{\mu}_x$ 包含均值 $\mathbb{E}[\mathbf{x}]$。该矩阵在位置 $(i, j)$ 的元素代表了变量 $x_i$ 和 $x_j$ 之间的协方差。

### C.2.3 方差恒等式

期望的规则（附录C.2.1）可以用来证明以下恒等式，它允许我们以不同的形式写出方差：

$$
\mathbb{E}[(x-\mu)^2] = \mathbb{E}[x^2] - \mathbb{E}[x]^2.
\tag{C.17}
$$

**证明**:
$$
\mathbb{E}[(x-\mu)^2] = \mathbb{E}[x^2 - 2\mu x + \mu^2] = \mathbb{E}[x^2] - \mathbb{E}[2\mu x] + \mathbb{E}[\mu^2] \\
= \mathbb{E}[x^2] - 2\mu \mathbb{E}[x] + \mu^2 = \mathbb{E}[x^2] - 2\mu^2 + \mu^2 \\
= \mathbb{E}[x^2] - \mu^2 = \mathbb{E}[x^2] - \mathbb{E}[x]^2,
\tag{C.18}
$$

我们在第一行和第二行之间使用了规则3，在第二行和第三行之间使用了规则1和2，并在剩下的两行中使用了定义 $\mu = \mathbb{E}[x]$。

### C.2.4 标准化

将一个随机变量的均值设为零，方差设为一，被称为**标准化 (standardization)**。这是通过以下变换实现的：

$$
z = \frac{x-\mu}{\sigma},
\tag{C.19}
$$

其中 $\mu$ 是 $x$ 的均值，$\sigma$ 是标准差。

**证明**: $z$ 上新分布的均值由下式给出：

$$
\mathbb{E}[z] = \mathbb{E}\left[\frac{x-\mu}{\sigma}\right] = \frac{1}{\sigma}\mathbb{E}[x-\mu] = \frac{1}{\sigma}(\mathbb{E}[x]-\mathbb{E}[\mu]) = \frac{1}{\sigma}(\mu-\mu)=0,
\tag{C.20}
$$

我们再次使用了操作期望的四个规则。新分布的方差由下式给出：

$$
\mathbb{E}[(z-\mu_z)^2] = \mathbb{E}[(z-\mathbb{E}[z])^2] = \mathbb{E}[z^2] = \mathbb{E}\left[\left(\frac{x-\mu}{\sigma}\right)^2\right] = \frac{1}{\sigma^2}\mathbb{E}[(x-\mu)^2] = \frac{1}{\sigma^2} \cdot \sigma^2 = 1.
\tag{C.21}
$$

通过类似的论证，我们可以取一个均值为零、方差为一的标准化变量 $z$，并使用以下公式将其转换为一个均值为 $\mu$、方差为 $\sigma^2$ 的变量 $x$：

$$
x = \mu + \sigma z.
\tag{C.22}
$$

在多变量情况下，我们可以使用以下公式来标准化一个均值为 $\boldsymbol{\mu}$、协方差矩阵为 $\mathbf{\Sigma}$ 的变量 $\mathbf{x}$：

$$
\mathbf{z} = \mathbf{\Sigma}^{-1/2}(\mathbf{x}-\boldsymbol{\mu}).
\tag{C.23}
$$

结果将有一个均值 $\mathbb{E}[\mathbf{z}] = \mathbf{0}$ 和一个单位协方差矩阵 $\mathbb{E}[(\mathbf{z}-\mathbb{E}[\mathbf{z}])(\mathbf{z}-\mathbb{E}[\mathbf{z}])^T] = \mathbf{I}$。为了逆转这个过程，我们使用：

$$
\mathbf{x} = \boldsymbol{\mu} + \mathbf{\Sigma}^{1/2}\mathbf{z}.
\tag{C.24}
$$

## C.3 正态概率分布

本书中使用的概率分布包括伯努利分布（图5.6）、分类分布（图5.9）、泊松分布（图5.15）、冯·米塞斯分布（图5.13）以及高斯混合分布（图5.14和17.1）。然而，机器学习中最常见的分布是**正态分布 (normal distribution)** 或**高斯分布 (Gaussian distribution)**。

### C.3.1 单变量正态分布

一个在标量变量 $x$ 上的**单变量正态分布 (univariate normal distribution)**（图5.3）有两个参数，均值 $\mu$ 和方差 $\sigma^2$，其定义为：

$$
\text{Pr}(x) = \text{Norm}_x[\mu, \sigma^2] = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right].
\tag{C.25}
$$

不出所料，一个正态分布变量的均值 $\mathbb{E}[x]$ 由均值参数 $\mu$ 给出，方差 $\mathbb{E}[(x-\mathbb{E}[x])^2]$ 由方差参数 $\sigma^2$ 给出。当均值为零、方差为一时，我们称之为**标准正态分布 (standard normal distribution)**。

正态分布的形状可以从以下论证中推断出来。项 $-(x-\mu)^2/2\sigma^2$ 是一个二次函数，当 $x=\mu$ 时为零，并以一个随着 $\sigma$ 变小而增加的速率衰减。当我们将此通过指数函数（图B.1）时，我们得到一个钟形曲线，它在 $x=\mu$ 处的值为1，并向两侧衰减。除以常数 $\sqrt{2\pi\sigma^2}$ 确保了函数积分为1，并且是一个有效的分布。从这个论证中可以得出，均值 $\mu$ 控制了钟形曲线中心的位置，而方差的平方根（标准差）控制了钟形曲线的宽度。

### C.3.2 多元正态分布

**多元正态分布 (multivariate normal distribution)** 将正态分布推广到描述一个长度为 $D$ 的向量 $\mathbf{x}$ 上的概率。它由一个 $D \times 1$ 的均值向量 $\boldsymbol{\mu}$ 和一个对称正定 $D \times D$ 的协方差矩阵 $\mathbf{\Sigma}$ 定义：

$$
\text{Norm}_{\mathbf{x}}[\boldsymbol{\mu}, \mathbf{\Sigma}] = \frac{1}{(2\pi)^{D/2}|\mathbf{\Sigma}|^{1/2}} \exp\left[-\frac{(\mathbf{x}-\boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}{2}\right].
\tag{C.26}
$$

其解释与单变量情况类似。二次项 $-(\mathbf{x}-\boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})/2$ 返回一个标量，该标量随着 $\mathbf{x}$ 离均值 $\boldsymbol{\mu}$ 越远而减小，其减小速率取决于矩阵 $\mathbf{\Sigma}$。这被指数函数转换成一个钟形曲线的形状，并且除以 $(2\pi)^{D/2}|\mathbf{\Sigma}|^{1/2}$ 确保了该分布积分为1。

协方差矩阵可以采取**球形 (spherical)**、**对角 (diagonal)** 和**全形 (full)** 的形式：

$$
\mathbf{\Sigma}_{\text{spher}} = \begin{pmatrix} \sigma^2 & 0 \\ 0 & \sigma^2 \end{pmatrix}, \quad
\mathbf{\Sigma}_{\text{diag}} = \begin{pmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{pmatrix}, \quad
\mathbf{\Sigma}_{\text{full}} = \begin{pmatrix} \sigma_{11}^2 & \sigma_{12}^2 \\ \sigma_{21}^2 & \sigma_{22}^2 \end{pmatrix}.
\tag{C.27}
$$

在二维空间中（图 C.4），球形协方差产生圆形的等密度轮廓线，对角协方差产生与坐标轴对齐的椭圆形等密度轮廓线。全协方差则产生一般的椭圆形等密度轮廓线。当协方差是球形或对角形时，各个变量是独立的：

$$
\text{Pr}(x_1, x_2) = \frac{1}{2\pi\sqrt{|\mathbf{\Sigma}|}} \exp\left[-\frac{1}{2}\begin{pmatrix}x_1 & x_2\end{pmatrix}\mathbf{\Sigma}^{-1}\begin{pmatrix}x_1 \\ x_2\end{pmatrix}\right] \\
= \frac{1}{2\pi\sigma_1\sigma_2} \exp\left[-\frac{1}{2}\begin{pmatrix}x_1 & x_2\end{pmatrix}\begin{pmatrix}1/\sigma_1^2 & 0 \\ 0 & 1/\sigma_2^2\end{pmatrix}\begin{pmatrix}x_1 \\ x_2\end{pmatrix}\right] \\
= \frac{1}{\sqrt{2\pi}\sigma_1}\exp\left[-\frac{x_1^2}{2\sigma_1^2}\right] \cdot \frac{1}{\sqrt{2\pi}\sigma_2}\exp\left[-\frac{x_2^2}{2\sigma_2^2}\right] \\
= \text{Pr}(x_1) \cdot \text{Pr}(x_2).
\tag{C.28}
$$

<br>

**图 C.4 二元正态分布**
a-b) 当协方差矩阵是单位矩阵的倍数时，等高线是圆形的，我们称之为球形协方差。c-d) 当协方差是一个任意的对角矩阵时，等高线是与坐标轴对齐的椭圆，我们称之为对角协方差。e-f) 当协方差是一个任意的对称正定矩阵时，等高线是一般的椭圆，我们称之为全协方差。

<br>

### C.3.3 两个正态分布的乘积

两个正态分布的乘积与第三个正态分布成正比，其关系如下：

$$
\text{Norm}_\mathbf{x}[\mathbf{a}, \mathbf{A}]\text{Norm}_\mathbf{x}[\mathbf{b}, \mathbf{B}] \propto \text{Norm}_\mathbf{x}[(\mathbf{A}^{-1}+\mathbf{B}^{-1})^{-1}(\mathbf{A}^{-1}\mathbf{a}+\mathbf{B}^{-1}\mathbf{b}), (\mathbf{A}^{-1}+\mathbf{B}^{-1})^{-1}].
\tag{C.29}
$$

这可以通过展开指数项并配方法来轻松证明（见问题18.5）。

### C.3.4 变量变换

当一个多元正态分布在 $\mathbf{x}$ 中的均值是第二个变量 $\mathbf{y}$ 的线性函数 $\mathbf{Ay}+\mathbf{b}$ 时，这与另一个在 $\mathbf{y}$ 中的正态分布成正比，其中均值是 $\mathbf{x}$ 的一个线性函数：

$$
\text{Norm}_{\mathbf{x}}[\mathbf{Ay}+\mathbf{b}, \mathbf{\Sigma}] \propto \text{Norm}_{\mathbf{y}}[(\mathbf{A}^T\mathbf{\Sigma}^{-1}\mathbf{A})^{-1}\mathbf{A}^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{b}), (\mathbf{A}^T\mathbf{\Sigma}^{-1}\mathbf{A})^{-1}].
\tag{C.30}
$$

乍一看，这个关系相当晦涩，但图 C.5 展示了对于标量 $x$ 和 $y$ 的情况，这很容易理解。与前一个关系一样，这可以通过展开指数项中的二次积并配方法，使其成为一个在 $\mathbf{y}$ 中的分布来证明。（见问题18.4）。

<br>

**图 C.5 变量变换**
a) 条件分布 $\text{Pr}(x|y)$ 是一个具有恒定方差和线性依赖于 $y$ 的均值的正态分布。青色分布显示了 $y=-0.2$ 的一个例子。b) 这与条件概率 $\text{Pr}(y|x)$ 成正比，它是一个具有恒定方差和线性依赖于 $x$ 的均值的正态分布。青色分布显示了 $x=-3$ 的一个例子。

<br>

## C.4 采样

为了从一个单变量分布 $\text{Pr}(x)$ 中采样，我们首先计算**累积分布函数 (cumulative distribution function)** $F[x]$（即 $\text{Pr}(x)$ 的积分）。然后我们从 $$ 范围内的均匀分布中抽取一个样本 $z^*$，并根据累积分布的逆来评估它，这样就创建了样本 $x^*$：

$$
x^* = F^{-1}[z^*].
\tag{C.31}
$$

### C.4.1 从正态分布采样

上述方法可以用来从一个单变量标准正态分布生成一个样本 $x^*$。然后可以使用方程C.22从一个均值为 $\mu$、方差为 $\sigma^2$ 的正态分布创建一个样本。类似地，一个来自 $D$ 维多元标准分布的样本 $x^*$ 可以通过独立采样 $D$ 个单变量标准正态变量来创建。然后可以使用方程C.24从一个均值为 $\boldsymbol{\mu}$、协方差为 $\mathbf{\Sigma}$ 的多元正态分布创建一个样本。

### C.4.2 祖先采样

当联合分布可以被分解为一系列条件概率时，我们可以使用**祖先采样 (ancestral sampling)** 来生成样本。基本思想是从根变量（们）生成一个样本，然后基于这个实例化从随后的条件分布中采样。这个过程被称为祖先采样，通过一个例子最容易理解。考虑三个变量 $x, y, z$ 上的一个联合分布 $\text{Pr}(x, y, z)$，它（在这种特殊情况下）可以分解为：

$$
\text{Pr}(x, y, z) = \text{Pr}(x)\text{Pr}(y|x)\text{Pr}(z|y).
\tag{C.32}
$$

为了从这个联合分布中采样，我们首先从 $\text{Pr}(x)$ 中抽取一个样本 $x^*$。然后我们从 $\text{Pr}(y|x^*)$ 中抽取一个样本 $y^*$。最后，我们从 $\text{Pr}(z|y^*)$ 中抽取一个样本 $z^*$。

## C.5 概率分布之间的距离

监督学习可以被框定为最小化模型所隐含的概率分布与样本所隐含的离散概率分布之间的距离（5.7节）。无监督学习通常可以被框定为最小化真实样本的概率分布与来自模型的数据的分布之间的距离。在这两种情况下，我们都需要一个衡量两个概率分布之间距离的度量。本节考虑了几种不同距离度量之间的性质（另见图15.8关于瓦瑟斯坦或推土机距离的讨论）。

### C.5.1 库尔贝克-莱布勒散度

概率分布 $p(x)$ 和 $q(x)$ 之间最常用的距离度量是**库尔贝克-莱布勒 (Kullback-Leibler)** 或 **KL散度 (KL divergence)**，定义为：

$$
D_{KL}[p(x)||q(x)] = \int p(x) \log\frac{p(x)}{q(x)} dx.
\tag{C.33}
$$

这个距离总是大于或等于零，这可以通过注意到 $-\log[y] \ge 1-y$ 来轻松证明（图 C.6），因此：

$$
D_{KL}[p(x)||q(x)] = \int p(x)\log\frac{p(x)}{q(x)}dx = -\int p(x)\log\frac{q(x)}{p(x)}dx \\
\ge \int p(x)\left(1-\frac{q(x)}{p(x)}\right)dx = \int (p(x)-q(x))dx = 1-1=0.
\tag{C.34}
$$

如果存在 $q(x)$ 为零但 $p(x)$ 非零的地方，KL散度是无穷大的。当我们基于这个距离最小化一个函数时，这可能会导致问题。

### C.5.2 詹森-香农散度

KL散度是不对称的（即，$D_{KL}[p(x)||q(x)] \neq D_{KL}[q(x)||p(x)]$）。**詹森-香农散度 (Jensen-Shannon divergence)** 是一种通过构造对称的距离度量：

$$
D_{JS}[p(x)||q(x)] = \frac{1}{2}D_{KL}\left[p(x)\left|\left|\frac{p(x)+q(x)}{2}\right.\right.\right] + \frac{1}{2}D_{KL}\left[q(x)\left|\left|\frac{p(x)+q(x)}{2}\right.\right.\right].
\tag{C.35}
$$

它是 $p(x)$ 和 $q(x)$ 到这两个分布的平均值的平均散度。

<br>

**图 C.6 负对数的下界**
函数 $1-y$ 总是小于函数 $-\log[y]$。这个关系被用来证明库尔贝克-莱布勒散度总是大于或等于零。

<br>

### C.5.3 弗雷歇距离

两个分布 $p(x)$ 和 $q(x)$ 之间的**弗雷歇距离 (Fréchet distance)** $D_{Fr}$ 由下式给出：

$$
D_{Fr}[p(x)||q(x)] = \min_{\pi(x,y)} \left[\sqrt{\iint \pi(x,y)|x-y|^2 dx dy}\right].
\tag{C.36}
$$

其中 $\pi(x,y)$ 表示与边缘分布 $p(x)$ 和 $q(y)$ 兼容的联合分布集合。弗雷歇距离也可以被表述为累积概率曲线之间的最大距离的度量。

### C.5.4 正态分布之间的距离

我们经常想要计算两个均值为 $\boldsymbol{\mu}_1$ 和 $\boldsymbol{\mu}_2$，协方差为 $\mathbf{\Sigma}_1$ 和 $\mathbf{\Sigma}_2$ 的多元正态分布之间的距离。在这种情况下，各种距离度量可以以封闭形式写出。

KL散度可以计算为：

$$
D_{KL}[\text{Norm}[\boldsymbol{\mu}_1, \mathbf{\Sigma}_1] || \text{Norm}[\boldsymbol{\mu}_2, \mathbf{\Sigma}_2]] = \frac{1}{2}\left(\log\frac{|\mathbf{\Sigma}_2|}{|\mathbf{\Sigma}_1|} - D + \text{tr}[\mathbf{\Sigma}_2^{-1}\mathbf{\Sigma}_1] + (\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^T\mathbf{\Sigma}_2^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)\right),
\tag{C.37}
$$

其中 $\text{tr}[\cdot]$ 是矩阵参数的迹。弗雷歇/2-瓦瑟斯坦距离由下式给出：

$$
D^2_{Fr/W_2}[\text{Norm}[\boldsymbol{\mu}_1, \mathbf{\Sigma}_1] || \text{Norm}[\boldsymbol{\mu}_2, \mathbf{\Sigma}_2]] = ||\boldsymbol{\mu}_1-\boldsymbol{\mu}_2||^2 + \text{tr}[\mathbf{\Sigma}_1+\mathbf{\Sigma}_2-2(\mathbf{\Sigma}_1\mathbf{\Sigma}_2)^{1/2}].
\tag{C.38}
$$
