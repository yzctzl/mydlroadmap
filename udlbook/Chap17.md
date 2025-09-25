好的，请看我的翻译和解答。

***

# 第十七章
# 变分自编码器

生成对抗网络学习一种创建样本的机制，这些样本与训练样本 $\{\mathbf{x}_i\}$ 无法区分。相比之下，与归一化流类似，**变分自编码器 (variational autoencoders)** 或简称 **VAEs**，是**概率生成模型 (probabilistic generative models)**；它们旨在学习数据上的一个分布 $\text{Pr}(\mathbf{x})$（见图14.2）。训练后，可以从这个分布中抽取（生成）样本。然而，VAE的特性意味着，不幸的是，无法精确评估新样本 $\mathbf{x}^*$ 的概率。

通常人们谈论VAE时，就好像它就是 $\text{Pr}(\mathbf{x})$ 的模型，但这具有误导性；VAE是一种旨在帮助学习 $\text{Pr}(\mathbf{x})$ 模型的神经网络架构。最终的 $\text{Pr}(\mathbf{x})$ 模型既不包含“变分”部分，也不包含“自编码器”部分，或许更适合被描述为一个**非线性潜变量模型 (nonlinear latent variable model)**。

本章首先介绍一般的潜变量模型，然后考虑非线性潜变量模型的具体情况。我们会发现，对该模型的最大似然学习并非易事。尽管如此，可以为似然定义一个下界，而VAE架构使用蒙特卡洛（采样）方法来近似这个下界。本章最后将介绍VAE的几种应用。

## 17.1 潜变量模型

潜变量模型采用一种间接的方法来描述一个多维变量 $\mathbf{x}$ 上的概率分布 $\text{Pr}(\mathbf{x})$。它们不直接写出 $\text{Pr}(\mathbf{x})$ 的表达式，而是对数据 $\mathbf{x}$ 和一个未观测到的隐藏或**潜变量 (latent variable)** $\mathbf{z}$ 的联合分布 $\text{Pr}(\mathbf{x}, \mathbf{z})$ 进行建模。然后，它们将 $\text{Pr}(\mathbf{x})$ 的概率描述为该联合概率的**边缘化 (marginalization)**，因此：

$$
\text{Pr}(\mathbf{x}) = \int \text{Pr}(\mathbf{x}, \mathbf{z}) d\mathbf{z}.
\tag{17.1}
$$

通常，联合概率 $\text{Pr}(\mathbf{x}, \mathbf{z})$ 会根据**条件概率 (conditional probability)** 的规则，被分解为数据相对于潜变量的**似然 (likelihood)** 项 $\text{Pr}(\mathbf{x}|\mathbf{z})$ 和**先验 (prior)** 项 $\text{Pr}(\mathbf{z})$：参考：Appendix C.1.2 Marginalization, Appendix C.1.3 Conditional probability

$$
\text{Pr}(\mathbf{x}) = \int \text{Pr}(\mathbf{x}|\mathbf{z})\text{Pr}(\mathbf{z}) d\mathbf{z}.
\tag{17.2}
$$

这是一种相当间接的描述 $\text{Pr}(\mathbf{x})$ 的方法，但它很有用，因为相对简单的 $\text{Pr}(\mathbf{x}|\mathbf{z})$ 和 $\text{Pr}(\mathbf{z})$ 表达式可以定义复杂的分布 $\text{Pr}(\mathbf{x})$。

### 17.1.1 示例：高斯混合模型

在一维高斯混合模型中（图17.1a），潜变量 $z$ 是离散的，先验 $\text{Pr}(z)$ 是一个**分类分布 (categorical distribution)**（图5.9），对于 $z$ 的每个可能值都有一个概率 $\lambda_n$。给定潜变量 $z$ 取值为 $n$ 时，数据 $\mathbf{x}$ 的似然 $\text{Pr}(x|z=n)$ 服从均值为 $\mu_n$、方差为 $\sigma_n^2$ 的正态分布：参考：Problem 17.1

$$
\begin{aligned}
\text{Pr}(z=n) &= \lambda_n \\
\text{Pr}(x|z=n) &= \text{Norm}_x[\mu_n, \sigma_n^2].
\end{aligned}
\tag{17.3}
$$

如方程17.2所示，概率 $\text{Pr}(x)$ 是通过对潜变量 $z$ 进行边缘化给出的（图17.1b）。在这里，潜变量是离散的，因此我们对其所有可能的值求和来进行边缘化：

$$
\text{Pr}(x) = \sum_{n=1}^N \text{Pr}(x, z=n) = \sum_{n=1}^N \text{Pr}(x|z=n) \cdot \text{Pr}(z=n) = \sum_{n=1}^N \lambda_n \cdot \text{Norm}_x[\mu_n, \sigma_n^2].
\tag{17.4}
$$

通过似然和先验的简单表达式，我们描述了一个复杂的多峰概率分布。

---

> **图 17.1 高斯混合模型 (MoG)**
> a) MoG将一个复杂的概率分布（青色曲线）描述为高斯分量（虚线曲线）的加权和。b) 这个和是连续观测数据 $x$ 和离散潜变量 $z$ 之间联合密度 $\text{Pr}(x, z)$ 的边缘化。

---

## 17.2 非线性潜变量模型

在非线性潜变量模型中，数据 $\mathbf{x}$ 和潜变量 $\mathbf{z}$ 都是连续且多变量的。先验 $\text{Pr}(\mathbf{z})$ 是一个标准多元正态分布：参考：Appendix C.3.2 Multivariate normal

$$
\text{Pr}(\mathbf{z}) = \text{Norm}_\mathbf{z}[\mathbf{0}, \mathbf{I}].
\tag{17.5}
$$

似然 $\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})$ 也是正态分布的；其均值是潜变量的非线性函数 $\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]$，其协方差 $\sigma^2\mathbf{I}$ 是球形的：

$$
\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi}) = \text{Norm}_\mathbf{x}[\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}], \sigma^2\mathbf{I}].
\tag{17.6}
$$

函数 $\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]$ 由一个带参数 $\boldsymbol{\phi}$ 的深度网络描述。潜变量 $\mathbf{z}$ 的维度低于数据 $\mathbf{x}$。模型 $\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]$ 描述了数据的重要方面，而剩余的未建模方面则归因于噪声 $\sigma^2\mathbf{I}$。
数据概率 $\text{Pr}(\mathbf{x}|\boldsymbol{\phi})$ 是通过对潜变量 $\mathbf{z}$ 进行边缘化找到的：

$$
\text{Pr}(\mathbf{x}|\boldsymbol{\phi}) = \int \text{Pr}(\mathbf{x}, \mathbf{z}|\boldsymbol{\phi})d\mathbf{z} = \int \text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi}) \cdot \text{Pr}(\mathbf{z}) d\mathbf{z} = \int \text{Norm}_\mathbf{x}[\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}], \sigma^2\mathbf{I}] \cdot \text{Norm}_\mathbf{z}[\mathbf{0}, \mathbf{I}] d\mathbf{z}.
\tag{17.7}
$$

这可以被看作是具有不同均值的球形高斯分布的无限加权和（即，无限混合），其中权重是 $\text{Pr}(\mathbf{z})$，均值是网络输出 $\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]$（图17.2）。参考：Notebook 17.1 Latent variable models

### 17.2.1 生成

可以使用**祖先采样 (ancestral sampling)** 生成一个新样本 $\mathbf{x}^*$（图17.3）。我们从先验 $\text{Pr}(\mathbf{z})$ 中抽取 $\mathbf{z}^*$，并将其通过网络 $\mathbf{f}[\mathbf{z}^*, \boldsymbol{\phi}]$ 来计算似然 $\text{Pr}(\mathbf{x}|\mathbf{z}^*, \boldsymbol{\phi})$ 的均值（方程17.6），然后从中抽取 $\mathbf{x}^*$。由于先验和似然都是正态分布，这个过程是直接的。参考：Appendix C.4.2 Ancestral sampling

---

> **图 17.2 非线性潜变量模型**
> 一个复杂的二维密度 $\text{Pr}(\mathbf{x})$（右）是通过对联合分布 $\text{Pr}(\mathbf{x}, \mathbf{z})$（左）对潜变量 $z$ 进行边缘化创建的；为了创建 $\text{Pr}(\mathbf{x})$，我们将三维体沿 $z$ 维度积分。对于每个 $z$，$\mathbf{x}$ 上的分布是一个球形高斯分布（显示了两个切片），其均值 $\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]$ 是 $z$ 的非线性函数，并依赖于参数 $\boldsymbol{\phi}$。分布 $\text{Pr}(\mathbf{x})$ 是这些高斯分布的加权和。

> **图 17.3 从非线性潜变量模型生成**
> a) 我们从潜变量上的先验概率 $\text{Pr}(\mathbf{z})$ 中抽取一个样本 $\mathbf{z}^*$。b) 然后从 $\text{Pr}(\mathbf{x}|\mathbf{z}^*, \boldsymbol{\phi})$ 中抽取一个样本 $\mathbf{x}^*$。这是一个球形高斯分布，其均值是 $\mathbf{z}^*$ 的一个非线性函数 $\mathbf{f}[\cdot, \boldsymbol{\phi}]$，并具有固定的方差 $\sigma^2\mathbf{I}$。c) 如果我们多次重复这个过程，我们就能恢复密度 $\text{Pr}(\mathbf{x}|\boldsymbol{\phi})$。

---

## 17.3 训练

为了训练模型，我们针对模型参数，在训练数据集 $\{\mathbf{x}_i\}_{i=1}^I$ 上最大化对数似然。为简单起见，我们假设似然表达式中的方差项 $\sigma^2$ 是已知的，并专注于学习 $\boldsymbol{\phi}$：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\mathrm{argmax}}\left[\sum_{i=1}^I \log[\text{Pr}(\mathbf{x}_i|\boldsymbol{\phi})]\right],
\tag{17.8}
$$

其中：

$$
\text{Pr}(\mathbf{x}_i|\boldsymbol{\phi}) = \int \text{Norm}_{\mathbf{x}_i}[\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}], \sigma^2\mathbf{I}] \cdot \text{Norm}_{\mathbf{z}}[\mathbf{0}, \mathbf{I}] d\mathbf{z}.
\tag{17.9}
$$

不幸的是，这是**难解的 (intractable)**。这个积分没有封闭形式的表达式，也没有简单的方法来为特定的 $\mathbf{x}$ 值评估它。

### 17.3.1 证据下界 (ELBO)

为了取得进展，我们为对数似然定义一个**下界 (lower bound)**。这是一个对于给定的 $\boldsymbol{\phi}$ 值总是小于或等于对数似然，并且还将依赖于一些其他参数 $\boldsymbol{\theta}$ 的函数。最终，我们将构建一个网络来计算这个下界并对其进行优化。为了定义这个下界，我们需要**詹森不等式 (Jensen's inequality)**。

### 17.3.2 詹森不等式

詹森不等式指出，对于一个**凹函数 (concave function)** $g[\cdot]$，数据 $y$ 的期望的函数值，大于或等于该数据函数的期望值：参考：Appendix B.1.2 Concave functions

$$
g[\mathbb{E}[y]] \ge \mathbb{E}[g[y]].
\tag{17.10}
$$

在这种情况下，凹函数是对数，所以我们有：

$$
\log[\mathbb{E}[y]] \ge \mathbb{E}[\log[y]],
\tag{17.11}
$$

或者，完全写出期望的表达式，我们有：

$$
\log\left[\int \text{Pr}(y)ydy\right] \ge \int \text{Pr}(y)\log[y]dy.
\tag{17.12}
$$

这在图17.4-17.5中进行了探讨。实际上，一个稍微更通用的陈述是成立的：

$$
\log\left[\int \text{Pr}(y)h[y]dy\right] \ge \int \text{Pr}(y)\log[h[y]]dy.
\tag{17.13}
$$

其中 $h[y]$ 是 $y$ 的一个函数。这是因为 $h[y]$ 是另一个具有新分布的随机变量。由于我们从未指定 $\text{Pr}(y)$，这个关系保持成立。参考：Problems 17.2-17.3

---

> **图 17.4 詹森不等式（离散情况）**
> 对数函数（黑色曲线）是一个凹函数；你可以在曲线上任意两点之间画一条直线，这条线将始终位于曲线下方。由此可知，对数函数上六个点的任何凸组合（权重为正且和为一的加权和）都必须位于曲线下的灰色区域内。在这里，我们对这些点进行了等权重处理（即，取平均值）以得到青色点。由于这个点位于曲线下方，$\log[\mathbb{E}[y]] \ge \mathbb{E}[\log[y]]$。

> **图 17.5 詹森不等式（连续情况）**
> 对于一个凹函数，计算一个分布 $\text{Pr}(y)$ 的期望，然后将其通过该函数，得到的结果大于或等于将变量 $y$ 通过该函数变换后再计算新变量的期望。在对数函数的情况下，我们有 $\log[\mathbb{E}[y]] \ge \mathbb{E}[\log[y]]$。图的左侧对应于这个不等式的左侧，图的右侧对应于右侧。一种思考方式是，我们正在取定义在 $y \in$ 上的橙色分布中各点的凸组合。根据图17.4的逻辑，这必须位于曲线下方。或者，我们可以认为凹函数相对于低值压缩了 $y$ 的高值，所以当我们先将 $y$ 通过函数时，期望值会更低。

---

### 17.3.3 推导下界

我们现在使用詹森不等式来推导对数似然的下界。我们首先将对数似然乘以并除以一个在潜变量上的任意概率分布 $q(\mathbf{z})$：

$$
\log[\text{Pr}(\mathbf{x}|\boldsymbol{\phi})] = \log\left[\int \text{Pr}(\mathbf{x}, \mathbf{z}|\boldsymbol{\phi})d\mathbf{z}\right] = \log\left[\int q(\mathbf{z}) \frac{\text{Pr}(\mathbf{x}, \mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z})} d\mathbf{z}\right],
\tag{17.14}
$$

然后我们使用詹森不等式对对数函数（方程17.12）来找到一个下界：

$$
\log\left[\int q(\mathbf{z}) \frac{\text{Pr}(\mathbf{x}, \mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z})} d\mathbf{z}\right] \ge \int q(\mathbf{z}) \log\left[\frac{\text{Pr}(\mathbf{x}, \mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z})}\right] d\mathbf{z},
\tag{17.15}
$$

其中右侧被称为**证据下界 (evidence lower bound)** 或 **ELBO**。它得名于在贝叶斯规则的背景下，$\text{Pr}(\mathbf{x}|\boldsymbol{\phi})$ 被称为**证据 (evidence)**（方程17.19）。
在实践中，分布 $q(\mathbf{z})$ 有参数 $\boldsymbol{\theta}$，所以ELBO可以写成：

$$
\text{ELBO}[\boldsymbol{\theta}, \boldsymbol{\phi}] = \int q(\mathbf{z}|\boldsymbol{\theta}) \log\left[\frac{\text{Pr}(\mathbf{x}, \mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right] d\mathbf{z}.
\tag{17.16}
$$

为了学习非线性潜变量模型，我们最大化这个量，作为 $\boldsymbol{\phi}$ 和 $\boldsymbol{\theta}$ 的函数。计算这个量的神经网络架构就是VAE。

---

> **图 17.6 证据下界 (ELBO)**
> 目标是最大化关于参数 $\boldsymbol{\phi}$ 的对数似然 $\log[\text{Pr}(\mathbf{x}|\boldsymbol{\phi})]$（黑色曲线）。ELBO是一个始终位于对数似然下方的函数。它是 $\boldsymbol{\phi}$ 和第二组参数 $\boldsymbol{\theta}$ 的函数。对于固定的 $\boldsymbol{\theta}$，我们得到一个关于 $\boldsymbol{\phi}$ 的函数（两条彩色曲线代表不同值的 $\boldsymbol{\theta}$）。因此，我们可以通过 a) 改进关于新参数 $\boldsymbol{\theta}$ 的ELBO（从一条彩色曲线移动到另一条）或 b) 改进关于原始参数 $\boldsymbol{\phi}$ 的ELBO（沿着当前的彩色曲线移动）来增加对数似然。

---

## 17.4 ELBO 的性质

初次接触时，ELBO是一个有些神秘的对象，所以我们现在提供一些关于其性质的直观解释。考虑到数据的原始对数似然是参数 $\boldsymbol{\phi}$ 的函数，我们想要找到它的最大值。对于任何固定的 $\boldsymbol{\theta}$，ELBO仍然是参数 $\boldsymbol{\phi}$ 的一个函数，但它必须位于原始似然函数的下方。当我们改变 $\boldsymbol{\theta}$ 时，我们修改了这个函数，根据我们的选择，下界可能更接近或更远离对数似然。当我们改变 $\boldsymbol{\phi}$ 时，我们沿着下界函数移动（图17.6）。

### 17.4.1 界的紧致性

当对于一个固定的 $\boldsymbol{\phi}$ 值，ELBO和对数似然函数重合时，ELBO是**紧的 (tight)**。为了找到使界紧致的分布 $q(\mathbf{z}|\boldsymbol{\theta})$，我们使用条件概率的定义来分解ELBO中对数项的分子：参考：Appendix C.1.3 Conditional probability

$$
\text{ELBO}[\boldsymbol{\theta}, \boldsymbol{\phi}] = \int q(\mathbf{z}|\boldsymbol{\theta}) \log\left[\frac{\text{Pr}(\mathbf{x}, \mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right] d\mathbf{z} \\
= \int q(\mathbf{z}|\boldsymbol{\theta}) \log\left[\frac{\text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})\text{Pr}(\mathbf{x}|\boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right] d\mathbf{z} \\
= \int q(\mathbf{z}|\boldsymbol{\theta}) \log[\text{Pr}(\mathbf{x}|\boldsymbol{\phi})]d\mathbf{z} + \int q(\mathbf{z}|\boldsymbol{\theta}) \log\left[\frac{\text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right] d\mathbf{z} \\
= \log[\text{Pr}(\mathbf{x}|\boldsymbol{\phi})] + \int q(\mathbf{z}|\boldsymbol{\theta}) \log\left[\frac{\text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right] d\mathbf{z} \\
= \log[\text{Pr}(\mathbf{x}|\boldsymbol{\phi})] - D_{KL}[q(\mathbf{z}|\boldsymbol{\theta}) || \text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})].
\tag{17.17}
$$

这里，第一个积分在第三行和第四行之间消失了，因为 $\log[\text{Pr}(\mathbf{x}|\boldsymbol{\phi})]$ 不依赖于 $\mathbf{z}$，而概率分布 $q(\mathbf{z}|\boldsymbol{\theta})$ 的积分为1。在最后一行，我们刚刚使用了**库尔贝克-莱布勒（KL）散度 (Kullback-Leibler (KL) divergence)** 的定义。参考：Appendix C.5.1 KL divergence

这个方程表明，ELBO是原始对数似然减去KL散度 $D_{KL}[q(\mathbf{z}|\boldsymbol{\theta}) || \text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})]$。KL散度衡量了分布之间的“距离”，并且只能取非负值。由此可知，ELBO是对数似然 $\log[\text{Pr}(\mathbf{x}|\boldsymbol{\phi})]$ 的一个下界。当 $q(\mathbf{z}|\boldsymbol{\theta}) = \text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$ 时，KL距离将为零，界将是紧的。这是给定观测数据 $\mathbf{x}$ 后，潜变量 $\mathbf{z}$ 的**后验分布 (posterior distribution)**；它指出了哪些潜变量的值可能对该数据点负责（图17.7）。

---

> **图 17.7 潜变量的后验分布**
> a) 后验分布 $\text{Pr}(\mathbf{z}|\mathbf{x}^*, \boldsymbol{\phi})$ 是对可能对数据点 $\mathbf{x}^*$ 负责的潜变量 $\mathbf{z}$ 的值的分布。我们通过贝叶斯规则 $\text{Pr}(\mathbf{z}|\mathbf{x}^*, \boldsymbol{\phi}) \propto \text{Pr}(\mathbf{x}^*|\mathbf{z}, \boldsymbol{\phi})\text{Pr}(\mathbf{z})$ 来计算它。b) 我们通过评估 $\mathbf{x}^*$ 相对于与每个 $z$ 值相关联的对称高斯分布的概率来计算右侧的第一项（似然）。在这里，它更有可能是从 $z_1$ 而不是 $z_2$ 创建的。第二项是潜变量上的先验概率 $\text{Pr}(\mathbf{z})$。结合这两个因素并进行归一化，使得分布总和为一，就得到了我们的后验 $\text{Pr}(\mathbf{z}|\mathbf{x}^*, \boldsymbol{\phi})$。

---

### 17.4.2 ELBO 作为重构损失减去与先验的KL距离

方程17.16和17.17是表达ELBO的两种不同方式。第三种方式是将下界视为**重构误差 (reconstruction error)** 减去与**先验的距离 (distance to the prior)**：

$$
\text{ELBO}[\boldsymbol{\theta}, \boldsymbol{\phi}] = \int q(\mathbf{z}|\boldsymbol{\theta}) \log\left[\frac{\text{Pr}(\mathbf{x}, \mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right] d\mathbf{z} \\
= \int q(\mathbf{z}|\boldsymbol{\theta}) \log\left[\frac{\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})\text{Pr}(\mathbf{z})}{q(\mathbf{z}|\boldsymbol{\theta})}\right] d\mathbf{z} \\
= \int q(\mathbf{z}|\boldsymbol{\theta}) \log[\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})]d\mathbf{z} + \int q(\mathbf{z}|\boldsymbol{\theta}) \log\left[\frac{\text{Pr}(\mathbf{z})}{q(\mathbf{z}|\boldsymbol{\theta})}\right] d\mathbf{z} \\
= \int q(\mathbf{z}|\boldsymbol{\theta}) \log[\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})]d\mathbf{z} - D_{KL}[q(\mathbf{z}|\boldsymbol{\theta}) || \text{Pr}(\mathbf{z})],
\tag{17.18}
$$

其中联合分布 $\text{Pr}(\mathbf{x}, \mathbf{z}|\boldsymbol{\phi})$ 在第一行和第二行之间被分解为条件概率 $\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})\text{Pr}(\mathbf{z})$，并且在最后一行再次使用了KL散度的定义。参考：Problem 17.4

在这个公式中，第一项衡量了潜变量和数据的平均一致性 $\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})$。这衡量了**重构精度 (reconstruction accuracy)**。第二项衡量了辅助分布 $q(\mathbf{z}|\boldsymbol{\theta})$ 与先验的匹配程度。这个公式是变分自编码器中使用的那个。

## 17.5 变分近似

我们在方程17.17中看到，当 $q(\mathbf{z}|\boldsymbol{\theta})$ 是后验分布 $\text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$ 时，ELBO是紧的。原则上，我们可以使用贝叶斯规则计算后验：

$$
\text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) = \frac{\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})\text{Pr}(\mathbf{z})}{\text{Pr}(\mathbf{x}|\boldsymbol{\phi})},
\tag{17.19}
$$

但在实践中，这是难解的，因为我们无法评估分母中的证据项 $\text{Pr}(\mathbf{x}|\boldsymbol{\phi})$（见17.3节）。

一种解决方案是做一个**变分近似 (variational approximation)**：我们为 $q(\mathbf{z}|\boldsymbol{\theta})$ 选择一个简单的参数形式，并用它来近似真实的后验。在这里，我们选择一个均值为 $\boldsymbol{\mu}$、协方差为对角矩阵 $\mathbf{\Sigma}$ 的多元正态分布。这不总能很好地匹配后验，但对于某些 $\boldsymbol{\mu}$ 和 $\mathbf{\Sigma}$ 的值会比其他值更好。在训练期间，我们将找到与真实后验 $\text{Pr}(\mathbf{z}|\mathbf{x})$ “最接近”的正态分布（图17.8）。这对应于最小化方程17.17中的KL散度，并将图17.6中的彩色曲线向上移动。

由于 $q(\mathbf{z}|\boldsymbol{\theta})$ 的最优选择是后验 $\text{Pr}(\mathbf{z}|\mathbf{x})$，而这又依赖于数据样本 $\mathbf{x}$，所以变分近似也应该这样做，因此我们选择：参考：Appendix C.3.2 Multivariate normal

$$
q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) = \text{Norm}_\mathbf{z}[\mathbf{g}_\mu[\mathbf{x}, \boldsymbol{\theta}], \mathbf{g}_\Sigma[\mathbf{x}, \boldsymbol{\theta}]],
\tag{17.20}
$$

其中 $\mathbf{g}[\mathbf{x}, \boldsymbol{\theta}]$ 是一个带有参数 $\boldsymbol{\theta}$ 的第二个神经网络，它预测正态变分近似的均值 $\boldsymbol{\mu}$ 和方差 $\mathbf{\Sigma}$。

## 17.6 变分自编码器

最后，我们可以描述VAE了。我们构建一个计算ELBO的网络：

$$
\text{ELBO}[\boldsymbol{\theta}, \boldsymbol{\phi}] = \int q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})\log[\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})]d\mathbf{z} - D_{KL}[q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) || \text{Pr}(\mathbf{z})],
\tag{17.21}
$$

其中分布 $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$ 是来自方程17.20的近似。

第一项仍然涉及一个难解的积分，但由于它是关于 $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$ 的一个**期望 (expectation)**，我们可以通过采样来近似它。对于任何函数 $a[\cdot]$ 我们有：参考：Appendix C.2 Expectation

$$
\mathbb{E}_{q}[a[\mathbf{z}]] = \int a[\mathbf{z}]q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})d\mathbf{z} \approx \frac{1}{N}\sum_{n=1}^N a[\mathbf{z}_n^*],
\tag{17.22}
$$

其中 $\mathbf{z}_n^*$ 是从 $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$ 中抽取的第 $n$ 个样本。这被称为**蒙特卡洛估计 (Monte Carlo estimate)**。对于一个非常粗略的估计，我们可以只使用一个从 $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$ 中抽取的样本 $\mathbf{z}^*$：

$$
\text{ELBO}[\boldsymbol{\theta}, \boldsymbol{\phi}] \approx \log[\text{Pr}(\mathbf{x}|\mathbf{z}^*, \boldsymbol{\phi})] - D_{KL}[q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) || \text{Pr}(\mathbf{z})].
\tag{17.23}
$$

第二项是变分分布 $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) = \text{Norm}_\mathbf{z}[\boldsymbol{\mu}, \mathbf{\Sigma}]$ 和先验 $\text{Pr}(\mathbf{z}) = \text{Norm}_\mathbf{z}[\mathbf{0}, \mathbf{I}]$ 之间的KL散度。两个正态分布之间的KL散度可以以封闭形式计算。对于一个分布的参数是 $\boldsymbol{\mu}, \mathbf{\Sigma}$，另一个是标准正态分布的特殊情况，它由下式给出：参考：Appendix C.5.4 KL divergence between normal distributions

$$
D_{KL}[q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) || \text{Pr}(\mathbf{z})] = \frac{1}{2}\left(\text{Tr}[\mathbf{\Sigma}] + \boldsymbol{\mu}^T\boldsymbol{\mu} - D_z - \log[\det[\mathbf{\Sigma}]]\right).
\tag{17.24}
$$

其中 $D_z$ 是潜空间的维度。

---

> **图 17.8 变分近似**
> 后验 $\text{Pr}(\mathbf{z}|\mathbf{x}^*, \boldsymbol{\phi})$ 无法以封闭形式计算。变分近似选择一个分布族 $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$（这里是高斯分布），并试图找到这个族中与真实后验最接近的成员。a) 有时，近似（青色曲线）是好的，并且接近真实后验（橙色曲线）。b) 然而，如果后验是多峰的（如图17.7），那么高斯近似将会很差。

---

### 17.6.1 VAE 算法

总而言之，我们的目标是构建一个计算数据点 $\mathbf{x}$ 的证据下界的模型。然后我们使用一个优化算法来最大化这个下界，从而改善对数似然。为了计算ELBO，我们：
*   使用网络 $\mathbf{g}[\mathbf{x}, \boldsymbol{\theta}]$ 计算该数据点 $\mathbf{x}$ 的变分后验分布 $q(\mathbf{z}|\boldsymbol{\theta}, \mathbf{x})$ 的均值 $\boldsymbol{\mu}$ 和方差 $\mathbf{\Sigma}$，
*   从这个分布中抽取一个样本 $\mathbf{z}^*$，并且
*   使用方程17.23计算ELBO。

相关的架构如图17.9所示。现在应该清楚为什么这被称为**变分自编码器**了。它是**变分的 (variational)**，因为它计算了对后验分布的一个高斯近似。它是一个**自编码器 (autoencoder)**，因为它从一个数据点 $\mathbf{x}$ 开始，从中计算一个低维的潜向量 $\mathbf{z}$，然后使用这个向量来尽可能地重建数据点 $\mathbf{x}$。在这种情况下，从数据到潜变量的映射由网络 $\mathbf{g}[\mathbf{x}, \boldsymbol{\theta}]$ 完成，被称为**编码器 (encoder)**，而从潜变量到数据的映射由网络 $\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]$ 完成，被称为**解码器 (decoder)**。

VAE将ELBO计算为 $\boldsymbol{\theta}$ 和 $\boldsymbol{\phi}$ 的函数。为了最大化这个下界，我们通过网络运行小批量的样本，并使用像SGD或Adam这样的优化算法更新这些参数。ELBO相对于参数的梯度像往常一样使用自动微分来计算。在这个过程中，我们既在彩色曲线之间移动（改变 $\boldsymbol{\theta}$），也在它们之上移动（改变 $\boldsymbol{\phi}$）（图17.10）。在这个过程中，参数 $\boldsymbol{\phi}$ 改变以在非线性潜变量模型中为数据分配更高的似然。

---

> **图 17.9 变分自编码器**
> 编码器 $\mathbf{g}[\mathbf{x}, \boldsymbol{\theta}]$ 接收一个训练样本 $\mathbf{x}$ 并预测变分分布 $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$ 的参数 $\boldsymbol{\mu}, \mathbf{\Sigma}$。我们从这个分布中采样，然后使用解码器 $\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]$ 来预测数据 $\mathbf{x}$。损失函数是负ELBO，它取决于这个预测的准确性以及变分分布 $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$ 与先验 $\text{Pr}(\mathbf{z})$ 的相似程度（方程17.21）。

---

## 17.7 重参数化技巧

还有一个复杂之处；网络涉及一个采样步骤，很难通过这个随机组件进行微分。然而，微分过这个步骤对于更新网络中它之前的参数 $\boldsymbol{\theta}$ 是必要的。

幸运的是，有一个简单的解决方案；我们可以将随机部分移动到网络的一个分支中，该分支从 $\text{Norm}_{\boldsymbol{\epsilon}}[\mathbf{0}, \mathbf{I}]$ 中抽取一个样本 $\boldsymbol{\epsilon}^*$，然后使用以下关系：参考：Problem 17.5

$$
\mathbf{z}^* = \boldsymbol{\mu} + \mathbf{\Sigma}^{1/2} \boldsymbol{\epsilon}^*,
\tag{17.25}
$$

从预期的的高斯分布中抽样。现在我们可以像往常一样计算导数，因为反向传播算法不需要沿着随机分支向下传递。这被称为**重参数化技巧 (reparameterization trick)**（图17.11）。参考：Notebook 17.2 Reparameterization trick

---

> **图 17.10 VAE在每次迭代中更新决定下界的两个因素。**
> 解码器的参数 $\boldsymbol{\phi}$ 和编码器的参数 $\boldsymbol{\theta}$ 都被操纵以增加这个下界。

> **图 17.11 重参数化技巧**
> 使用原始架构（图17.9），我们无法轻易地通过采样步骤进行反向传播。重参数化技巧将采样步骤从主流程中移除；我们从一个标准正态分布中抽取，并将其与预测的均值和协方差相结合，以获得一个来自变分分布的样本。

---

## 17.8 应用

变分自编码器有许多用途，包括去噪、异常检测和压缩。本节回顾了图像数据的几个应用。

### 17.8.1 近似样本概率

在17.3节中，我们论证了用VAE评估一个样本的概率是不可能的，它描述了这个概率为：

$$
\text{Pr}(\mathbf{x}) = \int \text{Pr}(\mathbf{x}|\mathbf{z})\text{Pr}(\mathbf{z})d\mathbf{z} = \mathbb{E}_{\mathbf{z}}[\text{Pr}(\mathbf{x}|\mathbf{z})] = \mathbb{E}_{\mathbf{z}}[\text{Norm}_\mathbf{x}[\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}], \sigma^2\mathbf{I}]].
\tag{17.26}
$$

原则上，我们可以通过从 $\text{Pr}(\mathbf{z}) = \text{Norm}_\mathbf{z}[\mathbf{0}, \mathbf{I}]$ 中抽取样本并计算来近似这个概率：

$$
\text{Pr}(\mathbf{x}) \approx \frac{1}{N}\sum_{n=1}^N \text{Pr}(\mathbf{x}|\mathbf{z}_n).
\tag{17.27}
$$

然而，维度的诅咒意味着我们抽取的几乎所有 $\mathbf{z}_n$ 的值都会有非常低的概率 $\text{Pr}(\mathbf{x}|\mathbf{z}_n)$；我们将不得不抽取极大量的样本才能得到一个可靠的估计。一个更好的方法是使用**重要性采样 (importance sampling)**。在这里，我们从一个辅助分布 $q(\mathbf{z})$ 中采样 $\mathbf{z}$，评估 $\text{Pr}(\mathbf{x}|\mathbf{z}_n)$，并根据新分布下的概率 $q(\mathbf{z})$ 来重新缩放得到的值：

$$
\text{Pr}(\mathbf{x}) = \int \text{Pr}(\mathbf{x}|\mathbf{z})\text{Pr}(\mathbf{z})d\mathbf{z} = \int \frac{\text{Pr}(\mathbf{x}|\mathbf{z})\text{Pr}(\mathbf{z})}{q(\mathbf{z})}q(\mathbf{z})d\mathbf{z} = \mathbb{E}_{q(\mathbf{z})}\left[\frac{\text{Pr}(\mathbf{x}|\mathbf{z})\text{Pr}(\mathbf{z})}{q(\mathbf{z})}\right] \approx \frac{1}{N}\sum_{n=1}^N \frac{\text{Pr}(\mathbf{x}|\mathbf{z}_n)\text{Pr}(\mathbf{z}_n)}{q(\mathbf{z}_n)},
\tag{17.28}
$$

其中现在我们从 $q(\mathbf{z})$ 中抽取样本。如果 $q(\mathbf{z})$ 接近于 $\text{Pr}(\mathbf{x}|\mathbf{z})$ 具有高似然的 $\mathbf{z}$ 的区域，那么我们将把采样集中在相关的空间区域，并更有效地估计 $\text{Pr}(\mathbf{x})$。

我们试图积分的乘积 $\text{Pr}(\mathbf{x}|\mathbf{z})\text{Pr}(\mathbf{z})$ 与后验分布 $\text{Pr}(\mathbf{z}|\mathbf{x})$（通过贝叶斯规则）成正比。因此，一个明智的辅助分布 $q(\mathbf{z})$ 的选择是由编码器计算的变分后验 $q(\mathbf{z}|\mathbf{x})$。

通过这种方式，我们可以近似新样本的概率。只要有足够的样本，这将提供比下界更好的估计，并且可以用来通过评估测试数据的对数似然来评估模型的质量。或者，它可以被用作确定新样本是否属于该分布或是否是异常的标准。

### 17.8.2 生成

VAEs建立了一个概率模型，从这个模型中采样很容易，通过从潜变量上的先验 $\text{Pr}(\mathbf{z})$ 中抽样，将这个结果通过解码器 $\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}]$，并根据 $\text{Pr}(\mathbf{x}|\mathbf{f}[\mathbf{z}, \boldsymbol{\phi}])$ 添加噪声。不幸的是，来自普通VAEs的样本通常质量较低（图17.12a-c）。这部分是由于天真的球形高斯噪声模型，部分是由于用于先验和变分后验的高斯模型。改善生成质量的一个技巧是，从聚合后验 $q(\mathbf{z}|\boldsymbol{\theta}) = (1/I)\sum_i q(\mathbf{z}|\mathbf{x}_i, \boldsymbol{\theta})$ 而不是先验中采样；这是所有样本上的平均后验，是更能代表潜空间中真实分布的高斯混合。

现代VAEs可以产生高质量的样本（图17.12d），但只能通过使用层次化先验和专门的网络架构和正则化技术。扩散模型（第18章）可以被看作是具有层次化先验的VAEs。这些也创建了非常高质量的样本。

---

> **图 17.12 从在CELEBA上训练的标准VAE中采样**
> 在每列中，一个潜变量 $\mathbf{z}^*$ 被抽取并通过模型来预测均值 $\mathbf{f}[\mathbf{z}^*, \boldsymbol{\phi}]$，然后再添加独立的高斯噪声（见图17.3）。a) 一组样本，它们是 b) 预测的均值和 c) 球形高斯噪声向量的和。在添加噪声之前，图像看起来太平滑，之后又太嘈杂。这是典型的，通常显示的是无噪声版本，因为噪声被认为是代表图像中未被建模的方面。改编自 Dorta et al. (2018)。d) 现在可以使用层次化先验、专门的架构和仔细的正则化从VAEs生成高质量的图像。改编自 Vahdat & Kautz (2020)。

---

### 17.8.3 再合成

VAEs也可以用来修改真实数据。一个数据点 $\mathbf{x}$ 可以被投影到潜空间中，通过 (i) 取编码器预测的分布的均值，或 (ii) 使用一个优化过程来找到最大化后验概率的潜变量 $\mathbf{z}$，贝叶斯规则告诉我们这与 $\text{Pr}(\mathbf{x}|\mathbf{z})\text{Pr}(\mathbf{z})$ 成正比。

在图17.13中，多个标记为“中性”或“微笑”的图像被投影到潜空间。代表这种变化的向量是通过取这两组均值之间的潜空间差异来估计的。第二个向量被估计来代表“嘴巴闭合”与“嘴巴张开”。
现在，感兴趣的图像被投影到潜空间，然后通过加或减这些向量来修改表示。为了生成中间图像，使用了**球面线性插值 (Slerp)** 而不是线性插值。在3D中，这将是沿着球面插值与挖一条直线隧道穿过其身体的区别。

在再次解码之前编码（并可能修改）输入数据的过程被称为**再合成 (resynthesis)**。这也可以用GANs和归一化流来完成。然而，在GANs中，没有编码器，所以必须使用一个单独的程序来找到与观测数据相对应的潜变量。

### 17.8.4 解耦

在上面的再合成例子中，代表可解释属性的空间方向必须使用标记的训练数据来估计。其他工作试图改善潜空间的特性，使其坐标方向对应于现实世界的属性。当每个维度代表一个独立的现实世界因素时，潜空间被描述为**解耦的 (disentangled)**。例如，在建模人脸图像时，我们可能希望将头部姿势或头发颜色揭示为独立的因素。

鼓励解耦的方法通常向损失函数添加基于 (i) 潜变量 $\mathbf{z}$ 上的后验 $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$ 或 (ii) 聚合后验 $q(\mathbf{z}|\boldsymbol{\theta}) = (1/I)\sum_i q(\mathbf{z}|\mathbf{x}_i, \boldsymbol{\theta})$ 的正则化项：

$$
L_{new} = -\text{ELBO}[\boldsymbol{\theta}, \boldsymbol{\phi}] + \lambda_1 \mathbb{E}_{\text{Pr}(\mathbf{x})}[r_1[q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})]] + \lambda_2 r_2[q(\mathbf{z}|\boldsymbol{\theta})].
\tag{17.29}
$$

这里正则化项 $r_1[\cdot]$ 是后验的一个函数，并由 $\lambda_1$ 加权。项 $r_2[\cdot]$ 是聚合后验的一个函数，并由 $\lambda_2$ 加权。
例如，**beta VAE** 对ELBO中的第二项进行了加权（方程17.18）：

$$
\text{ELBO}[\boldsymbol{\theta}, \boldsymbol{\phi}] \approx \log[\text{Pr}(\mathbf{x}|\mathbf{z}^*, \boldsymbol{\phi})] - \beta \cdot D_{KL}[q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) || \text{Pr}(\mathbf{z})],
\tag{17.30}
$$

其中 $\beta > 1$ 决定了与重构误差相比，与先验 $\text{Pr}(\mathbf{z})$ 的偏差被加权的程度。由于先验通常是一个具有球形协方差矩阵的多元正态分布，其维度是独立的。因此，加权该项鼓励了后验分布变得更不相关。另一个变体是**总相关VAE (total correlation VAE)**，它增加一个项来减少潜空间中变量之间的总相关性（图17.14），并最大化潜变量的一小部分子集与观测之间的互信息。

## 17.9 总结

VAE是一种帮助学习关于 $\mathbf{x}$ 的非线性潜变量模型的架构。这个模型可以通过从潜变量中采样，将结果通过一个深度网络，然后添加独立的高斯噪声来生成新的样本。
以封闭形式计算一个数据点的似然是不可能的，这给使用最大似然进行训练带来了问题。然而，我们可以为似然定义一个下界并最大化这个下界。不幸的是，为了使下界紧致，我们需要计算给定观测数据的潜变量的后验概率，这也是难解的。解决方案是做一个变分近似。这是一个更简单的分布（通常是高斯分布），它近似于后验，其参数由第二个编码器网络计算。
为了从VAE创建高质量的样本，似乎有必要用比高斯先验和后验更复杂的概率分布来建模潜空间。一种选择是使用层次化先验（其中一个潜变量生成另一个）。下一章讨论了扩散模型，它能产生非常高质量的样本，并可以被看作是层次化的VAEs。

---

> **图 17.13 再合成**
> 左边的原始图像使用编码器被投影到潜空间，并选择预测的高斯分布的均值来表示该图像。网格中左上方的图像是输入的重构。其他图像是在操纵潜空间中代表微笑/中性（水平）和嘴巴张开/闭合（垂直）方向后的重构。改编自 White (2016)。

> **图 17.14 总相关VAE中的解耦**
> VAE模型被修改，使得损失函数鼓励潜变量的总相关性被最小化，从而鼓励解耦。当在一个椅子图像的数据集上训练时，几个潜维度具有清晰的现实世界解释，包括 a) 旋转，b) 整体大小，和 c) 腿（转椅与普通椅）。在每种情况下，中心列描绘了来自模型的样本，当我们向左或向右移动时，我们正在潜空间中减去或加上一个坐标向量。改编自 Chen et al. (2018d)。

---

### 注释

VAE最初由 Kingma & Welling (2014) 引入。关于变分自编码器的全面介绍可以在 Kingma et al. (2019) 中找到。

**应用 (Applications)**：VAE及其变体已被应用于图像 (Kingma & Welling, 2014; Gregor et al., 2016; Gulrajani et al., 2016; Akuzawa et al., 2018)、语音 (Hsu et al., 2017b)、文本 (Bowman et al., 2015; Hu et al., 2017; Xu et al., 2020)、分子 (Gómez-Bombarelli et al., 2018; Sultan et al., 2018)、图 (Kipf & Welling, 2016; Simonovsky & Komodakis, 2018)、机器人学 (Hernández et al., 2018; Inoue et al., 2018; Park et al., 2018)、强化学习 (Heess et al., 2015; Van Hoof et al., 2016)、3D场景 (Eslami et al., 2016, 2018; Rezende Jimenez et al., 2016) 和手写 (Chung et al., 2015)。
应用包括再合成和插值 (White, 2016; Bowman et al., 2015)、协同过滤 (Liang et al., 2018) 和压缩 (Gregor et al., 2016)。Gómez-Bombarelli et al. (2018) 使用VAE构建化学结构的连续表示，然后可以为期望的属性进行优化。Ravanbakhsh et al. (2017) 模拟天文学观测以校准测量。

**与其他模型的关系 (Relation to other models)**：**自编码器 (autoencoder)** (Rumelhart et al., 1985; Hinton & Salakhutdinov, 2006) 将数据通过一个编码器到一个瓶颈层，然后使用一个解码器来重构它。瓶颈类似于VAE中的潜变量，但动机不同。在这里，目标不是学习一个概率分布，而是创建一个捕获数据本质的低维表示。自编码器也有各种应用，包括去噪 (Vincent et al., 2008) 和异常检测 (Zong et al., 2018)。
如果编码器和解码器是线性变换，自编码器就是**主成分分析 (principal component analysis, PCA)**。因此，非线性自编码器是PCA的推广。也存在概率形式的PCA。**概率PCA (Probabilistic PCA)** (Tipping & Bishop, 1999) 向重构添加球形高斯噪声以创建一个概率模型，而**因子分析 (factor analysis)** 添加对角高斯噪声（见 Rubin & Thayer, 1982）。如果我们将这些概率变体的编码器和解码器设为非线性的，我们就回到了变分自编码器。

**架构变体 (Architectural variations)**：**条件VAE (conditional VAE)** (Sohn et al., 2015) 将类别信息 $\mathbf{c}$ 传入编码器和解码器。结果是潜空间不需要编码类别信息。例如，当MNIST数据以数字标签为条件时，潜变量可能编码数字的方向和宽度，而不是数字类别本身。Sønderby et al. (2016a) 引入了**梯形变分自编码器 (ladder variational autoencoders)**，它用一个依赖于数据的近似似然项递归地校正生成分布。

**修改似然 (Modifying likelihood)**：其他工作研究了更复杂的似然模型 $\text{Pr}(\mathbf{x}|\mathbf{z})$。**PixelVAE** (Gulrajani et al., 2016) 在输出变量上使用了一个自回归模型。Dorta et al. (2018) 对解码器输出的协方差和均值进行了建模。Lamb et al. (2016) 通过添加额外的正则化项来提高重构质量，这些项鼓励重构在一个图像分类模型的一层激活空间中与原始图像相似。这个模型鼓励保留语义信息，并被用来生成图17.13中的结果。Larsen et al. (2016) 使用一个对抗性损失进行重构，这也改善了结果。

**潜空间、先验和后验 (Latent space, prior, and posterior)**：已经研究了许多不同形式的变分近似后验，包括归一化流 (Rezende & Mohamed, 2015; Kingma et al., 2016)、有向图模型 (Maaløe et al., 2016)、无向模型 (Vahdat et al., 2020) 和用于时序数据的递归模型 (Gregor et al., 2016, 2019)。
其他作者研究了使用离散潜空间 (Van Den Oord et al., 2017; Razavi et al., 2019b; Rolfe, 2017; Vahdat et al., 2018a,b)。例如，Razavi et al. (2019b) 使用一个向量量化的潜空间，并用一个自回归模型（方程12.15）来建模先验。从中采样很慢，但可以描述非常复杂的分布。
Jiang et al. (2016) 使用一个高斯混合作为后验，允许聚类。这是一个层次化潜变量模型，它添加一个离散潜变量以提高后验的灵活性。其他作者 (Salimans et al., 2015; Ranganath et al., 2016; Maaløe et al., 2016; Vahdat & Kautz, 2020) 实验了使用连续变量的层次化模型。这些与扩散模型（第18章）有密切的联系。

**与其他模型的组合 (Combination with other models)**：Gulrajani et al. (2016) 将VAEs与自回归模型结合以产生更逼真的图像。Chung et al. (2015) 将VAE与循环神经网络结合以建模时变测量。
如上所述，对抗性损失已被用来直接告知似然项。然而，其他模型以不同的方式将生成对抗网络（GANs）的思想与VAEs结合起来。Makhzani et al. (2015) 在潜空间中使用对抗性损失；其思想是判别器将确保聚合后验分布 $q(\mathbf{z})$ 与先验分布 $\text{Pr}(\mathbf{z})$ 无法区分。Tolstikhin et al. (2018) 将此推广到先验和聚合后验之间的更广泛的距离族。Dumoulin et al. (2017) 引入了**对抗性学习推理 (adversarially learned inference)**，它使用一个对抗性损失来区分两对潜/观测数据点。在一个案例中，潜变量从潜后验分布中抽取，在另一个案例中，从先验中抽取。VAEs和GANs的其他混合体由 Larsen et al. (2016), Brock et al. (2016), 和 Hsu et al. (2017a) 提出。

**后验坍塌 (Posterior collapse)**：训练中的一个潜在问题是后验坍塌，即编码器总是预测先验分布。这由 Bowman et al. (2015) 发现，并且可以通过在训练期间逐渐增加鼓励后验和先验之间KL距离小的项来缓解。已经提出了几种其他方法来防止后验坍塌 (Razavi et al., 2019a; Lucas et al., 2019b,a)，这也是使用离散潜空间的动机之一 (Van Den Oord et al., 2017)。

**模糊的重构 (Blurry reconstructions)**：Zhao et al. (2017c) 提供了证据，表明模糊的重构部分是由于高斯噪声，也部分是由于变分近似引起的次优后验分布。也许并非巧合，一些最好的合成结果来自于使用由复杂的自回归模型建模的离散潜空间 (Razavi et al., 2019b) 或使用层次化潜空间 (Vahdat & Kautz, 2020; 见图17.12d)。图17.12a-c使用了在CELEBA数据库 (Liu et al., 2015) 上训练的VAE。图17.12d使用了一个在CELEBA HQ数据集 (Karras et al., 2018) 上训练的层次化VAE。

**其他问题 (Other problems)**：Chen et al. (2017) 指出，当使用更复杂的似然项，如PixelCNN (Van den Oord et al., 2016c) 时，输出可能完全不再依赖于潜变量。他们将此称为**信息偏好问题 (information preference problem)**。这由 Zhao et al. (2017b) 在**InfoVAE**中解决，该VAE增加了一个最大化潜分布和观测分布之间互信息的额外项。
VAE的另一个问题是潜空间中可能存在不对应于任何现实样本的“洞”。Xu et al. (2020) 引入了**约束后验VAE (constrained posterior VAE)**，它通过添加一个正则化项来帮助防止潜空间中的这些空置区域。这允许从真实样本进行更好的插值。

**解耦潜表示 (Disentangling latent representation)**：解耦潜表示的方法包括 **beta VAE** (Higgins et al., 2017) 和其他方法 (e.g., Kim & Mnih, 2018; Kumar et al., 2018)。Chen et al. (2018d) 进一步分解了ELBO，以显示一个衡量潜变量之间总相关性的项的存在（即，聚合后验与其边际乘积之间的距离）。他们用此来激发**总相关VAE (total correlation VAE)**，它试图最小化这个量。**因子VAE (Factor VAE)** (Kim & Mnih, 2018) 使用不同的方法来最小化总相关性。Mathieu et al. (2019) 讨论了在解耦表示中重要的因素。

**重参数化技巧 (Reparameterization trick)**：考虑计算某个函数的期望，其中期望所依据的概率分布依赖于某些参数。重参数化技巧计算这个期望相对于这些参数的导数。本章介绍了这种方法，作为通过近似期望的采样过程进行微分的方法；存在替代方法（见问题17.5），但重参数化技巧给出的估计器（通常）方差较低。这个问题在 Rezende et al. (2014), Kingma et al. (2015), 和 Roeder et al. (2017) 中有讨论。

**下界与EM算法 (Lower bound and the EM algorithm)**：VAE训练基于优化证据下界（有时也称为ELBO、变分下界或负变分自由能）。Hoffman & Johnson (2016) 和 Lücke et al. (2020) 以几种阐明其性质的方式重新表达了这个下界。其他工作旨在使这个界更紧 (Burda et al., 2016; Li & Turner, 2016; Bornschein et al., 2016; Masrani et al., 2019)。例如，Burda et al. (2016) 使用一个基于使用来自近似后验的多个重要性加权样本来形成目标函数的修改后的界。
当分布 $q(\mathbf{z}|\boldsymbol{\theta})$ 匹配后验 $\text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$ 时，ELBO是紧的。这是**期望最大化 (expectation maximization, EM)** 算法 (Dempster et al., 1977) 的基础。在这里，我们交替地 (i) 选择 $\boldsymbol{\theta}$ 使得 $q(\mathbf{z}|\boldsymbol{\theta})$ 等于后验 $\text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$ 和 (ii) 改变 $\boldsymbol{\phi}$ 以最大化下界（图17.15）。这对于像高斯混合这样的模型是可行的，我们可以以封闭形式计算后验分布。不幸的是，对于非线性潜变量模型，情况并非如此，因此不能使用此方法。

---

> **图 17.14 总相关VAE中的解耦**
> VAE模型被修改，以使损失函数鼓励潜变量的总相关性最小化，从而鼓励解耦。当在一个椅子图像数据集上训练时，几个潜维度具有清晰的现实世界解释，包括 a) 旋转，b) 整体大小，和 c) 腿（转椅与普通椅）。在每种情况下，中心列描绘了来自模型的样本，当我们向左或向右移动时，我们正在潜空间中减去或加上一个坐标向量。改编自 Chen et al. (2018d)。

> **图 17.15 期望最大化（EM）算法**
> EM算法交替地调整辅助参数 $\boldsymbol{\theta}$（在彩色曲线之间移动）和模型参数 $\boldsymbol{\phi}$（沿着彩色曲线移动），直到达到最大值。这些调整分别被称为E步和M步。因为E步使用后验分布 $\text{Pr}(\mathbf{h}|\mathbf{x}, \boldsymbol{\phi})$ 作为 $q(\mathbf{h}|\mathbf{x}, \boldsymbol{\theta})$，所以界是紧的，并且在每个E步之后，彩色曲线接触到黑色的似然曲线。

---

***

### 习题

**问题 17.1** 创建一个具有 $n=5$ 个分量的一维高斯混合模型（方程17.4）需要多少参数？每个参数可能取值的范围是什么？

**思路与解答：**

一个有 $N$ 个分量的一维高斯混合模型由下式定义：
$\text{Pr}(x) = \sum_{n=1}^N \lambda_n \cdot \text{Norm}_x[\mu_n, \sigma_n^2]$

对于每个分量 $n$，有三个参数：
1.  **权重 $\lambda_n$**: 混合比例。
2.  **均值 $\mu_n$**: 高斯分量的中心。
3.  **方差 $\sigma_n^2$**: 高斯分量的宽度。

*   **参数数量**:
    *   $N$ 个权重 $\lambda_n$。
    *   $N$ 个均值 $\mu_n$。
    *   $N$ 个方差 $\sigma_n^2$。
    *   总计 $3N$ 个参数。但权重有一个约束：$\sum_{n=1}^N \lambda_n = 1$。所以独立参数是 $3N-1$ 个。
    *   对于 $n=5$，有 $3 \times 5 = 15$ 个参数（或14个独立参数）。

*   **参数取值范围**:
    *   **$\lambda_n$**: $0 < \lambda_n < 1$，且 $\sum_{n=1}^5 \lambda_n = 1$。
    *   **$\mu_n$**: $\mu_n \in \mathbb{R}$ (可以是任何实数)。
    *   **$\sigma_n^2$**: $\sigma_n^2 > 0$ (方差必须是正数)。

**问题 17.2** 如果一个函数的二阶导数处处小于或等于零，则该函数是凹的。证明函数 $g[x] = \log[x]$ 是凹的。

**思路与解答：**

1.  求一阶导数: $g'[x] = \frac{d}{dx} \log[x] = \frac{1}{x}$。
2.  求二阶导数: $g''[x] = \frac{d}{dx} \frac{1}{x} = -\frac{1}{x^2}$。
3.  函数 $\log[x]$ 的定义域是 $x > 0$。
4.  在定义域内，$x^2$ 总是正的。因此，$-\frac{1}{x^2}$ 总是负的。
5.  由于二阶导数 $g''[x] < 0$ 处处成立，所以函数 $g[x] = \log[x]$ 是凹函数。

**问题 17.3** 对于凸函数，詹森不等式反向成立... 证明函数 $g[x]=x^{2n}$ 对于任意 $n \in \{1,2,3,\dots\}$ 是凸的。并用此结果证明一个分布 $\text{Pr}(x)$ 的均值的平方 $\mathbb{E}[x]^2$ 必须小于或等于其二阶矩 $\mathbb{E}[x^2]$。

**思路与解答：**

1.  **证明 $g[x]=x^{2n}$ 是凸的**:
    *   一阶导数: $g'[x] = 2n \cdot x^{2n-1}$。
    *   二阶导数: $g''[x] = 2n(2n-1) \cdot x^{2n-2}$。
    *   因为 $n \ge 1$，所以 $2n(2n-1) \ge 0$。
    *   项 $x^{2n-2} = (x^{n-1})^2$ 总是非负的。
    *   因此，二阶导数 $g''[x] \ge 0$ 处处成立，所以函数 $g[x]=x^{2n}$ 是凸的。

2.  **证明 $\mathbb{E}[x]^2 \le \mathbb{E}[x^2]$**:
    *   选择 $n=1$，则 $g[x]=x^2$ 是一个凸函数。
    *   对于凸函数，詹森不等式为 $g[\mathbb{E}[y]] \le \mathbb{E}[g[y]]$。
    *   将 $g[x]=x^2$ 代入，得到 $(\mathbb{E}[x])^2 \le \mathbb{E}[x^2]$。
    *   证明完毕。这实际上是方差 $\text{Var}(x) = \mathbb{E}[x^2] - \mathbb{E}[x]^2 \ge 0$ 的另一种表述。

**问题 17.4*** 证明以方程17.18形式表示的ELBO，可以从变分分布 $q(\mathbf{z}|\mathbf{x})$ 和真实后验分布 $\text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$ 之间的KL散度推导出来。

**思路与解答：**

1.  从KL散度的定义开始:
    $D_{KL}[q(\mathbf{z}|\mathbf{x}) || \text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})] = \int q(\mathbf{z}|\mathbf{x}) \log\frac{q(\mathbf{z}|\mathbf{x})}{\text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} d\mathbf{z}$。
2.  使用贝叶斯规则展开后验: $\text{Pr}(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) = \frac{\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})\text{Pr}(\mathbf{z})}{\text{Pr}(\mathbf{x}|\boldsymbol{\phi})}$。
    $D_{KL} = \int q(\mathbf{z}|\mathbf{x}) \log\frac{q(\mathbf{z}|\mathbf{x})\text{Pr}(\mathbf{x}|\boldsymbol{\phi})}{\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})\text{Pr}(\mathbf{z})} d\mathbf{z}$
3.  分解对数项:
    $D_{KL} = \int q(\mathbf{z}|\mathbf{x}) \left( \log\frac{q(\mathbf{z}|\mathbf{x})}{\text{Pr}(\mathbf{z})} - \log\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi}) + \log\text{Pr}(\mathbf{x}|\boldsymbol{\phi}) \right) d\mathbf{z}$
4.  拆分积分:
    $D_{KL} = \int q(\mathbf{z}|\mathbf{x}) \log\frac{q(\mathbf{z}|\mathbf{x})}{\text{Pr}(\mathbf{z})} d\mathbf{z} - \int q(\mathbf{z}|\mathbf{x}) \log\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi}) d\mathbf{z} + \int q(\mathbf{z}|\mathbf{x}) \log\text{Pr}(\mathbf{x}|\boldsymbol{\phi}) d\mathbf{z}$
5.  识别各项:
    *   第一项是 $D_{KL}[q(\mathbf{z}|\mathbf{x}) || \text{Pr}(\mathbf{z})]$。
    *   第二项是 $-\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})]$，即负的重构项。
    *   第三项中，$\log\text{Pr}(\mathbf{x}|\boldsymbol{\phi})$ 与 $\mathbf{z}$ 无关，可以提到积分外。$\int q(\mathbf{z}|\mathbf{x})d\mathbf{z}=1$。所以第三项是 $\log\text{Pr}(\mathbf{x}|\boldsymbol{\phi})$。
6.  整理得到:
    $D_{KL}[q || p] = D_{KL}[q || \text{Pr}(z)] - \mathbb{E}_q[\log\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})] + \log\text{Pr}(\mathbf{x}|\boldsymbol{\phi})$
7.  移项:
    $\log\text{Pr}(\mathbf{x}|\boldsymbol{\phi}) - D_{KL}[q || p] = \mathbb{E}_q[\log\text{Pr}(\mathbf{x}|\mathbf{z}, \boldsymbol{\phi})] - D_{KL}[q || \text{Pr}(z)]$
8.  左边是方程17.17定义的ELBO，右边是方程17.18定义的ELBO。证明完毕。

**问题 17.5** 重参数化技巧计算期望的导数... 证明该导数也可以... 这被称为REINFORCE算法。

**思路与解答：**
这需要用到对数导数技巧 (Log-derivative trick)。
1.  目标: $\nabla_{\boldsymbol{\phi}} \mathbb{E}_{\text{Pr}(\mathbf{x}|\boldsymbol{\phi})}[f(\mathbf{x})]$
2.  展开期望: $\nabla_{\boldsymbol{\phi}} \int \text{Pr}(\mathbf{x}|\boldsymbol{\phi}) f(\mathbf{x}) d\mathbf{x}$
3.  交换积分和微分: $\int \nabla_{\boldsymbol{\phi}} \text{Pr}(\mathbf{x}|\boldsymbol{\phi}) f(\mathbf{x}) d\mathbf{x}$
4.  使用对数导数技巧: $\nabla_{\boldsymbol{\phi}} \text{Pr}(\mathbf{x}|\boldsymbol{\phi}) = \text{Pr}(\mathbf{x}|\boldsymbol{\phi}) \nabla_{\boldsymbol{\phi}} \log \text{Pr}(\mathbf{x}|\boldsymbol{\phi})$。
5.  代入: $\int \text{Pr}(\mathbf{x}|\boldsymbol{\phi}) \nabla_{\boldsymbol{\phi}} \log \text{Pr}(\mathbf{x}|\boldsymbol{\phi}) f(\mathbf{x}) d\mathbf{x}$
6.  写回期望形式: $\mathbb{E}_{\text{Pr}(\mathbf{x}|\boldsymbol{\phi})} [f(\mathbf{x}) \nabla_{\boldsymbol{\phi}} \log \text{Pr}(\mathbf{x}|\boldsymbol{\phi})]$
7.  近似为蒙特卡洛估计: $\approx \frac{1}{I} \sum_{i=1}^I f(\mathbf{x}_i) \nabla_{\boldsymbol{\phi}} \log \text{Pr}(\mathbf{x}_i|\boldsymbol{\phi})$。
    证明完毕。

**问题 17.6** 在潜空间中移动点时，为什么使用球面线性插值比常规线性插值更好？提示：考虑图8.13。

**思路与解答：**

*   VAE的先验通常是标准多元高斯分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$。这个分布是**球对称**的。这意味着在任何一个以原点为中心的超球面上，概率密度都是相等的。
*   **常规线性插值**: $\mathbf{z}_{interp} = (1-t)\mathbf{z}_1 + t\mathbf{z}_2$。这条路径是一条**直线**，会穿过超球体的内部。如果 $\mathbf{z}_1, \mathbf{z}_2$ 都在一个高概率的球壳上，它们的线性插值点（特别是中点）会更靠近原点，而这个区域的概率密度可能非常**低**（如图8.13所示的高维空间“空心”现象）。这会导致生成的中间图像质量下降或不真实。
*   **球面线性插值 (Slerp)**: 它在连接 $\mathbf{z}_1, \mathbf{z}_2$ 的**超球面**的**弧线**上进行插值。这保证了所有插值点与原点的距离是平滑变化的（通常保持在一个相似的范数范围内），从而使它们始终处于一个**高概率密度区域**。
*   **结论**: 使用Slerp可以确保插值路径始终位于潜空间的高概率区域，从而生成更平滑、更真实的过渡图像。

**问题 17.7*** 推导拟合N个分量的一维高斯混合模型的EM算法。

**思路与解答：**

EM算法包含E步和M步。
1.  **(E-Step) 期望步**: 计算后验概率。
    给定当前参数 $\{\lambda_n, \mu_n, \sigma_n^2\}_{n=1}^N$，对于每个数据点 $x_i$，我们计算它属于每个分量 $n$ 的后验概率（责任, responsibility）：
    $\gamma(z_{in}) = p(z=n|x_i; \boldsymbol{\theta}) = \frac{p(x_i|z=n)p(z=n)}{\sum_{j=1}^N p(x_i|z=j)p(z=j)} = \frac{\lambda_n \mathcal{N}(x_i|\mu_n, \sigma_n^2)}{\sum_{j=1}^N \lambda_j \mathcal{N}(x_i|\mu_j, \sigma_j^2)}$

2.  **(M-Step) 最大化步**: 更新参数以最大化期望的对数似然。
    利用E步计算出的后验概率 $\gamma(z_{in})$，我们更新参数：
    *   **更新均值 $\mu_n$**: $\mu_n^{new} = \frac{\sum_{i=1}^I \gamma(z_{in}) x_i}{\sum_{i=1}^I \gamma(z_{in})}$ (每个分量的加权均值)
    *   **更新方差 $\sigma_n^2$**: $\sigma_n^{2, new} = \frac{\sum_{i=1}^I \gamma(z_{in}) (x_i - \mu_n^{new})^2}{\sum_{i=1}^I \gamma(z_{in})}$ (每个分量的加权方差)
    *   **更新权重 $\lambda_n$**: $\lambda_n^{new} = \frac{1}{I} \sum_{i=1}^I \gamma(z_{in})$ (每个分量负责的数据点的平均比例)
    (这里没有使用拉格朗日乘子，但结果是一样的)。

3.  **迭代**: 重复E步和M步直到收敛。