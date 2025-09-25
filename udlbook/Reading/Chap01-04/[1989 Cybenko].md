
# 通过S型函数的叠加进行逼近*

**G. Cybenko**†

**摘要**  在本文中，我们证明了一个固定的单变量函数与一组仿射泛函的复合所形成的有限线性组合，可以在单位超立方体内一致地逼近任何n个实变量的连续函数；我们只对该单变量函数施加了温和的条件。我们的结果解决了一个关于单隐层神经网络表示能力的悬而未决的问题。特别地，我们证明了任意的决策区域都可以被只有一个内部隐藏层和任何连续S型非线性函数的连续前馈神经网络任意好地逼近。本文还讨论了可能由人工神经网络实现的其他类型非线性函数的逼近性质。

**关键词**  神经网络、逼近、完备性。

## 1. 引言

许多不同的应用领域都关注于用以下形式的有限线性组合来表示一个n维实变量 $x \in \mathbb{R}^n$ 的一般函数：
$$
\sum_{j=1}^N \alpha_j \sigma(\mathbf{y}_j^T \mathbf{x} + \theta_j),
\tag{1}
$$
其中 $\mathbf{y}_j \in \mathbb{R}^n$ 且 $\alpha_j, \theta_j \in \mathbb{R}$ 是固定的。（$\mathbf{y}^T$ 是 $\mathbf{y}$ 的转置，因此 $\mathbf{y}^T \mathbf{x}$ 是 $\mathbf{y}$ 和 $\mathbf{x}$ 的内积。）这里的单变量函数 $\sigma$ 很大程度上取决于应用背景。我们主要关注的是所谓的S型函数（sigmoidal）$\sigma$：
$$
\sigma(t) \to \begin{cases} 1 & \text{as } t \to +\infty, \\ 0 & \text{as } t \to -\infty. \end{cases}
$$
这类函数在神经网络理论中自然地作为神经节点（或现在更倾向于称之为“单元(unit)”）的激活函数而出现 [L1], [RHM]。本文的主要结果是证明了，如果 $\sigma$ 是任何连续的S型函数，那么形如 (1) 的和在单位立方体上的连续函数空间中是稠密的。

函数的讨论在细节上最为详尽，但我们也陈述了在其他可能的 $\sigma$ 函数上保证类似结果的一般条件。

在信号处理和控制应用中，人工神经网络的可能用途最近引起了广泛关注[B], [G]。不严格地说，一个人工神经网络是由一个单一、简单的非线性激活或响应函数的复合与叠加形成的。因此，网络的输出是该特定复合与叠加所产生的函数的值。特别地，最简单的非平凡网络类别是那些具有一个内部层的网络，它们实现了由 (1) 给出的函数类。在诸如模式分类[L1]和时间序列的非线性预测[LF]等应用中，目标是适当地选择复合与叠加，以实现期望的网络响应（分别旨在实现一个分类函数或非线性预测器）。

这就引出了一个问题：确定哪些函数类可以被人工神经网络有效地实现。类似的问题在电路理论和滤波器设计中是相当熟悉且被充分研究的，在这些领域中，简单的非线性设备被用来合成或逼近期望的传递函数。例如，数字信号处理中的一个基本结果是，由单位延迟和常数乘法器构成的数字滤波器可以任意好地逼近任何连续的传递函数。从这个意义上说，本文的主要结果表明，只有一个内部层和任意连续S型非线性的网络享有同一种**普适性 (universality)**。

要求像 (1) 这样的有限线性组合**精确地**表示一个给定的连续函数是要求过高了。在希尔伯特第13个问题的一个著名解决方案中，Kolmogorov证明了所有 $n$ 个变量的连续函数都可以用有限个单变量函数的叠加和复合来精确表示[K], [L2]。然而，Kolmogorov的表示涉及不同的非线性函数。在投影寻踪方法用于统计数据分析的背景下，精确表示性的问题在[DS]中被进一步探讨[H]。

我们的兴趣在于涉及**相同**单变量函数的有限线性组合。此外，我们满足于**逼近**而非精确表示。很容易看出，在这种情况下，(1) 仅仅是有限傅里叶级数逼近的一般化。用于证明这种完备性性质的数学工具通常分为两类：一类涉及函数代数（导致Stone-Weierstrass论证[A]），另一类涉及平移不变子空间（导致Tauberian定理[R2]）。我们在本文中给出了这两种情况的例子。

我们的主要结果解决了一个长期存在的问题，即关于连续值、单隐层神经网络可以实现的确切决策区域类别。最近对这个问题的一些讨论见于[HL1], [HL2], [MSJ]和[WL]，而[N]包含了早期的一个严格分析。在[N]中，Nilsson证明了任何 $M$ 个点的集合都可以被一个具有一个内部层的网络划分为两个任意的子集。通过例子和特殊情况，越来越多的证据表明这类网络可以实现更一般的决策区域，但一直缺少一个普遍的理论。在[MSJ]中，Makhoul等人对一些可以由单层精确构造的决策区域进行了详细的几何分析。相比之下，我们这里的工作表明，$\mathbb{R}^n$ 的任何紧致、不相交子集的集合都可以以任意精度被区分开。该结果包含在定理3和随后的讨论中。

目前还有一些其他工作致力于解决本文中提出的同类问题。在[HSW]中，Hornik等人证明了在单层网络中的单调S型函数在连续函数空间中是完备的。Carroll和Dickinson [CD]证明了完备性可以通过使用Radon变换的思想来构造性地证明。Jones [J]概述了一个对任意有界S型函数的完备性的简单构造性证明。Funahashi [F]给出了一个涉及傅里叶分析和Paley-Wiener理论的证明。在早期的工作[C]中，我们给出了一个构造性的数学证明，证明了具有两个隐藏层的连续神经网络可以逼近任意的连续函数。

我们使用的主要技术来自于标准的泛函分析。主要定理的证明过程如下。我们首先注意到，形式 (1) 的有限求和确定了 $\mathbb{R}^n$ 单位超立方体上所有连续函数空间中的一个子空间。利用Hahn-Banach和Riesz表示定理，我们证明了这个子空间被一个有限测度所零化。这个测度也必须零化 (1) 中的每一项，这导致了对 $\sigma$ 的必要条件。我们使用的所有基本泛函分析都可以在[A], [R2]中找到。

本文的组织结构如下。在第2节中，我们处理预备知识，陈述并证明了本文的主要结果。本文的大部分技术细节都在第2节中。在第3节中，我们专注于神经网络理论中感兴趣的情况，并阐述其推论。第4节是关于其他类型的函数 $\sigma$ 也能导致类似结果的讨论，而第5节是讨论和总结。

## 2. 主要结果

令 $I_n$ 表示 $n$ 维单位立方体 $^n$。$I_n$ 上的连续函数空间用 $C(I_n)$ 表示，我们用 $||f||$ 来表示 $f \in C(I_n)$ 的上确界（或一致）范数。通常，我们用 $||\cdot||$ 来表示一个函数在其定义域上的最大值。$I_n$ 上的有限、带号正则Borel测度空间用 $M(I_n)$ 表示。关于这些和其他泛函分析构造的介绍，请参见[R2]。

本文的主要目标是研究在何种条件下，形如
$$
G(\mathbf{x}) = \sum_{j=1}^N \alpha_j \sigma(\mathbf{y}_j^T \mathbf{x} + \theta_j)
$$
的和在 $C(I_n)$ 中关于上确界范数是稠密的。

**定义.** 我们称 $\sigma$ 对于一个测度 $\mu \in M(I_n)$ 是**判别性的 (discriminatory)**，如果
$$
\int_{I_n} \sigma(\mathbf{y}^T \mathbf{x} + \theta) d\mu(\mathbf{x}) = 0
$$
对于所有的 $\mathbf{y} \in \mathbb{R}^n$ 和 $\theta \in \mathbb{R}$ 蕴含了 $\mu=0$。

**定义.** 我们称 $\sigma$ 是 **S型的 (sigmoidal)**，如果
$$
\sigma(t) \to \begin{cases} 1 & \text{as } t \to +\infty, \\ 0 & \text{as } t \to -\infty. \end{cases}
$$

**定理 1.** 令 $\sigma$ 是任意连续的判别性函数。那么形如
$$
G(\mathbf{x}) = \sum_{j=1}^N \alpha_j \sigma(\mathbf{y}_j^T \mathbf{x} + \theta_j)
\tag{2}
$$
的有限和在 $C(I_n)$ 中是**稠密的**。换句话说，给定任意 $f \in C(I_n)$ 和 $\epsilon > 0$，存在一个上述形式的和 $G(\mathbf{x})$，使得
$$
|G(\mathbf{x}) - f(\mathbf{x})| < \epsilon \quad \text{for all } \mathbf{x} \in I_n.
$$
**证明.** 令 $S \subset C(I_n)$ 是形如(2)的函数 $G(\mathbf{x})$ 的集合。显然 $S$ 是 $C(I_n)$ 的一个线性子空间。我们声称 $S$ 的闭包是整个 $C(I_n)$。
假设 $S$ 的闭包不是整个 $C(I_n)$。那么 $S$ 的闭包，比如说 $R$，是 $C(I_n)$ 的一个闭真子空间。根据Hahn-Banach定理，存在一个在 $C(I_n)$ 上的有界线性泛函，称之为 $L$，它具有 $L \neq 0$ 但 $L(R)=L(S)=0$ 的性质。
根据Riesz表示定理，这个有界线性泛函 $L$ 具有以下形式：
$$
L(h) = \int_{I_n} h(\mathbf{x}) d\mu(\mathbf{x})
$$
对于某个 $\mu \in M(I_n)$，对所有 $h \in C(I_n)$ 成立。特别地，由于 $\sigma(\mathbf{y}^T\mathbf{x}+\theta)$ 对于所有的 $\mathbf{y}$ 和 $\theta$ 都在 $R$ 中，我们必须有
$$
\int_{I_n} \sigma(\mathbf{y}^T\mathbf{x}+\theta) d\mu(\mathbf{x}) = 0
$$
对于所有的 $\mathbf{y}$ 和 $\theta$ 成立。
然而，我们假设了 $\sigma$ 是判别性的，所以这个条件蕴含了 $\mu=0$，这与我们的假设相矛盾。因此，子空间 $S$ 必须在 $C(I_n)$ 中是稠密的。

这证明了只要 $\sigma$ 是连续且判别性的，形如
$$
G(\mathbf{x}) = \sum_{j=1}^N \alpha_j \sigma(\mathbf{y}_j^T \mathbf{x} + \theta_j)
$$
的和在 $C(I_n)$ 中就是稠密的。


这个论证过程相当通用，并且可以应用于第四节讨论的其他情况。现在，我们专注于这个结果，以证明任何形如我们之前讨论的连续S型函数 $\sigma$，即
$$
\sigma(t) \to \begin{cases} 1 & \text{as } t \to +\infty, \\ 0 & \text{as } t \to -\infty, \end{cases}
$$
都是判别性的。值得注意的是，在神经网络应用中，连续的S型激活函数通常被认为是单调递增的，但在我们的结果中并不要求任何单调性。

**引理 1.** 任何有界、可测的S型函数 $\sigma$ 都是判别性的。特别地，任何连续的S型函数都是判别性的。

**证明.** 为了证明这一点，请注意对于任意的 $x, y, \theta, \varphi$，我们有
$$
\sigma(\lambda(\mathbf{y}^T\mathbf{x} + \theta) + \varphi) \to \begin{cases} \to 1 & \text{for } \mathbf{y}^T\mathbf{x} + \theta > 0 & \text{as } \lambda \to +\infty, \\ \to 0 & \text{for } \mathbf{y}^T\mathbf{x} + \theta < 0 & \text{as } \lambda \to +\infty, \\ = \sigma(\varphi) & \text{for } \mathbf{y}^T\mathbf{x} + \theta = 0 & \text{for all } \lambda. \end{cases}
$$
因此，函数序列 $\sigma_\lambda(\mathbf{x}) = \sigma(\lambda(\mathbf{y}^T\mathbf{x} + \theta) + \varphi)$ 逐点且有界地收敛于函数
$$
\gamma(\mathbf{x}) = \begin{cases} = 1 & \text{for } \mathbf{y}^T\mathbf{x} + \theta > 0, \\ = 0 & \text{for } \mathbf{y}^T\mathbf{x} + \theta < 0, \\ = \sigma(\varphi) & \text{for } \mathbf{y}^T\mathbf{x} + \theta = 0, \end{cases}
$$
当 $\lambda \to +\infty$ 时。
令 $\Pi_{\mathbf{y}, \theta}$ 为由 $\{\mathbf{x} | \mathbf{y}^T\mathbf{x} + \theta = 0\}$ 定义的超平面，令 $H_{\mathbf{y}, \theta}$ 为由 $\{\mathbf{x} | \mathbf{y}^T\mathbf{x} + \theta > 0\}$ 定义的开半空间。那么根据勒贝格有界收敛定理，我们有
$$
0 = \int_{I_n} \sigma_\lambda(\mathbf{x}) d\mu(\mathbf{x}) = \int_{I_n} \gamma(\mathbf{x}) d\mu(\mathbf{x}) = \sigma(\varphi)\mu(\Pi_{\mathbf{y}, \theta}) + \mu(H_{\mathbf{y}, \theta})
$$
对于所有的 $\varphi, \theta, \mathbf{y}$ 成立。
我们现在证明，所有半空间的测度都为0意味着测度 $\mu$ 本身必须为0。如果 $\mu$ 是一个正测度，这将是平凡的，但这里并非如此。
固定 $\mathbf{y}$。对于一个有界可测函数 $h$，定义线性泛函 $F$ 如下：
$$
F(h) = \int_{I_n} h(\mathbf{y}^T\mathbf{x}) d\mu(\mathbf{x})
$$
并注意 $F$ 是 $L^\infty(\mathbb{R})$ 上的有界泛函，因为 $\mu$ 是一个有限带号测度。令 $h$ 为区间 $[0, \infty)$ 上的指示函数（即，如果 $u \ge 0$ 则 $h(u)=1$，如果 $u < 0$ 则 $h(u)=0$），那么
$$
F(h) = \int_{I_n} h(\mathbf{y}^T\mathbf{x}) d\mu(\mathbf{x}) = \mu(\Pi_{\mathbf{y}, -\theta}) + \mu(H_{\mathbf{y}, -\theta}) = 0.
$$
类似地，如果 $h$ 是开区间 $(0, \infty)$ 上的指示函数，则 $F(h)=0$。根据线性性，$F(h)=0$ 对于任何区间的指示函数都成立，因此对于任何简单函数（即区间指示函数的和）也成立。由于简单函数在 $L^\infty(\mathbb{R})$ 中是稠密的（见[A]中第90页），$F=0$。
特别地，有界可测函数 $s(u)=\sin(m \cdot u)$ 和 $c(u)=\cos(m \cdot u)$ 给出
$$
F(s+ic) = \int_{I_n} (\cos(m^T\mathbf{x}) + i\sin(m^T\mathbf{x})) d\mu(\mathbf{x}) = \int_{I_n} \exp(im^T\mathbf{x})d\mu(\mathbf{x}) = 0
$$
对于所有的 $m$ 成立。因此，$\mu$ 的傅里叶变换为0，所以 $\mu$ 必须也为零[R2, p. 176]。因此，$\sigma$ 是判别性的。

### 3. 在人工神经网络中的应用

在本节中，我们将前面的结果应用于神经网络理论中最感兴趣的情况。定理1和引理1的直接组合表明，只有一个内部层和任意连续S型函数的网络，可以以任意精度逼近连续函数，前提是对节点的数量或权重的大小没有限制。这就是下面的定理2。该结果对于一般决策区域的逼近所带来的后果将在其后阐述。

**定理 2.** 令 $\sigma$ 是任意连续的S型函数。那么形如
$$
G(\mathbf{x}) = \sum_{j=1}^N \alpha_j \sigma(\mathbf{y}_j^T \mathbf{x} + \theta_j)
$$
的有限和在 $C(I_n)$ 中是稠密的。换句话说，给定任意 $f \in C(I_n)$ 和 $\epsilon > 0$，存在一个上述形式的和 $G(\mathbf{x})$，使得
$$
|G(\mathbf{x}) - f(\mathbf{x})| < \epsilon \quad \text{for all } \mathbf{x} \in I_n.
$$
**证明.** 结合定理1和引理1，注意到连续的S型函数满足该引理的条件。

我们现在在决策区域的背景下展示这些结果的含义。令 $m$ 表示 $I_n$ 中的勒贝格测度。令 $P_1, P_2, \dots, P_k$ 是 $I_n$ 到 $k$ 个不相交、可测子集的划分。根据
$$
f(\mathbf{x}) = j \quad \text{当且仅当} \quad \mathbf{x} \in P_j
$$
来定义决策函数 $f$。
这个函数 $f$ 可以被看作是分类的决策函数：如果 $f(\mathbf{x})=j$，那么我们知道 $\mathbf{x} \in P_j$，我们可以相应地对 $\mathbf{x}$ 进行分类。问题在于，这样的一个决策函数是否可以由一个具有单个内部层的网络来实现。

我们有以下基本结果。

**定理 3.** 令 $\sigma$ 是一个连续的S型函数。令 $f$ 是 $I_n$ 的任何有限可测划分的决策函数。对于任何 $\epsilon > 0$，存在一个形如
$$
G(\mathbf{x}) = \sum_{j=1}^N \alpha_j \sigma(\mathbf{y}_j^T \mathbf{x} + \theta_j)
$$
的有限和，以及一个集合 $D \subset I_n$，使得 $m(D) \ge 1-\epsilon$ 且
$$
|G(\mathbf{x}) - f(\mathbf{x})| < \epsilon \quad \text{for } \mathbf{x} \in D.
$$
**证明.** 根据卢津定理 (Lusin's theorem) [R1]，存在一个连续函数 $h$ 和一个集合 $D$，使得 $m(D) \ge 1-\epsilon$ 且对于 $\mathbf{x} \in D$ 有 $h(\mathbf{x}) = f(\mathbf{x})$。现在 $h$ 是连续的，因此根据定理2，我们可以找到一个上述形式的和 $G$ 来满足 $|G(\mathbf{x})-h(\mathbf{x})| < \epsilon$ 对于所有 $\mathbf{x} \in I_n$ 成立。那么对于 $\mathbf{x} \in D$，我们有
$$
|G(\mathbf{x}) - f(\mathbf{x})| = |G(\mathbf{x}) - h(\mathbf{x})| < \epsilon.
$$

由于连续性，我们总是不得不对某些点做出一些不正确的决策。这个结果表明，被错误分类的点的总测度可以被做得任意小。有鉴于此，定理2似乎是这类结果中可能的最强形式。

我们可以通过考虑一个单一闭集 $D \subset I_n$ 的决策问题来进一步发展这个逼近思想。那么，如果 $\mathbf{x} \in D$ 则 $f(\mathbf{x})=1$，否则 $f(\mathbf{x})=0$；$f$ 是集合 $D \subset I_n$ 的指示函数。假设我们希望找到一个形如(1)的和来逼近这个决策函数。令
$$
\Delta(\mathbf{x}, D) = \min \{||\mathbf{x}-\mathbf{y}||, \mathbf{y} \in D\}
$$
因此 $\Delta(\mathbf{x}, D)$ 是 $\mathbf{x}$ 的一个连续函数。现在设
$$
f_\epsilon(\mathbf{x}) = \max\left\{0, \frac{\epsilon-\Delta(\mathbf{x}, D)}{\epsilon}\right\}
$$
因此，对于距离 $D$ 超过 $\epsilon$ 的点 $\mathbf{x}$，$f_\epsilon(\mathbf{x})=0$，而对于 $\mathbf{x} \in D$，$f_\epsilon(\mathbf{x})=1$。此外，$f_\epsilon(\mathbf{x})$ 在 $\mathbf{x}$ 中是连续的。
根据定理2，找到一个形如(1)的 $G(\mathbf{x})$ 使得 $|G(\mathbf{x})-f_\epsilon(\mathbf{x})|<1/4$，并使用这个 $G$ 作为一个近似的决策函数：$G(\mathbf{x}) < 1/2$ 猜测 $\mathbf{x} \in D^c$ 而 $G(\mathbf{x}) \ge 1/2$ 猜测 $\mathbf{x} \in D$。这个决策过程对于所有 $\mathbf{x} \in D$ 和所有距离 $D$ 至少为 $\epsilon$ 的点 $\mathbf{x}$ 都是正确的。如果 $\mathbf{x}$ 在 $D$ 的 $\epsilon$ 距离内，它的分类取决于 $G(\mathbf{x})$ 的具体选择。
这些观察表明，离闭合决策区域足够远的点和内部的点都可以被正确分类。相比之下，定理3表明，存在一个网络，使得被错误分类的点的测度可以任意小，但不保证它们的位置。

### 4. 其他激活函数的结果

在本节中，我们讨论其他类别的激活函数，它们具有与连续S型函数相似的逼近性质。由于这些其他例子在实践中的兴趣稍小，我们只简要勾勒相应的证明。

对于不连续的S型函数，如硬限幅器（当 $x>0$ 时 $\sigma(x)=1$，当 $x<0$ 时 $\sigma(x)=0$），存在着相当大的兴趣。不连续的S型函数不如连续的常用（因为缺乏好的训练算法），但它们在理论上具有重要意义，因为它们与经典的感知机和Gamba网络[MP]有密切关系。

假设 $\sigma$ 是一个有界的、可测的S型函数。我们有定理2的一个类似物，如下所示：

**定理 4.** 令 $\sigma$ 是有界可测的S型函数。那么形如
$$
G(\mathbf{x}) = \sum_{j=1}^N \alpha_j \sigma(\mathbf{y}_j^T \mathbf{x} + \theta_j)
$$
的有限和在 $L^1(I_n)$ 中是稠密的。换句话说，给定任意 $f \in L^1(I_n)$ 和 $\epsilon > 0$，存在一个上述形式的和 $G(\mathbf{x})$，使得
$$
||G-f||_1 = \int_{I_n} |G(\mathbf{x}) - f(\mathbf{x})| dx < \epsilon.
$$

证明遵循定理1和2的证明，只需做一些显而易见的变化，比如用可积函数替换连续函数，并使用 $L^\infty(I_n)$ 是 $L^1(I_n)$ 的对偶空间这一事实。判别性的概念也相应地改变为：对于 $h \in L^\infty(I_n)$，条件
$$
\int_{I_n} \sigma(\mathbf{y}^T\mathbf{x}+\theta)h(\mathbf{x}) dx = 0
$$
对于所有的 $\mathbf{y}$ 和 $\theta$ 蕴含了 $h(\mathbf{x})=0$ 几乎处处成立。一般的S型函数在这种意义上是判别性的，正如在引理1中已经看到的，因为形如 $h(\mathbf{x})dx$ 的测度属于 $M(I_n)$。
由于在 $L^1$ 中的收敛蕴含了在测度上的收敛[A]，我们有定理3的一个类似物，如下所示：

**定理 5.** 令 $\sigma$ 是一个一般的S型函数。令 $f$ 是 $I_n$ 的任何有限可测划分的决策函数。对于任何 $\epsilon > 0$，存在一个形如
$$
G(\mathbf{x}) = \sum_{j=1}^N \alpha_j \sigma(\mathbf{y}_j^T \mathbf{x} + \theta_j)
$$
的有限和，以及一个集合 $D \subset I_n$，使得 $m(D) \ge 1-\epsilon$ 且
$$
|G(\mathbf{x}) - f(\mathbf{x})| < \epsilon \quad \text{for } \mathbf{x} \in D.
$$

通过简单地使用斯通-魏尔斯特拉斯定理[A]，可以证明许多其他可能的激活函数具有与定理1中相似的逼近性质。这些包括正弦和余弦函数，因为 $\sin(mt)$ 和 $\cos(mt)$ 的线性组合生成了所有有限的三角多项式，这些在经典上被认为是完备的。有趣的是，三角多项式的完备性在引理1中被含蓄地使用了，当时使用了傅里叶变换的一对一映射性质（在分布上）。另一个经典的例子是指数函数 $\exp(mt)$，其证明同样直接来自于斯通-魏尔斯特拉斯定理的应用。指数激活函数由Palm在[P]中研究，其中展示了它们的完备性。

另一大类可能的激活函数由于维纳-陶伯定理[R2]而在 $L^1(I_n)$ 中具有完备性。例如，假设 $\sigma$ 是任何具有非零积分的 $L^1(\mathbb{R})$ 函数。那么形如(1)的和在 $L^1(\mathbb{R}^n)$ 中是稠密的，如下文所述。
定理1的类似物成立，但我们将 $C(I_n)$ 更改为 $L^1(I_n)$，$M(I_n)$ 更改为相应的对偶空间 $L^\infty(I_n)$。定理3的类似物成立，如果我们能证明一个具有非零积分的可积函数 $\sigma$ 在
$$
\int_{I_n} \sigma(\mathbf{y}^T\mathbf{x}+\theta)h(\mathbf{x}) dx = 0
$$
对于所有的 $\mathbf{y}$ 和 $\theta$ 蕴含了 $h=0$ 的意义上是判别性的。
为此，我们如下进行。如引理1中，定义在 $L^1(\mathbb{R})$ 上的有界线性泛函 $F$ 为
$$
F(g) = \int_{I_n} g(\mathbf{y}^T\mathbf{x})h(\mathbf{x}) dx.
$$
（注意积分是存在的，因为它是在 $I_n$ 上，并且 $h$ 是有界的。具体来说，如果 $g \in L^1(\mathbb{R})$，那么对于任何 $\mathbf{y}$，$g(\mathbf{y}^T\mathbf{x}) \in L^1(I_n)$。）
令 $g_{\theta,s}(t) = \sigma(st+\theta)$，我们看到
$$
F(g_{\theta,s}) = \int_{I_n} \sigma((s\mathbf{y})^T\mathbf{x}+\theta)h(\mathbf{x}) dx = 0
$$
所以 $F$ 零化了 $g_{0,1}$ 的每一个平移和缩放。令 $\hat{f}$ 是 $f$ 的傅里叶变换。通过标准的傅里叶变换论证，$\hat{g}_{\theta,s}(z) = \exp(iz\theta/s)\hat{\sigma}(z/s)/s$。因为 $s$ 的缩放，唯一一个所有 $g_{\theta,s}$ 的傅里叶变换都可以为零的 $z$ 是 $z=0$，但我们假设 $\int\sigma(t)dt = \hat{g}_{0,1}(0) \neq 0$。根据维纳-陶伯定理[R2]，由函数 $g_{\theta,s}$ 生成的子空间在 $L^1(\mathbb{R})$ 中是稠密的。由于 $F(g_{\theta,s})=0$，我们必须有 $F=0$。同样，这意味着
$$
F(\exp(imt)) = \int_{I_n} \exp(imt)h(t)dt=0
$$
对于所有的 $m$ 成立，所以 $h$ 的傅里叶变换为0。因此 $h$ 本身为0。（注意，尽管指数函数在整个 $\mathbb{R}$ 上不可积，但它在有界区域上是可积的，并且由于 $h$ 在 $I_n$ 上有支撑，这就足够了。）

维纳-陶伯定理的使用导致了一些其他相当奇特的激活函数，它们在 $L^1(I_n)$ 中具有完备性。考虑以下 $n$ 个变量的激活函数：如果 $\mathbf{x}$ 位于 $\mathbb{R}^n$ 中一个边与坐标轴平行的有限固定矩形内，则 $\sigma(\mathbf{x})=1$，否则为0。令 $\mathbf{U}$ 是一个 $n \times n$ 的正交矩阵，$\mathbf{y} \in \mathbb{R}^n$。现在 $\sigma(\mathbf{Ux}+\mathbf{y})$ 是一个任意定向的矩形的指示函数。注意，不允许对矩形进行缩放——只允许在欧几里得空间中的刚体运动！我们有，形如
$$
\sum_{j=1}^N \alpha_j \sigma(\mathbf{U}_j\mathbf{x}+\mathbf{y}_j)
$$
的和在 $L^1(\mathbb{R}^n)$ 中是稠密的。这直接来自于维纳-陶伯定理[R2]的应用，以及 $\sigma$ 的傅里叶变换在一个不包括原点的 $\mathbb{R}^n$ 网格上为零的观察。所有这些网格的可能旋转的交集是空的，所以 $\sigma$ 连同它的旋转和平移，生成了一个在 $L^1(\mathbb{R}^n)$ 中稠密的空间。
这最后一个结果与经典的Pompeiu问题[BST]密切相关，利用[BST]的结果，我们推测上述段落中的矩形可以被任何具有[BST]中定义的角的凸集所取代。

### 5. 总结

我们已经证明了，一个固定的、判别性的单变量函数的有限叠加，可以一致地逼近任何在单位超立方体上有支撑的 $n$ 个实变量的连续函数。在实值神经网络理论中常用的连续S型函数是判别性的。这个结果的组合表明，任何连续函数都可以由一个只有一个内部隐藏层和任意连续S型非线性的连续神经网络来一致地逼近（定理2）。定理3及其随后的讨论精确地说明了，任意的决策函数都可以由一个具有一个内部层和连续S型非线性的神经网络来任意好地逼近。
表1总结了我们所知的各种贡献。

<br>

**表1**

| 函数类型和变换 | 函数空间 | 参考文献 |
| :--- | :--- | :--- |
| $\sigma(\mathbf{y}^T\mathbf{x} + \theta)$, $\sigma$ 连续S型, $\mathbf{y} \in \mathbb{R}^n, \theta \in \mathbb{R}$ | $C(I_n)$ | 本文 |
| $\sigma(\mathbf{y}^T\mathbf{x} + \theta)$, $\sigma$ 单调S型, $\mathbf{y} \in \mathbb{R}^n, \theta \in \mathbb{R}$ | $C(I_n)$ | [F], [HSW] |
| $\sigma(\mathbf{y}^T\mathbf{x} + \theta)$, $\sigma$ S型, $\mathbf{y} \in \mathbb{R}^n, \theta \in \mathbb{R}$ | $C(I_n)$ | [J] |
| $\sigma(\mathbf{y}^T\mathbf{x} + \theta)$, $\sigma \in L^1(\mathbb{R})$, $\int\sigma(t)dt \neq 0, \mathbf{y} \in \mathbb{R}^n, \theta \in \mathbb{R}$ | $L^1(I_n)$ | 本文 |
| $\sigma(\mathbf{y}^T\mathbf{x} + \theta)$, $\sigma$ 连续S型, $\mathbf{y} \in \mathbb{R}^n, \theta \in \mathbb{R}$ | $L^2(I_n)$ | [CD] |
| $\sigma(\mathbf{Ux}+\mathbf{y}), \mathbf{U} \in \mathbb{R}^{n \times n \text{*}}, \mathbf{y} \in \mathbb{R}^n, \sigma$ 是矩形的指示函数 | $L^1(I_n)$ | 本文 |
| $\sigma(t\mathbf{x}+\mathbf{y}), t \in \mathbb{R}, \sigma \in L^1(\mathbb{R}^n), \mathbf{y} \in \mathbb{R}^n, \int\sigma(\mathbf{x})d\mathbf{x} \neq 0$ | $L^1(\mathbb{R}^n)$ | 维纳-陶伯定理 [R2] |

<br>

尽管我们描述的逼近性质非常强大，但我们只关注了**存在性**。尚待回答的重要问题涉及**可行性**，即需要多少项（或等效地，多少神经节点）才能达到给定质量的逼近？被逼近的函数的哪些性质在决定项数方面起作用？在这一点上，我们只能说，我们强烈怀疑绝大多数的逼近问题将需要**天文数字**的项数。这种感觉是基于困扰多维逼近理论和统计学的**维度诅咒**。关于被逼近的函数与适合的逼近所需的项数之间关系的一些最新进展，可以在[MSJ]和[BH], [BEHW], [V]中找到相关问题。鉴于本文结果的简洁性，我们相信这些研究途径值得更多关注。

**致谢.** 作者感谢Brad Dickinson, Christopher Chase, Lee Jones, Todd Quinto, Lee Rubel, John Makhoul, Alex Samarov, Richard Lippmann, 以及匿名审稿人对本文的评论、额外的参考文献和在表述上的改进。

> \* 收稿日期：1988年10月21日。修订日期：1989年2月17日。本研究部分由NSF基金DCR-8619103、ONR合同N000-86-G-0202和DOE基金DE-FG02-85ER25001支持。
>
> † 伊利诺伊大学厄巴纳-香槟分校，超级计算研究与发展中心及电气与计算机工程系，伊利诺伊州厄巴纳，61801，美国。
