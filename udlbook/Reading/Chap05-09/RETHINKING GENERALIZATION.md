理解深度学习需重新审视泛化能力
---

**Chiyuan Zhang**
麻省理工学院
chiyuan@mit.edu

**Benjamin Recht**
加州大学伯克利分校
brecht@berkeley.edu

**Samy Bengio**
Google Brain
bengio@google.com

**Oriol Vinyals**
Google DeepMind
vinyals@google.com

**Moritz Hardt**
Google Brain
mrtz@google.com

### 摘要

尽管深度人工神经网络规模庞大，但那些成功的模型在训练与测试性能上却展现出惊人的微小差异。传统观点将这种小泛化误差归因于模型族本身的特性，或是训练中使用的正则化技术。

通过大量系统的实验，我们证明了这些传统方法无法充分解释为何大型神经网络在实践中泛化良好。具体来说，我们的实验表明，当前最先进的用于图像分类的卷积网络，即使采用随机梯度方法进行训练，也能轻易地拟合训练数据的随机标签。即便使用显式正则化，这种现象也未受到实质性影响；甚至当我们用完全非结构化的随机噪声替换真实图像时，该现象依然存在。我们通过理论构建进一步验证了这些实验发现，证明了简单的两层神经网络，一旦其参数数量超过数据点数量（这在实践中通常如此），就已经具备了完美的有限样本表达能力。

我们将实验发现与传统模型进行对比，并提出了一种新的解释。

### 1 引言

深度人工神经网络的参数数量通常远超其训练样本的数量。然而，部分模型仍能展现出惊人的小泛化误差，即“训练误差”与“测试误差”之间的差异。与此同时，也很容易构建泛化能力差的自然模型架构。那么，到底是什么因素区分了泛化能力好的神经网络和那些不好的呢？如果能对这个问题给出令人满意的答案，不仅有助于提高神经网络的可解释性，也可能为更具原则性、更可靠的模型架构设计提供指导。

为了回答这个问题，统计学习理论提出了一系列不同的复杂性度量，这些度量能够控制泛化误差。其中包括 **VC维** (Vapnik, 1998)、**Rademacher 复杂度** (Bartlett & Mendelson, 2003) 和 **均匀稳定性** (Mukherjee et al., 2002; Bousquet & Elisseeff, 2002; Poggio et al., 2004)。此外，当参数数量很大时，理论表明需要某种形式的正则化来确保小的泛化误差。正则化也可以是隐式的，例如 **早停**（early stopping）就是如此。

#### 1.1 我们的贡献

在这项工作中，我们通过展示传统泛化观点的局限性来对其提出挑战，因为这种观点无法区分泛化性能迥异的神经网络。我们的方法核心是一种源自非参数统计学中广为人知的随机化测试的变体 (Edgington & Onghena, 2007)。

在第一组实验中，我们使用标准架构训练网络，但用随机标签替换了真实数据中的原始标签。我们的核心发现可归纳为：**深度神经网络能轻易拟合随机标签**。更精确地说，当使用完全随机标记的真实数据进行训练时，神经网络的训练误差可达到零。当然，由于训练标签与测试标签之间不存在任何关联，测试误差的表现不会优于随机猜测。换句话说，仅通过随机化标签，我们就可以在不改变模型、其大小、超参数或优化器的情况下，显著增大模型的泛化误差。我们在 CIFAR10 和 ImageNet 分类基准上，用几种不同的标准架构验证了这一事实。

这个观察虽然简单，但从统计学习的角度来看却有着深远的影响：

1.  神经网络的有效容量足以记忆整个数据集。
2.  即使是随机标签上的优化也依然容易。事实上，与使用真实标签训练相比，训练时间仅增加了一个小的常数因子。
3.  随机化标签仅仅是一种数据转换，学习问题的其他所有属性均未改变。

我们在这第一组实验的基础上进行了扩展，用完全随机的像素（例如高斯噪声）替换真实图像，并观察到卷积神经网络依然能够以零训练误差完美拟合数据。这表明，尽管其结构化特性，卷积神经网络仍能拟合随机噪声。

我们进一步改变了随机化的程度，在无噪声和完全噪声之间进行平滑插值。这带来了一系列中间学习问题，其中标签中仍保留一定程度的信号。我们观察到，随着噪声水平的增加，泛化误差稳步恶化。这表明神经网络能够捕捉数据中剩余的信号，同时通过“暴力”方式拟合噪声部分。我们将在下文进一步详细讨论这些观察结果如何排除了 VC 维、Rademacher 复杂度和均匀稳定性等传统理论，作为解释当前最先进神经网络泛化性能的可能原因。

**显式正则化的作用** 如果模型架构本身不足以作为充分的正则化器，那么显式正则化能有多大帮助呢？我们证明，诸如 **权重衰减**（weight decay）、**Dropout** 和 **数据增强**（data augmentation）等显式正则化形式，并不能充分解释神经网络的泛化误差。换句话说：**显式正则化可以改善泛化性能，但既不是控制泛化误差的必要条件，也非其本身所能完全胜任**。

与传统的凸经验风险最小化（其中显式正则化是排除平凡解所必需的）不同，我们发现正则化在深度学习中扮演着截然不同的角色。它更像是一个调优参数，通常有助于改善模型的最终测试误差，但其缺失并不必然意味着泛化误差变差。正如 Krizhevsky 等人 (2012) 报告的，$\ell_2$ 正则化（权重衰减）有时甚至有助于优化，这说明了其在深度学习中的性质尚未被充分理解。

**有限样本表达能力** 我们通过一个理论构建来补充我们的实验观察，表明在通常情况下，大型神经网络可以表达训练数据的任何标签。更正式地说，我们展示了一个非常简单的两层 ReLU 网络，其参数量为 $p=2n+d$，可以表达任意 $d$ 维、大小为 $n$ 的样本的任何标签。Livni 等人 (2014) 先前的一项构建获得了类似的结果，但参数量远多，为 $O(dn)$。虽然我们的两层网络不可避免地具有大宽度，但我们也可以构建一个深度为 $k$ 的网络，其中每一层仅有 $O(n/k)$ 个参数。

虽然先前的表达能力研究主要关注神经网络在整个域上能够表示哪些函数，但我们则专注于神经网络对于一个有限样本的表达能力。与函数空间中现有的深度分离 (Delalleau & Bengio, 2011; Eldan & Shamir, 2016; Telgarsky, 2016; Cohen & Shashua, 2016) 相比，我们的结果表明，即使是线性大小的两层网络也已能表示训练数据的任何标签。

**隐式正则化的作用** 虽然像 Dropout 和权重衰减这样的显式正则化器对于泛化可能并非必不可少，但并非所有能很好拟合训练数据的模型都能很好地泛化，这当然是事实。事实上，在神经网络中，我们几乎总是将通过运行随机梯度下降（SGD）得到的输出作为我们的模型。我们借助于线性模型，分析了 SGD 如何充当一种隐式正则化器。对于线性模型，SGD 总是收敛到一个具有小范数的解。因此，算法本身就隐式地对解进行了正则化。实际上，我们在小数据集上证明，即使是高斯核方法在没有正则化的情况下也能很好地泛化。尽管这不能解释为什么某些架构比其他架构泛化得更好，但它确实表明需要进行更多研究，以确切了解通过 SGD 训练得到的模型究竟继承了哪些特性。

#### 1.2 相关工作

Hardt 等人 (2016) 根据梯度下降所用的步数，给出了一个用随机梯度下降训练的模型的泛化误差上限。他们的分析通过了 **均匀稳定性** (Bousquet & Elisseeff, 2002) 的概念。正如我们在这项工作中指出的，一个学习算法的均匀稳定性与训练数据的标签无关。因此，这个概念不足以区分在真实标签上训练的模型（小泛化误差）和在随机标签上训练的模型（大泛化误差）。这也凸显了为什么 Hardt 等人 (2016) 对非凸优化的分析相当悲观，只允许对数据进行非常少的遍历。我们的结果表明，即使在经验上，神经网络的训练在多次遍历数据时也并不均匀稳定。因此，需要一种更弱的稳定性概念才能沿着这个方向取得进一步进展。

关于神经网络的表达能力已有很多研究，从多层感知器的通用逼近定理开始 (Cybenko, 1989; Mhaskar, 1993; Delalleau & Bengio, 2011; Mhaskar & Poggio, 2016; Eldan & Shamir, 2016; Telgarsky, 2016; Cohen & Shashua, 2016)。所有这些结果都处于“总体层面”，描述了某些类别的神经网络在整个域上可以表达哪些数学函数。而我们则研究了神经网络对于大小为 $n$ 的有限样本的表达能力。这导出了一个非常简单的证明，表明即使是 $O(n)$ 大小的两层感知器也具有通用的有限样本表达能力。

Bartlett (1998) 证明了具有 Sigmoid 激活函数的多层感知器的 **“胖碎维数”**（fat shattering dimension）的界限，该界限以每个节点权重的 $\ell_1$ 范数表示。这项重要成果给出了一个与网络大小无关的神经网络泛化界。然而，对于 ReLU 网络，$\ell_1$ 范数不再具有信息量。这引出了一个问题：是否存在一种不同的容量控制形式，能够对大型神经网络的泛化误差进行约束？Neyshabur 等人 (2014) 在一项发人深省的工作中提出了这个问题，他们通过实验论证了网络大小并非神经网络容量控制的主要形式。与矩阵分解的类比，说明了隐式正则化的重要性。

### 2 神经网络的有效容量

我们的目标是理解前馈神经网络的有效模型容量。为此，我们采用了一种受非参数随机化测试启发的方法论。具体来说，我们选取一个候选架构，并分别在真实数据和一份将真实标签替换为随机标签的数据副本上对其进行训练。在后一种情况下，实例与类别标签之间不再存在任何关系。因此，学习是不可能的。直觉上，这种不可能性应在训练过程中明确显现出来，例如训练不收敛或显著减慢。然而令我们惊讶的是，对于多种标准架构，训练过程的几个属性几乎未受这种标签转换的影响。这提出了一个概念性的挑战：我们最初期望的小泛化误差的任何理由，对随机标签的情况都不再适用。

---

图1：在 CIFAR10 数据集上拟合随机标签和随机像素。
（a）显示了不同实验设置下训练损失随训练步骤的衰减情况。
（b）显示了不同标签损坏率下的相对收敛时间。
（c）显示了在不同标签损坏率下，收敛后的测试误差（由于训练误差为0，这与泛化误差相同）。

为了进一步深入了解这一现象，我们实验了不同程度的随机化，探索从无标签噪声到完全损坏标签的连续变化。我们还尝试了对输入（而非标签）进行不同的随机化，得出了相同的普遍结论。实验在两个图像分类数据集上进行：CIFAR10 (Krizhevsky & Hinton, 2009) 和 ImageNet (Russakovsky et al., 2015) ILSVRC 2012 数据集。我们在 ImageNet 上测试了 Inception V3 (Szegedy et al., 2016) 架构，并在 CIFAR10 上测试了较小版本的 Inception、Alexnet (Krizhevsky et al., 2012) 和多层感知器（MLP）。关于实验设置的更多细节，请参阅附录 A 节。

#### 2.1 拟合随机标签和像素

我们对标签和输入图像进行以下修改来运行实验：

-   **真实标签**：未经修改的原始数据集。
-   **部分损坏标签**：以概率 $p$ 独立地将每张图像的标签损坏为一个均匀随机的类别。
-   **随机标签**：所有标签均被随机替换。
-   **像素洗牌**：选择一个随机的像素排列，然后将此相同的排列应用于训练集和测试集中的所有图像。
-   **随机像素**：对每张图像独立地应用不同的随机排列。
-   **高斯噪声**：使用高斯分布（均值和方差与原始图像数据集匹配）为每张图像生成随机像素。

令人惊讶的是，即使随机标签完全破坏了图像与标签之间的关系，随机梯度下降在超参数设置不变的情况下，仍能完美地优化权重以拟合随机标签 [cite: 1018, 1019]。我们通过打乱图像像素，甚至从高斯分布中完全重新采样随机像素，进一步破坏了图像的结构，但我们测试的网络仍然能够进行拟合。

图 1a 展示了 Inception 模型在 CIFAR10 数据集上各种设置下的学习曲线。我们预期在随机标签上，目标函数需要更长时间才能开始下降，因为最初每个训练样本的标签分配都是不相关的。因此，大的预测误差通过反向传播产生大的梯度，用于参数更新。然而，由于随机标签是固定的且在不同 Epoch 中保持一致，网络在多次遍历训练集后便开始拟合。我们发现拟合随机标签有以下几个有趣的观察结果：a) 我们无需更改学习率调度；b) 一旦开始拟合，它会迅速收敛；c) 它收敛到完美（过）拟合训练集 [cite: 1025, 1026]。此外，“随机像素”和“高斯噪声”比“随机标签”更快开始收敛。这可能是因为在随机像素的情况下，输入彼此之间比原本属于同一类别的自然图像更分散，因此更容易构建一个网络来拟合任意的标签分配。

在 CIFAR10 数据集上，Alexnet 和 MLP 均能收敛到训练集上的零损失。表 1 中的阴影行显示了具体的数值和实验设置。我们还在 ImageNet 数据集上测试了随机标签。如附录表 2 的最后三行所示，尽管未能达到完美的 100% Top-1 准确率，但对于来自 1000 个类别的百万个随机标签来说，95.20% 的准确率仍然非常令人惊讶。请注意，我们从真实标签切换到随机标签时，并未进行任何超参数调优。很可能通过一些超参数修改，可以在随机标签上实现完美的准确率。即使开启了显式正则化器，该网络仍能达到约 90% 的 Top-1 准确率。

**部分损坏标签** 我们进一步研究了神经网络在 CIFAR10 数据集上，从 0（无损坏）到 1（完全随机标签）的标签损坏水平变化下的训练行为。网络在所有情况下都能完美拟合损坏的训练集。图 1b 显示了随着标签噪声水平的增加，收敛时间变慢。图 1c 描绘了收敛后的测试误差。由于训练误差始终为零，测试误差与泛化误差相同。当噪声水平接近 1 时，泛化误差收敛到 CIFAR10 上随机猜测性能的 90%。

#### 2.2 影响

鉴于我们的随机化实验结果，我们讨论了这些发现如何对几种传统的泛化推理方法提出挑战。

**Rademacher 复杂度和 VC 维** Rademacher 复杂度是一种常用的、灵活的假设空间复杂性度量。假设空间 $\mathcal{H}$ 在数据集 $\{z_1, \dots, z_n\}$ 上的经验 Rademacher 复杂度定义为：

$$\hat{\mathfrak{R}}_n(\mathcal{H}) = \mathbb{E}_\sigma \left[ \sup_{h \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^{n} \sigma_i h(x_i) \right] \quad (1)$$

其中 $\sigma_1, \dots, \sigma_n \in \{\pm1\}$ 是独立同分布的均匀随机变量 [cite: 1047, 1049]。这个定义与我们的随机化测试非常相似。具体来说，$\hat{\mathfrak{R}}_n(\mathcal{H})$ 衡量了 $\mathcal{H}$ 拟合随机 $\pm1$ 二元标签分配的能力。虽然我们考虑的是多类问题，但很容易将其转换为相应的二元分类问题，并得出相同的实验观察结果。由于我们的随机化测试表明许多神经网络可以完美拟合带有随机标签的训练集，我们预期相应的模型类别 $\mathcal{H}$ 的 $\hat{\mathfrak{R}}_n(\mathcal{H}) \approx 1$。这当然是一个平凡的 Rademacher 复杂度上限，在实际环境中并不能得出有用的泛化界限。

类似的推理也适用于 VC 维及其连续模拟 **胖碎维数**（fat-shattering dimension），除非我们进一步限制网络。尽管 Bartlett (1998) 证明了基于网络权重 $\ell_1$ 范数的胖碎维数界限，但此界限不适用于我们在此考虑的 ReLU 网络。该结果后来被 Neyshabur 等人 (2015) 推广到其他范数，但即使这些结果似乎也无法解释我们观察到的泛化行为。

**均匀稳定性** 抛开假设空间的复杂性度量不谈，我们可以转而考虑用于训练的算法的属性。这通常使用某种稳定性概念来完成，例如 **均匀稳定性** (Bousquet & Elisseeff, 2002)。算法 $\mathcal{A}$ 的均匀稳定性衡量了算法对替换单个样本的敏感程度。然而，它仅仅是算法的一个属性，没有考虑到数据的具体细节或标签的分布。可以定义更弱的稳定性概念 (Mukherjee et al., 2002; Poggio et al., 2004; Shalev-Shwartz et al., 2010)。最弱的稳定性度量直接等价于限制泛化误差，并且确实考虑了数据。然而，要有效利用这个更弱的稳定性概念一直很困难。

### 3 正则化的作用

我们的大多数随机化测试都是在关闭显式正则化的情况下进行的。在参数数量多于数据点的情况下，正则化器是理论和实践中减轻过拟合的标准工具 (Vapnik, 1998) [cite: 1066, 1072]。基本思想是，尽管原始假设空间过于庞大，无法很好地泛化，但正则化器有助于将学习限制在具有可控复杂度的假设空间的子集内。通过添加显式正则化器（例如，通过惩罚最优解的范数），可能解的有效 Rademacher 复杂度会显著降低。

我们将看到，在深度学习中，显式正则化似乎扮演着一个截然不同的角色。正如附录表 2 的底部几行所示，即使有 Dropout 和权重衰减，Inception V3 仍然能够非常好地（甚至完美地）拟合随机训练集。虽然未明确显示，但在 CIFAR10 上，Inception 和 MLP 即使开启了权重衰减，也仍能完美地拟合随机训练集。然而，开启权重衰减的 AlexNet 未能收敛在随机标签上。为了探究正则化在深度学习中的作用，我们明确比较了有无正则化器的情况下深度网络的学习行为。

我们没有对深度学习中引入的所有正则化技术进行全面调查，而是简单地选取了几种常用的网络架构，并比较了在关闭其所配备的正则化器时的表现。涵盖的正则化器如下：

-   **数据增强**：通过特定领域的转换来扩充训练集。对于图像数据，常用的转换包括随机裁剪、亮度、饱和度、色相和对比度的随机扰动 [cite: 1081, 1082]。
-   **权重衰减**：等同于对权重施加 $\ell_2$ 正则化器。也等同于将权重严格约束在一个欧几里得球内，其半径由权重衰减量决定。
-   **Dropout** (Srivastava et al., 2014)：以给定的 Dropout 概率随机遮挡层输出的每个元素。在我们的实验中，只有用于 ImageNet 的 Inception V3 使用了 Dropout。

表 1 展示了在 CIFAR10 上，切换数据增强和权重衰减使用情况的 Inception、Alexnet 和 MLP 的结果。这两种正则化技术都有助于提高泛化性能，但即使关闭了所有正则化器，所有模型仍然泛化得非常好 [cite: 1087, 1123]。附录表 2 显示了在 ImageNet 数据集上类似的实验。当我们关闭所有正则化器时，Top-1 准确率下降了 18%。具体来说，无正则化时的 Top-1 准确率为 59.80%，而 ImageNet 上随机猜测的 Top-1 准确率仅为 0.1%。更令人震惊的是，开启数据增强但关闭其他显式正则化器后，Inception 能够达到 72.95% 的 Top-1 准确率。这确实表明，利用已知对称性来增强数据的能力，似乎比仅仅调整权重衰减或防止低训练误差更为强大。

在没有正则化的情况下，Inception 达到了 80.38% 的 Top-5 准确率，而 ILSVRC 2012 冠军 (Krizhevsky et al., 2012) 报告的数字为 83.6%。因此，虽然正则化很重要，但通过简单地改变模型架构可以获得更大的收益。很难说正则化是深度网络泛化能力发生根本性“相变”（phase change）的原因。

#### 3.1 隐式正则化

早停被证明在某些凸学习问题上具有隐式正则化作用 (Yao et al., 2007; Lin et al., 2016)。在附录表 2 中，我们用括号显示了训练过程中的最佳测试准确率。这证实了早停可能可以提高泛化性能。图 2a 展示了 ImageNet 上的训练和测试准确率。阴影区域表示累积的最佳测试准确率，可作为早停潜在性能增益的参考。然而，在 CIFAR10 数据集上，我们没有观察到早停的任何潜在好处。

**批量归一化**（Batch Normalization）(Ioffe & Szegedy, 2015) 是一种在每个 Mini-batch 内对层响应进行归一化的操作。它已被广泛应用于许多现代神经网络架构，如 Inception (Szegedy et al., 2016) 和残差网络 (He et al., 2016)。尽管并非明确设计用于正则化，但批量归一化通常被发现能改善泛化性能。

Inception 架构使用了大量的批量归一化层。为了测试批量归一化的影响，我们创建了一个“Inception w/o BatchNorm”架构，它与图 3 中的 Inception 完全相同，只是移除了所有批量归一化层。图 2b 比较了在关闭所有显式正则化器的情况下，Inception 的两个变体在 CIFAR10 上的学习曲线。该归一化操作有助于稳定学习动态，但对泛化性能的影响仅为 3-4%。具体的准确率也列在表 1 的“Inception w/o BatchNorm”部分。

总而言之，我们对显式和隐式正则化的观察结果都一致表明，如果调优得当，正则化器可以帮助改善泛化性能。然而，正则化器不太可能是泛化的根本原因，因为即使移除所有正则化器，网络仍然表现良好。

---
图2：隐式正则化器对泛化性能的影响。`aug` 表示数据增强，`wd` 表示权重衰减。`BN` 是批量归一化。阴影区域表示累积的最佳测试准确率，作为早停潜在性能增益的指标。
（a）当没有其他正则化器时，早停可能潜在地提高泛化能力。
（b）早停在 CIFAR10 上不一定有帮助，但批量归一化稳定了训练过程并改善了泛化能力。

### 4 有限样本表达能力

人们在表征神经网络的表达能力方面投入了大量精力，例如 Cybenko (1989)、Mhaskar (1993)、Delalleau & Bengio (2011)、Mhaskar & Poggio (2016)、Eldan & Shamir (2016)、Telgarsky (2016)、Cohen & Shashua (2016) 等人的工作 [cite: 1154, 1155]。这些结果几乎都处于“总体层面”，旨在表征特定类别的神经网络在整个域上能够表示哪些函数。例如，众所周知，在总体层面上，深度为 $k$ 的网络通常比深度为 $k-1$ 的网络更强大。我们认为，在实践中更相关的是神经网络对大小为 $n$ 的有限样本的表达能力。

有可能使用一致收敛定理将总体层面的结果转移到有限样本结果。然而，这样的一致收敛界限要求样本数量与输入维度呈多项式关系，与网络深度呈指数关系，这在实践中显然是不现实的要求。我们转而直接分析神经网络的有限样本表达能力，并注意到这极大地简化了问题。具体来说，只要网络的参数数量 $p$ 大于 $n$，即使是简单的两层神经网络也可以表示输入样本的任何函数。

我们说一个神经网络 $C$ 可以表示大小为 $n$ 的样本的任何函数，如果对于每个大小为 $|S|=n$ 且 $S \subseteq \mathbb{R}^d$ 的样本，以及每个函数 $f: S \to \mathbb{R}$，都存在一组 $C$ 的权重设置，使得对于每个 $x \in S$，都有 $C(x) = f(x)$ [cite: 1163, 1164]。

**定理 1**
存在一个具有 ReLU 激活函数和 $2n+d$ 个权重的两层神经网络，可以表示在 $d$ 维空间中，大小为 $n$ 的样本上的任何函数。

证明在附录 C 节中给出，其中我们还讨论了如何通过深度 $k$ 实现宽度 $O(n/k)$。我们指出，对我们构造中的系数向量的权重给出界限是一个简单的练习。引理 1 给出了矩阵 $\mathcal{A}$ 的最小特征值的界限。这可以用来对解向量 $w$ 的权重给出合理的界限。

### 5 隐式正则化：线性模型的启示

尽管深度神经网络因诸多原因依然神秘，但我们在此节中指出，即使是线性模型，其泛化来源也未必容易理解。事实上，借鉴简单的线性模型来探寻是否有相似的见解，可以帮助我们更好地理解神经网络，这是很有益的。

假设我们收集了 $n$ 个不同的数据点 $\{ (x_i, y_i) \}$，其中 $x_i$ 是 $d$ 维特征向量，$y_i$ 是标签。设 $loss$ 是一个非负损失函数，且 $loss(y, y) = 0$，考虑经验风险最小化（ERM）问题：

$$\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^{n} loss(w^T x_i, y_i) \quad (2)$$

如果 $d \ge n$，那么我们可以拟合任何标签。但是，在模型类别如此丰富且没有显式正则化的情况下，是否可能泛化呢？

设 $X$ 表示 $n \times d$ 的数据矩阵，其第 $i$ 行是 $x_i^T$。如果 $X$ 的秩为 $n$，那么方程组 $Xw = y$ 有无数个解，无论右侧如何。我们可以通过简单地求解这个线性系统来找到 ERM 问题 (2) 的一个全局最小值。但是，所有全局最小值都同样好地泛化吗？有没有办法确定何时一个全局最小值会泛化而另一个不会？

一种理解最小值质量的流行方法是观察解处损失函数的曲率。但在线性情况下，所有最优解的曲率都是相同的 (Choromanska et al., 2015)。要理解这一点，请注意当 $y_i$ 是一个标量时：

$$\nabla^2 \frac{1}{n} \sum_{i=1}^{n} loss(w^T x_i, y_i) = \frac{1}{n} X^T \text{diag}(\beta) X$$

其中 $\beta_i = \frac{\partial^2 loss(z, y)}{\partial z^2} \big|_{z=y}$。当 $y$ 是向量值时，也可以找到类似的公式。特别地，Hessian 矩阵不是 $w$ 选择的函数。此外，Hessian 在所有全局最优解处都是退化的。

如果曲率无法区分全局最小值，那么什么可以呢？一个有希望的方向是考察主要工作算法——随机梯度下降（SGD），并检查 SGD 最终收敛到哪个解。由于 SGD 更新的形式为 $w_{t+1} = w_t - \eta_t e_t x_{i_t}$，其中 $\eta_t$ 是步长，$e_t$ 是预测误差。如果 $w_0 = 0$，那么解必须具有形式 $w = \sum_{i=1}^{n} \alpha_i x_i$。因此，如果我们运行 SGD，我们得到 $w$ 位于数据点的张成空间中。如果我们也完美地插值了标签，那么我们有 $Xw = y$。将这两个恒等式结合起来，就简化为单个方程：

$$XX^T \alpha = y \quad (3)$$

它有一个唯一的解。注意，这个方程只取决于数据点 $x_i$ 之间的点积。因此，我们以一种迂回的方式推导出了“核技巧”（kernel trick）(Schölkopf et al., 2001)。因此，我们可以通过对数据形成 Gram 矩阵（即核矩阵）$K = XX^T$ 并求解线性系统 $K\alpha = y$ 来完美地拟合任何一组标签，从而得到 $\alpha$。这是一个 $n \times n$ 的线性系统，当 $n$ 小于十万时，可以在标准工作站上求解，例如 CIFAR10 和 MNIST 等小基准数据集。

令人惊讶的是，对于凸模型，精确拟合训练标签会产生出色的性能。在没有预处理的 MNIST 数据集上，我们只需求解 (3) 就能达到 1.2% 的测试误差。需要注意的是，这并不完全简单，因为核矩阵需要 30GB 的内存来存储。尽管如此，该系统可以在一个具有 24 个核心和 256 GB RAM 的商用工作站上，使用常规 LAPACK 调用在不到 3 分钟内求解。通过首先对数据应用 Gabor 小波变换，然后求解 (3)，MNIST 上的误差降至 0.6%。令人惊讶的是，添加正则化并没有改善任何一个模型的性能！CIFAR10 也出现了类似的结果。简单地在像素上应用高斯核而不使用正则化，即可达到 46% 的测试误差。通过使用包含 32,000 个随机滤波器的随机卷积神经网络进行预处理，测试误差降至 17%。添加 $\ell_2$ 正则化进一步将这个数字降低到 15% 的误差。请注意，这没有任何数据增强。

值得注意的是，这种核解在隐式正则化方面具有吸引人的解释。简单的代数运算表明，它等同于 $Xw=y$ 的最小 $\ell_2$ 范数解。也就是说，在所有精确拟合数据的模型中，SGD 通常会收敛到范数最小的解。很容易构造出不泛化的 $Xw=y$ 解：例如，可以对数据拟合一个高斯核，并将中心放置在随机点。另一个简单的例子是强制数据拟合测试数据上的随机标签。在这两种情况下，解的范数都明显大于最小范数解。不幸的是，这种最小范数的概念并不能预测泛化性能。例如，回到 MNIST 的例子，没有预处理的最小范数解的 $\ell_2$ 范数约为 220。而经过小波预处理后，范数跃升至 390。然而测试误差却...跃升至390，而测试误差却从1.2%降至0.6%。这说明，仅凭最小范数的概念并不能直接用于理解泛化。

### 6 总结与讨论

我们的实验结果表明，尽管深度神经网络具有极高的有效容量，足以拟合完全随机的训练数据，但当使用随机梯度下降（SGD）进行训练时，它们在真实数据上仍然能够很好地泛化。显式正则化，如权重衰减和 Dropout，在某些情况下可以改善泛化能力，但并非泛化的根本原因。

这些观察挑战了传统的泛化理论，因为这些理论未能区分在真实数据上训练（泛化良好）和在随机标签上训练（泛化能力差）的神经网络。此外，传统的复杂性度量，如 Rademacher 复杂度，对这两种情况的预测是相同的。这表明，我们需要一种新的、更适合深度学习的泛化理论。

我们提出，新的理论需要考虑以下几个方面：

-   **学习动态**：算法本身，特别是随机梯度下降（SGD）在非凸优化中的行为，在决定最终解的质量方面起着关键作用。正如我们在第5节中所示，SGD 对于线性模型是一种隐式正则化器，倾向于收敛到具有小范数的解。
-   **输入数据**：网络的泛化能力似乎对输入数据的结构敏感，这表明需要一个依赖于数据的复杂性度量。
-   **架构**：某些架构（如卷积网络）似乎比其他架构更倾向于泛化，即使它们在容量上足够庞大，可以拟合随机数据。

我们希望这项工作能激励研究人员在这些方向上进行更多探索，从而更好地理解深度学习中的泛化能力。

### 参考文献

-   Bartlett, P. L. (1998). The sample complexity of neural network learning with sigmoidal units. IEEE Transactions on Information Theory, 44(2), 525–530.
-   Bartlett, P. L., & Mendelson, A. (2003). Rademacher and gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research, 3, 463–482.
-   Bousquet, O., & Elisseeff, A. (2002). Stability of learning algorithms. Journal of Machine Learning Research, 2, 499–532.
-   Choromanska, A., Henaff, M., Mathieu, M., Arous, G. B., & LeCun, Y. (2015). The loss surfaces of multilayer networks. Journal of Machine Learning Research, 16(1), 2215–2257.
-   Cohen, N., & Shashua, A. (2016). Deep learning and the capacity to generalize. ICLR Workshop Track.
-   Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems, 2(4), 303–314.
-   Delalleau, O., & Bengio, Y. (2011). Shallow vs. deep networks. Advances in Neural Information Processing Systems, 24, 690–698.
-   Edgington, E. S., & Onghena, P. (2007). Randomization Tests. Chapman & Hall/CRC.
-   Eldan, R., & Shamir, O. (2016). The power of depth for feedforward neural networks. arXiv:1512.06239.
-   Hardt, M., Recht, B., & Singer, Y. (2016). Train faster, generalize better: Stability of stochastic gradient descent. arXiv:1512.06239.
-   He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770–778).
-   Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv:1502.03167.
-   Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, 25, 1097–1105.
-   Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical Report, University of Toronto.
-   Lin, H., Mairal, J., & Harchaoui, Z. (2016). A universal convergence result for gradient methods on non-convex objectives with applications to deep learning. arXiv:1603.02925.
-   Livni, R., Shalev-Shwartz, S., & Shamir, O. (2014). On the finite sample expressive power of deep neural networks. In Proceedings of The 31st International Conference on Machine Learning (pp. 1106–1114).
-   Mhaskar, H. N. (1993). Approximation of functions by a neural network with one hidden layer. Advances in Applied Mathematics, 14(2), 145–171.
-   Mhaskar, H., & Poggio, T. (2016). Deep versus shallow networks: An approximation theory perspective. In Proceedings of The 29th Annual Conference on Learning Theory (pp. 950–971).
-   Mukherjee, S., Rifkin, R., Hardt, M., & Poggio, T. (2002). The role of stability in learning theory. Journal of Machine Learning Research, 2, 85–104.
-   Neyshabur, B., Tomioka, R., & Srebro, N. (2014). The role of overparametrization in deep learning. In Advances in Neural Information Processing Systems, 27, 2404–2412.
-   Neyshabur, B., Tomioka, R., & Srebro, N. (2015). Norm-based capacity control in neural networks. In Advances in Neural Information Processing Systems, 28, 597–605.
-   Poggio, T., Rifkin, R., Mukherjee, S., & Niyogi, P. (2004). Generalization bounds for regularized least squares. MIT-CSAIL-TR-2004-032.
-   Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Berg, A. C. (2015). Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3), 211–252.
-   Schölkopf, B., Smola, A. J., & Bach, F. (2001). Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond. MIT Press.
-   Shalev-Shwartz, S., & Ben-David, S. (2010). Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press.
-   Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1), 1929–1958.
-   Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2818–2826).
-   Telgarsky, M. (2016). The expressive power of deep neural networks. arXiv:1606.00282.
-   Vapnik, V. N. (1998). Statistical Learning Theory. Wiley-Interscience.
-   Yao, Y., Rosasco, L., & Caponnetto, A. (2007). On the stability of early stopping in kernel methods. In Proceedings of the 20th Annual Conference on Neural Information Processing Systems (pp. 1593–1600).

---

### 附录 A

#### 实验设置

所有实验均使用 TensorFlow 框架进行。在所有实验中，我们都使用了 Adam 优化器 (Kingma & Ba, 2014)，并采用了一组经过仔细调整的默认超参数，且在随机和真实标签的训练过程中保持一致。我们使用 Adam 优化器的默认设置：学习率 $\alpha=0.001$，$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$。

**Inception** (CIFAR10) 我们使用了 Inception V3 架构，但移除了全连接层和辅助分类器，并根据 CIFAR10 的输入和输出尺寸对其进行了调整。具体的网络结构见图3。

**AlexNet** (CIFAR10) 我们使用了 Krizhevsky等人（2012）提出的原始 AlexNet 架构的简化版本，其结构见图4。这个简化版本将特征图的数量减半，并去除了局部响应归一化层。

**MLP** (CIFAR10) 我们测试了两种不同的多层感知器（MLP）架构：
1.  **MLP 3x512**：一个具有三个512个单元的隐藏层的MLP。
2.  **MLP 1x512**：一个具有一个512个单元的隐藏层的MLP。
所有MLP隐藏层都使用ReLU激活函数。

**Inception V3** (ImageNet) 我们使用了与Szegedy等人（2016）描述的相同的标准Inception V3架构，包括所有全连接层和辅助分类器。我们使用了Adam优化器，学习率设为0.001，并运行了150,000次迭代，批量大小为64。

图 3：用于CIFAR10分类的Inception架构。
图 4：用于CIFAR10分类的简化版AlexNet架构。

#### 附录 B

表 2：ImageNet 上的结果。`aug` 表示数据增强，`wd` 表示权重衰减。括号内的数值表示训练过程中的最佳测试准确率，这代表了早停的潜在性能。

表 3：ImageNet 上拟合随机标签的结果。`wd` 是权重衰减。

#### 附录 C

#### 定理 1 的证明

我们说一个网络可以表示一个有限样本上的任何函数，如果对于每个大小为 $n$ 的样本 $S \subset \mathbb{R}^d$，以及每个函数 $f: S \to \mathbb{R}$，都存在一组网络的权重设置，使得对于每个 $x \in S$，都有 $C(x)=f(x)$。
设 $x_1, \dots, x_n$ 是样本中的 $n$ 个点。我们想找到一个网络 $C$ 和一组权重，使得 $C(x_i)=y_i$ 对于任意给定的 $y_1, \dots, y_n$ 都成立。

考虑一个具有 $2n+d$ 个权重的两层网络 $C(x) = \sum_{j=1}^{n} a_j \text{ReLU}(w_j^T x + b_j)$，其中 $w_j \in \mathbb{R}^d, a_j, b_j \in \mathbb{R}$，并且我们固定 $b_j$。我们希望找到权重 $a_j, w_j$ 使得 $C(x_i) = y_i$。

选择 $w_j$ 使得对于 $i=1,\dots,n$，我们有 $w_j^T x_i \ge 0$ 当 $i \ge j$ 且 $w_j^T x_i < 0$ 当 $i < j$。
设 $x_j$ 的范数是1。选择 $w_j = \alpha x_j - \beta_j \sum_{k=1,k\ne j}^n x_k$，其中 $\alpha, \beta_j$ 是常数。
这可以通过选择足够大的 $\alpha$ 和适当的 $\beta_j$ 来实现。

如果 $d \ge n$，我们可以通过解决线性系统 $Xw=y$ 找到一个解。

---
本文件未包含习题。

### 附录 D

#### 关于拟合随机标签的更多实验

我们补充了一些实验，以探究显式正则化器如何影响拟合随机标签的能力。

表 4：使用权重衰减和数据增强在 CIFAR10 数据集上拟合随机标签的结果。

| 模型 | 正则化器 | 训练准确率 |
| :--- | :--- | :--- |
| Inception | 权重衰减 | 100% |
| Alexnet | | 未能收敛 |
| MLP 3x512 | | 100% |
| MLP 1x512 | | 99.21% |
| Inception | 随机裁剪$^1$ | 99.93% |
| Inception | 数据增强$^2$ | 99.28% |

根据表4，我们可以看到，在使用每个模型的默认权重衰减系数时，除了Alexnet之外，所有其他模型仍然能够拟合随机标签。我们还测试了在Inception架构上使用随机裁剪和数据增强。通过将默认的权重衰减因子从0.95更改为0.999，并运行更多的Epoch，我们在这两种情况下都观察到了对随机标签的过拟合。这可能需要更长的时间来收敛，因为数据增强会使训练集的大小爆炸性增长（尽管许多样本不再是独立同分布的）。

$^1$ 在随机裁剪和数据增强中，每个Epoch都使用一张新的随机修改过的图像，但（随机分配的）标签在所有Epoch中保持一致。“训练准确率”在这里意味着略有不同的东西，因为每个Epoch的训练集都不同。此处报告的是每个mini-batch上在增强样本上的在线准确率的全局平均值。
$^2$ 数据增强包括随机左右翻转和高达25度的随机旋转。
