# Seconds-Rule Tiered ANN：面向混合 DRAM+SSD 向量检索的秒级热集分层策略评测

Lucas Ding



## 0. 摘要

### 0.1 问题背景

在语义搜索、检索增强生成等现代人工智能应用中，系统往往要在一个规模巨大的向量集合中，根据用户查询向量，找出若干个距离最近的向量。向量检索，或近似最近邻检索，需要同时兼顾两个基本目标：一是检索质量高，也就是召回率要足够好；二是响应足够快，尤其是尾部请求的延迟要可控。

在小规模场景下，向量索引可以完整放入 DRAM，延迟和吞吐往往不是瓶颈。但在实际的线上系统中，向量数量容易达到亿级甚至更高，索引会远远超过单机 DRAM 容量。这时，系统通常采用分层存储：把一部分索引放在访问速度很快但价格昂贵的 DRAM，把剩余部分放在容量大但访问延迟高的 SSD。这样一来，如何利用有限的 DRAM 容量就成了核心系统问题之一。

已有系统中常见的做法可以粗略分为几类。最简单的一类是“静态切分”，在离线阶段预先指定哪些索引分片常驻 DRAM，哪些分片放在 SSD。这种方法实现简单，但难以适应请求模式随时间变化的情况。另一类是直接套用传统缓存策略，例如基于最近访问时间的 LRU，或基于访问频率的 LFU，把索引条目视作普通缓存对象，谁“热”谁进 DRAM。还有一些系统采用窗口化频率统计，试图兼顾“近期热点”和“长期高频”之间的平衡。

这些做法在工程实践中可以在一定程度上缓解问题，但在大规模向量检索场景下仍然存在几方面不足。首先，传统缓存策略往往以单个键值对象为粒度，而向量索引中的访问模式更复杂，一个查询会在短时间内访问一组高度相关的数据块，单纯按条目计数很难捕捉这种“团块式”的热点结构。其次，向量检索的查询分布通常具有明显的时间突发性，一小段时间内会出现非常集中的热点，随后又迅速冷却，过于粗糙的时间尺度会让策略反应滞后。最后，复杂的 ANN 算法本身就有多层结构，分层策略如果与索引结构脱节，容易导致额外的 I/O 放大和迁移开销，拉高尾延迟。

综上，需要一种更适配向量检索特点的分层策略，既能在秒级时间尺度上迅速识别真正的热点集合，又能控制 DRAM 和 SSD 之间的数据迁移成本，在有限的 DRAM 容量下尽可能提升召回率并压低尾延迟。Seconds-Rule Tiered ANN 正是在这一背景下提出，重点关注“秒级热集”的识别与利用。



### 0.2 方法概述

围绕上述问题，本工作搭建了一套可复现的评测环境，整体方法由三个核心部分组成：端到端实验流水线、混合存储模拟与评测框架、以及一组对比策略和评价指标。

首先，构建了一条端到端的实验流水线，用于从配置到结果自动化地完成一次完整实验。流水线从一个统一的配置入口开始，配置中包含数据集路径、索引类型、工作负载轨迹、候选分层策略列表、DRAM 容量扫描范围以及基础系统参数。流水线依次完成以下步骤：准备或加载向量索引，初始化模拟的 DRAM 和 SSD 视图，按配置重放查询请求序列，在每一次查询过程中记录访问路径和延迟，以及策略做出的搬迁决策；实验完成后，对单次运行的原始记录进行整理，输出结构化的 CSV 文件，再在此基础上进行聚合统计和可视化，生成用于分析的图表。整个过程中，重要的环境信息和每一阶段的关键信息都会写入日志，以便他人根据同一份 Report Bundle 和软件版本重现实验。

其次，构建了一个面向混合 DRAM+SSD 配置的分层 ANN 模拟与评测框架。框架抽象出两级存储：上层 DRAM 表示容量有限但访问延迟极低的区域，下层 SSD 表示容量较大但访问延迟显著更高的区域。在模拟中，每个查询都被展开为对若干索引单元或数据块的访问，分层策略根据当前状态决定每次访问是在 DRAM 中命中还是需要向 SSD 发起 I/O。当访问落在 SSD 时，框架会累积相应的访问时延和 I/O 统计，并在策略需要时触发数据迁移，将部分数据从 SSD 提升到 DRAM，或者从 DRAM 驱逐到 SSD。通过在不同的 DRAM 容量下重复这一过程，可以观察策略在资源约束变化时对延迟、I/O 放大和迁移成本的综合影响。

在策略层面，框架中实现并比较了多种典型策略与 Seconds-Rule 策略，形成一组有代表性的对照组：

* All-DRAM 策略：假设所有索引数据都常驻 DRAM，不依赖 SSD 提供任何查询路径。这一极端配置给出了理想情况下的延迟和 I/O 表现，可以视作性能上界，用于衡量其他策略“距离最好情况还差多少”。

* All-SSD 策略：假设所有索引数据都放在 SSD 中，每次查询都需要从 SSD 读取必需的数据结构。这一配置给出了延迟和 I/O 的下界表现，提供了“完全不做分层”的参考水平，有助于量化分层策略带来的绝对改善。

* LRU 策略：采用最近最少使用原则，以最近访问时间作为保留或驱逐的依据。每次访问都会更新对象的时间戳，在需要释放 DRAM 空间时优先驱逐最久未被访问的数据。该策略强调时间局部性，即“最近访问过的数据在不久的将来仍有较大概率再次访问”。

* LFU 策略：采用最少使用频率原则，以访问次数作为保留或驱逐的主要依据。访问频率较高的数据会倾向于保留在 DRAM 中，而长期访问频率较低的数据在需要腾挪空间时更容易被驱逐。该策略强调长期热门数据的价值，更适合访问模式相对稳定的场景。

* Window-LFU 策略：在 LFU 的基础上引入时间窗口，只统计一定时间范围内的访问次数，把过久远的访问逐步“遗忘”。这样可以避免历史早期的高频访问长期占据 DRAM，有利于捕捉工作负载在时间上的动态变化，提高适应突发热点的能力。

* Seconds-Rule 策略：这是本工作重点研究的主要策略。其核心思想是在秒级时间尺度上跟踪访问行为，把在最近若干秒内持续活跃的一组数据定义为“热集”，优先放入 DRAM；而在这一时间尺度上显著沉寂的数据则更易被驱逐到 SSD。与单纯依赖“最近一次访问”或“长期累计频率”的策略相比，Seconds-Rule 更强调短周期内的持续活跃程度，目的是更敏锐地捕捉短时热点和阶段性访问模式，从而在相同 DRAM 配置下获得更好的延迟与召回权衡。

在评价指标方面，实验重点关注以下几个维度，并在多种 DRAM 容量下对所有策略进行系统扫描
* p95 延迟：用于衡量尾部请求的性能，而非简单平均值
* I/O 放大量：用于描述逻辑上一次查询在底层存储层面触发了多少额外的读写操作
* 数据迁移字节数：用于量化策略在 DRAM 与 SSD 之间移动数据所付出的成本；
* 召回率与延迟：通过将不同配置的结果绘制在同一坐标系中，直观展示“更快”与“更准”之间的权衡；
* 给定服务等级协议延迟约束下可达到的最大召回率：用于衡量在现实系统约束内各策略能够提供的最佳检索质量。
这些指标组合在一起，构成了一个面向混合存储向量检索场景的系统级评价框架，可以较全面地回答“某个分层策略究竟好不好”这一问题。



### 0.3 实际成果

实验流水线已经在 Linux/WSL2 环境中成功执行完成，并形成了一套可复现的输出结果。一次完整运行大致可以分为三个阶段：准备与初始化、请求回放与记录、结果汇总与可视化。

在准备阶段，系统从统一的配置入口读取数据集位置、分层策略集合、DRAM 容量列表以及工作负载轨迹，对索引结构和模拟环境进行初始化。随后，在请求回放阶段，系统按照轨迹顺序逐条模拟查询，记录每一次访问命中 DRAM 还是 SSD、触发了多少次底层 I/O、产生了多少迁移，以及单次查询的端到端延迟。在这一阶段，所有原始事件和统计量都会以结构化方式写入中间结果。

在结果汇总与可视化阶段，流水线对原始记录进行聚合统计，并输出一组可直接用于分析的工件，包括：

* 一组 CSV 文件，既包含逐查询的原始数据，也包含按策略和 DRAM 容量聚合后的统计结果。例如，对于每一组配置，表中会给出平均延迟、第九十五百分位延迟、总 I/O 次数、I/O 放大量、数据迁移字节数以及召回率等指标。

* 一组图像文件，对应聚合结果中的关键关系图。一部分图展示在不同 DRAM 容量下，各策略的 p95 延迟变化；另一部分图展示 I/O 放大量、迁移开销以及召回–延迟前沿等，用来直观体现“增加 DRAM 容量”或“更换分层策略”带来的系统行为变化。

* 一份完整的日志文件，记录了实验运行过程中各阶段的状态，包括环境信息、配置参数、每个策略与容量组合的执行情况，以及可能出现的异常或警告。日志不仅为调试提供依据，也为第三方复现实验提供了必要的上下文。

即使在不深入分析具体数值的前提下，这些结构化 CSV、图像和日志已经构成系统性成果：代码可以被执行，行为可以被量化，过程可以被复现。任何评审或后续研究者，只要在相似环境中加载同一份 Report Bundle，即可重新获得同样的输出，从而验证结果的稳定性和实验方法的可靠性。



### 0.4 预期行为

在一个设计合理的分层策略下，随着 DRAM 容量从低到高逐步提升，系统行为在宏观上应当表现出若干方向一致的趋势。这些预期不要求每个数据点都严格单调，但在整体上应当符合直觉和系统结构的基本约束。

首先，p95 延迟应当随 DRAM 容量提高而整体下降。由于 DRAM 的访问延迟远低于 SSD，DRAM 容量增大后，可以容纳更多经常被访问的索引数据，查询命中 DRAM 的概率上升，对 SSD 的依赖减弱。尤其是在尾部请求上，更多原本可能触发多次随机 SSD 访问的查询，开始在 DRAM 中被满足，以较小的延迟完成。这样，大部分“最慢”的那一批请求会被显著“拉快”，从而使第九十五百分位延迟整体向下移动。即使在某些容量点附近因为负载波动或统计样本差异出现局部起伏，从整体趋势来看，曲线仍应保持明显的下降形态。

其次，I/O 放大量应当随 DRAM 容量提高而整体降低。逻辑上，一个查询只需要访问有限数量的索引单元。如果这些单元在查询间被高效地缓存于 DRAM，底层 SSD 实际承担的 I/O 次数就会接近“刚好足够”的水平。反之，当 DRAM 空间不足时，即使多次访问的是同一片数据，因为无法长时间留在 DRAM，系统被迫反复从 SSD 读取同一个块，于是“为了一次查询发起过多 I/O”的现象会频繁出现。随着 DRAM 扩容，热点数据可以更稳定地留在内存中，重复访问不再转化为重复 I/O，放大量便会随之下降。

再次，在相同的查询负载下，给定 SLA 延迟约束下可达的最大召回率应当随 DRAM 容量增加而提高。更大的 DRAM 并不直接改变向量空间的几何结构，但它改善了每次查询的时间预算分配。由于请求平均和尾部延迟下降，系统在相同 SLA 内可以进行更深、更广的检索，例如增加探测的候选数量、使用更精细的后过滤逻辑，或者采用更昂贵但更准确的重排步骤。这些额外的“计算余裕”通常可以换取更高的召回率。因此，当 DRAM 容量从低到高增长时，在同一个 SLA 门槛下，各策略理论上都有能力达到更高的召回水平。

在比较不同策略时，不仅需要检查它们是否遵循上述趋势，还需要考察在同一 DRAM 容量和同一 SLA 要求下，谁能提供更好的权衡。一个好的分层策略应当具备如下特征：在相同的 DRAM 容量下，p95 延迟更低，或者在相同 p95 的前提下召回更高；在相同 SLA 下，能够达到的最大召回率更高；当把所有策略的结果绘制在同一坐标系中时，该策略对应的数据点整体更集中于“低延迟、高召回”的区域，形成一条更接近理想角点的前沿。这样的表现，构成判断实验结果是否合理、策略是否有效的重要依据。



### 0.5 偏差度量

为了将上述“预期趋势”转化为可执行、可复现的分析步骤，本工作给出了一组完全基于数据的偏差度量方法，可以直接在生成的 CSV 文件上计算，而不依赖主观目测。

第一，单调性检查。对于每一种分层策略，将实验中的 DRAM 容量按从小到大排序，观察对应的 p95 延迟序列。如果在容量增加的地方出现 p95 上升的情况，则记录为一次单调性违背。对每一次违背，可以进一步记录上升幅度和相对于邻近点的比例。通过统计违背次数和幅度，可以判断曲线与“整体下降”这一预期的吻合程度，区分偶然的统计噪声和明显的异常行为。

第二，与理想上界的差距。All-DRAM 策略可以视作在给定工作负载下的理想基准：它提供了最低的 p95 和最高的召回。对于任意策略和任意 DRAM 容量配置，可以定义与 All-DRAM 的差距，例如以某策略的 p95 减去 All-DRAM 的 p95，得到一个非负数值，表示在延迟上距离理想还有多少毫秒；或者以 All-DRAM 的召回减去该策略的召回，得到在召回上的差距。通过对这些差距在不同容量上的统计，可以直观地比较各策略“离最优还差多远”，也可以通过平均差距或最大差距来概括其总体表现。

第三，SLA 差距。给定一个具体的 SLA 延迟门槛，例如某个毫秒级上限，可以对每种策略在所有 DRAM 容量配置下进行筛选，只保留实际 p95 不超过这一门槛的配置。在满足 SLA 的所有点中，选出召回率最高的一点，作为该策略在该 SLA 下的“可达召回”。不同策略在同一 SLA 下可达召回的差异，反映出它们在现实服务约束内能提供的最佳检索质量。如果需要从召回的角度反推延迟，也可以固定一个目标召回率，比较不同策略要达到这一水平各自需要的 p95，与 SLA 之间的差额便是另一种形式的“延迟差距”。

通过这些偏差度量，可以把“曲线看起来还不错”这样的直观印象转化为具体数字：单调性被打破了多少次，与 All-DRAM 在延迟和召回上的差距有多大，在给定 SLA 下损失了多少本可以达到的召回。相关的计算公式以及示例代码将在后续章节中给出，以便读者直接在本次实验生成的 CSV 上复现同样的分析流程。


## 1. 背景入门

### 1.1 向量、距离和 k 近邻

在向量检索（vector search）问题中，每一条数据（文档、图片、商品等）会先被一个模型编码成一个实数向量，可以把它想象成高维空间中的一个点。形式上可以写成：

$$
\mathbf{x} = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d
$$

这里的 $d$ 是向量的维度，例如常见的有 $128$ 维、$512$ 维、$768$ 维等，取决于所用的 embedding 模型。

数据库中会有很多这样的向量，构成一个集合：

$$
{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N}
$$

当用户发出一个查询（例如一句话、一个问题），系统先把它编码成一个查询向量：

$$
\mathbf{q} \in \mathbb{R}^d
$$

向量检索要解决的问题就是：在所有数据向量里，找到和 $\mathbf{q}$ 最接近的 $k$ 个向量。这一类问题通常叫做 k 最近邻搜索（k-NN）。

要比较“接近程度”，需要定义“距离”或“相似度”。常见的两种是：

* 欧氏距离（L2 距离）：

$$
d_{\text{L2}}(\mathbf{q}, \mathbf{x})
= |\mathbf{q} - \mathbf{x}|*2
= \sqrt{\sum*{i=1}^{d} (q_i - x_i)^2}
$$

* 余弦相似度（数值越大说明越相似）：

$$
\cos(\mathbf{q}, \mathbf{x})
= \frac{\mathbf{q} \cdot \mathbf{x}}{|\mathbf{q}|_2 , |\mathbf{x}|_2}
$$

在很多系统里，会把“相似度”转换成“距离”，例如用 $1 - \cos(\mathbf{q}, \mathbf{x})$ 之类，这样就可以统一用“距离越小越好”的语义。

形式上，k 近邻搜索可以写成：

$$
\text{kNN}(\mathbf{q})
= \arg\min_{\mathcal{S} \subset {1,\dots,N},,|\mathcal{S}|=k}
\sum_{i \in \mathcal{S}} d(\mathbf{q}, \mathbf{x}_i)
$$

直观理解就是：在所有数据点里，选出离查询点 $\mathbf{q}$ 最近的 $k$ 个点。

如果做“精确搜索”，最直接的做法是：对所有 $N$ 个向量都算一次距离，算完之后排序，取前 $k$ 个
这种方法的计算复杂度大致可以写成：

$$
O(N \cdot d)
$$

当 $N$ 达到几百万、几亿，甚至更大时，每次查询都对全部向量做距离计算，延迟会非常高，根本无法支持实时在线服务。

因此，实际系统中普遍采用的是 近似最近邻搜索（Approximate Nearest Neighbor, ANN）：
允许结果和“真正的精确 top-$k$”之间有一点点误差，用少量精度损失换取巨大的性能提升。

常见的精度指标是 $\text{recall@}k$，定义为：

$$
\text{recall@}k
= \frac{\text{返回结果中真正属于精确 top-}k\text{ 的个数}}{k}
$$

当 $\text{recall@}k$ 接近 $1$ 时，说明近似方法和精确搜索几乎一样，但查询速度往往快了好几倍甚至一个数量级。



### 1.2 IVF：倒排文件索引

要让 ANN 快起来，需要利用专门的索引结构。本项目使用的是一类非常常见的结构：IVF（Inverted File）倒排文件索引。

直观上，可以把 IVF 理解成“先把空间切成很多桶，再只检查和查询最相关的几个桶”。

#### 1.2.1 建索引

第一步是对向量空间做聚类，得到若干个簇中心。设有 $n_{\text{list}}$ 个簇中心：

$$
{\mathbf{c}*1, \mathbf{c}*2, \dots, \mathbf{c}*{n*{\text{list}}}}
$$

对于数据库中每一个向量 $\mathbf{x}$，找到离它最近的那个簇中心：

$$
j^*(\mathbf{x})
= \arg\min_{1 \le j \le n_{\text{list}}} d(\mathbf{x}, \mathbf{c}_j)
$$

然后，就把这个向量 $\mathbf{x}$ 放进第 $j^*(\mathbf{x})$ 个簇对应的倒排链表（也叫 inverted list），记为 $L_{j^*(\mathbf{x})}$。
最终会得到很多个倒排链表：

$$
L_1, L_2, \dots, L_{n_{\text{list}}}
$$

每个链表里存放的是“更接近同一个中心点”的那一堆向量，可以理解为一个“向量桶”。

#### 1.2.2 查询

当有查询向量 $\mathbf{q}$ 时，不再对所有向量做距离计算，而是分两步：

* 先计算 $\mathbf{q}$ 对所有簇中心的距离，找到最接近的 $n_{\text{probe}}$ 个簇：

   $$
   \mathcal{P}(\mathbf{q})
   = { j_1, j_2, \dots, j_{n_{\text{probe}}} }
   $$

* 只在这些簇对应的倒排链表

   $$
   L_{j_1} \cup L_{j_2} \cup \dots \cup L_{j_{n_{\text{probe}}}}
   $$

   中做更精细的距离计算和排序，从而得到近似的 top-$k$ 结果。

这样一来，每次查询需要检查的候选向量数目，就从原来的 $N$，降到了远小于 $N$ 的一个数（取决于 $n_{\text{list}}$、$n_{\text{probe}}$ 和每个列表的长度），这就是 IVF 能够加速 ANN 的核心原因。

#### 1.2.3 Faiss 是什么，为什么要提到它

IVF 不是一个孤立的概念，而是集成在具体的向量检索库中使用的。
其中最重要、最常被提到的一个库叫 Faiss：

* 由 Meta 开源；
* 专门用于大规模相似度搜索和密集向量的聚类；
* 核心实现使用 C++，同时提供 Python 接口；
* 内部实现了多种索引结构，包括 IVF、HNSW、PQ（产品量化）、IVF-PQ 等。

可以简单地把 Faiss 理解为一个已经帮你写好、并高度优化好的向量检索引擎，可以直接拿来做 ANN


### 1.3 为什么要用混合 DRAM + SSD

理解分层策略之前，需要先看清楚两种典型硬件的特点。

* DRAM（内存）：访问速度非常快，可以在纳秒级完成一次随机访问，但容量有限、价格昂贵。
* SSD（固态硬盘）：容量大得多、单位容量成本低很多，但随机访问延迟比 DRAM 高出几百到几千倍，通常是微秒到毫秒级。

可以用一个极简化的公式来近似表达一次查询的时间：

$$
T_{\text{query}}= T_{\text{CPU}} + n_{\text{DRAM}} \cdot t_{\text{DRAM}} + n_{\text{SSD}} \cdot t_{\text{SSD}}
$$

其中：
* $n_{\text{DRAM}}$：该次查询从 DRAM 读取的次数；
* $n_{\text{SSD}}$：该次查询从 SSD 读取的次数；
* $t_{\text{DRAM}}$：一次 DRAM 访问的平均延迟；
* $t_{\text{SSD}}$：一次 SSD 访问的平均延迟，通常远大于 $t_{\text{DRAM}}$。

现实中基本可以认为：

$$
t_{\text{SSD}} \gg t_{\text{DRAM}}
$$

这意味着哪怕 $n_{\text{SSD}}$ 很小，只要存在几次随机 SSD 访问，就有可能主导整次查询的尾部延迟。
尤其是在 p95、p99 这类尾延迟指标上，少数“需要多次 SSD 访问”的请求会把整体尾部拉得很高。

在大规模 ANN 系统中，向量和索引的总大小往往很难完全放进 DRAM。
如果强行把全部索引放在 DRAM 中，延迟会非常好，但机器成本会非常夸张，甚至单机根本放不下。
如果把索引完全放在 SSD 上，成本可控，但延迟和尾延迟会明显恶化，很难满足在线业务的 SLA。

因此，一个折中方案是采用 混合 DRAM+SSD 的分层存储。可以做这样一个抽象：
给定一个 DRAM 容量限制 $C_{\text{DRAM}}$，需要从所有数据中挑选一部分放到 DRAM 中，其余放在 SSD 上：

$$
\text{给定容量约束 } C_{\text{DRAM}},\quad
\text{寻找一个集合 } S_{\text{DRAM}} \subseteq \text{AllData}
$$

使得：

$$
\text{Size}(S_{\text{DRAM}}) \le C_{\text{DRAM}}
$$

并在这个约束下，尽量优化以下目标：

* 对于每次查询，尽量减小 $n_{\text{SSD}}$，从而降低 $T_{\text{query}}$，尤其是尾部的 $T_{\text{query}}$；
* 控制 DRAM 与 SSD 之间的数据迁移成本，不要频繁搬来搬去导致额外 I/O。

现代的磁盘化/混合化 ANN 系统，例如 DiskANN、SPANN 等，都是在这个“内存 vs SSD”的权衡框架下设计的，只不过采用的索引结构不同、分层策略不同。



### 1.4 Seconds-Rule 的直觉

在缓存与存储系统领域，有一个非常经典的法则，叫 “五分钟法则（five-minute rule）”。
它的核心思想是：比较“把某页数据常驻内存”与“每次需要时都从磁盘读一次”这两种方案的经济成本，从而推导出一个“值得缓存”的重用时间阈值。

用一个极简化的模型来表示：

* 假设把某页数据放在内存里的成本（单位时间）为 $C_{\text{mem}}$；
* 每次从磁盘读取这页数据的成本为 $C_{\text{io}}$；
* 这页数据的平均重用间隔为 $\Delta T$ 。

如果这页数据常驻内存，那么单位时间的成本大致是：

$$
\text{Cost}*{\text{mem}} \approx C*{\text{mem}}
$$

如果完全不缓存，每次都从磁盘读，那么在平均每 $\Delta T$ 秒访问一次的前提下，单位时间因 I/O 产生的成本大致是：

$$
\text{Cost}*{\text{disk}} \approx \frac{C*{\text{io}}}{\Delta T}
$$

当两种方案的成本相等时，有：

$$
C_{\text{mem}} = \frac{C_{\text{io}}}{\Delta T^*}
$$

解出临界重用间隔 $\Delta T^*$：

$$
\Delta T^* = \frac{C_{\text{io}}}{C_{\text{mem}}}
$$

含义是：

* 如果实际重用间隔 $\Delta T < \Delta T^*$，说明访问很频繁，缓存这页更划算；
* 如果 $\Delta T > \Delta T^*$，说明访问很稀疏，没必要用内存长期“占位”。

在现代硬件（包括 SSD）和 AI 工作负载的背景下，上述比值中的 $C_{\text{io}}$ 和 $C_{\text{mem}}$ 都发生了变化，对应的“临界间隔” $\Delta T^*$ 也会变化。
一些研究发现，在某些场景下，这个关键时间不再是“分钟级”，而是降到了 秒级，也就是“在若干秒内是否被重用”就足以决定是否值得缓存。

本项目提出的 Seconds-Rule，就是把这种“经济模型 + 重用间隔”的思想迁移到 ANN 的倒排列表上，形成一个在线分层策略。

可以定义一个“秒级热度”的简单示例。对于某个倒排列表 $\ell$，在时间 $t$ 时刻，统计它在最近 $S$ 秒内被访问的次数：

$$
h_\ell(t)
= \#\{\tau \mid t - S \le \tau \le t,\ \text{在时间 } \tau \text{ 访问了列表 } \ell\}
$$

给定一个阈值 $H$，可以采用类似的决策规则：

* 如果
  $$
  h_\ell(t) \ge H
  $$
  说明列表 $\ell$ 在最近 $S$ 秒内非常“热”，更倾向于将它放在 DRAM 中；

* 如果
  $$
  h_\ell(t) < H
  $$
  说明列表 $\ell$ 在这一时间尺度上的活跃度不足，可以优先让它留在 SSD 或被驱逐。

真实的 Seconds-Rule 策略比这个示意公式更复杂：
需要考虑 DRAM 总容量约束、每个列表的大小、迁移成本、访问延迟等多种因素，但核心出发点可以概括为在秒级时间尺度上定义和检测“热集”，优先让这些热集对应的倒排列表常驻 DRAM，从而在混合 DRAM+SSD 的结构中取得更好的延迟与召回权衡。
本项目就是在 IVF 索引之上实现并评估这样的秒级分层策略，并与传统策略（如 LRU、LFU、Window-LFU 等）进行系统对比。


## 3. 研究问题与范围

### 3.1 研究问题

本报告的核心研究问题是：在已经固定使用 IVF 这一类 ANN 索引结构的前提下，是否可以通过一个在线的“秒级热度（Seconds-Rule）”分层策略，按时间上的“热度”选择哪些倒排列表放在 DRAM 中，从而在召回率、尾延迟和 SSD I/O 放大量之间，取得比经典缓存策略（如 LRU、LFU、Window-LFU）更好的综合权衡。

### 3.2 研究范围

本报告的工作范围集中在“给定 ANN 索引结构后，如何设计和评估 DRAM+SSD 分层策略”这一层面，而不是把索引算法创新、系统工程和产品化全部混在一起。具体来说，主要做了三类事情。

* 分层策略的设计与对比：报告中实现并评估了一组可直接运行的分层策略。
  其中，Seconds-Rule 是我提出并重点分析的在线策略，它在秒级时间窗口内跟踪每个倒排列表的访问情况，根据“最近若干秒内的活跃度”来判断一个列表是否应当留在 DRAM 中。策略在查询流到来时持续更新热度估计，并在 DRAM 容量受限的前提下做出晋升或驱逐决策。
  为了有清晰的对照，本报告同时实现了多种经典策略作为基线，包括 LRU、LFU 和 Window-LFU。这些策略代表了现实系统中非常常见的做法，能够用来判断 Seconds-Rule 是否真的带来了改进。
  此外，还使用 All-DRAM 和 All-SSD 作为极端“上界”和“下界”，用于给所有实验结果提供一个性能参考区间。

* 指标体系的构建与使用：报告没有只盯着某一个指标，而是构建了一组互补的度量，用来量化“好不好”这个问题。
  首先是召回率，用来衡量 ANN 结果相对于精确 kNN 的质量，判断分层策略是否对检索精度造成显著损伤。
  其次是 p95 延迟，用来反映尾部请求的响应时间，而不是只看平均值。这一指标对“有多少查询需要访问 SSD、需要访问几次 SSD”等因素非常敏感。
  再次是 SSD I/O 放大量，用来描述“一次逻辑查询在底层触发了多少额外的读写”，直接刻画策略对存储层的压力。
  最后是 DRAM 与 SSD 之间的数据迁移字节数，用来衡量策略在维持热集时付出的后台代价。通过这四类指标，可以从质量、速度、成本三个角度综合评价不同策略。

* 可复现的实验流水线与工件产出：报告围绕 Seconds-Rule 及其基线策略，搭建了一条完整的实验流水线，用来从配置到结果自动运行出一套可检查、可复现的工件。
  流水线以统一的配置文件为入口，明确数据集、索引结构、候选策略、DRAM 容量扫描范围和查询轨迹。随后，对每一种“策略 × 容量”的组合，在模拟环境中重放查询流，记录访问路径、延迟、I/O 和迁移等信息。
  运行结束后，流水线会自动生成原始和聚合后的 CSV 文件、对应的图像，以及完整日志。任何获得同一份 Report Bundle 的人，只要在相似环境中重新执行 one-click 入口脚本，就能再现这套结果。

### 3.3 不在研究范围内的内容

为了避免读者误解本工作的目标，本报告也明确说明哪些事情“不在当前研究范围内”。这些内容不是能力不足，而是为了聚焦问题而主动划定的边界。

* 不设计全新的 ANN 索引结构：报告不会尝试发明一种全新的 ANN 索引，比如新的图结构、新的层级索引或对 HNSW 的根本性改造。
  整个研究从一开始就假定：底层索引结构已经选定为类似 Faiss IVF 的标准形式。我要回答的问题是：在这个既定结构之上，如何通过分层策略更好地使用有限的 DRAM，而不是改变索引本身。

* 不构建完整的生产级向量数据库：报告不会把 Seconds-Rule 真实接入某个现成的向量数据库产品，也不会实现多租户、权限控制、分布式部署、服务治理等工程层面的功能。
  实验环境是一个模拟器加延迟模型的组合，目的是在一个可控的环境中观察分层策略对延迟、I/O 和迁移的影响，而不是完整覆盖线上系统的所有细节。将 Seconds-Rule 融入具体产品，属于后续工程落地的工作，不在这篇报告的直接范围内。

* 不精细建模真实 SSD 队列和设备内部行为：报告不会尝试构建一个与某款具体 SSD 一一对应的复杂模型，不会显式模拟操作系统 I/O 调度、设备内部并行度、写放大和 GC 等细节。
  在实验中使用的是抽象化的延迟模型和访问统计：通过记录访问次数和数据量，近似估算不同策略对 SSD 的负载，然后比较策略之间的相对差异。这样做的目的是保持模型足够透明，让“策略好坏”的结论主要来自于策略本身，而不是埋在复杂硬件细节中的偶然效应。


## 4. 系统 / 模拟器设计

### 4.1 分层 IVF 存储模型

评测从一个抽象的两级存储模型开始：
IVF 索引由许多倒排列表（inverted lists）组成，每一个倒排列表在任意时刻只属于两个层级之一：

* DRAM 层：访问延迟极低，读取一次倒排列表的代价很小，但总容量受到严格限制。
* SSD 层：访问延迟显著更高，每次读取倒排列表都需要触发一次 SSD I/O，带来明显的延迟成本，但容量近似可视为“足够大”。

在 IVF 结构下，一次查询不会访问全部倒排列表，而是只探测若干个与查询最相关的列表。可以把“一次查询”抽象成“探测一组列表”的过程：

* 对于被探测到、且当前位于 DRAM 的列表，只需要付出 DRAM 访问的成本。
* 对于被探测到、且当前位于 SSD 的列表，必须付出 SSD I/O 成本，这对尾延迟影响很大。

因此，从系统角度看，分层策略的职责可以概括为一句话：

* 在给定的 DRAM 容量约束下，选择一部分倒排列表常驻 DRAM，并随着时间推移根据访问热度变化适时迁移列表，使“真正热点”尽量留在 DRAM 中，从而降低整体和尾部的查询延迟，同时控制 I/O 和迁移代价。

模拟器所做的，就是在这种两级存储抽象下，重放真实或合成的查询序列，统计在不同策略、不同 DRAM 容量下的延迟、I/O 和迁移情况。



### 4.2 策略接口

无论是 Seconds-Rule 还是 LRU/LFU，一旦放进这个模拟框架里，都可以被统一视为“策略模块”。
一个分层策略在系统中的接口大致可以拆成三个部分：输入信号、内部状态、输出决策。

* 输入信号（策略可以观察到的信息）

  策略不会直接看到“未来的查询”，只能根据已经发生的事件做出判断。典型输入包括：

  * 倒排列表被访问的事件：每当某个列表被查询探测到，策略收到一条“列表 $\ell$ 在时间 $t$ 被访问”的记录。
  * 时间信息：每个访问事件都有时间戳，策略可以用它来估计重用时间间隔、空闲时长等特征。

  在实现层面，可以把输入理解成一条不断到来的事件流：“某时刻访问了某个列表”。

* 内部状态（策略在运行过程中维护的量）

  为了利用历史信息，策略需要为每个倒排列表维护一些状态量，例如：

  * 访问次数：用于实现 LFU 或 Window-LFU 类策略。
  * 最近一次访问时间：用于实现 LRU 类策略，或者估算空闲时长。
  * 平均重用间隔的估计值：用于判断长期热度。
  * 其他统计量：例如指数加权移动平均（EWMA）的间隔、秒级窗口内的计数等。

  这些状态不会直接暴露给外部系统，但会影响策略做出的迁移与驻留决策。

* 决策输出（策略真正“干了什么”）

  策略需要定期或在特定触发条件下做两类决策：

  * 给定当前 DRAM 容量约束，选择哪些倒排列表应当放在 DRAM 中，也就是维护一个“DRAM 集合”的候选排序或打分。
  * 当 DRAM 空间不足或列表热度发生明显变化时，生成迁移计划：驱逐一部分不再足够热的列表，把一些足够热的列表从 SSD 提升到 DRAM。

  在模拟器中，这些决策会转化为具体的迁移操作和层级变更，从而改变后续查询的访问路径和延迟。

通过这种统一接口，可以在同一套框架下插入不同分层策略，对它们进行公平对比：输入信号相同，容量约束相同，只是内部状态定义和决策逻辑不同。



### 4.3 Seconds-Rule 策略

Seconds-Rule 策略的核心思想，是同时利用“长期热度”和“短期活跃度”这两类信息，为每个倒排列表构造一个简单、可解释的综合评分，然后按评分挑选哪些列表应该放在 DRAM 中。

在这个策略下，每个列表 $\ell$ 都维护两个关键量：

* 一个是平均重用间隔（平均访问间隔），记为 $\text{avg\_interval}_\ell$，用来反映“从长时间尺度看，这个列表平均多久被访问一次”。这个量越小，说明长期访问频率越高，更偏向于“长期热门”（类似 LFU 的视角）。
* 另一个是空闲时间，记为 $\text{idle}_\ell$，表示“从最近一次访问到当前时刻已经过去了多长时间”。这个量越小，说明近期刚被访问过，更偏向于“短期刚活跃”（类似 LRU 的视角）。

Seconds-Rule 把这两个量线性组合成一个“有效间隔”：

$$
\mathrm{effective\_interval}_{\ell}=\mathrm{avg\_interval}_{\ell}\,\lambda\,\mathrm{idle}_{\ell}
$$


其中，$\lambda$ 是一个可调的权重，用来平衡长期频率和短期新鲜度在决策中的相对重要性：

* 当 $\lambda$ 取值非常小时，$\text{effective\_interval}*\ell$ 主要由 $\text{avg\_interval}*\ell$ 决定，策略更接近“按长期访问频率排序”，类似于 LFU。
* 当 $\lambda$ 取值较大时，$\text{idle}_\ell$ 的影响变得更显著，策略更接近“按最近访问时间排序”，类似于 LRU。

在这一定义下，可以给出一个简单的排序规则：

* 对所有列表按照 $\text{effective\_interval}_\ell$ 从小到大排序；
* 在 DRAM 容量允许的前提下，优先保留 $\text{effective\_interval}_\ell$ 最小的那一批列表在 DRAM 中，其余列表驻留在 SSD 上。

直观理解是：长期来看平均重用间隔越短、最近一次访问离现在越近的列表，会被视为“真正的热点”，更有资格占据有限的 DRAM 空间。

需要强调的一点是，Seconds-Rule 不是一个黑箱机器学习模型，而是一个结构非常简单的、可解释的策略：

* 所有状态变量和参数都有明确含义，方便调试和分析。
* 权重 $\lambda$ 可以根据工作负载特征进行调节，实质上是在“更看重长期频率”与“更看重近期活跃度”之间滑动。
* 如果业务侧有额外先验（例如某类列表必须常驻、某些列表可以低优先级处理），可以在这个打分框架中自然加入附加项，而不破坏整体结构。

在模拟器中，Seconds-Rule 就通过不断更新 $\text{avg\_interval}*\ell$ 和 $\text{idle}*\ell$，并定期重算 $\text{effective\_interval}_\ell$，来驱动 DRAM/SSD 之间的晋升与驱逐。

### 4.4 基线策略

为了判断 Seconds-Rule 是否真的带来了收益，本报告在同一套模拟框架下实现并对比了多种经典基线策略。这些策略代表了实践中常见的分层思路，也覆盖了“极端上界与下界”的参考点。

* All-DRAM 策略：把所有倒排列表都放在 DRAM 中，不使用 SSD。
  这并不是一个现实可行的部署方案，因为需要巨大的 DRAM 容量，但在模拟中它提供了理论上的延迟和 I/O 上界：在相同查询负载下，p95 延迟和 SSD I/O 基本可以视为“最理想”的情况，其它策略都不可能在这两个指标上超越它。

* All-SSD 策略：把所有倒排列表都放在 SSD 上，不使用 DRAM。
  这代表了“完全不做分层”的极端情况，通常会带来最高的延迟和 I/O 放大量，但 DRAM 成本最低。
  它为所有分层策略提供了一个对照下界：任何合理的策略，在相同 DRAM 预算下性能都不应比 All-SSD 更差。

* LRU 策略：通过“最近一次访问时间”来排序列表，优先让最近访问过的列表留在 DRAM 中，把最久未访问的列表驱逐到 SSD。
  这一策略完全基于“时间局部性”的假设，即“如果一个列表刚被访问，那么在不久的将来它还有较大概率再次被访问”。

* LFU 策略：通过“累计访问次数”来排序列表，把长期访问频率最高的那部分列表放在 DRAM 中，把访问频率最低的列表留在 SSD。
  这一策略强调的是“长期热门”，适合访问模式相对稳定、不频繁变化的场景。

* Window-LFU 策略：在 LFU 的基础上引入时间窗口，只统计最近一段时间内的访问次数。超过窗口的历史访问会逐渐被遗忘，从而更好地反映当前阶段的热点，而不是过度受很久以前的访问影响。
  相比纯 LFU，这种做法在工作负载出现阶段性变化、热点转移时，适应速度更快。

总体来看，这些基线策略构成了“只看最近访问的极端”（LRU）与“只看累计频率的极端”（LFU）之间的代表，同时也给出了“不分层”（All-SSD）和“极端理想”（All-DRAM）的上下界。
Seconds-Rule 则可以被视为在这两类极端之间寻找平衡的一种中间方案：既继承了频率视角，又引入了秒级的近期活跃度，从而有机会在实际工作负载下获得更好的综合表现。


## 5. End-to-end pipeline walkthrough（端到端流水线全流程）

本节从“数据准备”开始，一直讲到“打包输出”，只保留与本次实验核心相关的阶段。每个阶段都用完整的说明来描述：这一阶段大致输入了什么，做了哪些事情，产出了什么，又为什么和本报告的结论直接相关。本次运行的全部细节都可以在日志`results/logs/one_click_XXXX_XXXX.log`中对应找到。


### Stage 1 数据准备

在完成环境检查之后，流水线进入第一个和数据密切相关的阶段，也就是数据准备阶段。在这一阶段中，系统会根据配置文件中的路径，找到用于实验的向量数据集和查询集合，并尝试将它们加载为内存中的矩阵或等价的数据结构。这里会检查向量的维度是否与配置中声明的一致，检查数据条目数量是否达到预期，如果需要评估召回，还会在数据规模允许的情况下离线计算精确的 k 近邻结果，作为后续 recall 计算的基准。

这一阶段的输出并不是新的文件，而是一个“已通过验证的数据视图”，包括已加载的向量矩阵、查询矩阵，以及可能存在的精确 ground truth。日志中会明确记录本次运行到底加载了多少条向量、每条向量的维度是多少、查询有多少条，以及 ground truth 是否成功生成。只要数据准备阶段出现缺失或损坏，后续所有关于召回的指标就会失去意义，因此这一阶段相当于为整个实验写下“本次是在什么数据基础上得出结论”的说明。

---

### Stage 2 建立或加载 IVF 索引

数据准备完成后，流水线会为向量数据构建或加载一个 IVF 索引结构。若没有现成的索引文件，系统会根据配置中的参数，例如 nlist、距离度量方式和 k 值，对向量进行聚类训练，得到一组簇中心，然后将每条数据向量分配到对应的倒排列表中，从而形成完整的 IVF 索引。如果已有预训练索引，流水线则会从磁盘加载该索引，并检查索引内部参数与当前实验配置是否一致。

这一阶段的核心输出是一个可以用于查询的 IVF 索引结构，它定义了“每个向量在倒排列表中的位置”，并决定后续 Stage 中“查询将访问哪些列表”的基础拓扑。日志会记录索引训练或加载的耗时，索引中包含的向量数量，使用的关键参数以及随机种子。如果随机种子没有被记录，不同运行之间的聚类结果可能会略有差异，从而轻微影响实验结果，因此本报告会将本次运行使用的索引参数视为当前结论的前提假设。

---

### Stage 3 访问序列生成

在索引就绪之后，流水线需要将查询集合转化为一条实际的访问序列，也就是在时间轴上明确“每一步访问了哪些倒排列表”。在这一阶段，系统会对每条查询执行 IVF 的簇选择步骤，根据查询向量与各个簇中心的距离，确定需要探测的倒排列表集合。然后，按照查询出现的顺序，将这些列表访问事件串联起来，就得到了一条时间序列形式的访问轨迹。

根据具体配置，访问序列可以来自真实的查询日志，也可以是根据指定的概率分布生成的合成序列，甚至可以刻意设计成具有热点和冷点分区、并且在不同时间段发生热点迁移的模式。流水线会记录生成的访问序列长度、类型以及使用的随机种子。对于分层策略而言，这条访问序列就是其决策的全部外部输入，它必须仅凭这些历史访问信息判断哪些列表值得驻留在 DRAM 中。因此，访问序列既不能过于简单，也不能完全随机，本次流水线的这一阶段就是在构造一个既能反映热点结构，又能考验策略适应能力的工作负载。

---

### Stage 4 运行分层策略并采集指标

访问序列生成之后，流水线进入最关键的阶段，即在同一条访问轨迹上运行不同的分层策略，并对各项指标进行详细记录。对于每一种策略和每一个 DRAM 容量配置，模拟器都会对访问序列进行完整的重放。在每一个访问事件到来时，模拟器根据当前的 DRAM 内容和策略内部状态，判断被访问的倒排列表当前是在 DRAM 还是在 SSD，如果在 DRAM，则只计入内存访问的代价，如果在 SSD，则累加一次 SSD I/O 的延迟和数据量，同时根据策略规则决定是否要把该列表迁移到 DRAM 中或驱逐其他列表。

在这一阶段中，流水线会为每次查询记录必要的信息，以便后续统计，包括该查询的端到端延迟、是否命中 DRAM、触发了多少次 SSD 访问、迁移了多少字节，以及查询结果相对于精确 kNN 的召回情况。这样，每个“策略 × 容量”的组合都会生成一份原始结果表，完整反映该组合在这条访问序列上的表现。正是这些细粒度的记录，支撑了后面所有关于 p95 延迟、I/O 放大量、迁移开销和召回率的结论，也是比较 Seconds-Rule 与 LRU、LFU 等基线策略优劣的直接证据来源。

---

### Stage 5 聚合原始结果

原始结果文件的粒度往往是“每一次查询一行”，对于分析和呈现来说过于细碎，因此流水线在这一阶段会对它们进行聚合。聚合的方式通常是按策略名称、DRAM 容量以及其他关键配置进行分组，然后在每一组内计算各种统计量，例如平均延迟、p95 延迟、平均 I/O 放大量、总迁移字节数、平均召回率，等等。聚合后的结果会被写入新的 CSV 文件，例如 `agg/ann_policy_agg.csv`，这些文件中的每一行就对应一个“策略加容量”的整体表现。

在此基础上，流水线还会计算一些更高层次的派生指标，例如在给定 SLA 延迟约束下，某个策略在所有容量配置中可达到的最大召回率，并将这些结果写入诸如 `agg/sla_reachable_recall.csv` 这样的文件中。通过这一阶段，原本混杂的逐查询记录被压缩成结构清晰的统计表格，使得后续绘制曲线、阅读结果以及量化偏差都变得直观而方便，这也是报告中所有表格和数字背后真正的数据来源。

---

### Stage 6 绘制图像

在获取聚合结果之后，流水线会自动生成一系列图像文件，把核心指标之间的关系以图形方式展现出来。这些图像通常包括 p95 延迟随 DRAM 容量变化的曲线、I/O 放大量随 DRAM 容量变化的曲线、迁移开销与容量的关系图、p95 延迟与 I/O 之间的关系图，以及展示召回率与延迟权衡的前沿曲线和在不同 SLA 下可达召回率的比较图。绘图过程读取前一阶段输出的聚合 CSV 文件，将不同策略绘制为不同的线条或点集，以便直观比较。

这些图像的意义在于，它们把复杂的数值表格转化为人眼更容易抓住的模式，使读者可以一眼看出，例如 DRAM 容量增加时各策略的延迟是否整体下降，I/O 放大量是否随容量扩展而减小，Seconds-Rule 在同一容量下是否相对于 LRU 和 LFU 产生了明显优势，以及在给定 SLA 下哪一种策略能够提供更高的召回。报告中对“曲线形状”的任何描述，最终都可以在这一阶段生成的图像文件中直接进行核对。

---

### Stage 7 打包工件并生成 Report Bundle

最后，流水线会把本次运行中产生的所有关键输出，包括日志文件、原始和聚合的 CSV 文件、所有图像以及必要的元数据路径，汇总成一个自描述的“报告束”，也就是 Report Bundle。这一步不会再计算新的数值，而是在做一次系统性的整理，将本次实验的所有成果以及它们的存放位置固定下来，用一个唯一的标识符来引用，例如 `Report Bundle (20251215_194002)`。

通过这一阶段，读者和评审可以在不依赖作者记忆的前提下，准确地找到本次实验所用的数据、生成的索引、记录的结果和输出的图像。报告中引用任何具体文件路径时，都可以回溯到这个 Bundle 中的清单。对于之后希望复现实验或做进一步扩展的人来说，Report Bundle 就是一份标准的“运行快照”，它定义了本次流水线从输入到输出的完整边界，也构成了本报告所有结论的物理基础。




太懂你在吐槽什么了 😂
下面把第 6 节重写一版：
*保留具体的文件名和目录结构*，但**不再出现任何 `20251215_191512` 这类具体 run id**，全部换成通用占位符。

---

## 6. Artifact inventory（逐个文件解释：你产出了什么）

本节说明 Report Bundle 里都包含哪些工件, 每一类文件放在哪里, 在报告和复现里各自起什么作用。为了让内容更通用, 所有带时间戳的目录名都用占位形式表示, 不再写死某一次运行的具体编号。

---

### 6.1 日志文件

一键流水线每运行一次, 都会在日志目录下生成一份带时间戳的日志文件。路径形式类似于:

* `results/logs/one_click_<timestamp>.log`

其中 `<timestamp>` 是某次运行的时间戳, 这次报告对应的那一条日志就是这类文件中的一个。

这类日志文件是本次运行的完整“文字记录”, 里面包含了:

* 入口脚本何时启动, 使用了什么参数
* 各个阶段何时开始、何时结束
* 发生错误或告警时的具体栈信息

在报告中, 它一方面用来支撑“这次实验确实被完整跑过”的可复现性说明, 另一方面也为评审或后续调试提供了排错线索。

---

### 6.2 CSV 文件

CSV 是这次实验最核心的数据载体, 大致可以分成“原始结果”和“聚合结果”两种类型, 都放在以某次运行为单位的结果目录下面。每次运行的目录名类似:

* `results/tiered_ann_seconds_rule_<run_id>/...`

其中 `<run_id>` 是某次运行的标识符, 可以包含时间戳等内容。

#### 6.2.1 原始结果 CSV

每次完整运行都会在对应目录下生成一份原始结果文件, 路径形式为:

* `results/tiered_ann_seconds_rule_<run_id>/raw/ann_policy_raw.csv`

这一文件按“行”记录实验数据, 典型含义是“某个策略在某个 DRAM 配置下的一次测量结果”。常见列包括:

* 使用的策略名称, 例如 Seconds-Rule, LRU, LFU
* DRAM 容量配置, 例如以比例或绝对大小表示
* 召回指标, 如 `recall@k`
* 延迟指标, 如 p50、p95 等
* SSD I/O 次数或字节数
* DRAM 与 SSD 之间的迁移字节数

确切列名可以通过读取 CSV 并查看 `df.columns` 得到。
这些原始 CSV 是后面所有聚合与绘图的基础, 也是出现异常时回溯到“每次查询到底发生了什么”的唯一数据源。如果某个策略或某个容量点在原始文件中缺失, 对应的曲线和聚合结果自然也会不完整。

#### 6.2.2 聚合结果 CSV

在同一 `<run_id>` 目录下, 流水线会从 `raw/ann_policy_raw.csv` 出发做一次聚合, 把“细粒度记录”转换成“每个策略每个容量的一组统计指标”。聚合结果主要有两类:

* 策略聚合表

  * `results/tiered_ann_seconds_rule_<run_id>/agg/ann_policy_agg.csv`

  这类文件按策略、DRAM 容量等键进行分组, 每一行代表一个“策略 × 容量”的整体表现, 通常包含: 平均延迟、p95 延迟、平均 I/O 放大量、总迁移字节数、平均召回率等。
  报告中关于“某策略在某个 DRAM 预算下表现如何”的所有文字描述, 都可以在这个表中找到对应行。

* 在给定 SLA 下的可达召回表

  * `results/tiered_ann_seconds_rule_<run_id>/agg/sla_reachable_recall.csv`

  这一文件基于聚合结果, 进一步在固定 p95 延迟约束下为每个策略挑选“召回最高的那一点”。例如, 给定某个延迟门槛, 先筛掉所有超过 SLA 的配置, 然后从剩下点里取最大召回, 写入这一表中。
  它的作用是用一个 “SLA 视角” 把多维评价压缩成“在同一延迟约束下, 谁的召回更高”这一单一问题。

---

### 6.3 图像文件

图像文件放在各 `<run_id>` 目录下的 `figures/` 子目录中。对于每一次完成聚合的运行, 通常都会生成同样六种图像, 文件名固定, 目录结构形如:

* `results/tiered_ann_seconds_rule_<run_id>/figures/io_amp_vs_dram.png`
* `results/tiered_ann_seconds_rule_<run_id>/figures/migration_vs_dram.png`
* `results/tiered_ann_seconds_rule_<run_id>/figures/p95_vs_dram.png`
* `results/tiered_ann_seconds_rule_<run_id>/figures/p95_vs_iops.png`
* `results/tiered_ann_seconds_rule_<run_id>/figures/recall_latency_frontier.png`
* `results/tiered_ann_seconds_rule_<run_id>/figures/sla_reachable_recall.png`

每一种图的语义是固定的, 与 `<run_id>` 用的是什么时间戳无关:

* `p95_vs_dram.png`
  展示 p95 延迟相对于 DRAM 容量的变化关系, 用来直观回答“DRAM 越多, 尾延迟是否整体越好”以及“在同一容量下, 哪个策略的 p95 更低”。

* `io_amp_vs_dram.png`
  展示 I/O 放大量随 DRAM 容量变化的情况, 反映“增加 DRAM 是否确实减少了对 SSD 的多余读写”, 以及“在相同 DRAM 预算下, 谁的 SSD 压力更小”。

* `migration_vs_dram.png`
  展示在不同 DRAM 容量下, 各策略为维持热集所付出的迁移数据量。通过这张图可以判断, 某个策略的好指标是不是建立在非常激进的搬迁之上, 以及 DRAM 变大之后迁移成本是否有缓和。

* `p95_vs_iops.png`
  把 p95 延迟和 I/O 压力放在同一张图里, 通常一轴是延迟, 一轴是 I/O 次数或 IOPS。它用来观察“尾延迟是否主要受 SSD 负载驱动”, 以及“在相似 I/O 压力下, 哪个策略能做到更低的 p95”。

* `recall_latency_frontier.png`
  把不同策略和容量组合的“召回–延迟”点绘制在同一平面上, 关注的是谁的前沿曲线更靠近“高召回、低延迟”的理想角落。这张图体现的是一个多目标优化的前沿, 用来比较各策略的整体权衡能力。

* `sla_reachable_recall.png`
  在固定一个或多个 p95 SLA 门槛的前提下, 展示各策略能达到的最高召回率。它提供的是一种工程视角: 如果只能接受这么高的延迟, 在所有策略中应该优先选谁。

整体来看, 日志文件说明“这次实验是怎么跑的”, 原始 CSV 说明“实验一共算出了哪些细粒度数据点”, 聚合 CSV 和图像则把这些数据整理成更适合阅读和对比的形态。所有这些工件, 共同构成了本报告结论的证据基础。


## 7. Results interpretation guide (怎么写 results + 偏差怎么量化)

> Important: I cannot see the numeric values inside your CSVs from within this chat.  
> So this section gives you an exact, reproducible method to extract the key numbers and write crisp conclusions.

### 7.1 Print columns + preview (第一步：看 CSV 里到底有哪些列)
Run from repo root:

```bash
python - <<'PY'
import pandas as pd

path = "results/tiered_ann_seconds_rule_20251215_193936/agg/ann_policy_agg.csv"
df = pd.read_csv(path)

print("Columns:", list(df.columns))
print(df.head(10).to_string(index=False))
PY
```

This tells you:
- what metric names are available
- how policies are labeled
- what DRAM budget column is called (e.g., `dram_frac`, `dram_budget`, etc.)

### 7.2 Core expected trends (你写 results 时应该先写“预期趋势”)
You can copy this structure directly into your paper:

1) Trend 1: latency should drop with more DRAM  
   - Expected: p95 is (mostly) monotonic decreasing with DRAM.  
   - Deviation: non-monotonicity indicates measurement noise, workload phase changes, or thrashing (often for recency-only policies).

2) Trend 2: I/O amplification should drop with more DRAM  
   - Expected: fewer SSD-resident lists are accessed.  
   - Deviation: if not decreasing, the policy may be caching the wrong lists.

3) Trend 3: migration overhead increases when policy is “too reactive”  
   - Expected: LRU-like policies can churn; LFU-like policies are stable.  
   - Deviation: if seconds-rule churns more than LRU, the scoring/tuning is wrong.

### 7.3 Quantifying deviation (可直接抄进报告的方法)
Below are three quantitative deviation measures. Pick at least one (paper needs at least one).

#### (A) Monotonicity violation rate
Example for p95 vs DRAM:

```bash
python - <<'PY'
import pandas as pd
import numpy as np

path = "results/tiered_ann_seconds_rule_20251215_193936/agg/ann_policy_agg.csv"
df = pd.read_csv(path)

# - edit these to match your CSV column names -
POLICY = "policy"
DRAM   = "dram_frac"      # or dram_budget
P95    = "p95_us"         # or p95_ms
# --

for pol, g in df.groupby(POLICY):
    g = g.sort_values(DRAM)
    p = g[P95].to_numpy()
    # violation count: any increase when DRAM increases
    violations = np.sum(np.diff(p) > 0)
    total = max(len(p) - 1, 1)
    print(f"{pol:20s} monotonicity violations: {violations}/{total} = {violations/total:.2%}")
PY
```

This yields a clear “偏差有多大” number.

#### (B) Gap-to-All-DRAM (how far from best possible)
At each DRAM budget, compare policy latency to All-DRAM:

\[
\text{gap}(b) = \frac{p95_{\text{policy}}(b)}{p95_{\text{all-dram}}(b)} - 1
\]

You can report average gap across budgets, or the max gap.

#### (C) SLA-reachable recall difference
For a given SLA (e.g., 2ms p95), report:

\[
\Delta \text{recall} = \text{recall}_{\text{seconds-rule}} - \text{recall}_{\text{baseline}}
\]

Use `sla_reachable_recall.csv` to read it directly.

### 7.4 How to write “Results” section (模板)
Use exactly this structure:

- Figure X: what the axes mean  
- Expected: what should happen and why  
- Observed: what your data shows (one sentence, with a number)  
- Explanation: why this happened (mechanism, not feelings)  
- Takeaway: one sentence conclusion

Example template:

> Fig. 2 (p95 vs DRAM) shows how tail latency changes when we increase DRAM budget.  
> Expected: p95 should decrease as DRAM increases because fewer probed lists require SSD I/O.  
> Observed: At 20% DRAM, Seconds-Rule reduces p95 by X% compared to LRU and by Y% compared to LFU, while remaining within Z of All-DRAM.  
> Explanation: Seconds-Rule combines interval and idle, adapting faster than LFU but with less churn than LRU.  
> Takeaway: Seconds-Rule achieves a better tail-latency/DRAM trade-off in this workload.

Replace X/Y/Z using the scripts above.



## 8. Final “paper-style” results section with your actual figures embedded

Below I embed the latest aggregated run (20251215_193936) figures using relative paths.  
If you keep this report in repo root, the images should render in Markdown viewers.

### 8.1 Tail latency vs DRAM
Figure 1. p95 latency vs DRAM budget.
![](results/tiered_ann_seconds_rule_20251215_193936/figures/p95_vs_dram.png)

### 8.2 I/O amplification vs DRAM
Figure 2. I/O amplification vs DRAM budget.
![](results/tiered_ann_seconds_rule_20251215_193936/figures/io_amp_vs_dram.png)

### 8.3 Migration overhead vs DRAM
Figure 3. Migration bytes vs DRAM budget.
![](results/tiered_ann_seconds_rule_20251215_193936/figures/migration_vs_dram.png)

### 8.4 Tail latency vs IOPS
Figure 4. p95 latency vs SSD IOPS.
![](results/tiered_ann_seconds_rule_20251215_193936/figures/p95_vs_iops.png)

### 8.5 Recall-latency frontier
Figure 5. Recall-latency frontier (higher recall, lower latency is better).
![](results/tiered_ann_seconds_rule_20251215_193936/figures/recall_latency_frontier.png)

### 8.6 SLA reachable recall
Figure 6. SLA-reachable recall.
![](results/tiered_ann_seconds_rule_20251215_193936/figures/sla_reachable_recall.png)

> If you also want to include Run A (20251215_192741) as a second workload/dataset, copy Section 8 and change the directory name.



## 9. Threats to validity (为什么可能不准：偏差来源)

A clear paper must explain what could make results differ in the real world.

### 9.1 Simulator vs real system gap
If latency is modeled (instead of measured on real SSD), then:
- queueing effects (IO depth), OS page cache, and tail amplification may be simplified.

Mitigation in report: clearly state your latency model and assumptions.

### 9.2 Workload realism
If workloads are synthetic:
- a different real trace may change policy ranking.

Mitigation: include at least two workloads (you already have two aggregated runs).

### 9.3 Parameter sensitivity
Seconds-Rule depends on \(\lambda\), EWMA smoothing, rebalance interval, and DRAM budget.

Mitigation: report sensitivity for \(\lambda\) or window size if available in your sweep.



## 10. Conclusion (一段话总结你贡献了什么)

This project contributes:

1) A reproducible evaluation pipeline for tiered ANN policies (one-click execution + artifacts).
2) A seconds-scale tiering policy that unifies recency and frequency into an interpretable scoring function.
3) A standardized set of metrics and plots (p95/I/O/migration/frontier/SLA) for comparing policies.

The generated results and figures provide the evidence needed to analyze trade-offs between recall, tail latency, SSD I/O pressure, and migration overhead in hybrid DRAM+SSD vector search.



## References (paper-format)
[1] J. Gray and G. R. Putzolu, “The 5 Minute Rule for Trading Memory for Disk Accesses and The 10 Byte Rule for Trading Memory for CPU Time,” SIGMOD 1987.  
[2] G. Graefe, “(Five-minute rule revisits under flash memory),” DaMoN 2007 (flash memory variant of five-minute rule).  
[3] T. Zhang et al., “From Minutes to Seconds: Redefining the Five-Minute Rule for AI-Era Memory Hierarchies,” arXiv, 2025.  
[4] J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity search with GPUs,” arXiv:1702.08734, 2017.  
[5] H. Jégou, M. Douze, and J. Johnson, “Faiss: A library for efficient similarity search,” Engineering at Meta, 2017.  
[6] Faiss documentation: IVF / IndexIVF reference.  
[7] S. Jayaram Subramanya et al., “DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node,” NeurIPS 2019 (Microsoft Research publication page).  
[8] Q. Chen et al., “SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search,” NeurIPS 2021.



## Appendix A - Submission checklist (给你最后交作业用)
- [ ] Repo contains this report (`FINAL_REPORT.md` or PDF) at root or in `report/`.
- [ ] One-click script runs successfully from a clean environment.
- [ ] `results/logs/` contains the log referenced in Section 1.
- [ ] All 6 plots for the main run are present and referenced by relative paths.
- [ ] Report includes: background, problem statement, design, baselines, evaluation, results, limitations, references.
- [ ] (If required) GitHub link + commit hash recorded in report.



## Appendix B - How to cover *every* source-code file (逐个源代码文件说明的方法)

你在问题里强调“每个阶段、每个文件、每个步骤都要解释”。  
本报告已经对 Report Bundle 中列出的所有产物文件（log/CSV/PNG） 做了逐文件解释（Section 6），但如果你希望把 仓库里的每一个源代码文件 都纳入“论文式附录”，最可靠的方法是让附录 自动从你的真实仓库生成（避免我在聊天里凭空猜文件名）。

### B.1 生成自动附录（1 条命令）
我附带了一个零依赖脚本：`tools_generate_repo_inventory.py`（见本聊天附件/下载）。

把它放到你的 repo 根目录，然后运行：

```bash
python tools_generate_repo_inventory.py
```

它会生成：

- `appendix_repo_inventory.md`

里面是一个 Markdown 表格：逐文件列出路径、类型、以及（对 Python 文件）模块 docstring/顶层函数类名摘要。

### B.2 如何把它并入最终报告
- 方式 1：把 `appendix_repo_inventory.md` 直接作为提交的一部分（推荐，最简单）。
- 方式 2：把该文件内容复制粘贴到你的最终 PDF/Word 报告的 Appendix 部分。

> 这样可以实现你要求的“每个文件都解释”，而且保证 100% 与你真实 repo 对齐。



## Appendix C - Auto-fill “偏差有多大”的关键数字（建议必做）

我另外附带了一个脚本：`tools_summarize_results.py`，它会从你的 `agg/ann_policy_agg.csv` 自动提取：
- 策略数量、DRAM budget 点数
- p95 vs DRAM 的单调性违背率（偏差量化）
- （如果能识别到 All-DRAM）avg gap-to-oracle
- `sla_reachable_recall.csv` 是否存在、以及列名

### C.1 用法（1 条命令）
从 repo root 运行（把 RUN_DIR 换成你要写进报告的那次 run）：

```bash
python tools_summarize_results.py results/tiered_ann_seconds_rule_20251215_193936
```

它会生成：
- `report_snippet.md`（可直接粘贴到你的 Results 里）

> 这一步做完，你的 report 就会从“有图”升级到“有可复现的量化结论”。

