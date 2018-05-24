## （3）主流生成模型的推断

### （3.1）非线性独立主成分NICA

非线性独立主成分Nonlinear independent components analysis NIC 做明确定义的“连续”和“非线性”变换，如果已知隐变量z的概率分布，对于一个”连续、可微、可逆“的变换g，x由g\(z\)生成，那么生成变量x有如下的概率密度表达式：


$$
p_x(x)=p_z(g^{−1}(x))\left|det(\frac{∂g^{−1}(x)}{∂x})\right|
$$


从上面的式子中观察，如果密度函数 $$p_z $$ 是“可直接优化的Tractable”，并且 $$g$$ 逆变换的 $$Jacobian$$ 行列式是“可直接优化的Tractable”，那么密度函数 $$p_x$$ 也是Tractable的。换句话说，如果 $$g$$ 变换被精心设计，那么即便 $$z$$ 的分布相对简单，也会让生成变量 $$x$$ 具有一个相对复杂的分布。

非线性ICA模型的主要缺陷是它对 $$g$$ 的选择做了限制，比如其中的 $$z$$必须和x具有相同的维度，以满足 $$g$$ 的可逆性。如果要求变换 $$g$$ 的限制较少，比如允许$$z$$  有比 $$x$$ 更多的维度，由于 $$p_{model}$$ 无法明确表示，则为“隐式模型”，参考GANs。

### （3.2）完全观察置信网FVBs

完全观察置信网Fully visible belief networks FVBs 是一类模型用链式概率法则分解一个n维向量x的密度到一个一维概率密度乘机的形式：


$$
p_{model}(x)=∏_{i=1}^{n}p_{model}(x^{<i>}\ |\ x^{<1>},...,x^{<i−1>})
$$


FVBs是一系列 DeepMind 生成模型的基础，例如WaveNet。WaveNet能够生成真实的人类语音，它的缺陷在于一次只能生成一个样本，首先是$$x^{<1>}$$，下一个是$$x^{<2>}$$，时间复杂度是$$O(n)$$。WaveNet作为一种现代FVBNs变种，每一个$$x^{<i>}$$的分布都需要一个深度神经网络，因而每一步都需要复杂度较高的计算。此外，这些步骤没有办法并行化，它耗费两分钟的计算来生成一秒钟的音频，因而无法用于交互式对话。GANs被设计并行化地生成所有的 $$x^{<i>}$$，有着更快的解码速度。

**IMAGE3.2.1**

### （3.3）变分自编码VAE

解决一些具有明确概率密度表示，但不具有“直接优化的tractable”的密度函数\(如，隐变量模型不能直接凸优化求解\)，则需要使用近似推断方法。一般地讲，近似推断分为两大类，一类是“确定性近似”如“变分推断Variational Methods”，另一类是“随机近似Stochastic Approximations”，如“马尔科夫链蒙特卡洛方法Markov Chain Monte Carlo Methods”。

变分自编码Variational AutoEncoder使用近似推断技术，VAE构建的生成模型具有如下示意图：

**IMAGE3.3.1**

VAE选择多元高斯分布作为输出概率，生成变量x具有概率密度：


$$
p(x\ |\ z;θ)=N(x\ |\ f(z;θ),σ^2∗I)
$$


其中生成函数 $$f(z;θ)$$ 接受一个多元高斯随机变量，确定了 $$x$$ 的均值，$$σ$$ 是超参数确定了一个对角协方差矩阵。通过积分掉 $$z$$，推断其中的参数 $$\theta$$ 来最大化似然：


$$
∏_{i=1}^{m}p_{model}(x^{(i)}\ |\ θ)=\int\prod_{i=1}^{m}p_{model}(x^{(i)}\ |\ z;θ)\ p_{prior}(z)dz
$$


我们记$$X = \{x^{(1)},...,x^{(n)}\}$$，记$$P_{model}(X\ |\ \theta)=\prod_{i=1}^{m}p_{model}(x^{(i)}\ |\ θ)$$ 和$$P_{model}(X\ |\ z; \theta)=∏_{i=1}^{m}p_{model}(x^{(i)}\ |\ z;θ)$$，则似然表达式变为：


$$
P_{model}(X\ |\ \theta)=∫P_{model}(X\ |\ z;\theta)\ p_{prior}(z)dz
$$


“期望最大化算法Expectation Maximum”是标准的隐变量模型推断方法，将变量$$z$$ 看作是不完全观察变量。这里，由于$$f(z;θ)$$是一个深度生成神经网络，后验概率密度函数$$P_{posterior}(z\ |\ X;\theta)\propto P_{model}(X\ |\ z;\theta)\ p_{prior}(z)$$ 的结构很复杂，在VAE中我们构建 $$z$$ 的后验分布近似函数 $$Q(z\ |\ X;\phi) = \mathcal{N}(\phi_{\mu}(X), \phi_{\Sigma}(X))$$ 来进行推断。总得来说，利用概率结构更为简单的分布来近似结构复杂的分布，对似然函数进行EM推断，这种技术叫作“变分贝叶斯方法Variational Bayesian Methods”。

利用$$Jensen$$ 不等式，似然函数满足下述等式：
$$
\underbrace{log P_{model}(X\ |\ \theta)}_{upper\_bound}\ \geq\ \underbrace{E_{z\sim Q}[log P_{model}(X\ |\ z;\theta)\ p_{prior}(z)] + \mathbb{H}(Q)}_{lower\_bound}
$$
并且满足$$upper\_bound - lower
\_bound = D_{KL}\left(Q(z\ |\ X;\phi)\|P_{posterior}(z\ |\ X;\theta)\right)$$，即




$$



$$

$$
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ lower\_bound = E_{z\sim Q}[log P_{model}(X\ |\ z;\theta)]−D_{KL}(Q(z\ |\ X;\phi)\|p_{prior}(z))
$$


式子的右边是 $$log$$ 似然$$log P_{model}(X\ |\ \theta)$$的下界，也是VAE的基本原理。在VAE中，我们

