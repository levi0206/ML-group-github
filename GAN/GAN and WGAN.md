# Generative Adversarial Networks
GAN and WGAN
## GAN 
### Motivation
- Model-based reinforcement learning
  - In reinforcement learning, the purpose of model is to simulate our environment and generate samples for the training of agent.  
- Simulate possible futures
  - We could perform regression on these simulations to predict the future events or find the optimal strategy.
 
### Framework
There are two roles in GAN: **generator** and **discriminator**. They are usually neural networks respectively.
- Generator:
  - The counterfeiter trying to make fake currency
  - A differentiable function $G:\mathcal{Z}\to\mathcal{X}$ that inputs a latent vector and output a fake sample
  - $\mathcal{Z}$ is the latent space
  - Let $z$ be a random variable in $\mathcal{Z}$ with known distribution $\mathbb{P}_z$.
- Discriminator
  - The police, trying to allow legitimate money and catch counterfeit money
  - A differentiable function $D$ that inputs a fake or real sample and outputs the probability that the sample is **real**

Their interaction can be illustrated as below:
![image alt](https://github.com/levi0206/Machine-Learning-Topics/blob/9d9d5d572ed5370a824958249e3debe9abad41f6/image/GAN%20in%20a%20nutshell.png)

### Loss Functions
Designing the loss functions of generator and discriminator is a critical part of GAN. The inappropriate choice of loss functions would make the convergence of GAN much more difficult. 

Let $\theta^G$ and $\theta^D$ be the parameters of generator and discriminator. Let $J^G( \theta^G,\theta^D)$ and $J^D( \theta^G,\theta^D)$ be the loss functions of generator and discriminator. An interesting fact is that all of the different games designed for GANs so far use the same cost for
the discriminator, $J^D$. They differ only in terms of the cost used for the
generator, $J^G$. (Ian Goodfellow, 2016)

The loss of discriminator usually defined as
```math
J^D(\theta^D,\theta^G) = -\frac{1}{2}\mathbb{E}_{x\sim \mathbb{P}_{data}} \log D(x) -\frac{1}{2}\mathbb{E}_z \log (1-D(G(z))).
```
- $\log$: avoid long precision
- $D(x)$: the probability that $x$ is real
- $1-D(G(z))$: the probaility that $D$ tells $z$ is fake, i.e. $D$ is not deceived by $G$
- If the probability is close to 1, then the loss is small.

GAN itself is a zero-sum game. Thus, we'll naturally define $J^G$ as 
```math
J^G=-J^D.
```
In this case, we can describe GAN by $V(\theta^G,\theta^D)=-J^D(\theta^G,\theta^D)$, and the optimal generator is thus
```math
\theta^{G^*}=\arg\min_{\theta^G}\max_{\theta^D} V(\theta^G,\theta^D).
```
However, if we define the loss of $G$ in this way, we may encounter gradient vanishing. In the beginning of training, the generator is poor and the discriminator can tell fake samples easily, resulting in small gradient. The generator cannot improve efficiently, and the training of GAN may be very slow.

Recall that **the gradient points to the maximum of the function**. With this in mind, we can use another equivalent loss:
```math
J^G=-\mathbb{E}_z[\log D(G(z))].
```
$G(z)$ is the probability that $D$ is deceived, believing that $G(z)$ is a real sample. When $D$ can tell fake samples with high confidence (high probability), then the probability $D(G(z))$ is small and the loss $J^G$ is large. The gradient is also larger than before if $G$ is poor. 

### Training
The general training algorithm is:
![image alt](https://github.com/levi0206/ML-group-github/blob/88f793b2eddaeca829bd4e0d57c3e371001c5507/image/general%20GAN%20training.png)
(Replace $L_G$ and $L_D$ by $J^G$ and $J^D$)

We have to calucate the gradient of loss to update our parameters. However, it's intractable to calculate the expectation. It would be simpler if we can **take the gradient first and then take the expectation**. This operation is achievable and the idea can start from this famous calculus theorem:

**Theorem (Leibnitz integral rule)** If $a(x),b(x),f(x,y)$ are $C^1$, then 
```math
\frac{d}{dx}\int_{a(x)}^{b(x)} f(x,y)dy=f(x,b(x))b'(x)-f(x,a(x))a'(x)+\int_{a(x)}^{b(x)} \frac{\partial }{\partial x}f(x,y)dy.
```
If $a(x),b(x)$ are constant, then 
```math
\frac{d}{dx}\int_{a(x)}^{b(x)} f(x,y)dy=\frac{\partial }{\partial x}f(x,y)dy.
```
Expectation is an integral. We could expect that we can replace $y$ by a random variable $Y$ under some conditions. Indeed, such theorem exists.

**Theorem** Let $X$ be a random variable, $g\colon \mathbb{R}\times\Omega\longrightarrow \mathbb{R}$ a function such that $g(x,Y)$ is integrable for all $x$ and $g$ is continuously differentiable w.r.t. $x$. Assume that there is a random variable $Z$ such that $\frac{\partial}{\partial x}g(x,Y)\leq Z$ a.s. for all $x$ and $\mathbb{E}Z<\infty$. Then 
```math
\frac{d}{dx}\mathbb{E}[g(x,X)]=\mathbb{E}[\frac{\partial}{\partial x}g(x,X)].
```
You can replace $x$ by parameters $\theta$. There's a more general theorem with similar form of the conditions, but we're not going to mention it.

Thus, we have
```math
\begin{aligned}
    \nabla J^{(D)}(\theta^{(D)},\theta^{(G)}) & = \nabla\left(-\frac{1}{2}\mathbb{E}_{x\sim \mathbb{P}_{data}} \log D(x) -\frac{1}{2}\mathbb{E}_z \log (1-D(G(z)))\right) \\
    & = -\frac{1}{2}\mathbb{E}_{x\sim \mathbb{P}_{data}} \nabla\log D(x)-\frac{1}{2}\mathbb{E}_z \nabla\log (1-D(G(z)))
\end{aligned}
```
and 
```math
\begin{aligned}
    \nabla J^{(G)} & = \nabla\left(-\frac{1}{2}\mathbb{E}_z \log D(G(z))\right)\\
    & = -\frac{1}{2}\mathbb{E}_z \nabla\log D(G(z))
\end{aligned}
```
$\nabla \log D(x)$ and $\nabla \log D(G(z))$ can be calculated with **backpropagation**, and each expectation can be estimated using **Monte Carlo estimation**.

The more detained training algorithm is like:
![image alt](https://github.com/levi0206/ML-group-github/blob/d0f1833e2087c95d5c57e05e50e1af0084977642/image/GAN%20training%20algorithm.png)
After calculating the gradient of loss, we can update parameters by applying different optizers such as Adam, RMSProp and so on. 

## WGAN
### Likelihood and KL Divergence
Let's see generative models in a different way. For generative models, we want to
- maximize the **likelihood** of our training data, that is, to generate training data as possible.
Or we can say we want to
- minimize the **distance** of the two distributions.

**What is likelihood?**

Consider the senario that 
- there are 10 people
- the probabiity that one gets infected is $\theta$
- each person is independent.

Then the probability of **4 people get infected** is
```math
\mathbb{P}(X=4|\theta)= {10\choose 4}\theta^4(1-\theta)^6.
```
**What is the $\theta$ that maximizes the probability?**

Likelihood,
```math
L(\theta|X=4)=\mathbb{P}(X=4|\theta),
```
is **a function of $\theta$**, **determining the probability of the observation ($X$=4)**. Once $\theta$ is chosen, the probability is determined. Thus, the **maximum likelihood** is 
```math
\arg\max_{\theta} L(\theta|X=4).
```
For generative models, the **training data** $x^1,x^2,...,x^m$ is our observation. We want to maximize the likelihood of our training data. Likelihood here is **the probability that the model assigns to the training data** $\Pi_{i=1}^m \mathbb{P}_{model}(x^i;\theta)$. 
```math
\begin{aligned}
\theta^* & = \arg\max_{\theta}\Pi_{i=1}^m \mathbb{P}_{model}(x^i;\theta) \\
    & = \arg\max_{\theta} \Sigma_{i=1}^m \log \mathbb{P}_{model}(x^i;\theta).
\end{aligned}
```
The **divergence** can be intuitively thought as the "distance" of two probability distribution. **KL-divergence** is defined by
```math
\begin{aligned}
KL(p||q)& \dot{=}\int_{-\infty}^{\infty} p(x)\log\frac{p(x)}{q(x)}d\mu(x)\\
    & = \mathbb{E}_{x\sim p(x)} \log\frac{p(x)}{q(x)}.
\end{aligned}
```
**What's the relation between likelihood and KL divrgence**? Let's look at the proposition:

**Proposition** Minimizing KL divergence is equivalent to maximizing likelihood.
```math
\begin{aligned}
    & \arg\min_{\hat{\theta}} KL(p(x|\theta^*) || p(x|\hat{\theta})) \\
    & = \arg\min_{\hat{\theta}} \mathbb{E}_{x\sim p(x|\theta^*)} \left[\log\frac{p(x|\theta^*)}{p(x|\hat{\theta})}\right] \\
    & = \arg\min_{\hat{\theta}} \mathbb{E}_{x\sim p(x|\theta^*)}\left[\log p(x|\theta^*)-\log p(x|\hat{\theta})\right] \\
    & = \arg\min_{\hat{\theta}} \mathbb{E}_{x\sim p(x|\theta^*)}\left[-\log p(x|\hat{\theta})\right] \\
    & = \arg\max_{\hat{\theta}} \mathbb{E}_{x\sim p(x|\theta^*)}\left[\log p(x|\hat{\theta})\right]
\end{aligned}
```

The optimal parameters $\theta^\*$ is independent of the $\hat{\theta}$, so the equation holds if we get rid of $\log p(x|\theta^*)$. 

### Motivation
We've shown that minimizing KL divergence is equivalent to maximizing likelihood. Can we simply calculate KL divergence, calculate the gradient of KL divergence, and then update model parameters with gradient? The answer is no if we use KL divergence. 

Let $\mathcal{X}$ be a compact metric space such as $`[0,1]^d`$. In the following discussion, we only consider the probability measures defined on $\mathcal{X}$. We can view $\mathcal{X}$ as the sample space.

**Theorem** A sequence of distributions $`\mathbb{P}_t`$ converges with respect to $\rho$ if and only if there exists a distribution $\mathbb{P}_{\infty}$ such that $`\rho(\mathbb{P}_t,\mathbb{P}_{\infty})\to 0`$ as $t\to\infty$. 

Does such divergence exist? Yes, **Earth-Mover** (EM) distance or **Wasserstein-1** ($W_1$) distance has silimar property. $W_1$ distance is defined as
```math
W(\mathbb{P}_r,\mathbb{P}_g)=\inf_{\gamma\sim\Pi(\mathbb{P}_r,\mathbb{P}_g)} \mathbb{E}_{(x,y)\sim\gamma}\left[||x-y||\right].
```
where $\Pi(\mathbb{P}_r,\mathbb{P}_g)$ denotes the set of all joint distributions $\gamma(x,y)$ whose marginals are respectively $\mathbb{P}_r$ and $\mathbb{P}_g$, that is, 
```math
\begin{aligned}
& \int_{\mathcal{X}} \gamma(x,y)dy=\mathbb{P}_g \\
& \int_{\mathcal{X}} \gamma(x,y)dx=\mathbb{P}_r.
\end{aligned}
```
Intuitively, $\gamma(x,y)$ indicates how much “mass” must be transported from $x$ to $y$ in order to transform the distributions $\mathbb{P}_r$ into the distribution $\mathbb{P}_g$. The EM distance then is the “**cost**” of the optimal transport plan.

We'll compare $W_1$ distance with other popular distances. 

**KL divergence**
  ```math
  KL(\mathbb{P}_r || \mathbb{P}_g)=\int \log\frac{P_r(x)}{P_g(x)}P_r(x)d\mu(x)
  ```
where both $\mathbb{P}_r$ and $\mathbb{P}_g$ are assumed to be absolutely continuous, and therefore admit **densities**, with respect to a same measure $\mu$ defined on $\mathcal{X}$. 
  
**Jensen-Shannon** (JS) divergence
  ```math
  JS(\mathbb{P}_r,\mathbb{P}_g)=KL(\mathbb{P}_r||\mathbb{P}_m)+KL(\mathbb{P}_g||\mathbb{P}_m)
  ```
where $`\mathbb{P}_m=\frac{\mathbb{P}_r+\mathbb{P}_g}{2}`$.
  
**Total Variation** (TV) distance
  ```math
  \delta(\mathbb{P}_r,\mathbb{P}_g)=\sup_{A\in\Sigma} |\mathbb{P}_r(A)-\mathbb{P}_g(A)|
  ```
  $\Sigma$ denotes the set of all the Borel subsets of a compact metric set $\mathcal{X}$.

**Example** 
Let $Z\sim U[0,1]$ the uniform distribution on the unit interval. Let $`\mathbb{P}_0`$ be the distribution of $(0,Z) \in \mathbb{R}^2$ (a $0$ on the x-axis and the random variable $Z$ on the y-axis), uniform on a straight vertical line passing through the origin. Let $g_{\theta}(z) = (\theta,z)$ with $\theta$ a single real parameter. Let $`\mathbb{P}_{\theta}`$ be the distribution of $`g_{\theta}`$. In this case,


$W_1$ distance: $`W(\mathbb{P}_0,\mathbb{P}_{\theta})=|\theta|`$

**Proof**: Let $`x\sim\mathbb{P}_0`$, $`y\sim\mathbb{P}_{\theta}`$ and $`\gamma`$ be a joint distribution of $`\mathbb{P}_0`$ and $`\mathbb{P}_{\theta}`$. No matter what $\gamma$ is, $`\|x-y\|`$ is always $`\theta`$. $`\theta`$ is a constant with respect to $\gamma$, and the expectation of a probability distribution is always 1. Thus,
```math
\mathbb{E}_{(x,y)\sim\gamma} |\theta|=\int_{\omega\in\mathcal{X}} |\theta| d\omega = |\theta| \int_{\omega\in\mathcal{X}}d\omega=\|\theta\|\blacksquare
```


KL divergence:
```math
KL(\mathbb{P}_0||\mathbb{P}_{\theta})=KL(\mathbb{P}_{\theta}||\mathbb{P}_0)
    \begin{cases}
        \infty & \text{if }\theta\neq 0 \\
        0 & \theta=0
    \end{cases}
```
**Proof**: 
```math
\begin{aligned}
KL(\mathbb{P}_0||\mathbb{P}_{\theta}) & =\int_{\omega\in\mathcal{X}} \mathbb{P}_0(\omega)\log\frac{\mathbb{P}_0(\omega)}{\mathbb{P}_{\theta}(\omega)}=\int_{(x,y)\in\mathbb{R}^2} \mathbb{P}_0(x,y)\log\frac{\mathbb{P}_0(x,y)}{\mathbb{P}_{\theta}(x,y)} \\ & = \int_{(x,y)\in [0,\theta]\times [0,1]} \mathbb{P}_0(x,y)\log\frac{\mathbb{P}_0(x,y)}{\mathbb{P}_{\theta}(x,y)} = \int_{y\in[0,1]} \mathbb{P}_0(0,y)\log\frac{\mathbb{P}_0(0,y)}{\mathbb{P}_{\theta}(0,y)}+\int_{y\in[0,1]} \mathbb{P}_0(\theta,y)\log\frac{\mathbb{P}_0(\theta,y)}{\mathbb{P}_{\theta}(\theta,y)}
\end{aligned}
```
If $\theta\neq 0$, $`\mathbb{P}_{\theta}(0,y)`$ and $\mathbb{P}_0(\theta,y)$ are always 0 for all $y\in[0,1]$. The first term is arbitrarily large (infinity) and the second term is 0, so KL divergence is $`+\infty`$.

For $\theta=0$, 
```math
KL(\mathbb{P}_0||\mathbb{P}_{\theta})=\int_{y\in[0,1]} \mathbb{P}_0(0,y)\log\frac{\mathbb{P}_0(0,y)}{\mathbb{P}_{\theta}(0,y)}=\int_{y\in[0,1]} \mathbb{P}_0(0,y)\log 1 = 0.
```
Similar calculation for $`KL(\mathbb{P}_{\theta}||\mathbb{P}_0)\blacksquare`$.

JS divergence:
```math
JS(\mathbb{P}_0,\mathbb{P}_{\theta})=
        \begin{cases}
        \log 2 & \text{if }\theta\neq 0 \\
        0 & \theta=0
        \end{cases}
```
**Proof**: 
```math
  JS(\mathbb{P}_r,\mathbb{P}_g)=KL(\mathbb{P}_r||\mathbb{P}_m)+KL(\mathbb{P}_g||\mathbb{P}_m)
  ```
For $`\theta=0`$, $`\frac{\mathbb{P}_0+\mathbb{P}_{\theta}}{2}=\mathbb{P}_0`$, so JS divergence is 0. Suppose $\theta\neq 0$. 
```math
\begin{aligned}
& \int_{y\in[0,1]} \mathbb{P}_0(0,y)\log\frac{\mathbb{P}_0(0,y)}{\frac{\mathbb{P}_0(0,y)+\mathbb{P}_{\theta}(0,y)}{2}} \\
& = \int_{y\in[0,1]} \mathbb{P}_0(0,y)\log\frac{\mathbb{P}_0(0,y)}{\frac{\mathbb{P}_0(0,y)+0}{2}} \\
& = \int_{y\in[0,1]} \mathbb{P}_0(0,y)\log 2 \\
& = \log 2,
\end{aligned}
```
and the calculation is similar for $`KL(\mathbb{P}_{\theta} || \frac{\mathbb{P}_0+\mathbb{P}_{\theta}}{2})\blacksquare`$.

TV distance:
```math
\delta(\mathbb{P}_0,\mathbb{P}_{\theta})=
        \begin{cases}
        1 & \text{if }\theta\neq 0 \\
        0 & \theta=0
        \end{cases}
```
**Proof**: Recall that 
```math
\mathbb{P}_r(A)=\int_A \mathbb{P}_r(x)d\mu(x).
```
$`\mathbb{P}_0`$ and $`\mathbb{P}_{\theta}`$ are defined on $`\mathbb{R}^2`$ and are non-zero on $x=0$ and $x=\theta$ respectively. For each $y\in\mathbb{R}$, $`\{(x,y) | x\in\mathbb{R}\}`$ is a Borel subset. If $\theta=0$, then $`\mathbb{P}_0=\mathbb{P}_{\theta}`$ and $`\delta(\mathbb{P}_0,\mathbb{P}_{\theta})=0`$. For $\theta\neq 0$, 
```math
\delta(\mathbb{P}_0,\mathbb{P}_{\theta})_{x=0}=|1-0|=1,
```
and 
```math
\delta(\mathbb{P}_0,\mathbb{P}_{\theta})_{x=\theta}=|0-1|=1. \blacksquare
```
What can we learn from this example? We know that we can learn a probability distribution over a low dimensional manifold by doing gradient descent on the $`W_1`$ distance. This cannot be done with the other distances and divergences because the resulting loss function is not even continuous.

**Theorem** Let $`\mathbb{P}_r`$ be a fixed distribution over $`\mathcal{X}`$. Let $Z$ be a random variable (e.g Gaussian) over another space $\mathcal{Z}$. Let $g : \mathcal{Z} × \mathbb{R}^d \to \mathcal{X}$ be a function, that will be denoted $g_{\theta}(z)$ with $z$ the first coordinate and $\theta$ the second. Let $`\mathbb{P}_{\theta}`$ denote the distribution of $`g_{\theta}(Z)`$. Then,
1. $g$ is continuous in $`\theta`$, and so is $`W(\mathbb{P}_0,\mathbb{P}_{\theta})`$.
2. If $g$ is locally Lipschitz and satisfies regularity assumption 1, then $`W(\mathbb{P}_0,\mathbb{P}_{\theta})`$
is continuous everywhere, and differentiable almost everywhere.
3. Statements 1-2 are false for the Jensen-Shannon divergence $`JS(\mathbb{P}_0,\mathbb{P}_{\theta})`$ and all the KLs.

**Corollary** Let $`g_{\theta}`$ be any feedforward neural network4 parameterized by $`\theta`$, and $`p(z)`$ a prior over $z$ such that $`\mathbb{E}_{z\sim p(z)}[\|z\|]<\infty`$ (e.g. Gaussian, uniform, etc.). Then assumption 1 is satisfied and therefore $`W(\mathbb{P}_0,\mathbb{P}_{\theta})`$ is continuous everywhere and differentiable almost everywhere.

### WGAN
By the Kantorovich-Rubinstein duality,
```math
    W(\mathbb{P}_r,\mathbb{P}_{\theta}) = \sup_{||f||_L\leq 1} \mathbb{E}_{x\sim\mathbb{P}_r}[f(x)]-\mathbb{E}_{x\sim\mathbb{P}_{\theta}}[f(x)].
```
If we replace $`\|f\|_L\leq 1$ by $\|f\|_L\leq K`$, then we end up with $`K\cdot W(\mathbb{P}_r,\mathbb{P}_{\theta})`$. The gradient is scaled but its direction does not change. Let $f_w$ denote the neural network with parameters $w$. We could consider solving
```math
    \max_{w\in\mathcal{W}} \mathbb{E}_{x\sim\mathbb{P}_r}[f_w(x)]-\mathbb{E}_{z\sim \mathbb{P}_z}[f_w(g_{\theta}(z))]
```
where we restrict the range of parameters. For example, we restrict $\mathcal{W}\in [-0.01,0.01]^l$ so that $\frac{\partial f_w}{\partial w_i}$ are bounded and the gradient is Lipschitz. If the supremum is attained for some $w\in\mathcal{W}$, then this process would yield a calculation of $`W(\mathbb{P}_0,\mathbb{P}_{\theta})`$ up to a multiplicative constant.

***Theorem** Let $`\mathbb{P}_r`$ be any distribution. Let $`\mathbb{P}_{\theta}`$ be the distribution of $`g_{\theta}(Z)`$ with $Z$ a random variable with density $p$ and $`g_{\theta}`$ a function satisfying assumption 1. Then, there is a solution $`f \colon \mathcal{X} \to \mathbb{R}`$ to the problem
```math
\max_{||f||_L\leq 1} \mathbb{E}_{x\sim\mathbb{P}_r}[f(x)]-\mathbb{E}_{x\sim\mathbb{P}_{\theta}}[f(x)]
```
and we have
```math
\nabla_{\theta} W(\mathbb{P}_r,\mathbb{P}_{\theta})=-\mathbb{E}_{z\sim \mathbb{P}_z}[f_w(g_{\theta}(z))]
```
when both terms are well-defined.

### Training
This is the training algorithm in WGAN paper:
![image alt](https://github.com/levi0206/ML-group-github/blob/df741c3178be1fde18a5a00c3e1c6e2bb9ec6f46/image/WGAN%20training.png)
Note that
- we clip the domain of parameters to enforce Lipchitz continuity
- there's no $\log$ in loss functions
- the output layer is `Linear` since the discriminator now is doing **regression** on two distances, not logistic regression. 

### Comparison
![image alt](https://github.com/levi0206/ML-group-github/blob/1594689990bcf9ac1b799a64713950fa80f38157/image/WGAN%20figure5.png)
Algorithms trained with a DCGAN generator. Left: WGAN algorithm. Right: standard GAN formulation. Both algorithms produce high quality samples.
![image alt](https://github.com/levi0206/ML-group-github/blob/1594689990bcf9ac1b799a64713950fa80f38157/image/WGAN%20figure6.png)
 Algorithms trained with a generator without batch normalization and constant number of filters at every layer. Standard GAN failed to learn while the WGAN still was able to produce samples.
 ![image alt](https://github.com/levi0206/ML-group-github/blob/1594689990bcf9ac1b799a64713950fa80f38157/image/WGAN%20figure7.png)
 Algorithms trained with an MLP generator with 4 layers and 512 units with ReLU nonlinearities. WGAN: lower quality, GAN: worse quality and mode collapse
