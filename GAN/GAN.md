# Generative Adversarial Networks
GAN and WGAN
## Motivation
- Model-based reinforcement learning
  - In reinforcement learning, the purpose of model is to simulate our environment and generate samples for the training of agent.  
- Simulate possible futures
  - We could perform regression on these simulations to predict the future events or find the optimal strategy.
 
## Framework
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

## Loss Functions
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

## Training
The general training algorithm is as below:
![image alt](https://github.com/levi0206/ML-group-github/blob/88f793b2eddaeca829bd4e0d57c3e371001c5507/image/general%20GAN%20training.png)
(Replace $L_G$ and $L_D$ by $J^G$ and $J^D$)

We have to calucate the gradient of loss to update our parameters. However, it's intractable to calculate the expectation; calculating the gradient is easier.  
