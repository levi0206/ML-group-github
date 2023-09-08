# Adversarial Examples
When machine learning models are attacked by **adversarial examples**, they misclassify examples that are only slightly different from correctly classified examples. 
![image alt](https://github.com/levi0206/ML-group-github/blob/0f7eda42f618ef8854663d61ca506862cdc57870/image/panda%20gibbon.png)

## The Linear Adversarial Examples
In many problems, the precision of an individual input feature is limited. For digital images, each channel (red, blue, green) of digital images often use only 8 bits per pixel (256 colors in each channel), so they discard all information below $`\frac{1}{255}`$ of the dynamic range (the limits of luminance range that a given digital camera or film can capture). Because the precision is limited, the classifier will not respond differently to an input $x$ than an adversarial input $\tilde{x}=x+\eta$ (column vector) if the norm of the perturbation $`\| \eta\|`$ is smaller than the precision of features. 

Let $`\|\eta\|_{\infty}\leq\epsilon`$ where $`\epsilon`$ is smaller than the precision of features. Consider the inner product between a weight maxtrix $\mathbf{W}$ and an adversarial example $`\tilde{x}`$:
```math
\mathbf{W}^T_j\tilde{x}=\mathbf{W}^T_j x+\mathbf{W}_j^T\eta.
```
- $\mathbf{W}$: weight matrix
- $\mathbf{W}_j$: weight vector of the current layer to the jth neuon of the next layer, **$n$-dimensional column vector**
  
To quote the paper,
> We can maximize $`\mathbf{W}_j\eta`$ subject to the max norm constraint on $\eta$ by assigning $`\eta=\text{sign}(\mathbf{W})`$.

This is a typo. $\eta$ should be $`\eta=\epsilon\text{sign}(\mathbf{W}_j)`$. For example, if
```math
\mathbf{W}_j=(0.1,-2,0,-3,0.5),
```
then
```math
\text{sign}(\mathbf{W}_j)=(1,-1,0,-1,1).
```
- The `sign` function operates on each element of a vector, 1 for a positive value, -1 for a negative value.
- Recall: The $`\|\cdot\|_{\infty}`$ of $v=(v_1,...,v_n)$ is defined by
```math
\|v\|_{\infty}=\max_{1\leq i\leq n} |v_i|
```
- If $`\eta=\text{sign}(\mathbf{W})`$, then $`\|\eta\|=1`$, a contradiction.
Check: $`\|\eta\|=\epsilon\cdot 1=\epsilon`$.

If the average magnitude (absolute value) of each element of $`\mathbf{W}_j`$ is $m$, then the activation will grow by $\epsilon m n$ approximately compared to $`\mathbf{W} x`$ when we choose `ReLU` as our activation. 
```math
\begin{aligned}
\mathbf{W}_j^T\eta & = \mathbf{W}^T_j\cdot\epsilon\text{sign}(\mathbf{W}_j) \\
                 & = \epsilon\cdot\sum_{j=1}^n |w_{ij}| \\
                 & \approx \epsilon m n
\end{aligned}
```
Many infinitesmall changes to the input that add up to one large change to the output. Thus, a simple linear model can have adversarial examples if its input has sufficient dimensionality.

## Linear Perturbation of Non-linear Models
Linear structures such as LSTM, `ReLU` and maxout networks, which are easier to optimize, are more vulnerable to linear perturbation. For more non-linear models such as sigmoid networks, the behavior of non-saturating regime is close to linearity, and such non-linear models are still vulnerable to linear perturbation.

Let $\theta$ be the parameters of a model, $\mathbf{x}$ the input to the model, $y$ the targets associated with $\mathbf{x}$ (for machine learning tasks that have targets), $`J(\theta,\mathbf{x},y)`$ the loss function. Researchers found that the perturbation, called **fast gradient sign method**, 
```math
\eta=\epsilon\text{sign}(\nabla J(\theta,\mathbf{x},y))
```
causes many models to misclassify their input, for example, GoogLeNet.  

![image alt](https://github.com/levi0206/ML-group-github/blob/aaaeff18338dea0514f6e8fd90715a426ada979e/image/panda%20gibbon%20example.png)
Figure 1: With $`\epsilon=0.007`$, the GoogLeNet misclassifies the image. The amount 0.007 corresponds to the magnitude of the smallest bit of an 8 bit image encoding after GoogLeNet’s conversion to real numbers.

## Adversarial Training 
As the title suggests, we could train our model with adversary, which makes it become **resistant to adversarial examples** and **regularizes** the model. It's worth noting that there's slight difference betweenn $L1$ regularization and adversrial training. 

Consider a logistic regression task on which we train our model to recognoze $`y\in\{0,1\}`$ with $`P(y=1)=\sigma(\mathbf{w}x+b)`$, $\sigma$ logistic sigmoid function, and $`P(y=-1)`$ 
```math
\begin{aligned}
P(y=-1) & = 1-\sigma(\mathbf{w}^Tx+b) \\
        & = 1-\frac{1}{1+e^{(-\mathbf{w}^Tx+b)}} \\
        & = \frac{e^{(-\mathbf{w}^Tx+b)}}{1+e^{(-\mathbf{w}^Tx+b)}}
\end{aligned}
```
Since the sigmoid function has the property
```math
1-\sigma(z)=1-1/(1+e^{-z})=e^{-z}/(1+e^{-z})=1/(1+e^z)=\sigma(-z),
```
we have $`P(y=-1)=\sigma(-(\mathbf{w}^Tx+b))`$. 

Suppose we have $m$ training data points $`\{x_i,y_i\}_{i=1}^m`$ and write $`z_i=\mathbf{w}^Tx_i+b_i`$. 

The log of $`P(y_i|x_i)`$ is
```math
\begin{aligned}
\log P(y_i|x_i) & = \log(\frac{1}{1+e^{-z_i}}) \\
                & = \log 1-\log(1+e^{-z_i}) \\ 
                & = -\log(1+e^{-z_i}) \\
\end{aligned}
```

Our goal is to minimize the log likelihood of training data 
```math
\begin{aligned}
\max \mathbb{P}(y_1,...,y_n|x_1,...,x_n) & = \max_{\mathbf{w}} \Pi_{i=1}^m P(y_i|x_i) \\
                                         & = \max_{\mathbf{w}} \sum_{i=1}^m \log(P(y_i|x_i)) \\
                                         & = \min_{\mathbf{w}} -\sum_{i=1}^m \log(P(y_i|x_i)) \\
                                         & = \min_{\mathbf{w}} -\sum_{i=1}^m -\log(1+e^{-z_i}) \\
                                         & = \min_{\mathbf{w}} \sum_{i=1}^m \log(1+e^{-z_i}) \\
                                         & = \min_{\mathbf{w}} \sum_{i=1}^m \log(1+e^{-y_i(\mathbf{w}^Tx_i+b_i)}) \\ 
\end{aligned}
```
Let's introduce a new random variable $Y$ an empirical distribution of the sample. The above minimization is equivalent to
```math
\begin{aligned}
& \min_{\mathbf{w}} \sum_{i=1}^m \log(1+e^{-y_i(\mathbf{w}^Tx_i+b_i)}) \\
& = \min_{\mathbf{w}} m \sum_{i=1}^m \frac{1}{m}\log(1+e^{-y_i(\mathbf{w}^Tx_i+b_i)}) \\
& = \min_{\mathbf{w}} m \sum_{i=1}^m P(Y=x_i)\log(1+e^{-y_i(\mathbf{w}^Tx_i+b_i)}) \\
& = \min_{\mathbf{w}} m \mathbb{E}_{x\sim p_{data}} \log(1+e^{-y_i(\mathbf{w}^Tx_i+b_i)}) \\
& = \min_{\mathbf{w}} \mathbb{E}_{x\sim p_{data}} \log(1+e^{-y_i(\mathbf{w}^Tx_i+b_i)}) \\
& = \min_{\mathbf{w}} \mathbb{E}_{x\sim p_{data}} \zeta(-y_i(\mathbf{w}^Tx_i+b_i)) \\
\end{aligned}
```
where $`\zeta(z)=\log(1+e^z)`$ is the softplus function.

Thus, our training consists of gradient descent on 
```math
\begin{aligned}
& = \mathbb{E}_{\mathbf{x},y\sim p_{data}} \zeta\left(-y(\mathbf{w}^T\mathbf{x}+b)\right). 
\end{aligned}
```
We can derive a simple analytical form for training on the worst-case adversarial perturbation of $\mathbf{w}$ based on fast gradient sign method. The adversarial version of logistic regression is therefore to minimize
```math
\begin{aligned}
& \mathbb{E}_{\mathbf{x},y\sim p_{data}} \zeta\left(-y(-\epsilon\mathbf{w}^T\text{sign}(\mathbf{w})+\mathbf{w}^T\mathbf{x}+b)\right) \\
& = \mathbb{E}_{\mathbf{x},y\sim p_{data}} \zeta\left(-y(-\epsilon\|\mathbf{w}\|_1+\mathbf{w}^T\mathbf{x}+b)\right) \\
& = \mathbb{E}_{\mathbf{x},y\sim p_{data}} \zeta\left(y(\epsilon\|\mathbf{w}\|_1-\mathbf{w}^T\mathbf{x}-b)\right)
\end{aligned}
```
This is somewhat similar to $L^1$ regularization. However, there are some important differences.
$L^1$ penalty is added to the original lost function:
```math
\begin{aligned}
NewLoss & = Loss+\lambda\|w_i\|_1 \\
        & = Loss+\lambda\sum_{i=1}^n |w_i|
\end{aligned}
```

## Reference
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [14. Neural Networks, Structure, Weights and Matrices](https://python-course.eu/machine-learning/neural-networks-structure-weights-and-matrices.php)

  
