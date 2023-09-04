# Adversarial Examples
When machine learning models are attacked by **adversarial examples**, they misclassify examples that are only slightly different from correctly classified examples. 

## The Linear Adversarial Examples
In many problems, the precision of an individual input feature is limited. For digital images, each channel (red, blue, green) of digital images often use only 8 bits per pixel (256 colors in each channel), so they discard all information below $`\frac{1}{255}`$ of the dynamic range (the limits of luminance range that a given digital camera or film can capture). Because the precision is limited, the classifier will not respond differently to an input $x$ than an adversarial input $\tilde{x}=x+\eta$ if the norm of the perturbation $`\| \eta\|`$ is smaller than the precision of features. 

Let $`\|\eta\|_{\infty}\leq\epsilon`$ where $`\epsilon`$ is smaller than the precision of features. Consider the inner product between a weight maxtrix $\mathbf{W}$ and an adversarial example $`\tilde{x}`$:
```math
\mathbf{W}_j\tilde{x}=\mathbf{W}_j x+\mathbf{W}_j\eta.
```
- $\mathbf{W}$: weight matrix
![image alt](https://github.com/levi0206/ML-group-github/blob/f21e5edd05917ff7ab13a68bf84040db145a70b1/image/weight%20matrix.png)
```math
\left[
\begin{matrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23} \\
w_{31} & w_{32} & w_{33} \\
w_{41} & w_{42} & w_{43} 
\end{matrix}
\right]
```
- $\mathbf{W}_j$: weight vector of the current layer to the jth neuon of the next layer, $n$-dimensional row vector
  
To quote the paper,
> We can maximize $`\mathbf{W}\eta`$ subject to the max norm constraint on $\eta$ by assigning $`\eta=\text{sign}(\mathbf{W})`$.

This is a typo. $\eta$ should be $`\eta=\epsilon\text{sign}(\mathbf{W}_j)`$. For example, if
```math
\mathbf{W}_j=(0.1,-2,-3,0.5),
```
then
```math
\text{sign}(\mathbf{W}_j)=(1,-1,-1,1).
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
\mathbf{W}_j\eta & = \mathbf{W}_j\cdot\epsilon\text{sign}(\mathbf{W}_j) \\
                 & = \epsilon\cdot\sum_{j=1}^n |w_{ij}| \\
                 & \approx \epsilon m n
\end{aligned}
```
Many infinitesmall changes to the input that add up to one large change to the output. Thus, a simple linear model can have adversarial examples if its input has sufficient dimensionality.

## Linear Perturbation of Non-linear Models
Linear structures such as LSTM, `ReLU` and maxout networks, which are easier to optimize, are more vulnerable to linear perturbation. For more non-linear models such as sigmoid networks, the behavior of non-saturating regime is close to linearity, and such non-linear models are still vulnerable to linear perturbation.

Let $\theta$ be the parameters of a model, $\mathbf{x}$ the input to the model, $y$ the targets associated with $\mathbf{x}$ (for machine learning tasks that have targets), $`J(\theta,\mathbf{x},y)`$ the loss function. Researchers found that the perturbation 
```math
\eta=\epsilon\text{sign}(\nabla J(\theta,\mathbf{x},y))
```
causes many models to misclassify their input.  

![image alt](https://github.com/levi0206/ML-group-github/blob/aaaeff18338dea0514f6e8fd90715a426ada979e/image/panda%20gibbon%20example.png)
Figure 1: With $`\epsilon=0.07`$, the ImageNet misclassifies a 

## Reference
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [14. Neural Networks, Structure, Weights and Matrices](https://python-course.eu/machine-learning/neural-networks-structure-weights-and-matrices.php)

  
