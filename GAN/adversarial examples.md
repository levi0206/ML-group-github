# Adversarial Examples
When machine learning models are attacked by **adversarial examples**, they misclassify examples that are only slightly different from correctly classified examples. 

## The Linear Adversarial Examples
In many problems, the precision of an individual input feature is limited. For digital images, each channel (red, blue, green) of digital images often use only 8 bits per pixel (256 colors in each channel), so they discard all information below $`\frac{1}{255}`$ of the dynamic range (the limits of luminance range that a given digital camera or film can capture). Because the precision is limited, the classifier will not respond differently to an input $x$ than an adversarial input $\tilde{x}=x+\eta$ if the norm of the perturbation $`\| \eta\|`$ is smaller than the precision of features. 

Let $`\|\eta\|_{\infty}\leq\epsilon`$ where $`\epsilon`$ is smaller than the precision of features. Consider the dot product between a weight vector $\mathbf{W}_j$ and an adversarial example $`\tilde{x}`$:
```math
\mathbf{W}^T_j\tilde{x}=\mathbf{W}^T_j x+\mathbf{W}^T_j\eta.
```
- $\mathbf{W}$: weight matrix
- $`\mathbf{W}_j`$: weight vector of the current layer and the jth neuron in the next layer, $n$-dimensional
  
To quote the paper,
> We can maximize $`\mathbf{W}^T_j\eta`$ subject to the max norm constraint on $\eta$ by assigning $`\eta=\text{sign}(\mathbf{w})`$.

I think this $`\eta`$ should be like $`(\epsilon, -\epsilon, \epsilon, \epsilon, -\epsilon,...)`$, not $`(1,-1,1,1,-1,...)`$.
- The `sign` function operates on each element of $\mathbf{W}^T_j$, 1 for a positive value, -1 for a negative value.
- Recall: The $`\|\cdot\|_{\infty}`$ of $v=(x_1,...,x_n)$ is defined by
```math
\|v\|_{\infty}=\max_{i\in\{1,...,n\}} |x_i|.
```
- $`\|(1,-1,1,1,...)\|_{\infty}=1`$, greater than $`\epsilon`$, contradicting our assumption.

If the average of the absolute value of each element of $`\mathbf{W}_j`$ is $m$, then the activation will grow by $\epsilon m n$ approximately compared to $`\mathbf{W}^T_j x`$ when we choose `ReLU` as our activation. Many infinitesmall changes to the input that add up to one large change to the output. Thus, a simple linear model can have adversarial examples if its input has sufficient dimensionality.

## Linear Perturbation of Non-linear Models
Linear structures such as LSTM, `ReLU` and maxout networks, which are easier to optimize, are more vulnerable to linear perturbation. For more non-linear models such as sigmoid networks, the behavior of non-saturating regime is close to linearity, and such non-linear models are still vulnerable to linear perturbation.

Let $\theta$ be the parameters of a model, $\mathbf{x}$ the input to the model, $y$ the targets associated with $\mathbf{x}$ (for machine learning tasks that have targets), $`J(\theta,\mathbf{x},y)`$ the loss function. Researchers found that the perturbation 
```math
\eta=\epsilon\text{sign}(\nabla J(\theta,\mathbf{x},y))
```
causes many models to misclassify their input.  

![image alt](https://github.com/levi0206/ML-group-github/blob/aaaeff18338dea0514f6e8fd90715a426ada979e/image/panda%20gibbon%20example.png)
Figure 1: With $`\epsilon=0.07`$, the ImageNet misclassifies a 
