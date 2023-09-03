# Adversarial Examples
When machine learning models are attacked by **adversarial examples**, they misclassify examples that are only slightly different from correctly classified examples. 
![image alt](https://github.com/levi0206/ML-group-github/blob/aaaeff18338dea0514f6e8fd90715a426ada979e/image/panda%20gibbon%20example.png)

## The Linear Adversarial Examples
In many problems, the precision of an individual input feature is limited. For digital images, each channel (red, blue, green) of digital images often use only 8 bits per pixel (256 colors in each channel), so they discard all information below $`\frac{1}{255}`$ of the dynamic range (the limits of luminance range that a given digital camera or film can capture). Because the precision is limited, the classifier will not respond differently to an input $x$ than an adversarial input $\tilde{x}=x+\eta$ if the norm of the perturbation $`\| \eta\|`$ is smaller than the precision of features. 

Let $`\|\eta\|_{\infty}<\epsilon`$ where $`\epsilon`$ is smaller than the precision of features. Consider the dot product between a weight vector $\mathbf{W}_j$ and an adversarial example $`\tilde{x}`$:
```math
\mathbf{W}^T_j\tilde{x}=\mathbf{W}^T_j x+\mathbf{W}^T_j\eta.
```
- $\mathbf{W}$: weight matrix
- $`\mathbf{W}_j`$: weight vector of the current layer and the jth neuron in the next layer, $n$-dimensional
  
If average of the absolute value of each element of $`\mathbf{W}_j`$ is $m$, then the activation will grow by $\epsilon m n$ approximately compared to $`\mathbf{W}^T_j x`$ when we choose `ReLU` as our activation. Many infinitesmall changes to the input that add up to one large change to the output.
