# Regression
HW1 source: [Hung-Yi Lee ML course 2023](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)

## Preface
My first deep learning task is COVID-19 Cases Prediction with deep neural network (DNN). Given survey results in the **past 3 days** in a specific state in U.S., then predict the percentage of **new tested positive cases in the 3rd day**. Even in this fundamental DL task, I stuck in many problems such as neural network architecture, the choice of optimizer, tuning hyperparameter. As a record, I write down what I learned in this notes.

## What is regression?
Regression is a supervised learning problem. Given a data set $`D=(x_i,y_i)_{i=1}^n`$ where $`x_i=(x^{(num)},x^{(cat)})\in\mathbb{X}`$ represents numerical $`x^{(num)}`$ and categorical $`x^{(cat)}`$ features and $`y_i\in\mathbb{Y}`$ denotes the corresponding object label. $`D`$ is split into three disjoint subsets: $`D\cup D_{train}\cup D_{val}\cup D_{test}`$. Regression can be viewed as a classification problem with $`\mathbb{Y}=\mathbb{R}`$.

## Neural Network Architecture
To solve a regression problem, people usually use MLP, a very simple model. A multilayer perceptron with three layers is like:
```
nn.Sequential(
            nn.Linear(input_dim,a),
            nn.ReLU(),
            nn.Linear(a, b),
            nn.ReLU(),
            nn.Linear(b, 1),
)
```
The first layer is the input layer where the parameter `in_features` must be equal to the dimension of the input vector. The last layer is the output layer. Our task is to output a real value, so `out_features` is 1. It's worth being noted that no activation function is put after the output layer since the output could be any real number and we don't want to restrict the output in some range.



## Optimizer
The choice of optimizer may effects the convergence rate and the exploration of local minimum. Which optimizer is the best one? There are many studies and Stackoverflow posts about this question based on different settings. However, I think the lastest algorithms such as **Adam**, **Adagrad**, **RMSProp** are generally (but not absolutely) better since they usually report results on standard datasets and may beat previous algorithms.

**SGD** (Stochastic gradient descent) is the most basic optimzer for model optimization, but it's seldem used now. One reason is that it's learning rate is fixed. 
> Hence it doesn't work well when the parameters are in different scales since a low learning rate will make the learning slow while a large learning rate might lead to oscillations.

Moreover, it's generally considered having a hard time to escape saddle points. Probably this is the reason why people avoid using SGD nowadays. 

To improve SGD, we could use **momentum** and **nesterov** acceleration together with SGD. Vanilla SGD updates parameters by
```math
\theta_{t+1}=\theta_k-\eta\nabla Loss(\theta_k)
```
SGD together with momentum could be like
```math
\begin{aligned}
& \theta_{t+1}=\gamma\theta_k-v_t \\
& v_t=\gamma v_{t-1}+\eta\nabla Loss(\theta_t),\quad \gamma\in[0,1]
\end{aligned}
```
We usually set the momentum term 0.9.

Intuitively, omentum somehow reserve the previous updating direction by multiplying a constant. As the formula indicates, momentum can amplify the spped of correct direction and slow down the wrong direction. 
In practice, SGD with momentum indeed reduce oscillation and stabilize convergence. To accelerate optimization, we usually set momentum large. However, when we're close to the local minima, our momentum is still large and don't know it should stop. This may cause the algorithm to miss the local minima. 

>Essentially, when using momentum, we push a ball down a hill. The ball accumulates momentum as it rolls downhill, becoming faster and faster on the way.

![image alt](https://github.com/levi0206/Deep_Learning_Notes/blob/3919a90b0a32a10db7382bebaf30bb7d252c429e/image/SGD%20with%3Awithout%20momentum.png)

To find a local minima, it's dangerous to let the ball roll straightly down a hill. It could be better if there's another ball knows slowing down before the hill slopes up again. This smarter ball is called Nesterov accelerated gradient (NAG).

```math
\begin{aligned}
& \theta_{t+1}=\gamma\theta_k-v_t \\
& v_t=\gamma v_{t-1}+\eta\nabla Loss(\theta_t-\gamma v_{t-1}),\quad \gamma\in[0,1]
\end{aligned}
```
![image alt](https://github.com/levi0206/Deep_Learning_Notes/blob/6219bbc24054903186dd42520cdc771e30fbcc69/image/Nesterov%20update.png)
Again the momentum term is usually set to 0.9. Momentum first computes the current gradient (small blue vector $`\nabla Loss(\theta_t)`$) and then takes a big jump in the direction of the updated accumulated gradient (big blue vector $`\gamma v_{t-1}`$). NAG on the otherhand first makes a big jump in the direction of the previous accumulated gradient (brown vector $`\gamma v_{t-1}`$), measures the gradient and then makes a correction (green vector $`\nabla Loss(\theta_t-\gamma v_{t-1})`$).

**RMSProp** is my choice is this regression task. Let $`g_t`$ denote the gradient of loss at time $`t`$. RMSProp is simply
```math
\begin{aligned}
& \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{E[g^2]_t+\epsilon}}g_t \\
& E[g^2]_t=\gamma E[g^2]_{t-1}+(1-\gamma)g_t^2.
\end{aligned}
```
$`\gamma`$ is suggested to be 0.9, and $`\eta`$ is suggested to be 0.001. RMSprop divides the learning rate by an exponentially decaying average of squared gradients. 

This is a comparison of SGD with/without momentum and RMSProp made by myself when I first know people think RMSSProp is generally better than SGD. This is not a rigorous test, just a quick check whether it holds true for me. 
### Parameter

Neural network: 
```
self.layers = nn.Sequential(
            nn.Linear(input_dim, 56),
            nn.ReLU(),
            nn.Linear(56, 16),
            nn.ReLU(),
            nn.Linear(16,1),
        )
```

Optimizer parameters: `momentum=0.9`, `weight_decay=1e-5`, `lr=1e-5`
- Purple: SGD+momentum
- Green: SGD+momentum+NAG
- Orange: RMSProp
All the other hyperparameter are the same. It turns out that RMSProp eventually has lower loss, and the green one outperforms the purple one. 

![image alt](https://github.com/levi0206/Deep_Learning_Notes/blob/c910202f7426eade11c003fe77640cdc8b466ead/image/SGD%20vs%20RMSProp.png)


## References
[Stackoverflow: Gradient Descent vs Adagrad vs Momentum in TensorFlow](https://stackoverflow.com/questions/36162180/gradient-descent-vs-adagrad-vs-momentum-in-tensorflow)
[Stack Exchange: Guidelines for selecting an optimizer for training neural networks](https://datascience.stackexchange.com/questions/10523/guidelines-for-selecting-an-optimizer-for-training-neural-networks)
[An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747), Sebastian Ruder
