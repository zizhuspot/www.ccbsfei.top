---
title: fm Factorization Machine - Rendle 
date: 2023-09-09 10:01:00
categories:
  - 排序模型
tags:
  - 多任务模型 
  - fm
  - Recommender systems
description: The FM model (Factorization Machine) is an improvement over Poly2 to better handle data sparsity issues. Additionally, the FM model employs matrix factorization techniques, allowing it to train models with near-linear time complexity efficiency 
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_200255.png
---



## fm model 

The FM model (Factorization Machine) is an improvement over Poly2 to better handle data sparsity issues. Additionally, the FM model employs matrix factorization techniques, allowing it to train models with near-linear time complexity efficiency. For each feature, the FM model introduces a hidden vector of dimensions \(k\), where \(k\) is much smaller than the number of features. As a result, the model equation for the FM model is as follows:

\[ \hat{y}(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j \]

In this equation:

- \(\hat{y}(x)\) represents the predicted output.
- \(w_0\) is the bias term.
- \(w_i\) represents the weight associated with the \(i\)-th feature \(x_i\).
- \(v_i\) represents the \(i\)-th feature's hidden vector of dimension \(k\).
- \(n\) is the total number of features.

The key innovation of the FM model lies in the second summation term, which captures second-order feature interactions efficiently through inner products of feature vectors. This allows the model to capture pairwise interactions between features without the need to explicitly enumerate all possible feature pairs, making it suitable for handling sparse data efficiently.



## FM (Factorization Machine) model's principle

The FM model aims to approximate the interaction between features in a second-order polynomial form. For a given input instance \(x\), the prediction \(\hat{y}(x)\) in FM is defined as:

\[ \hat{y}(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j \]

Here's the breakdown of the terms:

- \(w_0\) is the bias term.
- \(w_i\) represents the weight associated with the \(i\)-th feature \(x_i\).
- \(v_i\) represents the latent vector associated with the \(i\)-th feature, with \(d\) dimensions (where \(d\) is the dimensionality of the latent vectors).
- \(x_i\) is the \(i\)-th feature's value.

Now, let's focus on the second-order interaction term:

\[ \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j \]

This term captures the pairwise interactions between features. The inner product \(\langle v_i, v_j \rangle\) represents the interaction strength between the latent vectors \(v_i\) and \(v_j\), and \(x_i x_j\) captures the product of the feature values \(x_i\) and \(x_j\).

The key insight here is that the inner product \(\langle v_i, v_j \rangle\) quantifies the interaction between the latent vectors \(v_i\) and \(v_j\), which in turn reflects the interaction between the corresponding features \(x_i\) and \(x_j\). This allows the FM model to capture pairwise feature interactions without the need to explicitly enumerate all possible feature pairs, making it an efficient and effective approach for modeling complex relationships in data.


## 损失函数

Certainly, for the FM (Factorization Machine) model, the loss function and optimization method are crucial components. The objective of training the FM model is to find the model parameters that minimize a certain loss function. In the case of FM, the commonly used loss function for binary classification problems is the logistic loss (also known as the log loss or cross-entropy loss), which is defined as follows:

\[ \mathcal{L}(\Theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] \]

Here's what each term represents:

- \(\mathcal{L}(\Theta)\): The logistic loss, which measures the difference between the predicted probabilities \(\hat{y}_i\) and the true labels \(y_i\) for a set of \(N\) training examples.

- \(\Theta\): The set of model parameters, including the weights \(w_0\) and \(w_i\) for linear terms and the latent vectors \(v_i\) for feature interactions.

- \(N\): The number of training examples in the dataset.

- \(y_i\): The true label (0 or 1) for the \(i\)-th example.

- \(\hat{y}_i\): The predicted probability of the positive class (class 1) for the \(i\)-th example, which is the output of the FM model \(\hat{y}(x_i)\) after passing through a sigmoid function to ensure it falls in the range [0, 1].

To optimize the FM model, commonly used optimization algorithms include gradient descent variants like stochastic gradient descent (SGD), mini-batch gradient descent, or Adam. The goal is to find the model parameters \(\Theta\) that minimize the logistic loss \(\mathcal{L}(\Theta)\) by iteratively updating the parameters in the direction that reduces the loss.

The specific optimization method and hyperparameters (learning rate, batch size, etc.) depend on the problem, dataset size, and the desired trade-off between training speed and convergence.




## 原文link

https://www.researchgate.net/publication/348321049_User_Response_Prediction_in_Online_Advertising




