## Cheat Sheet for ML - basic overview.

General Topics/Concepts:

**1. Linear Algebra:**

- In machine learning, linear algebra is a foundational mathematical framework used to represent and manipulate data and models. It provides tools to understand relationships between variables, optimize models, and perform various operations efficiently. Key concepts include vectors and matrices for data representation, linear transformations for feature engineering, eigenvalues and eigenvectors for dimensionality reduction, and matrix operations for model training and optimization. Understanding linear algebra is crucial for grasping the inner workings of machine learning algorithms, from linear regression to deep neural networks, and for effectively working with data in higher-dimensional spaces.

**2. Gradient Descent:**

- Gradient descent is a fundamental optimization algorithm in machine learning used to minimize the cost or loss function of a model during training. It operates by iteratively adjusting model parameters in the direction of the steepest descent of the cost function, as determined by the gradient (derivative). In each iteration, it calculates the gradient of the cost function with respect to the model parameters and updates the parameters by taking a step proportional to the negative gradient. This process continues until convergence, where the cost function reaches a minimum or a predetermined number of iterations is reached. Gradient descent is vital for training various machine learning models, including neural networks, linear regression, and logistic regression, and is crucial for finding the optimal model parameters that best fit the data.

Objective Function (Cost Function): Typically denoted as J(θ), where θ represents the model parameters (weights and biases).

Gradient of the Cost Function: ∇J(θ), which is the vector of partial derivatives of J(θ) with respect to each parameter.

Update Rule: θ := θ - α * ∇J(θ), where α (alpha) is the learning rate, a hyperparameter that controls the step size in each iteration.

Iteration: This update rule is applied iteratively until convergence is reached or a specified number of iterations is completed.

- Stoachstic Gradient Descent adds an element of randomness so that the gradient does not get stuck. It uses one data point at a time and uses smaller subsect of datapoint.

**3. Model Evaluation and Selection**

In model evaluation, the primary objective is to measure a model's performance using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, or mean squared error, depending on the problem type (classification or regression). Cross-validation techniques, like k-fold cross-validation, are often employed to obtain a robust estimate of a model's performance and assess its generalization ability. It is essential to carefully select the right evaluation metrics that align with the problem's goals and interpret the results in the context of the problem domain.

- It is also important to recognize how well the TEST SET performs on the TRAIN SET (usually 80%)

In model selection, the focus shifts to comparing multiple candidate models to identify the one that performs the best. Common techniques include grid search and randomized search, which explore different hyperparameter settings to find the optimal configuration. Additionally, domain knowledge and intuition can guide the selection process. The chosen model should strike a balance between complexity and generalization, ensuring it can make accurate predictions on new, unseen data. Overall, model evaluation and selection are iterative processes that require a deep understanding of the data, problem, and the strengths and weaknesses of different algorithms, making them essential topics for machine learning interviews.

**4. Bias Variance Tradeoff**

Bias: Bias represents the error introduced by approximating a real-world problem, which may be complex, by a simplified model. High bias can lead to underfitting, where the model is too simplistic and fails to capture important relationships in the data. Models with high bias have poor training and test performance.

Variance: Variance represents the error introduced by the model's sensitivity to small fluctuations or noise in the training data. High variance can lead to overfitting, where the model captures noise in the training data and does not generalize well to new data. Models with high variance perform well on the training data but poorly on the test data.

The bias-variance trade-off is a balance between these two sources of error. Ideally, you want a model that has enough complexity to capture the underlying patterns in the data (low bias) but not too much complexity that it fits the noise (low variance). Achieving this balance typically involves tuning model hyperparameters, selecting appropriate algorithms, and adjusting the model's complexity.

Error = Bias² + Variance + Irreducible Error

Error: The overall error of the model on the test data.
Bias²: The squared bias term, representing how much the model's predictions systematically deviate from the true values.
Variance: The variance term, representing how much the model's predictions vary when trained on different subsets of the data.
Irreducible Error: The error that cannot be reduced, as it is inherent to the problem's complexity and noise in the data.

Bias-Variance Trade-off in Practice:

High-Bias Model Example: Linear regression with only one feature may have high bias. It assumes a simple linear relationship between the feature and target, which might be overly simplistic for complex data with nonlinear patterns. The model tends to underfit, resulting in high bias but low variance.

High-Variance Model Example: A deep neural network with many layers and parameters may exhibit high variance. Such a model can fit the training data very closely, capturing even the noise. However, it may fail to generalize to new data, leading to overfitting and high variance.

The goal in machine learning is to find the right model complexity and hyperparameters that strike a balance between bias and variance, minimizing the overall error on unseen data. Techniques like cross-validation, regularization, and model selection play a crucial role in managing the bias-variance trade-off. In interviews, understanding this trade-off and its associated formulas demonstrates a solid grasp of model evaluation and selection.


**5. Regularizaition**

**Regularization** is a fundamental technique in machine learning and statistics used to prevent overfitting, improve the generalization ability of models, and control model complexity. It involves adding a penalty term to the model's cost function, encouraging the model to be less complex and reducing the impact of large parameter values. Regularization is crucial when training complex models with many parameters to ensure they perform well on new, unseen data. Here are some common forms of regularization:

1. **L1 Regularization (Lasso):**
   - L1 regularization adds a penalty term equal to the absolute sum of the model's coefficients to the cost function.
   - It encourages sparsity in the model by driving some coefficients to exactly zero, effectively performing feature selection.
   - Lasso regression is a classic example of L1 regularization.

2. **L2 Regularization (Ridge):**
   - L2 regularization adds a penalty term equal to the sum of the squares of the model's coefficients to the cost function.
   - It discourages large coefficients and tends to distribute the impact of features more evenly.
   - Ridge regression is a classic example of L2 regularization.

3. **Elastic Net Regularization:**
   - Elastic Net combines both L1 and L2 regularization, adding a penalty term that is a combination of the absolute sum and the sum of squares of coefficients.
   - It provides a balance between feature selection (L1) and coefficient shrinkage (L2).

Regularization techniques help in several ways:
- **Preventing Overfitting:** Regularization discourages models from fitting the training data too closely, which can lead to overfitting, where the model performs well on training data but poorly on new data.
- **Improving Generalization:** By controlling model complexity, regularization helps models generalize better to unseen data, improving their predictive performance.
- **Handling Multicollinearity:** Regularization is effective at handling multicollinearity (high correlation between features) by reducing the impact of correlated features.

The choice between L1, L2, or Elastic Net regularization depends on the specific problem, the desired model behavior (e.g., feature selection or coefficient shrinkage), and careful tuning of hyperparameters. Regularization is a crucial tool in a data scientist's toolkit, ensuring that machine learning models are robust and capable of handling real-world data with noisy and complex patterns.
