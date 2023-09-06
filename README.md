
# Machine Learning Process

This is a place for me to understand and learning the whole machine learning process.

I have attached [Concepts](MLConcepts.md) for me to read all of the important concepts.

I have also attached [Sample Interview Questions](MLInterviewQuestions.md) for FAQs in interviews.


<img width="468" alt="image" src="https://github.com/jasoncchandra/MachineLearningProcess/assets/141464490/bdc50859-f3bf-4d58-a3df-18059c158478">

(from https://github.com/stefan-jansen/machine-learning-for-trading)

## Machine Learning Overview

1. **Problem Definition:**

    
    -   Clearly define the problem you aim to solve. Understand the business context, objectives, and desired outcomes.
    -   Identify whether the problem requires supervised, unsupervised, or reinforcement learning.
		            **Always recognize the use case.** 
		            o	Sometimes this is statistical inference, [eg. assess how likely observation is due to noise]
		            o	Or causal relationship between variables.
		            o	Or predict outcome
		            -	Nature of output:
		            o	Continuous output variable = **regression** problem
		            o	Categorical variable = **classification** problem (etc, binary classification)
		            o	Ordered categorical variables = ranking problem

2.  **Data Collection:**
    
    -   Gather relevant data from various sources. This could involve databases, APIs, web scraping, or manual collection.
    -   Ensure data quality, accuracy, and consistency.
3.  **Data Preprocessing:**
    
    -   Cleanse the data by handling missing values, outliers, and inconsistencies.
    -   Transform categorical variables into numerical representations.
    -   Normalize or standardize numerical features.
4.  **Feature Engineering:**
    -   Feature engineering in ML consists of four main steps: **Feature Creation, Transformations, Feature Extraction, and Feature Selection.**
    -   Select relevant features that contribute to the model's predictive power.
    -   Create new features that capture important patterns or relationships.
    -   Perform dimensionality reduction if necessary.
5.  **Data Splitting:**
    
    -   Divide the dataset into training, validation, and test subsets.
    -   Ensure that each subset maintains the distribution of classes or outcomes.
6.  **Model Selection:**
    
    -   Choose appropriate algorithms based on the problem type and characteristics of the data.
    -   Consider factors like interpretability, performance, and computational efficiency.

Available Types:

**Supervised Regression Models:**
1. **Linear Regression:** Models relationships between variables using linear equations for continuous numeric predictions.
2. **Polynomial Regression:** Extends linear regression to fit polynomial curves for continuous numeric predictions.
3. **Ridge Regression and Lasso Regression:** Techniques that handle multicollinearity and prevent overfitting in linear regression.
4. **Support Vector Regression (SVR):** Utilizes support vector machines for continuous numeric predictions.

**Supervised Classification Models:**
1. **Logistic Regression:** Used for binary or multi-class classification by estimating class probabilities.
2. **Decision Trees:** Divides data into subsets based on feature values for classification tasks.
3. **Random Forest:** Ensemble method combining multiple decision trees for improved classification.
4. **Naive Bayes:** A probabilistic algorithm often used for text classification and categorical data.
5. **k-Nearest Neighbors (k-NN):** Predicts class by majority vote of k nearest neighbors in feature space.
6. **Support Vector Machines (SVM):** Constructs hyperplanes to separate different classes with maximum margin.
7. **Neural Networks:** Multilayered networks for complex classification tasks, including deep learning architectures.

**Unsupervised Learning Models:**
1. **K-Means Clustering:** Divides data into clusters based on similarity.
2. **Principal Component Analysis (PCA):** Reduces dimensionality while preserving variance.
3. **t-Distributed Stochastic Neighbor Embedding (t-SNE):** Visualizes high-dimensional data in lower dimensions.
4. **Isolation Forest:** Detects anomalies by isolating data points in a tree structure.
5. **Apriori Algorithm:** Finds associations between items in transactional data (e.g., market basket analysis).
6. **Gaussian Mixture Models (GMM):** Models data as a mixture of Gaussian distributions for density estimation.
7. **Autoencoders:** Neural network architectures for feature extraction and dimensionality reduction.

**Reinforcement Learning Models:**
1. **Q-Learning:** Learns optimal policies by estimating Q-values for state-action pairs.
2. **Deep Q-Networks (DQN):** Extends Q-learning with deep neural networks for complex tasks.
3. **Policy Gradient Methods:** Learn policies directly to maximize expected rewards.
4. **Actor-Critic Methods:** Combine policy and value estimation for improved learning.
5. **Monte Carlo Tree Search (MCTS):** Used in decision-making processes, e.g., game-playing AI.
6. **Deep Deterministic Policy Gradient (DDPG):** Designed for continuous action spaces.
7. **A3C (Asynchronous Advantage Actor-Critic):** Distributed RL algorithm for efficient learning.
8. **TRPO and SAC:** Focus on optimizing policy stability and convergence in RL tasks.

This breakdown distinguishes between regression models, which predict continuous numeric values, and classification models, which assign data points to discrete categories or classes. Each of these models serves specific purposes within supervised learning.


Choosing the right machine learning model for a specific problem involves considering various factors such as the nature of the data, the problem type (classification, regression, clustering, etc.), and the model's strengths and weaknesses. While there isn't a one-size-fits-all mind map for model selection, you can follow a systematic process to make informed decisions:

1. **Understand the Problem:**
   - Begin by thoroughly understanding the problem and its requirements. Determine whether it's a classification, regression, clustering, or another type of task.

2. **Analyze the Data:**
   - Examine the characteristics of your dataset:
     - Is it structured or unstructured?
     - What are the types of features (numeric, categorical, text, etc.)?
     - Are there any missing values or outliers?
     - What is the data size (small, medium, large)?

3. **Consider Model Requirements:**
   - Each machine learning model has specific requirements and assumptions. Consider factors like:
     - Linearity: Linear models assume a linear relationship between features and the target variable.
     - Complexity: Some models are more complex and suitable for high-dimensional data or complex patterns.
     - Interpretability: Some models, like decision trees, are more interpretable than others, like neural networks.
     - Scalability: Consider the computational resources required for large datasets.
     - Type of Output: Ensure the model can produce the desired output (e.g., probabilities for classification).

4. **Model Performance Metrics:**
   - Think about the evaluation metrics relevant to your problem type (e.g., accuracy, mean squared error, F1-score). Different models may perform better with different metrics.

5. **Experiment and Compare:**
   - It's often beneficial to experiment with multiple models to see which one performs best. Use techniques like cross-validation to assess model performance.

6. **Consider Domain Knowledge:**
   - If you have domain expertise, consider whether certain models are well-suited to your industry or problem domain.

7. **Ensemble Methods:**
   - Ensemble methods like Random Forest or Gradient Boosting can often improve performance by combining multiple models. These are robust choices for many tasks.

Ensemble methods are powerful techniques in machine learning that combine predictions from multiple base models to produce a stronger, more robust model. The idea behind ensemble methods is that by aggregating the predictions of several weaker models, you can often achieve better performance than using a single model. Ensemble methods are particularly effective in reducing overfitting and improving generalization.

Here are some key ensemble methods:

1. **Bagging (Bootstrap Aggregating):**
   - Bagging involves training multiple base models independently on random subsets of the training data (with replacement) and then averaging their predictions (for regression) or taking a majority vote (for classification).
   - The most well-known bagging algorithm is the Random Forest, which applies this technique to decision trees.

2. **Boosting:**
   - Boosting is an iterative ensemble method where base models are trained sequentially, and each new model focuses on the examples that the previous models found difficult.
   - Popular boosting algorithms include AdaBoost (Adaptive Boosting), Gradient Boosting, and XGBoost (Extreme Gradient Boosting).

3. **Stacking (Stacked Generalization):**
   - Stacking combines predictions from multiple models by training a meta-model (also called a blender or aggregator) on top of the base models. The meta-model learns how to weigh the predictions of the base models to make the final prediction.
   - Stacking can be more complex to set up but often leads to improved performance.

4. **Voting Ensembles:**
   - In a voting ensemble, multiple base models make predictions on a given input, and the final prediction is determined by a majority vote (for classification) or an average (for regression).
   - Voting can be "hard" (simple majority) or "soft" (weighted average based on predicted probabilities).

5. **Bootstrap Aggregating for Model Selection (BAMS):**
   - BAMS is a technique that applies bagging not only to the data but also to the selection of the base models. It helps in robust model selection and can improve generalization.

6. **Ensemble of Neural Networks:**
   - For deep learning tasks, you can create an ensemble of neural networks by training multiple neural networks with different architectures or initializations and combining their predictions.

Key benefits of ensemble methods:

- Improved Performance: Ensemble methods often produce more accurate and robust predictions compared to individual models.
- Reduction of Overfitting: They help reduce overfitting by combining the strengths of multiple models and mitigating their weaknesses.
- Robustness: Ensembles can handle noisy data and outliers better than single models.
- Flexibility: You can use a variety of base models and combine them using different ensemble techniques.

However, ensemble methods come with some trade-offs:

- Increased Complexity: Ensembles can be computationally expensive and may require careful tuning.
- Interpretability: The combined predictions can be harder to interpret than those of individual models.
- Potential Overfitting: If not properly tuned, ensemble methods can overfit the training data.

When using ensemble methods, it's essential to consider the problem type (classification or regression), the characteristics of the data, and the computational resources available. Experimentation and cross-validation are crucial for optimizing ensemble models for your specific problem.



7.  **Model Training:**
    
    -   Train the selected model on the training data.
    -   Adjust hyperparameters to optimize performance.
    -   Use techniques like cross-validation to avoid overfitting.
8.  **Model Evaluation:**
    
    -   Evaluate the model's performance on the validation dataset using appropriate metrics (accuracy, precision, recall, F1-score, etc.).
    -   Tune the model based on evaluation results.
9.  **Model Interpretation:**
    
    -   Interpret the model's predictions to gain insights into its decision-making process.
    -   Analyze feature importances or coefficients to understand the factors influencing predictions.
10.  **Model Deployment:**

   -   Deploy the trained model to a production environment if applicable.
   -   Implement a user interface or API for model interaction.
   -   Monitor the model's performance in real-world scenarios.

12.  **Documentation and Communication:**
    
   -   Create comprehensive documentation outlining the entire process, from problem definition to deployment.
   -   Communicate results and insights to stakeholders in a clear and understandable manner.
13.  **Iterate and Improve:**
    
   -   Continuously monitor the model's performance and gather feedback.
   -   Iterate on the process to refine the model and address any issues that arise.
