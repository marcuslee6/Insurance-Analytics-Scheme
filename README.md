This repository demonstrates a complete data analytics pipeline applied to the Kaggle Medical Cost Personal Dataset. As an Information Engineer, I focus on transforming unprocessed health and demographic data into actionable business insights by linking traditional risk modeling (SOA Exam P, FAM, SRM logic) with modern algorithmic implementation.


Part 1: Unsupervised Customer Segmentation
Objective: Move beyond basic demographics to identify hidden behavioral clusters.
Techniques: K-Means Clustering, Agglomerative Hierarchical Clustering.
Key Action: Used the Elbow Method to optimize \(K=4\) and visualized the hierarchy using a Dendrogram.
Business Value: Identified a "High-Exposure" segment where lifestyle factors (BMI/Smoking) drive higher costs, enabling Risk-Adjusted Pricing.


Part 2: Dimensionality Reduction & Predictive Modeling
Objective: Simplify complex data and predict annual insurance charges.
Techniques: PCA (n=2), VIF Diagnostics, Multiple Linear Regression.
Key Action: Applied PCA to eliminate Multicollinearity between features while retaining approximately 85% of data variance.
Validation: Performed VIF (Variance Inflation Factor) checks to guarantee model stability and analyzed Residual Plots to confirm prediction reliability.
Business Value: 
1)Provides a stable forecasting engine for total Aggregate Loss.
2)Accounted for the Fat-tailed (Pareto) distribution of insurance charges to improve prediction accuracy for high-cost outliers

Part 3:Advanced Risk Classification
Objective:To develop a robust binary classification system that predicts high-cost outliers using a multi-model approach.
Techniques: Linear Discriminant Analysis (LDA), Single-Layer Perceptron, Support Vector Machines (SVM).
Methodology: 
1)Preprocessing: Transformed continuous insurance charges into binary risk levels (High vs. Low) to test categorical prediction power.
2)LDA: Utilized LDA to project features into a lower-dimensional space that maximizes the separation between risk classes.
3)Perceptron: Implemented a Single-Layer Perceptron as a baseline neural-logic model to evaluate linear separability.
4)SVM: Applied Support Vector Machines to identify the optimal hyperplane with the Maximum Magin, ensuring robust classification with overlapping data points.
IE Insight: This project simulates an automated underwriting system where the goal is to minimize misclassifcation of high-risk policyholders, thereby protecting the company's Aggregate Loss method.
