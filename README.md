This repository demonstrates a complete data analytics pipeline applied to the Kaggle Medical Cost Personal Dataset. As an Information Engineer, I focus on transforming unprocessed health and demographic data into actionable business insights by linking traditional risk modeling (SOA Exam P, FAM, SRM logic) with modern algorithmic implementation.


Part 1: Unsupervised Customer Segmentation
Objective: Move beyond basic demographics to identify hidden behavioral clusters.
Techniques: K-Means Clustering, Agglomerative Hierarchical Clustering.
Key Action: Used the Elbow Method to optimize \(K=4\) and visualized the hierarchy using a Dendrogram.
Business Value: Identified a "High-Exposure" segment where lifestyle factors (BMI/Smoking) drive higher costs, enabling Risk-Adjusted Pricing.


Part 2: Dimensionality Reduction & Predictive Modeling
Objective: Simplify complex data and predict annual insurance charges.
Techniques: PCA (n=2), VIF Diagnostics, Multiple Linear Regression.
Key Action: Applied PCA to eliminate Multicollinearity (redundancy) between features while retaining ~85% of data variance.
Validation: Performed VIF (Variance Inflation Factor) checks to ensure model stability and analyzed Residual Plots to confirm prediction reliability.
Business Value: Provides a stable forecasting engine for total Aggregate Loss.
