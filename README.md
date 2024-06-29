# Image_Clustering_and_Classification
shapely.py
This script reads a dataset (dataset.csv), processes polygon data using geometric libraries (shapely, numpy), and overlays these polygons onto images fetched from /content/drive/PNG/. It demonstrates data manipulation, geometric operations, and image processing techniques in Python.

clustering.py
This script utilizes VGG16 for feature extraction from images, applies t-SNE for dimensionality reduction, and performs K-means clustering to analyze image similarities. It also uses DBSCAN for density-based clustering and evaluates clustering performance using silhouette scores. Results are visualized using scatter plots and saved to CSV and PNG files.

classification.py
This script utilizes transfer learning with VGG16 on a custom image dataset, performs training and evaluation using ImageDataGenerator, and saves the trained model and its weights. It visualizes training progress with plots for loss and accuracy over epochs, demonstrating model performance and validation.

features_classfication.py
This script loads a pretrained VGG16 model, extracts features from images using its intermediate layer outputs, performs dimensionality reduction with PCA, and trains various classifiers like Logistic Regression, Decision Trees, Random Forests, K-Nearest Neighbors, SVM, Gaussian Naive Bayes, and XGBoost for image classification tasks using extracted features. It also includes hyperparameter tuning for RandomForest using GridSearchCV and evaluates classification performance with metrics such as accuracy, confusion matrix, and classification report.

histogram.py
This script utilizes pandas and visualization libraries like matplotlib, seaborn, and plotly to create various heatmaps displaying Kernel Density Estimate (KLD) values from an Excel dataset (Histogram.xlsx). It demonstrates both static and interactive visualization techniques for exploring relationships across different groups and variables.
