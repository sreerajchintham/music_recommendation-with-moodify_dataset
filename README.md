```markdown
# music recommendation.ipynb

## Music Recommendation System Analysis

This Google Colab notebook explores the development of a music recommendation system using both classification models and a content-based similarity approach. It demonstrates data preprocessing, model training, evaluation, and a basic recommendation function based on track features.

## Overview

The primary goal of this notebook is to analyze a dataset of music tracks, classify them based on their features, and build a system that can recommend similar tracks. It showcases two distinct approaches often used in recommendation systems:

*   **Supervised Learning for Classification:** Using Decision Trees and Random Forests to predict music labels (e.g., genres, moods) based on audio features. This allows for classifying new music or recommending songs belonging to a predicted category.
*   **Unsupervised Learning for Content-Based Recommendation:** Identifying similar tracks based purely on their intrinsic audio features (e.g., danceability, energy) using Euclidean distance. This provides a direct way to find musically similar songs.

## Key Features and Functionality

*   **Data Loading and Exploration:** Reads music track data from a CSV file and performs initial data inspection (shape, descriptive statistics, missing values, and target variable distribution).
*   **Data Preprocessing:** Handles duplicate entries, removes rows with missing values, and selects relevant features for modeling.
*   **Feature Scaling:** Standardizes numerical features using `StandardScaler` to ensure models perform optimally.
*   **Supervised Classification Model Training & Evaluation:**
    *   Trains and evaluates a `DecisionTreeClassifier`.
    *   Trains and evaluates a `RandomForestClassifier`.
    *   Provides comprehensive performance metrics including classification reports, confusion matrices (with visualizations), and accuracy scores for both models.
*   **Content-Based Recommendation System:** Implements a function to find the top 'N' most similar music tracks (identified by their URIs) to a given input track based on their scaled features and Euclidean distance.
*   **Data Visualization:** Uses `seaborn` and `matplotlib` to visualize confusion matrices for model performance.

## Technologies and Libraries Used

The notebook leverages the following Python libraries:

*   `pandas`: For efficient data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `matplotlib.pyplot`: For creating static, interactive, and animated visualizations.
*   `seaborn`: For high-level, aesthetically pleasing statistical data visualizations.
*   `sklearn.model_selection`: For splitting data into training and testing sets (`train_test_split`).
*   `sklearn.preprocessing.StandardScaler`: For standardizing features.
*   `sklearn.metrics`: For evaluating model performance (`accuracy_score`, `classification_report`, `confusion_matrix`) and calculating distances (`euclidean_distances`).
*   `sklearn.tree.DecisionTreeClassifier`: For implementing the Decision Tree classification model.
*   `sklearn.ensemble.RandomForestClassifier`: For implementing the Random Forest classification model.

## Main Sections and Steps

The notebook is structured into the following logical sections:

1.  **Setup and Data Loading:**
    *   Imports all necessary libraries.
    *   Loads the music track dataset from `/content/278k_labelled_uri 2.csv`.
2.  **Exploratory Data Analysis (EDA):**
    *   Displays the first few rows (`df.head()`).
    *   Checks the dataset dimensions (`df.shape`).
    *   Generates descriptive statistics (`df.describe()`).
    *   Identifies missing values (`df.isna().sum()`).
    *   Examines the distribution of target labels (`df["labels"].value_counts()`).
3.  **Data Preprocessing and Cleaning:**
    *   Removes duplicate rows (`df.drop_duplicates()`).
    *   Removes rows with any missing values (`df.dropna()`).
    *   Selects relevant columns for features and target, discarding initial non-feature columns (`df.iloc[:,2:]`).
4.  **Feature Engineering and Scaling:**
    *   Separates features (X) and the target variable ("labels"). The "uri" column is also separated for the content-based recommendation part.
    *   Applies `StandardScaler` to normalize numerical features.
    *   Splits the scaled data into training and testing sets using a stratified approach to maintain label proportions (`train_test_split`).
5.  **Model Training and Evaluation (Decision Tree):**
    *   Initializes and trains a `DecisionTreeClassifier` on the training data.
    *   Makes predictions on the test set.
    *   Prints a classification report.
    *   Generates and visualizes a confusion matrix using `seaborn.heatmap`.
    *   Calculates and prints the accuracy score.
6.  **Model Training and Evaluation (Random Forest):**
    *   Initializes and trains a `RandomForestClassifier` on the training data.
    *   Makes predictions on the test set.
    *   Prints a classification report.
    *   Calculates and prints the accuracy score.
    *   Generates and visualizes a confusion matrix using `seaborn.heatmap`.
7.  **Content-Based Recommendation System:**
    *   Prepares features and URIs for similarity calculation.
    *   Defines a function `get_closest_uris` that calculates Euclidean distances between an input URI's features and all other tracks to find the top 'N' closest ones.
    *   Demonstrates an example usage of the recommendation function with a sample URI from the dataset.

## Key Insights and Results

*   The notebook successfully implements and evaluates two common classification algorithms (Decision Tree and Random Forest) for predicting music labels based on audio features. Random Forest is generally expected to outperform Decision Tree due to its ensemble nature, offering better generalization and robustness.
*   Beyond classification, a functional content-based recommendation system is developed. This system can find musically similar tracks without relying on explicit user feedback, making it valuable for cold-start scenarios or exploring new music based on specific audio characteristics.
*   The comprehensive evaluation steps (classification report, confusion matrix, accuracy) provide clear insights into model performance for the classification task.

## How to Use/Run the Notebook

To run this notebook:

1.  **Open in Google Colab:**
    *   Navigate to Google Colaboratory ([colab.research.google.com](https://colab.research.google.com/)).
    *   Go to `File > Upload notebook` and select `music recommendation.ipynb` from your local machine, or `File > Open notebook` and choose it from your Google Drive if already uploaded.
2.  **Upload Data:**
    *   The notebook expects a CSV file named `278k_labelled_uri 2.csv` to be present in the Colab runtime's `/content/` directory.
    *   You will need to manually upload this file to your current Colab session using the "Files" tab (folder icon on the left sidebar) by clicking "Upload to session storage" and selecting your CSV file.
    *   Alternatively, you can modify the file path in Cell 2 if your data is located elsewhere (e.g., mounted from Google Drive).
3.  **Run All Cells:**
    *   Once the data is uploaded, you can execute all cells sequentially by going to `Runtime > Run all`.
    *   Observe the outputs, including data explorations, model performance metrics, visualizations, and the content-based recommendation example.
```