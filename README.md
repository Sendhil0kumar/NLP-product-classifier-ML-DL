# E-commerce Text Classification (ML vs. Deep Learning)

**Date:** October 26, 2025

---

## 1. Project Overview

This project's goal was to solve a common e-commerce challenge: automatically categorizing products using only their text descriptions. This automation saves manual labor and improves product discoverability.

The objective was to build and evaluate several machine learning and deep learning models to find the most accurate classifier for the given 4 categories:
* Books
* Clothing & Accessories
* Electronics
* Household

## 2. Methodology

This project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework, covering all stages from data understanding to evaluation.

### Data Preparation
1.  **Data Cleaning:** The raw CSV data was loaded, and "dirty" data (e.g., malformed columns, null values) was cleaned.
2.  **Text Preprocessing:** A standard NLP pipeline was built to clean the text descriptions:
    * Converted text to lowercase.
    * Removed all punctuation and numbers using regex.
    * Tokenized text into individual words.
    * Removed common English "stopwords."
    * **Lemmatized** words to their root form (e.g., "blazers" -> "blazer") for standardization.

### Feature Engineering
Two different feature engineering methods were used to prepare the text for the models:
* **For Classical ML (Q5):** Used **TF-IDF (Term Frequency-Inverse Document Frequency)** to create a sparse matrix of 5,000 features, highlighting important and specific words.
* **For Deep Learning (Q6):** Used a **Keras Tokenizer** to convert text into integer sequences, which were then **padded** to a uniform length. This preserves the word order, which is critical for an LSTM.

---

## 3. Final Model Performance

Six models were trained and evaluated. The **Support Vector Machine (SVM)** provided the highest accuracy, slightly outperforming the Deep Learning model.

| Model Type | Accuracy |
| :--- | :---: |
| Support Vector Machine | 96.29% |
| Logistic Regression | 95.99% |
| LSTM (Deep Learning) | 95.04% |
| Random Forest | 95.30% |
| Multinomial Naive Bayes | 93.69% |
| K-Nearest Neighbours | 67.22% |

---

## 4. Key Findings & Conclusion

* **Best Overall Model:** The **Support Vector Machine (SVM)** was the top-performing model with **96.29%** accuracy. This demonstrates that a well-tuned linear model on TF-IDF features is an incredibly strong and efficient baseline for text classification.

* **Deep Learning Performance:** The **LSTM** model also achieved a very strong accuracy of **95.04%**. The training plots showed excellent generalization with no overfitting.

* **A Key Lesson (The "Curse of Dimensionality"):** The **K-Nearest Neighbours (KNN)** model failed significantly (67.22%). This is a classic example of the "Curse of Dimensionality," where distance metrics become meaningless in high-dimensional (5,000+ features) sparse data, making KNN a poor choice for this type of NLP problem.

## 5. How to Run

1.  Clone this repository.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow
    ```
3.  Run the `.ipynb` notebook in Google Colab or a local Jupyter environment.
