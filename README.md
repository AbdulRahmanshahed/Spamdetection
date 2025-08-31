# 📧 SMS Spam Detection (Machine Learning Project)

This project builds a machine learning model to classify SMS messages as **Spam** or **Ham (Not Spam)** using the **Kaggle SMS Spam Collection Dataset**. It demonstrates text preprocessing, feature extraction with TF–IDF, and classification using multiple ML models.

---

## 🚀 Features
- Dataset: Kaggle SMS Spam Collection (`spam.csv`)
- Preprocessing: lowercase conversion, cleaning, stopword removal
- Feature extraction using **TF–IDF Vectorization**
- Models supported:
  - Logistic Regression (default)
  - Multinomial Naïve Bayes
  - Linear SVM
- Evaluation with Accuracy, Precision, Recall, and F1-Score
- Confusion Matrix visualization
- Interactive mode: enter a custom message → get prediction (Spam / Not Spam)

---

## 📊 Example Results
Typical performance with Logistic Regression:
- Accuracy: ~97%
- Precision: ~93%
- Recall: ~93%
- F1 Score: ~0.93

Confusion Matrix Example:

|              | Predicted Ham | Predicted Spam |
|--------------|---------------|----------------|
| **Actual Ham**  | 955           | 11             |
| **Actual Spam** | 10            | 139            |

---

## 🛠️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/spam-detection.git
cd spam-detection
pip install pandas numpy scikit-learn matplotlib
```

---

## ▶️ Usage
Place `spam.csv` in the project directory, then run:

```bash
python spam_classifier.py
```

- The model trains on the dataset.
- Evaluation metrics + confusion matrix are displayed.
- Enter a message to classify interactively.

Example:

```
Enter a message (or type 'exit' to quit): Congratulations! You won a free iPhone
Prediction: Spam
```

---

## ⚙️ Model Choice
Change the model in the script:

```python
MODEL_CHOICE = 'logreg'   # options: 'logreg', 'nb', 'svm'
```

---

## 📈 Future Improvements
- Save & load models with Joblib
- Deploy as a Flask/Django web app
- Extend to email/WhatsApp spam or phishing detection

---

👨‍💻 **Author**: Your Name  
📅 **Project**: AI/ML Internship Task – Spam Detection
