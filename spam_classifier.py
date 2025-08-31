# spam_classifier.py
# Task 1: SMS Spam Detection using Kaggle's spam.csv dataset

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ----------------------------
# 1. Load Dataset
# ----------------------------
print("Loading dataset...")
df = pd.read_csv("spam.csv", encoding="latin-1")

# Drop extra columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# ----------------------------
# 2. Preprocessing
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)           # keep only letters
    text = re.sub(r"\s+", " ", text).strip()        # remove extra spaces
    return text

df['text_clean'] = df['text'].apply(clean_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ----------------------------
# 3. Train/Test Split + TF-IDF
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ----------------------------
# 4. Train Model
# ----------------------------
MODEL_CHOICE = 'logreg'   # options: 'logreg', 'nb', 'svm'

if MODEL_CHOICE == 'logreg':
    model = LogisticRegression(max_iter=200, class_weight='balanced')
elif MODEL_CHOICE == 'nb':
    model = MultinomialNB()
elif MODEL_CHOICE == 'svm':
    model = LinearSVC()

model.fit(X_train_tfidf, y_train)

# ----------------------------
# 5. Evaluation
# ----------------------------
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1 Score :", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# 6. User Input + Confusion Matrix with Result
# ----------------------------
def classify_message(msg):
    clean = clean_text(msg)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    return pred  # 1 = spam, 0 = not spam

while True:
    user_msg = input("\nEnter a message (or type 'exit' to quit): ")
    if user_msg.lower() == "exit":
        break

    result = classify_message(user_msg)
    result_text = "Spam" if result == 1 else "Not Spam"
    print("Prediction:", result_text)

    # Show Confusion Matrix again + overlay result
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (User message → {result_text})")
    plt.colorbar()
    plt.xticks([0,1], ['Ham','Spam'])
    plt.yticks([0,1], ['Ham','Spam'])
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color="red")

    # Write F1 score and message classification at bottom
    plt.xlabel(f"Predicted (F1 Score: {f1:.2f})")
    plt.ylabel("True label")
    plt.figtext(0.5, -0.05, f"User message classified as: {result_text}", ha="center", fontsize=12, color="green")

    plt.show()
