import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Step 1: Load the datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Step 2: Add labels
fake['label'] = 0  # Fake news
real['label'] = 1  # Real news

# Step 3: Combine datasets and clean
data = pd.concat([fake, real], axis=0)
data = data[['text', 'label']].dropna()

# Step 4: Split into train and test sets
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Step 9: Interactive Prediction Loop
print("\n🤖 Model is ready! Type 'exit' to quit.")

while True:
    user_input = input("\n📝 Enter a news statement:\n")
    
    if user_input.lower() == "exit":
        print("👋 Exiting. Thank you!")
        break

    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]
    confidence = probabilities[prediction]

    label = "REAL" if prediction == 1 else "FAKE"
    print(f"🔎 Prediction: {label} (Confidence: {confidence:.2f})")

    if confidence < 0.65:
        print("⚠️ Warning: This news may be very new, rare, or unfamiliar to the model.")
