import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv(r"C:\Users\saita\OneDrive - UMBC\Desktop\733homework2\hamspam.csv")

# Print column names to verify correct column name
print("Column names in CSV:", data.columns)

# Rename 'Class' column to 'label' for consistency
data.rename(columns={'Class': 'label'}, inplace=True)

# Convert labels to binary (Ham = 0, Spam = 1)
data['label'] = data['label'].map({'Ham': 0, 'Spam': 1})  # Ensure correct capitalization

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=['label']),  # Features
    data['label'],  # Target
    test_size=0.3,
    random_state=42,
    stratify=data['label']
)

# Convert text features to TF-IDF vectors (assuming 'message' is the text column)
vectorizer = TfidfVectorizer(stop_words='english')

# If there is no "message" column, select appropriate text-based column
text_column = 'message' if 'message' in data.columns else 'Contains Link'  # Choose the right text column

X_train_tfidf = vectorizer.fit_transform(X_train[text_column])
X_test_tfidf = vectorizer.transform(X_test[text_column])

# Train Naive Bayes Classifier
nb_clf = MultinomialNB()
nb_clf.fit(X_train_tfidf, y_train)
y_pred_nb = nb_clf.predict(X_test_tfidf)

print("\nNaive Bayes Results:")
print(classification_report(y_test, y_pred_nb))
print("Accuracy:", accuracy_score(y_test, y_pred_nb))

# Train KNN Classifier (K=5)
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_tfidf, y_train)
y_pred_knn = knn_clf.predict(X_test_tfidf)

print("\nKNN Results:")
print(classification_report(y_test, y_pred_knn))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))