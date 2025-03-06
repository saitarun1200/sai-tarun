import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load the dataset
hamspam_data = pd.read_csv(r"C:\Users\saita\OneDrive - UMBC\Desktop\733homework2\hamspam.csv")

# Rename 'Class' column to 'label' for consistency
hamspam_data.rename(columns={'Class': 'label'}, inplace=True)

# Convert labels to binary (Ham = 0, Spam = 1)
hamspam_data['label'] = hamspam_data['label'].map({'Ham': 0, 'Spam': 1})  

# Convert categorical features into a single text representation
hamspam_data['text_features'] = hamspam_data[['Contains Link', 'Contains Money Words', 'Length']].astype(str).agg(' '.join, axis=1)

# Split dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    hamspam_data['text_features'],  # Features
    hamspam_data['label'],  # Target labels
    test_size=0.3,
    random_state=42,
    stratify=hamspam_data['label']
)

# Convert text features to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Choose a classifier: "DecisionTree", "NaiveBayes", or "KNN"
classifier_choice = "DecisionTree"  # Change to "NaiveBayes" or "KNN" if needed

# Initialize the classifier
if classifier_choice == "DecisionTree":
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)  # Reduced max_depth
elif classifier_choice == "NaiveBayes":
    clf = MultinomialNB()
elif classifier_choice == "KNN":
    clf = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
clf.fit(X_train_tfidf, y_train)

# Predict probabilities for the positive class (Spam)
y_prob = clf.predict_proba(X_test_tfidf)[:, 1]

# Compute ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, marker='o', linestyle='-', label=f'ROC Curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title(f'ROC Curve for {classifier_choice} on HamSpam Data')
plt.legend()
plt.grid(True)

# Show the ROC curve
plt.show()

# Display the AUC Score
print(f"AUC Score for {classifier_choice}: {auc_score:.3f}")
