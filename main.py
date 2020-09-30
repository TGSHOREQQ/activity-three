# Improvements
# - I don't like using X_train etc as global variables within the function
#    Is that bad practice ^^
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt
import time

SAMPLES = 100000
FEATURES = 10
NO_CLASSES = 5
TEST_SIZE = 0.3

# Creates machine learning models for dataset
def create_model(model_type, k):
    model_name = type(model_type).__name__
    time_start = time.perf_counter()
    model = model_type.fit(X_train, y_train)
    time_elapsed = (time.perf_counter() - time_start)

    prob = model_type.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    print(f"{model_name} Metrics")
    print("Computation Time:%5.4f seconds" % time_elapsed)
    print("Accuracy: %.2f" % accuracy, "\n")
    if model_name == 'KNeighborsClassifier':
        print("Optimal K value: %f" % k)
    confusion_matrix(model, model_name)

# Plots confusion matrix for ML model
def confusion_matrix(model, model_name):
    plot_confusion_matrix(model, X_test, y_test, normalize='true')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# Find optimal k value for kNN
# - Samples / what, is best?
def optimal_k(samples):
    k_acc_scores = []
    k_values = [i for i in range(1, int(samples / 500), 2)]
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        cross_scores = cross_val_score(knn, X_train, y_train, cv=None, scoring='accuracy')
        k_acc_scores.append(cross_scores.mean())
    optimal_k = k_values[k_acc_scores.index(max(k_acc_scores))]
    return optimal_k

# Generating 2D 3-class classification dataset using sklearn function
X, y = make_classification(n_samples=SAMPLES, n_features=FEATURES, n_classes=NO_CLASSES, weights=None, flip_y=0.01,
                           class_sep=2.1, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=42,
                           n_informative=3, n_redundant=2, n_repeated=0, n_clusters_per_class=1)

scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)

# PCA to reduce dimensionality
pca = PCA(n_components=3)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

# Visualise reduced data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=y, marker='o', s=0.8)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=TEST_SIZE, random_state=42)

models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(),
          KNeighborsClassifier(n_neighbors=optimal_k(SAMPLES))]

for model in models:
    create_model(model, optimal_k)
