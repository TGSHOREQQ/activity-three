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

# Generating 2D 3-class classification dataset using sklearn function
X, y = make_classification(n_samples=SAMPLES, n_features=FEATURES, n_classes=NO_CLASSES, weights=None, flip_y=0.01,
                           class_sep=2.1, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=42,
                           n_informative=3, n_redundant=2, n_repeated=0, n_clusters_per_class=1)

scaler=StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)

# PCA to reduce dimensionality
pca = PCA(n_components=3)
pca.fit(scaled_data)
x_pca = pca.transform((scaled_data))

#Visualise reduced data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=y, marker='o', s=0.8)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=TEST_SIZE, random_state=42)

# Plot Results
fig = plt.figure(figsize=(24, 6))
(ax1, ax2, ax3, ax4) = fig.subplots(1, 4)
fig.suptitle('ML Model Confusion Matrices', size=16)

# Logistic Regression
lr = LogisticRegression()
time_start_lr = time.perf_counter()
lr_model = lr.fit(X_train, y_train)
time_elapsed_lr = (time.perf_counter() - time_start_lr)
# Model Values
y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)*100
plot_confusion_matrix(lr_model, X_test, y_test, ax=ax1, normalize='pred')
ax1.title.set_text('Logistic Regression Confusion Matrix')

# Decision Tree
clf = DecisionTreeClassifier(random_state=0, max_depth=8)
time_start_dt = time.perf_counter()
dt_model = clf.fit(X_train, y_train)
time_elapsed_dt = (time.perf_counter() - time_start_dt)
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)*100
plot_confusion_matrix(dt_model, X_test, y_test, ax=ax2, normalize='pred')
ax2.title.set_text('Decision Tree Confusion Matrix')

# Random Forest
rf = RandomForestClassifier(random_state=0, max_depth=8 )
time_start_rf = time.perf_counter()
rf_model = rf.fit(X_train, y_train)
time_elapsed_rf = (time.perf_counter() - time_start_rf)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)*100
plot_confusion_matrix(rf_model, X_test, y_test, ax=ax3, normalize='pred')
ax3.title.set_text('Random Forest Confusion Matrix')

# k-Nearest Neighbour
k_acc_scores = []
k_values = [i for i in range(1, 30, 2)]
time_start_k = time.perf_counter()
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    cross_scores = cross_val_score(knn, X_train, y_train, cv=None, scoring='accuracy')
    k_acc_scores.append(cross_scores.mean())

optimal_k = k_values[k_acc_scores.index(max(k_acc_scores))]
time_elapsed_k = (time.perf_counter() - time_start_k)
# kNN model
knn = KNeighborsClassifier(n_neighbors=optimal_k, n_jobs=-1)
time_start_knn = time.perf_counter()
knn_model = knn.fit(X_train, y_train)
time_elapsed_knn = (time.perf_counter() - time_start_knn)
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn) * 100
plot_confusion_matrix(knn_model, X_test, y_test, ax=ax4, normalize='pred')
ax4.title.set_text('Random Forest Confusion Matrix')
plt.show()

# Logistic Regression Metrics
print("Logistic Regression Metrics")
print("LR Computation Time:%5.4f seconds" % time_elapsed_lr)
print("LR Accuracy: %.4f" % lr_accuracy)

# Decision Tree Metrics
print("Decision Tree Metrics")
print("DT Computation Time:%5.4f seconds" % time_elapsed_dt)
print("DT Accuracy: %.4f" % dt_accuracy)

# Random Forest Metrics
print("Decision Tree Random Forest Metrics")
print("DTRF Computation Time:%5.4f seconds" % time_elapsed_rf)
print("DTRF Accuracy: %.4f" % rf_accuracy)

# kNN Metrics
print("K-Nearest Neighbour Metrics")
print("kNN Computation Time:%5.4f seconds" % time_elapsed_rf)
print("kNN Accuracy: %.4f" % rf_accuracy)
print("kNN Optimal K: %.4f" % optimal_k)
