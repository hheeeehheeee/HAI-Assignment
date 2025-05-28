import numpy as np

# ----------------------------------------
# 1. 유클리드 거리 함수
# ----------------------------------------


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# ----------------------------------------
# 2. KNN 예측 함수
# ----------------------------------------


def knn_predict(x, X_train, y_train, k):
    distances = [euclidean_distance(x, xi) for xi in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]

    # 다수결
    labels, counts = np.unique(k_labels, return_counts=True)
    return labels[np.argmax(counts)]

# ----------------------------------------
# 3. 정확도 평가 함수
# ----------------------------------------


def compute_accuracy(X_val, y_val, X_train, y_train, k):
    correct = 0
    for x, y in zip(X_val, y_val):
        pred = knn_predict(x, X_train, y_train, k)
        if pred == y:
            correct += 1
    return correct / len(y_val)


# ----------------------------------------
# 4. 데이터 예시
# ----------------------------------------
X = np.array([
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
    [6.0, 9.0], [1.0, 0.6], [9.0, 11.0],
    [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]
])
y = np.array(['A', 'A', 'B', 'B', 'A', 'B', 'B', 'B', 'B'])

# ----------------------------------------
# 5. 훈련/검증 데이터 분할
# ----------------------------------------
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.7 * len(X))

train_idx = indices[:split]
val_idx = indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# ----------------------------------------
# 6. k값 최적화
# ----------------------------------------
k_candidates = range(1, 8)
best_k = None
best_acc = 0.0

print("k값별 정확도:")
for k in k_candidates:
    acc = compute_accuracy(X_val, y_val, X_train, y_train, k)
    print(f"k = {k} → 정확도: {acc:.2f}")
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"\n✅ 최적의 k값은: {best_k} (정확도: {best_acc:.2f})")
