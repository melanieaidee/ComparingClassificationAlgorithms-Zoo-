from sklearn.model_selection import RepeatedKFold, cross_val_score

# 10-fold CV repeated 100 times (1,000 train/eval rounds in total)

rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=42)
results = {}
for name, model in models.items():
scores = cross_val_score(model, X_scaled, y, cv=rkf,
scoring='accuracy', n_jobs=-1)
results[name] = (scores.mean(), scores.std())
print(f'{name:20s} mean={scores.mean():.4f} std={scores.std():.4f}')