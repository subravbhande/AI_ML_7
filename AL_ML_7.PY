import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# 1. Load dataset
df = pd.read_csv("BostonHousing.csv")

# 2. Binary classification target
median_price = df['medv'].median()
df['Price_Category'] = np.where(df['medv'] > median_price, 1, 0)

# 3. Separate features and target
X = df.drop(['medv', 'Price_Category'], axis=1)
y = df['Price_Category']

# 4. Impute missing values (mean strategy)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 7. Train SVMs
linear_svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
rbf_svm = SVC(kernel='rbf', C=1, gamma=0.5).fit(X_train, y_train)

# 8. Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)

# 9. Cross-validation
scores = cross_val_score(grid.best_estimator_, X_scaled, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

# 10. Evaluation
y_pred = grid.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
