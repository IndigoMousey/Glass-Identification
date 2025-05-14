import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("glass.csv")

# Select variables
X=df.iloc[:, :-1:].copy().to_numpy()
y=df.iloc[:, -1].copy().to_numpy()

clf = RandomForestClassifier(oob_score=True, verbose=3, n_jobs=-1, class_weight='balanced')
parameters = {"max_depth": range(2, 7)}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X, y)
score_clf = pd.DataFrame(grid_search.cv_results_)
#print(score_clf[['param_max_depth', 'mean_test_score', 'rank_test_score']])

# Fit data based on grid search results
max_depth = grid_search.best_params_["max_depth"]
clf = RandomForestClassifier(max_depth=max_depth, oob_score=True, verbose=3, n_jobs=-1, class_weight='balanced')
clf.fit(X, y)

importances = pd.DataFrame(clf.feature_importances_, index=df.columns[:-1:])
importances.plot.bar()
plt.show()

print(f"Score: {clf.score(X, y):.3f}")
print(f"OOB Score: {clf.oob_score_:.3f}")
print(max_depth)

cm = confusion_matrix(y, clf.predict(X), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
