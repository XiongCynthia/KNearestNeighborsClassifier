## K-Nearest Neighbors Classifier Implementation

[KNearestNeighborsClassifier](https://github.com/XiongCynthia/KNearestNeighborsClassifier/blob/main/KNearestNeighborsClassifier.py) is a class for classifying data using the k-nearest neighbors method.

### Usage

```python
from KNearestNeighborsClassifier import KNearestNeighborsClassifier
knnc = KNearestNeighborsClassifier()
knnc.fit(x_train, y_train)
y_pred = knnc.predict(x_test)
```

More example usages are included in [KNearestNeighborsClassifier_examples.ipynb](https://github.com/XiongCynthia/KNearestNeighborsClassifier/blob/main/KNearestNeighborsClassifier_examples.ipynb), which additionally showcases the performance of the model for different n_neighbors and sampling methods for compensating class imbalance in the data.
