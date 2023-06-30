``` python
from sklearn.metrics import RocCurveDisplay

display = RocCurveDisplay.from_estimator(lr, X_test, y_test)
display.line_.set_marker("o")
```
![[Pasted image 20230528180133.png]]