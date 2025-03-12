from sklearn.utils.estimator_checks import parametrize_with_checks

from tabicl import TabICLClassifier


# use n_estimators=2 to test other preprocessing as well
@parametrize_with_checks([TabICLClassifier(n_estimators=2)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
