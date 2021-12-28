"""
DOCSTRING
"""
import sklearn
import tensorflow

class MLApp:
    """
    DOCSTRING
    """
    def __init__(self):
        """
        DOCSTRING
        """
        self.iris = sklearn.datasets.load_iris()
        self.classifier = tensorflow.estimator.LinearClassifier(n_classes=3)
    
    def __call__(self):
        """
        DOCSTRING
        """
        self.classifier.fit(self.iris.data, self.iris.target)
        score = sklearn.metrics.accuracy_score(
            self.iris.target, self.classifier.predict(self.iris.data))
        print('Accuracy:%f' % score)

if __name__ == '__main__':
    MLApp()()
