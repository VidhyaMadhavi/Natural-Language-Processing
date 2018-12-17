

m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
m_confusion_test

pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])
