from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
import pickle


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(60000, 784)

X_test = X_test.reshape(10000, 784)

dt = DecisionTreeClassifier()

rf = RandomForestClassifier()
svc = SVC()

print('Training {}'.format('Random Forest'))
rf.fit(X_train, y_train)

print('Training {}'.format('Decision Tree'))
dt.fit(X_train, y_train)


print('Training {}'.format('SVC'))
svc.fit(X_train, y_train)

rf_score = rf.score(X_test, y_test)
print('{} Score: {}'.format('Random Forest', rf_score))

dt_score = dt.score(X_test, y_test)
print('{} Score: {}'.format('Decision Tree', dt_score))

svc_score = svc.score(X_test, y_test)
print('{} Score: {}'.format('SVC', svc_score))

print()

pickle.dump((rf, rf_score), open('RandomForest.pkl','wb'))
pickle.dump((dt, dt_score), open('DecisionTree.pkl','wb'))
pickle.dump((svc, svc_score), open('SVC.pkl','wb'))

