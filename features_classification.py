from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras.models import load_model
pretrained_model = load_model('/content/drive/MyDrive/output/Classifier1.h5')

from tensorflow.keras.models import load_model
loaded_model = load_model('/content/drive/MyDrive/output/Classifier1.h5')
layer_names = [layer.name for layer in loaded_model.layers]
print(layer_names)

import os
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

pretrained_model = VGG16(weights='imagenet', include_top=False)
feature_extraction_model = Model(inputs=pretrained_model.input, outputs=pretrained_model.get_layer('block5_pool').output)

image_folder = '/content/drive/'

extracted_features = []

for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(image_folder, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = feature_extraction_model.predict(img)
        extracted_features.append(features)

extracted_features = np.vstack(extracted_features)

import pandas as pd

excel_file_path = '/content/drive/Dataset.csv'
df = pd.read_csv(excel_file_path)

your_labels = df['temperature'].tolist()

label_to_int = {label: idx for idx, label in enumerate(set(your_labels))}
your_labels_numeric = [label_to_int[label] for label in your_labels]

from sklearn.model_selection import train_test_split

X_features = extracted_features
y_labels = your_labels

print("Shape of X_features:", X_features.shape)

num_samples, *other_dimensions = X_features.shape
X_features = X_features.reshape(num_samples, -1)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_auc_score,auc,roc_curve

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred))
confusion_matrix(y_test,y_pred)

des=DecisionTreeClassifier(criterion='gini')
des.fit(X_train,y_train)
y_pred = des.predict(X_test)
print(accuracy_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)

rand = RandomForestClassifier(n_estimators=100)
rand.fit(X_train, y_train)
y_pred = rand.predict(X_test)
print(accuracy_score(y_test, y_pred))
confusion_matrix(y_test,y_pred)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(accuracy_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)

param_grid = {
    'n_estimators': [50,70,100,200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=rand, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_

rand = RandomForestClassifier(criterion= 'gini',max_depth= 8,max_features= 'log2',n_estimators = 100)
rand.fit(X_train, y_train)
y_pred = rand.predict(X_test)
print(accuracy_score(y_test, y_pred))
confusion_matrix(y_test,y_pred)
rand.score(X_train, y_train)
rand.score(X_test, y_test)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_train.shape

from xgboost import XGBClassifier
xgbmodel = XGBClassifier()
xgbmodel.fit(X_train, y_train)
y_pred = xgbmodel.predict(X_test)
print(accuracy_score(y_test, y_pred))
confusion_matrix(y_test,y_pred)

xgbmodel = XGBClassifier(learning_rate=0.05, max_depth=6, n_estimators=60,objective='binary:logistic',booster='gbtree')
xgbmodel.fit(X_train, y_train)
y_pred = xgbmodel.predict(X_test)
print(accuracy_score(y_test, y_pred))
confusion_matrix(y_test,y_pred)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
accuracy = accuracy_score(y_test, test_predictions)
report = classification_report(y_test, test_predictions)
test_predictions = svm_classifier.predict(X_test)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)