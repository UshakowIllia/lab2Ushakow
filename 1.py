import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support


# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 1000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

if X.size == 0:
    raise ValueError("No data read from the file or all lines contained missing values ('?')")

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Створення та навчання SVМ-класифікатора з параметрами dual=False і збільшеною кількістю ітерацій
classifier = OneVsOneClassifier(LinearSVC(random_state=0, dual=False, max_iter=10000))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Обчислення F-міри для SVМ-класифікатора
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

# Передбачення результату для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

# Використання класифікатора для кодованої точки даних та виведення результату
predicted_class = classifier.predict(input_data_encoded)
print(label_encoder[-1].inverse_transform(predicted_class)[0])

# Обчислення показників якості класифікації
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")

# Створення SVM з поліноміальним ядром
poly_svm = SVC(kernel='poly', degree=8)
poly_svm.fit(X_train, y_train)
y_pred_poly = poly_svm.predict(X_test)

# Створення SVM з гаусовим ядром
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

# Створення SVM з сигмоїдальним ядром
sigmoid_svm = SVC(kernel='sigmoid')
sigmoid_svm.fit(X_train, y_train)
y_pred_sigmoid = sigmoid_svm.predict(X_test)

print("Shape of y_test:", y_test.shape)
print("Shape of y_test_pred:", y_test_pred.shape)

# Перевірка помилок під час побудови моделей SVM
if not hasattr(poly_svm, "fit"):
    raise ValueError("SVM model with polynomial kernel was not properly trained.")
if not hasattr(rbf_svm, "fit"):
    raise ValueError("SVM model with RBF kernel was not properly trained.")
if not hasattr(sigmoid_svm, "fit"):
    raise ValueError("SVM model with sigmoid kernel was not properly trained.")

# Обчислення показників якості класифікації для поліноміального SVM
accuracy_poly = accuracy_score(y_test, y_pred_poly)
precision_poly = precision_score(y_test, y_pred_poly)
recall_poly = recall_score(y_test, y_pred_poly)
f1_poly = f1_score(y_test, y_pred_poly)

# Обчислення показників якості класифікації для гаусового SVM
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
precision_rbf = precision_score(y_test, y_pred_rbf)
recall_rbf = recall_score(y_test, y_pred_rbf)
f1_rbf = f1_score(y_test, y_pred_rbf)

# Обчислення показників якості класифікації для сигмоїдального SVM
accuracy_sigmoid = accuracy_score(y_test, y_pred_sigmoid)
precision_sigmoid = precision_score(y_test, y_pred_sigmoid)
recall_sigmoid = recall_score(y_test, y_pred_sigmoid)
f1_sigmoid = f1_score(y_test, y_pred_sigmoid)


# Виведення результатів

print("\nPolynomial SVM:")
print(f"Accuracy: {accuracy_poly * 100:.2f}%")
print(f"Recall: {recall_poly * 100:.2f}%")
print(f"Precision: {precision_poly * 100:.2f}%")
print(f"F1 Score: {f1_poly * 100:.2f}%")

print("\nGaussian SVM:")
print(f"Accuracy: {accuracy_rbf * 100:.2f}%")
print(f"Recall: {recall_rbf * 100:.2f}%")
print(f"Precision: {precision_rbf * 100:.2f}%")
print(f"F1 Score: {f1_rbf * 100:.2f}%")

print("\nSigmoid SVM:")
print(f"Accuracy: {accuracy_sigmoid * 100:.2f}%")
print(f"Recall: {recall_sigmoid * 100:.2f}%")
print(f"Precision: {precision_sigmoid * 100:.2f}%")
print(f"F1 Score: {f1_sigmoid * 100:.2f}%")