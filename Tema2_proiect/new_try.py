import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('output5.csv')
X = df.iloc[:, 1:6]
y = df['Quality of patient care star rating']
scaler = StandardScaler()
X = scaler.fit_transform(X)

print()
print("RN -------------y is a continue value from 1 to 5 ---------------------")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = Sequential()
model.add(Dense(units=50, activation='relu', input_dim=5))
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=200, batch_size=100, validation_data=(X_test, y_test))
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Mean Absolute Error: {mae}')
predictions = model.predict(X_test)
print(f"Accuracy -  {1 -mae}")

plt.plot(history.history['val_mae'], label='Test MAE')
plt.title('Mean test error')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()

print()
print("RN -------------9 CLASSES WITH FIXED VALUES---------------------")

y = pd.get_dummies(y)
num_classes = 9
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(units=50, activation='relu', input_dim=5))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, batch_size=100, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy on Test Set')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print()
print("--------------------------------------BAYES NAIVE ------------------------------------------")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

label_encoder = LabelEncoder()
y = df['Quality of patient care star rating']
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:6], y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train.astype(str))
X_test_tfidf = vectorizer.transform(X_test.astype(str))


gnb = GaussianNB() # distr gausiana = distr normala
gnb.fit(X_train_scaled, y_train)

y_pred = gnb.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

print()
print("--------------------------------------BAYES NAIVE MULTINOMIAL------------------------------------------")

X = df.iloc[:, 1:6]
y = df['Quality of patient care star rating']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')
print()
print("--------------------------------------LINEAR REGRESSION------------------------------------------")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X = df.iloc[:, 1:]
y = df['Quality of patient care star rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)

y_pred = regressor.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test Mean Absolute Error: {mae}')
print(f'Accuracy: {1- mae}')