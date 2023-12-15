import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

file = "seeds_dataset.txt"
# file = "C:/Users/gabri/Desktop/tema4-AI/seeds_dataset.txt"

data = []
with open(file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        values_str = line.strip().split('\t')
        if values_str and len(values_str) == 8:
            values_float = [float(value) for value in values_str]
            data.append(values_float)

classes = [int(data[i][7]) for i in range(len(data))]
classes = list(set(classes))
print("Clasele: ", classes)
date_antrenament = []
date_test = []

for class_o in classes:
    data_output = []
    for line in data:
        if line[7] == class_o:
            data_output.append(line)
    nr_antrenament = int(0.75 * len(data_output))
    indici_antrenament = random.sample(range(len(data_output)), nr_antrenament)
    indici_test = [i for i in range(len(data_output)) if i not in indici_antrenament]
    date_antrenament.extend(data_output[i] for i in indici_antrenament)
    date_test.extend(data_output[i] for i in indici_test)

print("Numar date de antrenament: ", len(date_antrenament))
print("Numar date de test: ", len(date_test))

dict_nr_neurons = {}

entry_layer_dimension = len(date_antrenament[0]) - 1
print("entry_layer_dimension = ",entry_layer_dimension)
dict_nr_neurons[0] = entry_layer_dimension

output_layer_dimension = len(classes)
print("output_layer_dimension = ", output_layer_dimension)

nr_hidden_layers = 2
ant_nr = entry_layer_dimension
for i in range(nr_hidden_layers):
    nr_neurons = int((ant_nr + output_layer_dimension) / 2)
    while nr_neurons < output_layer_dimension:
        nr_neurons += nr_neurons
    while nr_neurons > entry_layer_dimension:
        nr_neurons -= nr_neurons
    dict_nr_neurons[i+1] = nr_neurons
    ant_nr = nr_neurons

dict_nr_neurons[len(dict_nr_neurons)] = output_layer_dimension

learning_rate = 0.001
epochs = 300

dict_w = {}
for i in range(nr_hidden_layers + 1):
    array = np.random.uniform(low=-0.1, high=0.1, size=(dict_nr_neurons[i], dict_nr_neurons[i + 1]))
    dict_w[i] = array

bias = []
for i in range(nr_hidden_layers + 1):
    bias.extend([[np.random.uniform(0.001, 0.4) for j in range(dict_nr_neurons[i+1])]])

output_expected = []
for i in range(len(date_antrenament)):
    output_expected_i = []
    for classs in classes:
        if int(date_antrenament[i][7]) == classs:
            output_expected_i.append(1.0)
        else:
            output_expected_i.append(0.0)
    output_expected.extend([output_expected_i])

print(bias)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    softmax_output = e_x / e_x.sum(axis=0)
    return softmax_output


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)


def error(y_output, y_network):
    err = -(y_output * np.log(y_network)+(1 - y_output) * np.log(1-y_network))
    return err


def error1(y_output, y_network):
    err = 0
    err += np.sum((y_output - y_network) ** 2) / len(y_network)
    return err


def mean_squared_error(y_output, y_network):
    return 0.5 * np.sum((y_output - y_network) ** 2)

dict_output = {}
dict_gradienti = {}
training_errors = []
test_errors = []
ep = [100, 200, 300, 400, 500]


def test_neural_network(test_data, dict_w, bias):
    predictions = []
    for ind, item in enumerate(test_data):
        dict_output = {}
        for i in range(1, nr_hidden_layers + 2):
            if i == 1:
                y_1 = []
                for j in range(dict_nr_neurons[i]):
                    suma = 0
                    for k, value in enumerate(item[:-1]):
                        suma += value * dict_w[i - 1][k][j]
                    out_funct = sigmoid(bias[i - 1][j] + suma)
                    y_1.append(out_funct)
                dict_output[i] = y_1

            elif i < nr_hidden_layers + 1:
                y_s = []
                for index in range(dict_nr_neurons[i]):
                    suma = 0
                    for j in range(dict_nr_neurons[i - 1]):
                        suma += dict_w[i - 1][j][index] * dict_output[i - 1][index]
                    out_funct = sigmoid(bias[i - 1][index] + suma)
                    y_s.append(out_funct)
                dict_output[i] = y_s

            else:
                y_s = []
                input_funct = []
                for index in range(dict_nr_neurons[i]):
                    suma = 0
                    for j in range(dict_nr_neurons[i - 1]):
                        suma += dict_w[i - 1][j][index] * dict_output[i - 1][index]
                    input_funct.append(bias[i - 1][index] + suma)
                out_funct = softmax(input_funct)
                y_s.append(out_funct)
                dict_output[i] = out_funct
        predictions.append(dict_output[nr_hidden_layers + 1])

    return predictions



for epochs in ep:
    for epoch in range(epochs):
        dict_w_new = copy.deepcopy(dict_w)
        bias_new = copy.deepcopy(bias)
        for ind, item in enumerate(date_antrenament):
            for i in range(1, nr_hidden_layers + 2):
                if i == 1:
                    y_1 = []
                    for j in range(dict_nr_neurons[i]):
                        suma = 0
                        for k, value in enumerate(item[:-1]):
                            suma += value * dict_w[i - 1][k][j]
                        out_funct = (bias[i - 1][j] + suma)
                        y_1.append(out_funct)
                    dict_output[i] = sigmoid(np.array(y_1))

                elif i < nr_hidden_layers + 1:
                    y_s = []
                    for index in range(dict_nr_neurons[i]):
                        suma = 0
                        for j in range(dict_nr_neurons[i - 1]):
                            suma += dict_w[i - 1][j][index] * dict_output[i - 1][index]
                        out_funct = (bias[i - 1][index] + suma)
                        y_s.append(out_funct)
                    dict_output[i] = sigmoid(np.array(y_s))

                else:
                    input_funct = []
                    for index in range(dict_nr_neurons[i]):
                        suma = 0
                        for j in range(dict_nr_neurons[i - 1]):
                            suma += dict_w[i - 1][j][index] * dict_output[i - 1][index]
                        input_funct.append(bias[i - 1][index] + suma)
                    out_funct = softmax(input_funct)
                    dict_output[i] = out_funct

            for i in reversed(range(nr_hidden_layers + 2)):

                if i == nr_hidden_layers + 1:
                    gradienti = []
                    input_funct = []
                    for index in range(dict_nr_neurons[i]):
                        suma = 0
                        for j in range(dict_nr_neurons[i - 1]):
                            suma += dict_w[i - 1][j][index] * dict_output[i - 1][j]
                        input_funct.append(bias[i - 1][index] + suma)
                    deriv = softmax_derivative(input_funct)
                    for index in range(dict_nr_neurons[i]):
                        gradient = (output_expected[ind][index] - dict_output[i][index])
                        gradienti.append(gradient)
                    dict_gradienti[i] = gradienti
                    for index in range(dict_nr_neurons[i - 1]):
                        for index1 in range(dict_nr_neurons[i]):
                            dict_w_new[i - 1][index][index1] += learning_rate * dict_output[i - 1][index] * \
                                                                dict_gradienti[i][index1]

                    for index1 in range(dict_nr_neurons[i]):
                        bias_new[i - 1][index1] += learning_rate * dict_gradienti[i][index1]


                elif i > 1:
                    y_s = []
                    for index in range(dict_nr_neurons[i]):
                        suma = 0
                        for j in range(dict_nr_neurons[i - 1]):
                            suma += dict_w[i - 1][j][index] * dict_output[i - 1][index]
                        out_funct = sigmoid_derivative(bias[i - 1][index] + suma)
                        suma2 = 0
                        for k in range(dict_nr_neurons[i+1]):
                            suma2 += dict_gradienti[i+1][k] * dict_w[i][index][k]
                        y_s.append(out_funct * suma2)
                    dict_gradienti[i] = y_s

                    for index in range(dict_nr_neurons[i-1]):
                        for index1 in range(dict_nr_neurons[i]):
                            dict_w_new[i-1][index][index1] = (dict_w_new[i-1][index][index1] + learning_rate * dict_output[i-1][index]
                                                              * dict_gradienti[i][index1])
                    for index1 in range(dict_nr_neurons[i]):
                        bias_new[i-1][index1] = bias_new[i-1][index1] + learning_rate * dict_gradienti[i][index1]

                elif i == 1:
                    y_1 = []
                    for j in range(dict_nr_neurons[i]):
                        suma = 0
                        for k, value in enumerate(item[:-1]):
                            suma += value * dict_w[i - 1][k][j]
                        out_funct = sigmoid_derivative(bias[i - 1][j] + suma)
                        suma2 = 0
                        for k in range(dict_nr_neurons[i+1]):
                            suma2 += dict_gradienti[i+1][k] * dict_w[i][j][k]
                        y_1.append(out_funct * suma2)
                    dict_gradienti[i] = y_1

                    for index in range(dict_nr_neurons[i + 1]):
                        for index1, value in enumerate(item[:-1]):
                            dict_w_new[i - 1][index1][index] = dict_w_new[i - 1][index1][index] + learning_rate * value * \
                                                               dict_gradienti[i][index]
                        bias_new[i - 1][index] = bias_new[i - 1][index] + learning_rate * dict_gradienti[i][index]
        dict_w = copy.deepcopy(dict_w_new)
        bias = copy.deepcopy(bias_new)
    test_predictions = test_neural_network(date_test, dict_w, bias)
    test_predictions_as_lists = [list(prediction) for prediction in test_predictions]
    total_error = 0.0
    print(epochs)
    for i, predicted_class_probabilities in enumerate(test_predictions_as_lists):
        predicted_class = np.argmax(test_predictions[i]) + 1
        true_class = int(date_test[i][7])
        total_error += mean_squared_error(output_expected[i], test_predictions[i])
    print(f"Error on Test Data: {total_error}")
    predicted_labels = [np.argmax(prediction) + 1 for prediction in test_predictions_as_lists]
    true_labels = [int(item[7]) for item in date_test]
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

