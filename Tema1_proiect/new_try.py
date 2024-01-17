import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def is_float(value):
    try:
        float_value = float(value)
        return True
    except ValueError:
        return False


def find_type(data):
    if is_float(data[1]):
        return "float"
    if not is_float(data[1]):
        return "str"


def process_value(value):
    if value == "-":
        return None
    else:
        return float(value.replace(',', ''))


df = pd.read_csv('data.csv')


def column_elimination(data):
    delete_columns = []
    for column in data.columns:
        if find_type(data[column]) == "str":
            if np.count_nonzero(data[column] == "-") > len(data[column]) * 0.1:
                delete_columns.append(column)
    delete_columns.extend(["CMS Certification Number (CCN)", "Provider Name", "Address", "ZIP Code", "Certification Date", "Telephone Number"])
    data = data.drop(columns=delete_columns)
    return data


df = column_elimination(df)


def csv_update(df):
    csv_file_name = "output5.csv"
    with open(csv_file_name, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header_row = list(df.columns)
        csv_writer.writerow(header_row)
        max_values_count = max(len(df[column]) for column in df.columns)

        for i in range(max_values_count):
            row = []

            for column in df.columns:
                if i < len(df[column]):
                    row.append(df[column].iloc[i])
                else:
                    row.append(None)

            csv_writer.writerow(row)


def eliminate_row_space(data):
    index_set = set()

    for column in data.columns:
        if find_type(data[column]) == "str":
            outliers = np.where((data[column] == '-'))[0]
            index_set.update(outliers)

    index_list = list(index_set)
    data = data.drop(index_list)
    data.reset_index(drop=True, inplace=True)
    return data

df = eliminate_row_space(df)


def eliminate_outliers(df):
    index_li = []
    for column in df.columns:
        if find_type(df[column]) == "float":
            my_new_data =[float(i.replace(',', '')) for i in df[column] if str(i) != '-']
            q1 = np.percentile(my_new_data, 25)
            q3 = np.percentile(my_new_data, 75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
            for index, i in enumerate(df[column]):
                if i == "-":
                    pass
                else:
                    x = float(i.replace(',', ''))
                    if (x < lower_limit) | (x > upper_limit):
                        index_li.append(index)
    index_set = set(index_li)
    index_list = list(index_set)
    df.drop(index_list, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = eliminate_outliers(df)

def nan_complete(df):
    for column in df.columns:
        if find_type(df[column]) == "float":
            new_array = [float(i.replace(',', '')) for i in df[column] if str(i) != '-']
            mean = np.mean(new_array)
            df[column] = df[column].apply(lambda x: str(mean) if str(x) == "-" else x)
    return df


df = nan_complete(df)
csv_update(df)

delete_columns = []
delete_columns.extend(["State", "City/Town", "Type of Ownership"])
df = df.drop(columns=delete_columns)

def bool_completation(df):
    for column in df:
        if find_type(df[column]) == "str":
            for index, i in enumerate(df[column]):
                if i == "Yes":
                    df[column][index] = str(1)
                else:
                    df[column][index] = str(0)
    return df

df = bool_completation(df)
csv_update(df)

def data_view(data):
    medians = []
    means = []

    for column in data.columns:
        array = [float(i.replace(',', '')) for i in data[column]]
        mean_val = np.mean(array)
        median_val = np.median(array)

        print(f"Mean {column}: {mean_val}")
        print(f"Median {column}: {median_val}")

        means.append(mean_val)
        medians.append(median_val)

    plt.figure(figsize=(15, 10))
    x = np.arange(len(data.columns))
    bar_width = 0.35

    plt.bar(x, means, width=bar_width, label='Mean', color='blue', alpha=0.7)
    plt.bar(x + bar_width, medians, width=bar_width, label='Median', color='orange', alpha=0.7)

    plt.title('Overall Means and Medians of Columns')
    plt.xlabel('Columns')
    plt.ylabel('Value')
    plt.xticks(x + bar_width / 2, data.columns, rotation=45, ha='right', fontsize=8)  # Set the fontsize
    plt.legend()

    plt.tight_layout()

    plt.show()

data_view(df)

def convert_to_float(df):
    for column in df.columns:
        array = [float(i.replace(',', '')) for i in df[column]]
        df[column] = array
    return df

df = convert_to_float(df)

# correlation_matrix = df.corr()
# correlation_with_target = correlation_matrix['Quality of patient care star rating'].sort_values(ascending=False)
# print("Scorurile de corelație cu variabila țintă:")
# print(correlation_with_target)
# selected_variables = correlation_with_target[correlation_with_target < 0].index.tolist()
# df = df.drop(columns=selected_variables)
# csv_update(df)

y = df['Quality of patient care star rating'].values.astype(float)


def mean(column):
    return np.mean(column)


def covariance(column1, column2):
    mean1 = mean(column1)
    mean2 = mean(column2)
    n = len(column1)
    covar = sum((column1[i] - mean1) * (column2[i] - mean2) for i in range(n))
    return covar / (n - 1)


def std_dev(column):
    mean_val = mean(column)
    n = len(column)
    variance = sum((x - mean_val) ** 2 for x in column) / (n - 1)
    return variance ** 0.5


def correlation(column1, column2):
    covar = covariance(column1, column2)
    std_dev1 = std_dev(column1)
    std_dev2 = std_dev(column2)
    correlation_coefficient = covar / (std_dev1 * std_dev2)
    return correlation_coefficient


del_column = []
matrix_correlation = []
for column in df.columns:
    X = df[column]
    correlationn = correlation(X, y)
    matrix_correlation.append(correlationn)
    if abs(correlationn) < 0.5:
        del_column.append(column)


df = df.drop(columns=del_column)
csv_update(df)
print(matrix_correlation)