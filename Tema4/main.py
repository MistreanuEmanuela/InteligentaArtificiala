import time
def variable_value(matrix_of_state, i, j):
    value = matrix_of_state[i][j][0]
    color = matrix_of_state[i][j][1]
    return value, color


def domain(element):
    if element[0] != 0:
        return [element[0]]
    elif element[1] == "white":
        return [1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif element[1] == "grey":
        return [2, 4, 6, 8]


def square(matrix_of_state, i, j):
    my_square = []
    if i % 3 == 0 and j % 3 == 0:
        for i1 in range(i, i + 3):
            for j1 in range(j, j + 3):
                if i1 != i and j1 != j:
                    my_square.append(matrix_of_state[i1][j1][0])
    elif i % 3 == 0 and j % 3 == 1:
        for i1 in range(i, i + 3):
            for j1 in range(j - 1, j + 2):
                if i1 != i and j1 != j:
                    my_square.append(matrix_of_state[i1][j1][0])
    elif i % 3 == 0 and j % 3 == 2:
        for i1 in range(i, i + 3):
            for j1 in range(j - 2, j + 1):
                if i1 != i and j1 != j:
                    my_square.append(matrix_of_state[i1][j1][0])
    elif i % 3 == 1 and j % 3 == 0:
        for i1 in range(i - 1, i + 2):
            for j1 in range(j, j + 3):
                if i1 != i and j1 != j:
                    my_square.append(matrix_of_state[i1][j1][0])
    elif i % 3 == 1 and j % 3 == 1:
        for i1 in range(i - 1, i + 2):
            for j1 in range(j - 1, j + 2):
                if i1 != i and j1 != j:
                    my_square.append(matrix_of_state[i1][j1][0])
    elif i % 3 == 1 and j % 3 == 2:
        for i1 in range(i - 1, i + 2):
            for j1 in range(j - 2, j + 1):
                if i1 != i and j1 != j:
                    my_square.append(matrix_of_state[i1][j1][0])
    elif i % 3 == 2 and j % 3 == 0:
        for i1 in range(i - 2, i + 1):
            for j1 in range(j, j + 3):
                if i1 != i and j1 != j:
                    my_square.append(matrix_of_state[i1][j1][0])
    elif i % 3 == 2 and j % 3 == 1:
        for i1 in range(i - 2, i + 1):
            for j1 in range(j - 1, j + 2):
                if i1 != i and j1 != j:
                    my_square.append(matrix_of_state[i1][j1][0])
    elif i % 3 == 2 and j % 3 == 2:
        for i1 in range(i - 2, i + 1):
            for j1 in range(j - 2, j + 1):
                if i1 != i and j1 != j:
                    my_square.append(matrix_of_state[i1][j1][0])
    return my_square


def validate_value(value, i, j, matrix_of_state):
    for k in range(9):
        if k != j:
            if value == matrix_of_state[i][k][0]:
                return False
        if k != i:
            if value == matrix_of_state[k][j][0]:
                return False

    my_square = square(matrix_of_state, i, j)
    for i in my_square:
        if i == value:
            return False
    return True


def instance_init(matrix_of_state):
    for i in range(9):
        for j in range(9):
            result = variable_value(matrix_of_state, i, j)
            print("Valoarea variabilei de pe pozitia [{}][{}] este: {}, iar culoarea este: {}".format(
                i, j, result[0], result[1]))

            domain1 = domain(matrix_of_state[i][j])
            print("Domeniul variabilei este: {}".format(domain1))

            available_elements = []
            if len(domain1) > 1:
                for value in domain1:
                    if validate_value(value, i, j, matrix_of_state) is True:
                        available_elements.append(value)

            print("Elementele ce se pot adauga pe aceasta pozitie sunt: {}".format(available_elements))


our_matrix = [
    [(8, "white"), (4, "white"), (0, "white"), (0, "white"), (5, "white"), (0, "white"), (0, "grey"), (0, "white"), (0, "white")],
    [(3, "white"), (0, "white"), (0, "white"), (6, "white"), (0, "white"), (8, "white"), (0, "white"), (4, "white"), (0, "white")],
    [(0, "white"), (0, "white"), (0, "grey"), (4, "white"), (0, "white"), (9, "white"), (0, "white"), (0, "white"), (0, "grey")],
    [(0, "white"), (2, "white"), (3, "white"), (0, "white"), (0, "grey"), (0, "white"), (9, "white"), (8, "white"), (0, "white")],
    [(1, "white"), (0, "white"), (0, "white"), (0, "grey"), (0, "white"), (0, "grey"), (0, "white"), (0, "white"), (4, "white")],
    [(0, "white"), (9, "white"), (8, "white"), (0, "white"), (0, "grey"), (0, "white"), (1, "white"), (6, "white"), (0, "white")],
    [(0, "grey"), (0, "white"), (0, "white"), (5, "white"), (0, "white"), (3, "white"), (0, "grey"), (0, "white"), (0, "white")],
    [(0, "white"), (3, "white"), (0, "white"), (1, "white"), (0, "white"), (6, "white"), (0, "white"), (0, "white"), (7, "white")],
    [(0, "white"), (0, "white"), (0, "grey"), (0, "white"), (2, "white"), (0, "white"), (0, "white"), (1, "white"), (3, "white")],
]

instance_init(our_matrix)

# FORWARD CHECKING


def is_complete(matrix_of_state):
    for i in range(9):
        for j in range(9):
            if matrix_of_state[i][j][0] == 0:
                return False
    return True


def next_unassigned_variable(matrix_of_state):
    for i in range(9):
        for j in range(9):
            if matrix_of_state[i][j][0] == 0:
                return matrix_of_state[i][j], i, j


def backtracking(assignment):
    # print("ASSIGNMENT: \n {} ".format(assignment))
    if is_complete(assignment) is True:
        return assignment

    variable = next_unassigned_variable(assignment)
    i = variable[1]
    j = variable[2]
    for value in domain(variable[0]):
        if validate_value(value, i, j, assignment) is True:
            new_assignment = [row[:] for row in assignment]
            new_assignment[i][j] = (value, assignment[i][j][1])
            res = backtracking(new_assignment)

            if res is not None:
                return res
    return None


print("------------BACKTRACKING--------------------")
start_time = time.time()
backtracking_result = backtracking(our_matrix)
end_time = time.time()
for i in range(len(backtracking_result)):
    print(backtracking_result[i])

def all_domains(matrix_of_state):
    matrix_of_domains = [[None for x in range(9)] for x in range(9)]
    for i in range(9):
        for j in range(9):
            domain1 = domain(matrix_of_state[i][j])

            available_elements = []
            if len(domain1) > 1:
                for value in domain1:
                    if validate_value(value, i, j, matrix_of_state) is True:
                        available_elements.append(value)
            else:
                available_elements.append(matrix_of_state[i][j][0])
            matrix_of_domains[i][j] = available_elements
    return matrix_of_domains


def is_domain_empty(matrix_of_domains):
    for i in range(9):
        for j in range(9):
            if not matrix_of_domains[i][j]:
                return True
    return False

def new_domain(matrix_of_domains, i, j, value):
    new_matrix_of_domains = [row[:] for row in matrix_of_domains]
    new_matrix_of_domains[i][j] = [value]
    for k in range(9):
        if k != j:
           if value in new_matrix_of_domains[i][k]:
               new_matrix_of_domains[i][k].remove(value)
        if k != i:
            if value in new_matrix_of_domains[k][j]:
                new_matrix_of_domains[k][j].remove(value)
    if i % 3 == 0 and j % 3 == 0:
        for i1 in range(i, i + 3):
            for j1 in range(j, j + 3):
                if i1 != i and j1 != j:
                    if value in new_matrix_of_domains[i1][j1]:
                        new_matrix_of_domains[i1][j1].remove(value)
    elif i % 3 == 0 and j % 3 == 1:
        for i1 in range(i, i + 3):
            for j1 in range(j - 1, j + 2):
                if i1 != i and j1 != j:
                    if value in new_matrix_of_domains[i1][j1]:
                        new_matrix_of_domains[i1][j1].remove(value)
    elif i % 3 == 0 and j % 3 == 2:
        for i1 in range(i, i + 3):
            for j1 in range(j - 2, j + 1):
                if i1 != i and j1 != j:
                    if value in new_matrix_of_domains[i1][j1]:
                        new_matrix_of_domains[i1][j1].remove(value)
    elif i % 3 == 1 and j % 3 == 0:
        for i1 in range(i - 1, i + 2):
            for j1 in range(j, j + 3):
                if i1 != i and j1 != j:
                    if value in new_matrix_of_domains[i1][j1]:
                        new_matrix_of_domains[i1][j1].remove(value)
    elif i % 3 == 1 and j % 3 == 1:
        for i1 in range(i - 1, i + 2):
            for j1 in range(j - 1, j + 2):
                if i1 != i and j1 != j:
                    if value in new_matrix_of_domains[i1][j1]:
                        new_matrix_of_domains[i1][j1].remove(value)
    elif i % 3 == 1 and j % 3 == 2:
        for i1 in range(i - 1, i + 2):
            for j1 in range(j - 2, j + 1):
                if i1 != i and j1 != j:
                    if value in new_matrix_of_domains[i1][j1]:
                        new_matrix_of_domains[i1][j1].remove(value)
    elif i % 3 == 2 and j % 3 == 0:
        for i1 in range(i - 2, i + 1):
            for j1 in range(j, j + 3):
                if i1 != i and j1 != j:
                    if value in new_matrix_of_domains[i1][j1]:
                        new_matrix_of_domains[i1][j1].remove(value)
    elif i % 3 == 2 and j % 3 == 1:
        for i1 in range(i - 2, i + 1):
            for j1 in range(j - 1, j + 2):
                if i1 != i and j1 != j:
                    if value in new_matrix_of_domains[i1][j1]:
                        new_matrix_of_domains[i1][j1].remove(value)
    elif i % 3 == 2 and j % 3 == 2:
        for i1 in range(i - 2, i + 1):
            for j1 in range(j - 2, j + 1):
                if i1 != i and j1 != j:
                    if value in new_matrix_of_domains[i1][j1]:
                        new_matrix_of_domains[i1][j1].remove(value)
    return new_matrix_of_domains

def BKT_with_FC(assignment, matrix_of_domains):
    if is_complete(assignment) is True:
        return assignment

    variable = next_unassigned_variable(assignment)
    i = variable[1]
    j = variable[2]
    for value in domain(variable[0]):
        if validate_value(value, i, j, assignment) is True:
            new_assignment = [row[:] for row in assignment]
            new_assignment[i][j] = (value, assignment[i][j][1])
            new_domains = new_domain(matrix_of_domains, i, j, value)
            if is_domain_empty(new_domains) is False:
                res = BKT_with_FC(new_assignment, new_domains)
                if res is not None:
                    return res
    return None


print("------------BACKTRACKING WITH FORWARD CHECKING--------------------")
start_time = time.time()
backtracking_with_FC_result = BKT_with_FC(our_matrix, all_domains(our_matrix))
end_time = time.time()

print(backtracking_with_FC_result)


def next_unassigned_variable_MRV(assignment):
    domains = all_domains(assignment)
    min = 10
    k = -1
    l = -1
    for i in range(9):
        for j in range(9):
            if len(domains[i][j]) < min and assignment[i][j][0] == 0:
                k = i
                l = j
                min = len(domains[i][j])

    return assignment[k][l], k ,l

def BKT_with_FC_MRV(assignment, matrix_of_domains):
    # print("ASSIGNMENT: \n {} ".format(assignment))
    if is_complete(assignment) is True:
        return assignment

    variable = next_unassigned_variable_MRV(assignment)
    i = variable[1]
    j = variable[2]
    for value in domain(variable[0]):
        if validate_value(value, i, j, assignment) is True:
            new_assignment = [row[:] for row in assignment]
            new_assignment[i][j] = (value, assignment[i][j][1])
            new_domains = new_domain(matrix_of_domains, i, j, value)
            if is_domain_empty(new_domains) is False:
                res = BKT_with_FC_MRV(new_assignment, new_domains)
                if res is not None:
                    return res
    return None

print("------------BACKTRACKING WITH FORWARD CHECKING MRV--------------------")
start_time = time.time()
backtracking_with_FC_MRV_result = BKT_with_FC_MRV(our_matrix, all_domains(our_matrix))
end_time = time.time()
for i in range(len(backtracking_with_FC_MRV_result)):
    print(backtracking_with_FC_MRV_result[i])

