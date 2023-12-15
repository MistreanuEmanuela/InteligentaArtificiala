
import numpy as np
import heapq
import math

def matrix_init(state):
    position = 0
    matrix = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            if position < len(state):
                matrix[i][j] = state[position]
                position += 1
    return matrix


def matrix_to_state(matrix_of_state):
    state = []
    for i in range(3):
        for j in range(3):
            state.append(matrix_of_state[i][j])
    return state


def neighbors(state):
    matrix_dict = {}
    for i in range(3):
        for j in range(3):
            matrix_dict[(i, j)] = []
    for i in range(3):
        for j in range(3):
            if (i > 0):
                matrix_dict[(i, j)].append(state[i - 1, j])
            if (i < 2):
                matrix_dict[(i, j)].append(state[i + 1, j])
            if (j > 0):
                matrix_dict[(i, j)].append(state[i, j - 1])
            if (j < 2):
                matrix_dict[(i, j)].append(state[i, j + 1])

    return (matrix_dict)



def neighbors_positions(state):
    matrix_dict = {}
    for i in range(3):
        for j in range(3):
            matrix_dict[(i, j)] = []
    for i in range(3):
        for j in range(3):
            if i > 0:
                matrix_dict[(i, j)].append([i - 1, j])
            if i < 2:
                matrix_dict[(i, j)].append([i + 1, j])
            if j > 0:
                matrix_dict[(i, j)].append([i, j - 1])
            if j < 2:
                matrix_dict[(i, j)].append([i, j + 1])

    return (matrix_dict)



def values(state):
    matrix_v = {}
    for i in range(3):
        for j in range(3):
            matrix_v[(i, j)] = 0
    for i in range(3):
        for j in range(3):
            matrix_v[(i, j)] = (state[i, j])

    return (matrix_v)


def values_0(state):
    for i in range(3):
        for j in range(3):
            if (state[i][j]) == 0:
                return (i, j)

def verify_final_state(state):
    for i in range(len(state) - 1):
        if (state[i + 1] == 0):
            if (i + 2 >= len(state)):
                return True
            elif (state[i] > state[i + 2]):
                # print(i)
                return False
        else:
            if (state[i] > state[i + 1]):
                # print(i)
                return False
    return True



def val_up(matrix, i, j):
    if (i == 0):
        return False
    elif (matrix[i - 1][j] != 0):
        return False
    return True


def val_down(matrix, i, j):
    if (i == 2):
        return False
    elif (matrix[i + 1][j] != 0):
        return False
    return True


def val_left(matrix, i, j):
    if (j == 0):
        return False
    elif (matrix[i][j - 1] != 0):
        return False
    return True


def val_right(matrix, i, j):
    if (j == 2):
        return False
    elif (matrix[i][j + 1] != 0):
        return False
    return True



def transition(state, i, j, direction, last_move):
    matrix = matrix_init(state)
    return_matrix = []
    if (matrix[i][j] == last_move):
        return return_matrix
    else:
        if direction == "up":
            if (val_up(matrix, i, j) == False):
                return return_matrix
            else:
                matrix[i][j], matrix[i - 1][j] = matrix[i - 1][j], matrix[i][j]
        elif direction == "down":
            if (val_down(matrix, i, j) == False):
                return return_matrix
            else:
                matrix[i][j], matrix[i + 1][j] = matrix[i + 1][j], matrix[i][j]
        elif direction == "left":
            if (val_left(matrix, i, j) == False):
                return return_matrix
            else:
                matrix[i][j], matrix[i][j - 1] = matrix[i][j - 1], matrix[i][j]
        elif direction == "right":
            if (val_right(matrix, i, j) == False):
                return return_matrix
            else:
                matrix[i][j], matrix[i][j + 1] = matrix[i][j + 1], matrix[i][j]

    return matrix

##-------------------------------------------
def get_neighbors_of_state(state, last_move):
    matrix_of_state = matrix_init(state)
    pos_0 = values_0(matrix_of_state)

    neighbors = set()

    neighbor_positions = neighbors_positions(matrix_of_state)[pos_0]

    for pos in neighbor_positions:
        i, j = pos

        directions = ["up", "down", "left", "right"]
        for direction in directions:
            new_state = transition(state, i, j, direction, last_move)
            if len(new_state) > 0:
                neighbors.add(tuple(new_state.ravel()))

    return neighbors


def is_matrix_in_visited(visited, target_matrix):
    searched = matrix_to_state(target_matrix)
    for matrix in visited:
        if np.array_equal(matrix, searched):
            return True
    return False


def get_last_moved_cell(current_state, next_state):
    current_matrix = matrix_init(current_state)
    next_matrix = matrix_init(next_state)

    for i in range(3):
        for j in range(3):
            if current_matrix[i][j] != next_matrix[i][j]:
                if current_matrix[i][j] != 0:
                    return current_matrix[i][j]
                else:
                    return next_matrix[i][j]

    return -1




# -----------------ALGORITM IDDFS-----------------
def IDDFS(init_state):
    max_depth = 1
    while True:
        result, moves = depth_limited_DFS(init_state, max_depth)
        if result is not None:
            return result, moves
        max_depth += 1

def depth_limited_DFS(state, max_depth):
    visited = set()
    last_move = -1

    def recursive_dfs(current_state, depth, last_move):
        if depth == 0:
            return None
        if verify_final_state(current_state):
            return [current_state]

        for neighbor in get_neighbors_of_state(current_state, last_move):
            state_hash = tuple(neighbor)

            if state_hash not in visited:
                visited.add(state_hash)
                new_last_move = get_last_moved_cell(current_state, neighbor)
                res = recursive_dfs(neighbor, depth - 1, new_last_move)
                if res is not None:
                    return res

    result = recursive_dfs(state, max_depth, last_move)
    if result is not None:
        return result, max_depth

    return None, max_depth


# print("-------------------ALGORITM IDDFS-------------------")
#
# #initial_state = [1, 2, 3, 5, 7, 0, 6, 4, 8]
# #initial_state = [1, 2, 3, 5, 7, 6, 0, 4, 8]
# #initial_state = [2, 5, 3, 1, 0, 6, 4, 7, 8]
# initial_state = [8, 6, 7, 2, 5, 4, 0, 3, 1]
# print(matrix_init(initial_state))
#
# max_depth = 30
#
# solution = IDDFS(initial_state, max_depth)
#
#
# if solution is not None:
#     print("Solution FOUND:", solution)
# else:
#     print("Solution NOT found for this depth")

# print("-------------------ALGORITM GREEDY-------------------")


def hamming_distance(state):
    matrix_of_state = matrix_init(state)
    i0 = values_0(matrix_of_state)[0]
    j0 = values_0(matrix_of_state)[1]

    distance = 0
    k = 1

    for i in range(3):
        for j in range(3):
            if i == i0 and j == j0:
                continue
            elif matrix_of_state[i][j] != k:
                distance += 1
                k += 1
            else:
                k += 1

    return distance


def values_x(state, x):
    for i in range(3):
        for j in range(3):
            if (state[i][j]) == x:
                return (i, j)


def manhattan_distance(state):
    matrix_of_state = matrix_init(state)

    goal_state = np.zeros((3, 3))

    i0 = values_0(matrix_of_state)[0]
    j0 = values_0(matrix_of_state)[1]

    k = 1

    for i in range(3):
        for j in range(3):
            if i == i0 and j == j0:
                goal_state[i][j] = 0
            else:
                goal_state[i][j] = k
                k += 1

    distance = 0

    for i in range(1, 8):
        modul1 = abs(values_x(matrix_of_state, i)[0] - values_x(goal_state, i)[0])
        modul2 = abs(values_x(matrix_of_state, i)[1] - values_x(goal_state, i)[1])
        distance += modul1 + modul2

    return distance


def euclidean_distance(state):
    matrix_of_state = matrix_init(state)

    goal_state = np.zeros((3, 3))

    i0 = values_0(matrix_of_state)[0]
    j0 = values_0(matrix_of_state)[1]

    k = 1

    for i in range(3):
        for j in range(3):
            if i == i0 and j == j0:
                goal_state[i][j] = 0
            else:
                goal_state[i][j] = k
                k += 1

    distance = 0

    for i in range(1, 8):
        modul1 = (values_x(matrix_of_state, i)[0] - values_x(goal_state, i)[0]) ** 2
        modul2 = (values_x(matrix_of_state, i)[1] - values_x(goal_state, i)[1]) ** 2
        distance = math.sqrt(modul1 + modul2)

    return distance


def greedy(init_state, heuristic_function):
    pq = []
    last_move = -1
    number_moves = 0
    heapq.heappush(pq, (heuristic_function(init_state), (init_state, last_move, number_moves)))

    visited = set()

    while pq:
        priority, (state, last_move, number_moves) = heapq.heappop(pq)
        if verify_final_state(state):
            return state, number_moves

        visited.add(tuple(matrix_init(state).ravel()))
        g = 0
        nmb_moves = number_moves
        for neighbor in get_neighbors_of_state(state, last_move):
            if neighbor not in visited:
                if g == 0:
                    nmb_moves = number_moves+1
                    g = 1
                new_last_move = get_last_moved_cell(state, neighbor)
                heuristic_value = heuristic_function(neighbor)
                heapq.heappush(pq, (heuristic_value, (neighbor, new_last_move, nmb_moves)))

    return None, number_moves



import time

def test(states):
    for state in states:
        print(matrix_init(state))
        print("---------------------------------")
        start_time = time.time()
        solution1 = IDDFS(state)
        end_time = time.time()
        if solution1 is not None:
            print("Solution FOUND by IDDFS:", solution1[0])
        else:
            print("Solution NOT found for this depth")
        print(f"Timpul cautarii:{end_time-start_time}")
        print(f"Numarul de mutari pe care le face:{solution1[1]}")

        print("---------------------------------")
        start_time = time.time()
        solution2 = greedy(state, hamming_distance)
        end_time = time.time()
        if solution2 is not None:
            print("Solution FOUND by Greedy using hamming_distance:", solution2[0])
        else:
            print("Solution NOT by Greedy using hamming_distance")
        print(f"Timpul cautarii:{end_time-start_time}")
        print(f"Numarul de mutari pe care le face:{solution2[1]}")

        print("---------------------------------")
        start_time = time.time()
        solution3 = greedy(state, manhattan_distance)
        end_time = time.time()
        if solution3 is not None:
            print("Solution FOUND by Greedy using manhattan_distance:", solution3[0])
        else:
            print("Solution NOT by Greedy using manhattan_distance")
        print(f"Timpul cautarii:{end_time-start_time}")
        print(f"Numarul de mutari pe care le face:{solution3[1]}")

        print("---------------------------------")
        start_time = time.time()
        solution4 = greedy(state, euclidean_distance)
        end_time = time.time()
        if solution4 is not None:
            print("Solution FOUND by Greedy using euclidean_distance:", solution4[0])
        else:
            print("Solution NOT by Greedy using euclidean_distance")
        print(f"Timpul cautarii:{end_time-start_time}")
        print(f"Numarul de mutari pe care le face:{solution4[1]}")


test([[8, 6, 7, 2, 5, 4, 0, 3, 1], [2, 5, 3, 1, 0, 6, 4, 7, 8], [2, 7, 5, 0, 8, 4, 3, 1, 6]])