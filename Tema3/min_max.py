import copy


class NumberScrabble:
    def __init__(self, player_number):
        self.turn = 1
        self.player_number = player_number
        self.available_moves = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.state = [[(0, 0), (0, 0), (0, 0)], [(0, 0), (0, 0), (0, 0)], [(0, 0), (0, 0), (0, 0)]]

    def get_turn(self):
        return self.turn

    def get_player_number(self):
        return self.player_number

    def get_curent_state(self):
        return self.state

    def validate_transition(self, move):
        if move < 1 or move > 9 or move not in self.available_moves:
            return False
        return True

    def transition(self, move, player):
        if self.validate_transition(move):
            if move == 1:
                self.state[1][2] = (1, player)
            elif move == 2:
                self.state[0][0] = (2, player)
            elif move == 3:
                self.state[2][1] = (3, player)
            elif move == 4:
                self.state[2][0] = (4, player)
            elif move == 5:
                self.state[1][1] = (5, player)
            elif move == 6:
                self.state[0][2] = (6, player)
            elif move == 7:
                self.state[0][1] = (7, player)
            elif move == 8:
                self.state[2][2] = (8, player)
            elif move == 9:
                self.state[1][0] = (9, player)

            self.available_moves.remove(move)
            if player == 1:
                self.turn = 2
            else:
                self.turn = 1

        else:
            print("You can't do this move")
            return -1

    def is_final(self):
        for i in range(3):
            ok = 1
            player = self.state[i][0][1]
            for j in range(3):
                if self.state[i][j][0] == 0 or player != self.state[i][j][1]:
                    ok = 0
                    break
            if ok == 1:
                return 1

        for i in range(3):
            ok = 1
            player = self.state[0][i][1]
            for j in range(3):
                if self.state[j][i][0] == 0 or player != self.state[j][i][1]:
                    ok = 0
                    break
            if ok == 1:
                return 1

        player = self.state[1][1][1]
        if (self.state[0][0][1] == self.state[2][2][1] == player and
                self.state[0][0][0] != 0
                and self.state[2][2][0] != 0 and self.state[1][1][0] != 0):
            return 1
        if (self.state[0][2][1] == self.state[2][0][1] == player and
                self.state[0][2][0] != 0 and self.state[2][0][0] != 0 and
                self.state[1][1][0] != 0):
            return 1

        for i in range(3):
            for j in range(3):
                if self.state[i][j][0] == 0:
                    return -1

        return 0

    def euristic_function(self, state, player):
        value_player = 0
        if player == 1:
            opponent = 2
        else:
            opponent = 1

        value_opponent = 0

        for i in range(3):
            ok = 1
            for j in range(3):
                if state[i][j][1] == opponent:
                    ok = 0
                    break
            if ok == 1:
                value_player += 1

        for i in range(3):
            ok = 1
            for j in range(3):
                if state[i][j][1] == player:
                    ok = 0
                    break
            if ok == 1:
                value_opponent += 1

        for i in range(3):
            ok = 1
            for j in range(3):
                if state[j][i][1] == opponent:
                    ok = 0
                    break
            if ok == 1:
                value_player += 1

        for i in range(3):
            ok = 1
            for j in range(3):
                if state[j][i][1] == player:
                    ok = 0
                    break
            if ok == 1:
                value_opponent += 1

        if ((state[0][0][1] == player or state[0][0][1] == 0) and
                (state[2][2][1] == player or state[2][2][1] == 0) and
                (state[1][1][1] == player or state[1][1][1] == 0)):
            value_player += 1

        if (state[0][0][1] == opponent or state[0][0][1] == 0) and (
                state[2][2][1] == opponent or state[2][2][1] == 0) and (
                state[1][1][1] == opponent or state[1][1][1] == 0):
            value_opponent += 1

        if ((state[0][2][1] == player or state[0][2][1] == 0)
                and (state[1][1][1] == player or state[1][1][1] == 0) and
                (state[2][0][1] == player or state[2][0][1] == 0)):
            value_player += 1

        if (state[0][2][1] == opponent or state[0][2][1] == 0) and (
                state[1][1][1] == opponent or state[1][1][1] == 0) and (
                state[2][0][1] == opponent or state[2][0][1] == 0):
            value_opponent += 1

        return value_player - value_opponent


    def is_final_state(self, state):
        for i in range(3):
            ok = 1
            player = state[i][0][1]
            for j in range(3):
                if state[i][j][0] == 0 or player != state[i][j][1]:
                    ok = 0
                    break
            if ok == 1:
                return 1

        for i in range(3):
            ok = 1
            player = state[0][i][1]
            for j in range(3):
                if state[j][i][0] == 0 or player != state[j][i][1]:
                    ok = 0
                    break
            if ok == 1:
                return 1

        player = state[1][1][1]
        if state[0][0][1] == state[2][2][1] == player and state[0][0][0] != 0 and state[2][2][0] != 0 and state[1][1][0] != 0:
            return 1
        if state[0][2][1] == state[2][0][1] == player and state[0][2][0] != 0 and state[2][0][0] != 0 and state[1][1][0] != 0:
            return 1

        for i in range(3):
            for j in range(3):
                if state[i][j][0] == 0:
                    return -1

        return 0

    def difference_move(self, state):
        for i in range(3):
            for j in range(3):
                if self.state[i][j][0] != state[i][j][0]:
                    return state[i][j][0]
        return None

    def available_child(self, state, player):
        child = []
        if state[0][0][0] == 0:
            matrix = copy.deepcopy(state)
            matrix[0][0] = (2, player)
            child.append(matrix)
        if state[0][1][0] == 0:
            matrix = copy.deepcopy(state)
            matrix[0][1] = (7, player)
            child.append(matrix)
        if state[0][2][0] == 0:
            matrix = copy.deepcopy(state)
            matrix[0][2] = (6, player)
            child.append(matrix)
        if state[1][0][0] == 0:
            matrix = copy.deepcopy(state)
            matrix[1][0] = (9, player)
            child.append(matrix)
        if state[1][1][0] == 0:
            matrix = copy.deepcopy(state)
            matrix[1][1] = (5, player)
            child.append(matrix)
        if state[1][2][0] == 0:
            matrix = copy.deepcopy(state)
            matrix[1][2] = (1, player)
            child.append(matrix)
        if state[2][0][0] == 0:
            matrix = copy.deepcopy(state)
            matrix[2][0] = (4, player)
            child.append(matrix)
        if state[2][1][0] == 0:
            matrix = copy.deepcopy(state)
            matrix[2][1] = (3, player)
            child.append(matrix)
        if state[2][2][0] == 0:
            matrix = copy.deepcopy(state)
            matrix[2][2] = (8, player)
            child.append(matrix)

        return child

    def minimax(self, state, depth, is_max_player, player):
        if self.is_final_state(state) == 1:
            if not is_max_player:
                a = 100
                return (a, [])
            else:
                a = -100
                return (a, [])
        if depth == 0:
            return (self.euristic_function(state, player), [])

        if is_max_player:
            best_child = None
            best_value = -float('inf')
            for child in self.available_child(state, 1):
                child_value = self.minimax(child, depth - 1, not is_max_player, player)[0]
                if child_value > best_value:
                    best_value = child_value
                    best_child = child
            return (best_value, best_child)
        else:
            best_child = None
            best_value = float('inf')
            for child in self.available_child(state, 2):
                child_value = self.minimax(child, depth - 1, not is_max_player, player)[0]
                if child_value < best_value:
                    best_value = child_value
                    best_child = child
            return (best_value, best_child)




def game():
    input_player = int(input("What player do you choose? (introduce 1 for player1 or 2 for player2):\n"))
    if input_player == 2:
        game1 = NumberScrabble(1)
        while game1.is_final() == -1:
            if game1.get_turn() == game1.get_player_number():
                state = game1.get_curent_state()
                move = game1.minimax(state, 3, True, 1)
                new_state = move[1]
                new_move = game1.difference_move(new_state)
                print("CALCULATOR move = ", new_move)
                game1.transition(new_move, game1.get_turn())
            else:
                move1 = int(input("Choose a move: "))
                game1.transition(move1, game1.get_turn())
            if game1.get_turn() == 1:
                result = 2
            else:
                result = 1
            if game1.is_final() == 1:
                print("Player {} won!".format(result))
            elif game1.is_final() == 0:
                print("It's draw")
    if input_player == 1:
        game1 = NumberScrabble(2)
        while game1.is_final() == -1:
            if game1.get_turn() == game1.get_player_number():
                state = game1.get_curent_state()
                move = game1.minimax(state, 3, False, 2)
                new_state = move[1]
                new_move = game1.difference_move(new_state)
                print("CALCULATOR move = ", new_move)
                game1.transition(new_move, game1.get_turn())
            else:
                move1 = int(input("Choose a move: "))
                game1.transition(move1, game1.get_turn())
            if game1.get_turn() == 1:
                result = 2
            else:
                result = 1
            if game1.is_final() == 1:
                print("Player {} won!".format(result))
            elif game1.is_final() == 0:
                print("It's draw")
game()
