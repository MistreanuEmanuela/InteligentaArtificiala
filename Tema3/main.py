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

    def euristic_function(self, move, player):
        matrix1 = copy.deepcopy(self.state)
        if move == 1:
            matrix1[1][2] = (1, player)
        elif move == 2:
            matrix1[0][0] = (2, player)
        elif move == 3:
            matrix1[2][1] = (3, player)
        elif move == 4:
            matrix1[2][0] = (4, player)
        elif move == 5:
            matrix1[1][1] = (5, player)
        elif move == 6:
            matrix1[0][2] = (6, player)
        elif move == 7:
            matrix1[0][1] = (7, player)
        elif move == 8:
            matrix1[2][2] = (8, player)
        elif move == 9:
            matrix1[1][0] = (9, player)

        value_player = 0
        if player == 1:
            opponent = 2
        else:
            opponent = 1

        value_opponent = 0

        for i in range(3):
            ok = 1
            for j in range(3):
                if matrix1[i][j][1] == opponent:
                    ok = 0
                    break
            if ok == 1:
                value_player += 1

        for i in range(3):
            ok = 1
            for j in range(3):
                if matrix1[i][j][1] == player:
                    ok = 0
                    break
            if ok == 1:
                value_opponent += 1

        for i in range(3):
            ok = 1
            for j in range(3):
                if matrix1[j][i][1] == opponent:
                    ok = 0
                    break
            if ok == 1:
                value_player += 1

        for i in range(3):
            ok = 1
            for j in range(3):
                if matrix1[j][i][1] == player:
                    ok = 0
                    break
            if ok == 1:
                value_opponent += 1

        if ((matrix1[0][0][1] == player or matrix1[0][0][1] == 0) and
                (matrix1[2][2][1] == player or matrix1[2][2][1] == 0) and
                (matrix1[1][1][1] == player or matrix1[1][1][1] == 0)):
            value_player += 1

        if (matrix1[0][0][1] == opponent or matrix1[0][0][1] == 0) and (
                matrix1[2][2][1] == opponent or matrix1[2][2][1] == 0) and (
                matrix1[1][1][1] == opponent or matrix1[1][1][1] == 0):
            value_opponent += 1

        if ((matrix1[0][2][1] == player or matrix1[0][2][1] == 0)
                and (matrix1[1][1][1] == player or matrix1[1][1][1] == 0) and
                (matrix1[2][0][1] == player or matrix1[2][0][1] == 0)):
            value_player += 1

        if (matrix1[0][2][1] == opponent or matrix1[0][2][1] == 0) and (
                matrix1[1][1][1] == opponent or matrix1[1][1][1] == 0) and (
                matrix1[2][0][1] == opponent or matrix1[2][0][1] == 0):
            value_opponent += 1

        return value_player - value_opponent

    def best_move(self, player):
        max_key = 0
        dict_moves = {}
        for item in self.available_moves:
            dict_moves[item] = self.euristic_function(item, player)

        max_value = max(dict_moves.values())

        for key, value in dict_moves.items():
            if value == max_value:
                max_key = key

        return max_key


def game():
    input_player = int(input("What player do you choose? (introduce 1 for player1 or 2 for player2):\n"))
    if input_player == 1:
        game1 = NumberScrabble(2)
        while game1.is_final() == -1:
            if game1.get_turn() == game1.get_player_number():
                move = game1.best_move(game1.get_turn())
                print("CALCULATOR move = ", move)
                game1.transition(move, game1.get_turn())
            else:
                move = int(input("Choose a move: "))
                game1.transition(move, game1.get_turn())
            if game1.get_turn() == 1:
                result = 2
            else:
                result = 1
            if game1.is_final() == 1:
                print("Player {} won!".format(result))

    elif input_player == 2:
        game1 = NumberScrabble(1)
        while game1.is_final() == -1:
            if game1.get_turn() == game1.get_player_number():
                move = game1.best_move(game1.get_turn())
                print("CALCULATOR move = ", move)
                game1.transition(move, game1.get_turn())
            else:
                move = int(input("Choose a move: "))
                game1.transition(move, game1.get_turn())
            if game1.get_turn() == 1:
                result = 2
            else:
                result = 1
            if game1.is_final() == 1:
                print("Player {} won!".format(result))
    else:
        print("Invalid input!")


game()
