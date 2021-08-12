import numpy as np
from math import inf


class tic_tac_toe:
    bot_turn = False

    def __init__(self):
        self.board = np.full((3, 3), "_")

    def print_board(self):
        idx = len(self.board) - 1
        for row in self.board[::-1]:
            print("{idx}|".format(idx=idx), end="")
            for tile in row[:-1]:
                print(tile, end="|")
            print(row[-1], end="|\n")
            idx -= 1
        for i in range(len(self.board)):
            print("", i, end=" ")
        print("")

    def valid_move(self, i, j):
        return (
            i >= 0
            and i < len(self.board)
            and j >= 0
            and j < len(self.board)
            and self.board[i][j] == "_"
        )

    def make_move(self, i, j):
        if not self.valid_move(i, j):
            print("Invalid Move!")
        else:
            if self.bot_turn:
                (self.board)[i][j] = "X"
                self.bot_turn = False
            else:
                (self.board)[i][j] = "O"
                self.bot_turn = True

    def row_won(self):
        for row in self.board:
            row = set(row)
            if len(row) == 1:
                if "X" in row:
                    return 1
                elif "O" in row:
                    return -1
        return 0

    def col_won(self):
        for i in range(len(self.board)):
            check = []
            for j in range(len(self.board)):
                check.append(self.board[j][i])
            col = set(check)
            if len(col) == 1:
                if "X" in col:
                    return 1
                elif "O" in col:
                    return -1
        return 0

    def diagonal_won(self):
        right_diag = set(np.diag(self.board))
        left_diag = set(np.diag(np.fliplr(self.board)))

        if len(right_diag) == 1:
            if "X" in right_diag:
                return 1
            elif "O" in right_diag:
                return -1
        if len(left_diag) == 1:
            if "X" in left_diag:
                return 1
            elif "O" in left_diag:
                return -1
        return 0

    def is_finished(self):
        # Check every winning possibilities in the board
        row_eval = self.row_won()
        col_eval = self.col_won()
        diag_eval = self.diagonal_won()

        # Return finish status and player property to decide who wins the game
        # Finish status is True if the game is over
        # Player property is 1 for Bothicc (X), -1 for Human (O), and 0 for Tie
        if row_eval == 1 or col_eval == 1 or diag_eval == 1:
            return (True, 1)
        elif row_eval == -1 or col_eval == -1 or diag_eval == -1:
            return (True, -1)
        elif not any("_" in row for row in self.board):
            return (True, 0)
        else:
            return (False, 0)

    def reset_board(self, i, j):
        self.board[i][j] = "_"

    def get_possible_moves(self):
        possible_moves = []
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.valid_move(i, j):
                    possible_moves.append((i, j))
        return possible_moves

    def maximizer(self, alpha, beta):
        # Initialize the variables
        max_value = -inf
        min_value = inf
        move_x = -1
        move_y = -1

        possible_moves = self.get_possible_moves()

        # Get the finish status and player property
        finished, player_won = self.is_finished()

        # Check the status
        if finished:
            if player_won == 1:
                return 10
            elif player_won == -1:
                return -10
            else:
                return 0

        if(player_won == 1):
            for moves in possible_moves:
                move_x, move_y = possible_moves


            for (var i = 0; i < moves.length; i++) {
      if (moves[i].score > bestScore) {
        bestScore = moves[i].score;
        bestMove = i;
      }
    }

        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.valid_move(i, j):
                    self.board[i][j] = "X"
                    (min_value, dummy_x, dummy_y) = self.minimizer(alpha, beta)
                    if min_value > max_value:
                        max_value = min_value
                        move_x, move_y = i, j
                    self.reset_board(i, j)

                    if max_value >= beta:
                        return (max_value, move_x, move_y)

                    if max_value > alpha:
                        alpha = max_value

        return (max_value, move_x, move_y)

    def minimizer(self, alpha, beta):
        # Initialize the variables
        min_value = inf
        move_x = -1
        move_y = -1

        # Get the finish status and player property
        finished, player_won = self.is_finished()

        # Check the status
        if finished:
            if player_won == 1:
                return (10, 0, 0)
            elif player_won == -1:
                return (-10, 0, 0)
            else:
                return (0, 0, 0)

        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.valid_move(i, j):
                    self.board[i][j] = "O"
                    (max_value, dummy_x, dummy_y) = self.maximizer(alpha, beta)
                    if min_value > max_value:
                        min_value = max_value
                        move_x, move_y = i, j
                    self.reset_board(i, j)

                    if min_value <= alpha:
                        return (min_value, move_x, move_y)
                    if min_value < beta:
                        beta = min_value

        return (min_value, move_x, move_y)

    def move_by_minimax(self, alpha=-inf, beta=inf):
        (max_value, move_x, move_y) = self.maximizer(alpha, beta)
        self.make_move(move_x, move_y)

    def play(self):
        print("Welcome to the deepest hell of TicTacToe")
        print("The thicc demon, Bothicc, will be your opponent!")
        print("FIGHT AND BE VICTORIOUS, SOLDIER!\n")
        finished, player = self.is_finished()

        while not finished:
            self.print_board()
            print("")
            if self.bot_turn:
                print("[BEHOLD, PEASANT!]")
                self.move_by_minimax()
            else:
                print("[MAKE YOUR MOVE!]")
                i = int(input("Enter i: "))
                j = int(input("Enter j: "))
                self.make_move(i, j)
            finished, player = self.is_finished()

        self.print_board()
        if player == 1:
            print("Bothicc is superior!")
        elif player == -1:
            print("You're just lucky!")
        else:
            print("Bothicc is still superior!")


if __name__ == "__main__":
    tic_tac_toe = tic_tac_toe().play()
