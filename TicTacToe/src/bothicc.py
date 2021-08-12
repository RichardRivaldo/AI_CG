import numpy as np
from math import inf
from random import randint


class tic_tac_toe:
    # Constructor of the game
    def __init__(self):
        # Randomize who starts first
        self.bot_turn = False if randint(0, 1) else True
        # Assume the board is 3x3 and fill it with "_", meaning that it is still empty
        self.board = np.full((3, 3), "_")

    # Print the board
    def print_board(self):
        # Initialize index for indexing the board
        idx = len(self.board) - 1
        # Print each row in the reversed board
        for row in self.board[::-1]:
            print("{idx}|".format(idx=idx), end="")
            for tile in row[:-1]:
                print(tile, end="|")
            print(row[-1], end="|\n")
            idx -= 1
        for i in range(len(self.board)):
            print("", i, end=" ")
        print("")

    # Check valid moves from a pair of index
    def valid_move(self, i, j):
        # Index not out of range (0 <= i, j <= len(self.board) - 1)
        # The element of the board is still empty ("_")
        return (
            i >= 0
            and i < len(self.board)
            and j >= 0
            and j < len(self.board)
            and self.board[i][j] == "_"
        )

    # Apply a move to a tile on position (i, j)
    def make_move(self, i, j):
        # Check if the move is valid
        if not self.valid_move(i, j):
            print("[Invalid Move!]")
        else:
            # Check whose turn the game is currently
            if self.bot_turn:
                # Apply the sign and change the turn
                (self.board)[i][j] = "X"
                self.bot_turn = False
            else:
                (self.board)[i][j] = "O"
                self.bot_turn = True

    # Check if a row is already won by a player
    def row_won(self):
        for row in self.board:
            # Check if the set is unique
            # 1 for Bothicc, -1 for Human, 0 for else cases
            row = set(row)
            if len(row) == 1:
                if "X" in row:
                    return 1
                elif "O" in row:
                    return -1
        return 0

    # Check if a column is already won by a player
    def col_won(self):
        for i in range(len(self.board)):
            # Append each column to a list
            check = []
            for j in range(len(self.board)):
                check.append(self.board[j][i])
            # Check the unique set
            col = set(check)
            if len(col) == 1:
                if "X" in col:
                    return 1
                elif "O" in col:
                    return -1
        return 0

    # Check if a diagonal is already won by a player
    def diagonal_won(self):
        # Get both main and secondary diagonal with Numpy Diag and Fliplr
        right_diag = set(np.diag(self.board))
        left_diag = set(np.diag(np.fliplr(self.board)))

        # Check the unique set
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

    # Check the status of the game
    # Return finish status and player property to decide who wins the game
    # Finish status is True if the game is over
    def is_finished(self):
        # Check every winning possibilities in the board
        row_eval = self.row_won()
        col_eval = self.col_won()
        diag_eval = self.diagonal_won()

        # Player property is 1 for Bothicc (X), -1 for Human (O), and 0 for Tie
        if row_eval == 1 or col_eval == 1 or diag_eval == 1:
            return (True, 1)
        elif row_eval == -1 or col_eval == -1 or diag_eval == -1:
            return (True, -1)
        elif not any("_" in row for row in self.board):
            return (True, 0)
        else:
            return (False, 0)

    # Reset the board at a certain position
    def reset_board(self, i, j):
        self.board[i][j] = "_"

    # Get all possible valid moves positions
    def get_possible_moves(self):
        possible_moves = []
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.valid_move(i, j):
                    possible_moves.append((i, j))
        return possible_moves

    # Minimax Maximizer with Alpha-Beta Pruning
    # Return the maximum value of a move at a position
    def maximizer(self, alpha, beta):
        # Initialize the variables
        max_value = -inf
        move_x = -1
        move_y = -1

        # Get the finish status and player property
        finished, player_won = self.is_finished()

        # Check whose the game is in advantage of
        # Evaluation Function
        if finished:
            if player_won == 1:
                return (10, 0, 0)
            elif player_won == -1:
                return (-10, 0, 0)
            else:
                return (0, 0, 0)

        # Get all possible moves
        possible_moves = self.get_possible_moves()

        # Check every moves
        for moves in possible_moves:
            i, j = moves

            # Try the move and make a dummy move at each position
            self.board[i][j] = "X"
            (min_value, dummy_x, dummy_y) = self.minimizer(alpha, beta)
            if min_value > max_value:
                max_value = min_value
                move_x, move_y = i, j
            # Reset the board as before
            self.reset_board(i, j)

            # Prune the check with Alpha-Beta Pruning if possible
            alpha = max(alpha, max_value)
            if alpha >= beta:
                return (alpha, move_x, move_y)

        return (max_value, move_x, move_y)

    # Minimax Minimizer with Alpha-Beta Pruning
    # Return the minimum value of a move at a position
    def minimizer(self, alpha, beta):
        # Initialize the variables
        min_value = inf
        move_x = -1
        move_y = -1

        # Get the finish status and player property
        finished, player_won = self.is_finished()

        # Check whose the game is in advantage of
        # Evaluation Function
        if finished:
            if player_won == 1:
                return (10, 0, 0)
            elif player_won == -1:
                return (-10, 0, 0)
            else:
                return (0, 0, 0)

        # Get all possible moves
        possible_moves = self.get_possible_moves()

        # Check every moves
        for moves in possible_moves:
            i, j = moves

            # Try the move and make a dummy move at each position
            self.board[i][j] = "O"
            (max_value, dummy_x, dummy_y) = self.maximizer(alpha, beta)
            if min_value > max_value:
                min_value = max_value
                move_x, move_y = i, j
            # Reset the board as before
            self.reset_board(i, j)

            # Prune the check with Alpha-Beta Pruning if possible
            beta = min(beta, min_value)
            if beta <= alpha:
                return (beta, move_x, move_y)

        return (min_value, move_x, move_y)

    # Move Bothicc with Minimax decision
    def move_by_minimax(self, alpha=-inf, beta=inf):
        # Initialize the algorithm with a maximizer to maximize Bothicc's advantage
        (max_value, move_x, move_y) = self.maximizer(alpha, beta)
        self.make_move(move_x, move_y)

    # Give the user a choice to retry the game again
    def retry(self):
        print("[Wanna try again, LOSER?]")
        choice = input("Enter anything to continue, NO to quit and go suicide.\n")
        if choice.lower() != "no":
            # Reset the game if the player wants to continue
            self.bot_turn = False if randint(0, 1) else True
            self.board = np.full((3, 3), "_")
            self.play()
        else:
            # Terminate the game
            print("\n[Now now, don't cry.]")

    # Simulates the game by integrating the functions and procedures above
    def play(self):
        print("Welcome to the deepest hell of TicTacToe")
        print("The thicc demon, Bothicc, will be your opponent!")
        print("FIGHT AND BE VICTORIOUS, SOLDIER!\n")
        finished, player = self.is_finished()

        # Repeat until the game is finished
        while not finished:
            self.print_board()
            print("")
            if self.bot_turn:
                # Bothicc's Turn
                # Use the Minimax
                print("[BEHOLD, PEASANT!]")
                self.move_by_minimax()
            else:
                # Player's Turn
                # Enter the index of the tile the player want to fill
                print("[MAKE YOUR MOVE!]")
                try:
                    i = int(input("Enter row index: "))
                    j = int(input("Enter column index: "))
                    self.make_move(i, j)
                except ValueError:
                    print("\n[Can't you differentiate number and literals!?]\n")
            finished, player = self.is_finished()

        # The game is over
        # Show who wins the game, the board, and give the player a chance to retry
        self.print_board()
        if player == 1:
            print("\n[Bothicc is superior!]\n")
        elif player == -1:
            print("\n[You're just lucky!]\n")
        else:
            print("\n[Bothicc is still superior!]\n")
        self.retry()


# Main code
if __name__ == "__main__":
    # Initialize the object and directly call the play method
    tic_tac_toe = tic_tac_toe().play()
