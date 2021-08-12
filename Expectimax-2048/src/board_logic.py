# Libraries
import numpy as np
from constant import UP, DOWN, LEFT, RIGHT, DIRECTIONS


class BoardLogic:
    # Constructor of the board logic
    def __init__(self):
        # Initialize empty matrix of 4x4 as the board
        # Fill it with zeros (empty tiles)
        self.board = np.zeros((4, 4), dtype=np.int32)

    # Extract x and y from a position type
    @staticmethod
    def extract_position(position):
        return position[0], position[1]

    # Insert new tile with certain value on certain position
    def insert_new_tile(self, value, position):
        x, y = self.extract_position(position)
        self.board[x][y] = value

    # Get the value of a certain tile
    def get_tile_value(self, position):
        x, y = self.extract_position(position)
        return self.board[x][y]

    # Copy the board to avoid shallow copy changing it accidentally
    def copy_board(self):
        copied_board = BoardLogic()
        copied_board.board = np.copy(self.board)
        return copied_board

    # Get all empty tiles' position, which is valued by 0
    def get_empty_cells(self):
        empty_cells = []
        for x in range(4):
            for y in range(4):
                if not self.board[x][y]:
                    empty_cells.append((x, y))

        return empty_cells

    # Merge a row recursively to the left of the board
    @staticmethod
    def merge_row(row, result):
        # Base of the recursion
        # There is no more row left to merge
        if len(row) == 0:
            return result

        # Another base of the recursion
        # Simply add the element to the list if it is the only one left
        # Since no merge can be done
        first_elmt = row[0]
        if len(row) == 1:
            return result + [first_elmt]
        # Recursion part of the algorithm
        else:
            # The second element is identical as the first, then these
            # elements can be added (merged) together by doubling its value
            # Add them to the result and merge the sliced list of the rest
            if first_elmt == row[1]:
                return BoardLogic.merge_row(row[2:], result + [first_elmt * 2])
            # Else, recursively just add the element to the result list and slice it
            else:
                return BoardLogic.merge_row(row[1:], result + [first_elmt])

    # Check if there is changes made to the board because of a move
    # by finding the equalities of the matrix
    def find_equalities(self, moved_board):
        return (self.board == moved_board).all()

    # Move to the LEFT direction
    def move_left(self):
        temp_board = []
        # Iterate over each row
        for row in self.board:
            # Filter all zeros in the row
            filtered_row = [element for element in row if element]
            # Merge the row recursively and store the result into an empty list
            merged_left = BoardLogic.merge_row(filtered_row, [])
            # Fill in extra zeros if needed to restore the dimension of the row
            merged_left = merged_left + [0] * (len(row) - len(merged_left))
            # Store the merged row to the temporary board
            temp_board.append(merged_left)

        # Convert the list into numpy array for better processing later
        return np.array(temp_board, dtype=np.int32)

    # Move to the RIGHT direction
    def move_right(self):
        # Flip the board to make it reversed horizontally
        copied_board = self.copy_board()
        copied_board.board = np.fliplr(copied_board.board)
        # Move the board to the left (basically moving it to the right)
        merged_left = copied_board.move_left()
        # Reverse flip the board to apply `move right`
        return np.fliplr(merged_left)

    # Move to the UP direction
    def move_up(self):
        # Transpose the board matrix
        copied_board = self.copy_board()
        copied_board.board = copied_board.board.T
        # Move the board to the left (basically moving it up)
        merged_left = copied_board.move_left()
        # Transpose the board again to apply `move up`
        return merged_left.T

    # Move to the DOWN direction
    def move_down(self):
        # Transpose the board matrix
        copied_board = self.copy_board()
        copied_board.board = copied_board.board.T
        # Move the board to the right (basically moving it down)
        merged_right = copied_board.move_right()
        # Transpose the board again to apply `move down`
        return merged_right.T

    # Wrapper for the move command on certain direction
    # Will return False if the move doesn't change anything of the board
    def move(self, direction):
        temp_board = self.copy_board().board
        if direction == UP:
            temp_board = self.move_up()
        elif direction == DOWN:
            temp_board = self.move_down()
        elif direction == LEFT:
            temp_board = self.move_left()
        elif direction == RIGHT:
            temp_board = self.move_right()

        # Check equalities and replace the board if the board produced
        # by applying the move is different
        if not self.find_equalities(temp_board):
            self.board = temp_board
            return True
        return False

    # Find available direction to transitiate
    def find_direction(self):
        # Initialized all needed variables
        [empty_up, empty_down, empty_left, empty_right] = [
            False for _ in range(4)]

        is_empty_row = [False for _ in range(4)]
        is_empty_col = [False for _ in range(4)]

        # Iterate over all tiles
        for x in range(4):
            found_zero = False
            found_not_zero = False
            for y in range(4):
                # Check if the tile's value is zero
                if self.board[x][y]:
                    found_not_zero = True
                    is_empty_col[y] = True

                    # Check left and up direction
                    if found_zero:
                        empty_left = True
                    if is_empty_row[y]:
                        empty_up = True
                elif not self.board[x][y]:
                    found_zero = True
                    is_empty_row[y] = True

                    # Check right and down direction
                    if found_not_zero:
                        empty_right = True
                    if is_empty_col[y]:
                        empty_down = True

        return [empty_up, empty_down, empty_left, empty_right]

    # Find all valid moves
    def find_valid_moves(self):
        # Get available transitions
        all_dir_transitions = self.find_direction()

        valid_moves = []
        # Check all directions
        for direction in DIRECTIONS:
            # If the transition is valid, then append it to the list
            if all_dir_transitions[direction]:
                valid_moves.append(direction)
            # If not, simulate a move to the current checked direction on the copied board
            # This is in case there are several tiles that can be joined because of identical values
            # but cannot be detected with the method from above
            else:
                copied_board = self.copy_board()
                # If the move is valid, then append it to the list
                if copied_board.move(direction):
                    valid_moves.append(direction)

        return valid_moves

    # Calculate score approximation based on the last matrix values
    def approximate_score(self):
        # A function to help calculating the sequence
        def sequence_helper(n):
            # Base recurrents
            if n == 1:
                return 0
            # Recursion
            else:
                return 2 * sequence_helper(n - 1) + np.power(2, n)

        # Initialize total point
        total_score = 0
        # Flatten the board for easier iteration
        flatten_board = list(self.board.flatten())

        # Iterate over all elements
        for value in flatten_board:
            if value:
                # Find the log2 of the value, which is the n for the sequence
                n_value = np.log2(value)

                # Add the total score given for making a tile with this value
                total_score += sequence_helper(n_value)

        return int(total_score)
