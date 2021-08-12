import numpy as np
from board_logic import BoardLogic


class Expectimax2048:
    # Constructor of the Expectimax algorithm agent
    def __init__(self, board_logic: BoardLogic):
        # Get the board logic
        self.board_logic = board_logic

    # First heuristic used in the algorithm
    # The sum of all elements squared in a board
    # We will want to generate tiles with bigger and bigger number
    @staticmethod
    def calculate_sum_of_square(board_logic: BoardLogic):
        # Get the board
        board = board_logic.board
        # Find the power of 2 to all tiles' values of the board
        all_elements_squared = np.power(board, 2)
        # Sum all of the squared elements
        return np.sum(all_elements_squared)

    # Second heuristic of the algorithm
    # The total amount of empty tiles left in the board
    # By having more empty tiles, we are safer from game over
    @staticmethod
    def get_amount_of_empty_tiles(board: BoardLogic):
        # Use the Numpy count_nonzero with additional condition
        # Which is counting only the element valued with 0
        return np.count_nonzero(board.board == 0)

    # Third heuristic of the algorithm
    # Monotonicity of the board
    # Instinctively, we would want to pile tiles with big value to the corner
    # Thus, making it easier for us to direct our direction to the gameplay
    # This is the heuristic to exactly measure that trick
    @staticmethod
    def calculate_monotonicity(board_logic: BoardLogic):
        # Get the board
        board = board_logic.board
        # Initialize the heuristic value for each direction
        [monotonic_up_dir, monotonic_down_dir, monotonic_right_dir,
            monotonic_left_dir] = [0 for _ in range(4)]

        # Iterate over all tiles in column direction
        # to calculate monotonicity in UP and DOWN direction
        for x in range(4):
            # Initialize the two indices used together later
            current_row_idx = 0
            next_row_idx = current_row_idx + 1

            # Iterate until the next index is less than 4
            # to avoid out of range index
            while next_row_idx < 4:
                # Increase the next index while the next tile is 0
                while not board[next_row_idx, x] and next_row_idx < 3:
                    next_row_idx += 1
                # Get the current tile and its log value
                current_tile_value = board[current_row_idx, x]
                log_tile_value = np.log2(
                    current_tile_value) if current_tile_value else 0

                # Do the same for the next tile
                next_tile_value = board[next_row_idx, x]
                log_next_value = np.log2(
                    next_tile_value) if next_tile_value else 0

                # Check monotonicity
                if log_tile_value > log_next_value:
                    monotonic_up_dir += log_next_value - log_tile_value
                elif log_tile_value < log_next_value:
                    monotonic_down_dir += log_tile_value - log_next_value

                # Increment the index
                current_row_idx = next_row_idx
                next_row_idx += 1

        # Iterate over all tiles in row direction
        # to calculate monotonicity in LEFT and RIGHT direction
        for y in range(4):
            # Initialize the two indices used together later
            current_col_idx = 0
            next_col_idx = current_col_idx + 1

            # Iterate until the next index is less than 4
            # to avoid out of range index
            while next_col_idx < 4:
                # Increase the next index and skip empty tile
                while not board[y, next_col_idx] and next_col_idx < 3:
                    next_col_idx += 1
                # Get the current tile and its log value
                current_tile_value = board[y, current_col_idx]
                log_tile_value = np.log2(
                    current_tile_value) if current_tile_value else 0

                # Do the same for the next tile
                next_tile_value = board[y, next_col_idx]
                log_next_value = np.log2(
                    next_tile_value) if next_tile_value else 0

                # Check monotonicity
                if log_tile_value > log_next_value:
                    monotonic_left_dir += log_next_value - log_tile_value
                elif log_tile_value < log_next_value:
                    monotonic_right_dir += log_tile_value - log_next_value

                # Increment the index
                current_col_idx = next_col_idx
                next_col_idx += 1

        # Calculate the sum of the maximum monotonicity in horizontal-vertical direction
        max_monotonicity_vertical = max(monotonic_up_dir, monotonic_down_dir)
        max_monotonicity_horizontal = max(
            monotonic_left_dir, monotonic_right_dir)

        # Return the sum of both of them
        return max_monotonicity_vertical + max_monotonicity_horizontal

    # Last heuristic of the algorithm
    # Smoothness of the board
    # Smoothness heuristic is about how easy it is to merge tiles
    # Basically calculating the differences between neighboring tiles
    # This heuristic is a handler of monotonicity where adjacent tiles
    # tend to be decreasing in values, without being able to be merged
    @staticmethod
    def calculate_smoothness(board_logic: BoardLogic):
        # Get the board
        board = board_logic.board
        # Initialize the heuristic score and root all elements of the board to 2
        smoothness_heuristic = 0
        squared_board = np.sqrt(board)

        # Calculate the smoothness of the board
        # Iterate over all index
        for idx in range(3):
            # Calculate row smoothness
            neighboring_differences = np.abs(
                squared_board[:, idx] - squared_board[:, idx + 1])
            smoothness_heuristic -= np.sum(neighboring_differences)

            # Calculate column smoothness
            neighboring_differences = np.abs(
                squared_board[idx, :] - squared_board[idx + 1, :])
            smoothness_heuristic -= np.sum(neighboring_differences)

        return smoothness_heuristic

    # Calculate the utility with heuristics defined above
    # Weight each heuristic to know which one is more important than the others
    @staticmethod
    def calculate_utility(board_logic: BoardLogic, empty_weight=100000, smoothness_weight=3, monotonic_weight=10000):
        # Calculate sum of square of all tiles utility
        sos_utility = Expectimax2048.calculate_sum_of_square(board_logic)
        # Calculate empty tiles utility
        empty_tiles_utility = Expectimax2048.get_amount_of_empty_tiles(
            board_logic) * empty_weight
        # Calculate monotonicity utility
        monotonicity_utility = Expectimax2048.calculate_monotonicity(
            board_logic) * monotonic_weight
        # Calculate smoothness utility
        smoothness_utility = Expectimax2048.calculate_smoothness(
            board_logic) ** smoothness_weight

        # Utility List
        utility_list = [sos_utility, empty_tiles_utility,
                        monotonicity_utility, smoothness_utility]

        # Sum them all together
        return sum(utility_list)

    # Add a new random tile to the board virtually
    # Simulate the chances of 2 or 4 random valued tile
    @staticmethod
    def simulate_random_tile(board_logic: BoardLogic, depth=0):
        # Get the empty tiles and get the amount of them
        empty_tiles = board_logic.get_empty_cells()
        num_empty_tiles = Expectimax2048.get_amount_of_empty_tiles(board_logic)

        # Determine certain conditions to directly return the utility calculation
        if depth >= 3 and num_empty_tiles >= 6:
            return Expectimax2048.calculate_utility(board_logic)
        if depth >= 5 and num_empty_tiles >= 0:
            return Expectimax2048.calculate_utility(board_logic)
        if num_empty_tiles == 0:
            _, max_utility = Expectimax2048.maximizing_utility(
                board_logic, depth + 1
            )
            return max_utility

        # Check every empty tiles
        total_utility = 0
        for pos in empty_tiles:
            for random_tile_value in [2, 4]:
                # Insert the tile to new copied board
                copied_board = board_logic.copy_board()
                copied_board.insert_new_tile(random_tile_value, pos)

                # Get the new maximum utility
                _, new_utility = Expectimax2048.maximizing_utility(
                    copied_board, depth + 1)

                # Calculate the new utility based on the random tile's value
                if random_tile_value == 2:
                    new_utility *= (0.9 / num_empty_tiles)
                else:
                    new_utility *= (0.1 / num_empty_tiles)

                # Add the new calculated utility to the total utility
                total_utility += new_utility

        return total_utility

    # A function to calculate the best move that will give us maximum utility for a certain board condition
    @staticmethod
    def maximizing_utility(board_logic: BoardLogic, depth=0):
        # Initialize default variables
        best_move_dir = None
        best_utility = -np.inf

        # Get all valid moves to `prune` the possible tree
        all_available_moves = board_logic.find_valid_moves()

        # Iterate over all valid moves and produce the board with the applied move
        for move_dir in all_available_moves:
            # Copy the board
            copied_board = board_logic.copy_board()
            # Apply the move to the copied board
            copied_board.move(move_dir)

            # Simulate adding random values
            new_utility = Expectimax2048.simulate_random_tile(
                copied_board, depth + 1)

            # Check and replace both best move and best utility if the new value is better
            if new_utility >= best_utility:
                best_utility = new_utility
                best_move_dir = move_dir

        return best_move_dir, best_utility

    # Get the best move with Expectimax maximizing algorithm
    @staticmethod
    def expectimax_get_best_move(board_logic: BoardLogic):
        best_move_dir, _ = Expectimax2048.maximizing_utility(board_logic)
        return best_move_dir
