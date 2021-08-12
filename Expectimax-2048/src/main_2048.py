# Librariess
import numpy as np
from board_logic import BoardLogic
from expectimax import Expectimax2048
from constant import SIZE, GRID_LEN, GRID_PADDING
from tkinter import Frame, Label, Tk, Button, CENTER
from constant import FONT, BACKGROUND_COLOR_DICT, CELL_COLOR_DICT
from constant import BACKGROUND_COLOR_GAME, BACKGROUND_COLOR_CELL_EMPTY


class Board2048(Frame):
    # Constructor of the 2048 GUI
    def __init__(self):
        # Call parent constructor
        Frame.__init__(self)

        # Initialize grid attributes of the GUI
        self.grid()
        self.master.title("2048")
        self.board_grid = []

        # Initialize the board grid
        self.init_board_gui()
        # Generate Board Logic
        self.generate_board_logic()
        # Update the board for the first time
        self.update_board()
        # Use the constructed AI
        self.expectimax = Expectimax2048(self.board)

        # Play the game
        self.play_game()
        # Event loop for the GUI
        self.mainloop()

    # Initialize grid background for the board GUI
    def init_board_gui(self):
        # Initialize grid background for the board
        board_background = Frame(
            self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        board_background.grid()

        # Set properties of every tile in the grid background view
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                # Initialize a frame for the tile
                tile = Frame(board_background, bg=BACKGROUND_COLOR_CELL_EMPTY,
                             width=SIZE / GRID_LEN, height=SIZE / GRID_LEN)
                tile.grid(row=i, column=j, padx=GRID_PADDING,
                          pady=GRID_PADDING)
                # Initialize the label for the tile
                tile_label = Label(master=tile, text="", bg=BACKGROUND_COLOR_CELL_EMPTY,
                                   justify=CENTER, font=FONT, width=GRID_LEN, height=int(GRID_LEN / 2))
                # Set the tile and label to be a grid and append it to the row
                tile_label.grid()
                grid_row.append(tile_label)

            # Append the grid of every row to the board
            self.board_grid.append(grid_row)

    # Generate a new tile on random position
    def generate_random_tile(self):
        # Randomize the value of the tile
        # 90% rate of getting 2, else 4
        tile_value = 2 if np.random.randint(100) < 90 else 4

        # Get the empty cells to put on
        empty_cells = self.board.get_empty_cells()
        # Randomize index of the empty cell position if there is still empty cells
        if empty_cells:
            empty_cell_idx = np.random.randint(0, len(empty_cells))
            # Generate new tile
            new_cell_pos = empty_cells[empty_cell_idx]
            self.board.insert_new_tile(tile_value, new_cell_pos)

    # Generate starting board logic and component
    def generate_board_logic(self):
        # Initialize the logic as the class attribute
        self.board = BoardLogic()

        # Add two random starting tiles to the board
        self.generate_random_tile()
        self.generate_random_tile()

    # Update every tile background color on the board based on the value
    def update_board(self):
        # Iterate over all grids
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                curr_tile = int(self.board.board[i][j])
                # Check if the tile is empty (valued 0)
                if not curr_tile:
                    self.board_grid[i][j].configure(
                        text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                # The tile is not empty
                else:
                    self.board_grid[i][j].configure(text=str(
                        curr_tile), bg=BACKGROUND_COLOR_DICT[curr_tile], fg=CELL_COLOR_DICT[curr_tile])
        # Enter the event loop and exit until idle tasks are completed
        self.update_idletasks()

    # Check if the game is over
    def is_game_over(self):
        return not len(self.board.find_valid_moves())

    # Reset all tiles background color
    def reset_all_tiles(self):
        for i in range(4):
            for j in range(4):
                # Set the background and text to be empty
                self.board_grid[i][j].configure(
                    text="", bg=BACKGROUND_COLOR_CELL_EMPTY)

    # Get top 4 tiles with highest value
    def get_top_tiles(self):
        # Flatten the matrix to make processing easier
        flatten_board = list(self.board.board.flatten())
        # Sort the board by descending order
        sorted_board = sorted(flatten_board, reverse=True)
        # Slice and return the four top tiles
        return sorted_board[:4]

    # Game over message
    def display_game_over(self):
        # Reset the board
        self.reset_all_tiles()

        # Update the game over notification in the GUI
        self.board_grid[1][1].configure(
            text="Top", bg=BACKGROUND_COLOR_CELL_EMPTY)
        self.board_grid[1][2].configure(
            text="Tiles", bg=BACKGROUND_COLOR_CELL_EMPTY)

        # Get the top tiles
        top_tiles = self.get_top_tiles()

        # Iterate to show the score
        for i in range(4):
            self.board_grid[2][i].configure(text=str(
                top_tiles[i]), bg=BACKGROUND_COLOR_DICT[top_tiles[i]], fg=CELL_COLOR_DICT[top_tiles[i]])

        # Show the approximated score with a popup message box
        score = self.board.approximate_score()
        popup = Tk()
        popup.wm_title("Expectimax Approximated Score:")
        label = Label(popup, text=str(score), font=FONT)
        label.pack(side="top", fill="x", pady=10)
        B1 = Button(popup, text="HOORAY!", command=popup.destroy)
        B1.pack()
        popup.mainloop()

        self.update()

    # Play the game and display the GUI
    def play_game(self):
        # Endless loop until the game is over
        while True:
            # Get the best move with Expectimax Algorithm
            best_move_dir = self.expectimax.expectimax_get_best_move(
                self.board)

            # Move the AI with analysis from Expectimax Algorithm
            self.board.move(best_move_dir)
            # Update the tiles from the move applied to the board
            self.update_board()
            # Generate random tile to the board
            self.generate_random_tile()
            # Update the tiles again from the generation
            self.update_board()
            if self.is_game_over():
                # The game is over, display the game over message
                self.display_game_over()
                break
            # Enter the event loop until all pending tasks are done
            self.update()


# Main Caller
if __name__ == "__main__":
    Board2048()
