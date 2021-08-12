# Libraries
import numpy as np


class QLearning:
    # Initiate actions available
    RIGHT, LEFT = 1, -1
    # Initiate special tiles property
    HOLE, BANANA, EMPTY, QTY = "X", "O", "-", "*"

    # Constructor of the Q-Learning for Q-ty
    def __init__(self):
        # Initialize the score and starting position of the bot
        self.score = 0
        self.current_pos = 3

        # Initialize the environment, rewards, and Q-Table
        self.init_environment()
        self.init_rewards()
        self.init_q_table()

        # Initialize hyperparameters to train the agent later
        # Exploration Rate
        self.epsilon = 1.0
        # Maximum and Minimum Epsilon Rate
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        # Discount Rate
        self.gamma = 0.95
        # Learning Rate
        self.alpha = 0.8
        # Epsilon Decay Rate
        self.decay_rate = 0.005

    # Initialize the environment
    def init_environment(self):
        # Initiate the empty board with the shape just as described
        self.env = np.full((10), QLearning.EMPTY)
        # Initiate all special tiles
        self.env[0] = QLearning.HOLE
        self.env[-1] = QLearning.BANANA
        self.env[self.current_pos] = QLearning.QTY

    # Output the environment
    def output_environment(self):
        print("The environment:")
        for tiles in self.env:
            print(tiles, end="|")
        print("\n")

    # Initialize all rewards
    def init_rewards(self):
        # Initialize zeros
        self.rewards = np.zeros(len(self.env))
        # Change the BANANA and HOLE tile reward
        self.rewards[0] = -1
        self.rewards[-1] = 1

    # Get rewards based on position
    def get_rewards(self, position):
        # The BANANA will give +1, HOLE -1, and the rest will be 0
        return self.rewards[position]

    # Output all rewards
    def output_rewards(self):
        print("The rewards:")
        print(self.rewards)

    # Get the current position of the bot
    def get_position(self):
        return self.current_pos

    # Do a move
    def move(self, direction):
        # Get first occurence of the bot
        # In this case, it will be the current position of the bot
        self.current_pos += direction

        # Modify the environment based on the direction
        self.env[self.current_pos] = QLearning.QTY

        # Add reward to the score
        self.score += self.get_rewards(self.get_position())

        # Reset the tile
        self.reset_tile(direction)

    # Reset the position
    def reset_tile(self, direction):
        # Reset the old position based on the direction
        if direction == QLearning.RIGHT:
            # Reset the previous occupied position by the bot
            self.env[self.current_pos - 1] = QLearning.EMPTY
            # Check if the BANANA tile is reached
            if self.get_position() == 9:
                # If yes, then reset the first occurrence + 1 to BANANA tile
                # In this case, the first occurence will be the index of the tile before BANANA
                self.env[self.get_position()] = QLearning.BANANA
                # Reset the player position to the starting tile
                self.current_pos = 3
                self.env[self.current_pos] = QLearning.QTY
        else:
            # Reset the previous occupied position by the bot
            self.env[self.current_pos + 1] = QLearning.EMPTY
            # Check if the HOLE tile is reached
            if self.get_position() == 0:
                # If yes, then reset the first occurrence to HOLE tle
                # In this case, the first occurrence is the index of the HOLE tile
                self.env[self.get_position()] = QLearning.HOLE
                # Reset the player position to the starting tile
                self.current_pos = 3
                self.env[self.current_pos] = QLearning.QTY

    # Initialize Q-Table
    def init_q_table(self):
        # Number of available actions are two, move left or right
        num_of_actions = 2
        # Number of state is the length of the environment
        # Column 0 will be LEFT, and column 1 will be RIGHT
        self.q_table = np.zeros((len(self.env), num_of_actions))

    # Update Q-Value of certain state-action pair
    def update_q_value(self, state, action):
        # Determine the direction based on the action, 0 to the LEFT, 1 to the RIGHT
        dir = self.determine_direction(action)

        # Update the Q-Table first before moving the agent
        # This is done to avoid the agent position got reset
        # after reaching BANANA or HOLE, and thus making it impossible
        # to get the appropiate rewards (always return zero if not)
        self.q_table[state, action] = self.q_table[state, action] * (
            1 - self.alpha
        ) + self.alpha * (
            self.get_rewards(state + dir)
            + self.gamma * np.max(self.q_table[state + dir, :])
        )

    # Output the Q-Table
    def output_q_table(self):
        print("Q-Table:")
        print(self.q_table)

    # Determine direction based on randomized action value
    def determine_direction(self, action):
        # The action is 0, then return the direction of LEFT
        if not action:
            return QLearning.LEFT
        # Else, return the direction of RIGHT
        else:
            return QLearning.RIGHT

    # Update Epsilon of the agent
    def update_epsilon(self, episode):
        # Reduce the epsilon based on the decay rate
        # So that the exploration-exploitation tradeoff can work
        self.epsilon = self.min_epsilon + (
            self.max_epsilon - self.min_epsilon
        ) * np.exp(-self.decay_rate * episode)

    # Check if the agent reached the terminal state, winning or losing the game
    def is_terminal_state(self):
        return self.score == 5 or self.score == -5

    # Train the agent using the hyperparameters
    # Total episodes are maximum iteration of each learning
    # Maximum steps are maximum number of actions taken in an episode
    def train(self, total_ep=1000, max_steps=100):
        # Initialize an empty list to contain all scores obtained in each episode
        all_scores = []

        # Iterate over all episodes
        for episode in range(total_ep):
            # Reset the environment
            self.init_environment()
            # Iterate over all steps
            for step in range(max_steps):
                # Get current position as the current state
                state = self.get_position()

                # Exploration-Exploitation Tradeoff
                # If the random is bigger than epsilon, then do exploitation
                # By choosing action with highest q-value
                if np.random.uniform(0, 1) > self.epsilon:
                    action = np.argmax(self.q_table[state, :])
                else:
                    action = np.random.randint(2)

                # Update the Q-Value of current state and action
                self.update_q_value(state, action)

                # Move the agent to the next state based on the direction
                self.move(self.determine_direction(action))

                # Break the steps if terminal state is reached
                # That is, the agent won or lost in the steps
                if self.is_terminal_state():
                    break

            # Update the epsilon rate of the agent
            self.update_epsilon(episode)

            # Append the score and reset it
            all_scores.append(self.score)
            self.score = 0

        # Output the average score
        print("Average Score:", sum(all_scores) / len(all_scores))

        # Output the Q-Table
        qty.output_q_table()

    # Decide whether Q-ty won or lost the game
    def win_or_lose(self):
        return self.score == 5

    # Q-ty wants to play!
    # Q-ty learned and now it wants to exploit the environment!
    def play_qty(self):
        # Track the total steps taken
        total_steps = 0
        while not self.is_terminal_state():
            # Output the environment
            self.output_environment()
            # Get the action
            action = np.argmax(self.q_table[self.get_position(), :])
            # Determine the direction
            direction = self.determine_direction(action)
            # Q-ty moves!
            self.move(direction)
            # Increase the total steps
            total_steps += 1

        # Output the total steps, the score, and the decision
        print("Total Steps:", total_steps)
        print("Total Score:", self.score)
        if self.win_or_lose():
            print("HOORAY! Q-ty won the game! EZPZ~")


# Main Caller
if __name__ == "__main__":
    # Create the agent
    qty = QLearning()
    # Train the agent
    print("Q-ty is training...")
    qty.train()
    # Play the agent(?)
    choice = input("Training is done! Do you want to let Q-ty play? (Y/N)\n")
    if choice == "Y":
        qty.play_qty()
    else:
        print("Phew, next time, then!")
