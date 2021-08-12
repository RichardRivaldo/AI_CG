# Expectimax-2048

An Artificial Intelligence model for 2048 with Expectimax Algorithm

### Description

`2048` is a classic modern board game that is played in a 4x4 grid board. And so, it will have 16 blocks where the block will contain a certain number that will always adhere to the power of 2.

Each turn, the board will be inserted with a block with both `randomized` value and position. The valid value that the new blocks will have is either `2` or `4`, with 2 having `90%` probabilities distribution, and the rest for 4. The position will be the randomization of empty tiles, and therefore it is absolute that new tiles will appear in each turn.

Because of its randomness and unpredictability, the game is highly compatible with the `Expectimax` algorithm. Previously, in `Bothicc`, the AI uses `Minimax` algorithm. `Expectimax` is the variation of `Minimax` algorithm since the latter one will have trouble in adapting to randomness and inoptimality.

Different than `Minimax`, `Expectimax` uses the term `Chance Node` and in that node, the algorithm will calculate the average value of its successors utility to take into account the unpredictability that the board will do in the next turn.

The utility itself is calculated with a certain `Evaluation Function` that uses 4 `Heuristics Value`. They are the `Sum of Square` of all elements in the board, the `Number of Empty Tiles` in the board, `Monotonicity` to determine tiles order either in increasing or decreasing manner, and lastly `Smoothness` to make it possible for the agent to handle over-monotonous board and will make the agent able to merge tiles while keeping the monotonous structure.

The AI successfully won (reaching 2048) in `9` out of `10` games, where the one time it lost, it still produced the `1024` tile. The average score in these games would be higher than `50.000` approximately (since I've not really calculated it).

### Guide

-   Go to the home directory of the project through `Terminal` or `Command Prompt` with the `cd <Expectimax-2048_directory>`. Don't get too deep to the `src` folder!
-   Enter the command needed to run an algorithm. The example of the command will be given below.
    -   `python src/main_2048.py`
-   that's it. Sit back and relax watching the AI plays the game. You might feel upset if your AI doesn't hear what you tell him inside, but that's fair. Who knows what does it have inside?
-   Anyway, the `Tkinter` GUI will show us the AI approximation score at the end of the game, also the 4 top tiles produced in our game.

### Creator

-   Richard Rivaldo / 13519185

### Possible Improvements

-   Better UI/UX
-   Better and faster algorithm
-   More heuristics and more precise weights

### EPIC GAME(S?)

- The best record is actually astounding and amazing, where it successfully generated both `8192` and `4096` blocks. The highest score nearly reaches `160.000`, WOOHOO!
![best_score](https://user-images.githubusercontent.com/60037073/125831973-2b56fe4f-f0fe-4fe6-885d-6a99f407a00a.PNG)


### References

-   [2048](https://github.com/gjdanis/2048)
-   [2048 Expectimax AI](https://github.com/kairess/2048_expectimax_ai)
-   [2048 Game Using Expectimax](https://github.com/Wahab16/2048-Game-Using-Expectimax)
-   [2048 Python](https://github.com/yangshun/2048-python)
-   [Adversarial Search](https://courses.cs.washington.edu/courses/cse473/11au/slides/cse473au11-adversarial-search.pdf)
-   [AIProg](https://github.com/hakon0601/AIProg/tree/master/AIProg_Module_4)
-   [An artificial intelligence for the 2048 game](https://iamkush.me/an-artificial-intelligence-for-the-2048-game/)
-   [Expectimax Algorithm in Game Theory.](https://www.geeksforgeeks.org/expectimax-algorithm-in-game-theory/)
-   [Games I](https://web.stanford.edu/class/archive/cs/cs221/cs221.1196/lectures/games1.pdf)
-   [Lecture7: Expectimax and Utilities](https://www.youtube.com/watch?v=r-RxPnnp__o)
-   [What is the optimal algorithm for the game 2048?](https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048/22498940#22498940)
-   [Writing a 2048 AI](https://www.robertxiao.ca/hacking/2048-ai/)
