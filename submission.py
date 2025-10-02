
import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """Drops a piece into the specified column."""
    new_board = board.copy()
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    return board # Should not reach here if is_valid_location is checked

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board[r * columns + c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board[(r + i) * columns + c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board[(r + i) * columns + c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board[(r - i) * columns + c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """Evaluates the score of a window of cells."""
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    if window.count(piece) == inarow:
        score += 100
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 5
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 2

    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 4

    return score

def score_position(board, piece, configuration):
    """Evaluates the score of the entire board for a given piece."""
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score Horizontal
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win or draw)."""
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """Minimax algorithm for finding the optimal move."""
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    is_terminal = is_terminal_node(board, configuration)

    if is_terminal:
        if check_win(board, 1, configuration): # Player 1 wins
            return (None, 100000000000000)
        elif check_win(board, 2, configuration): # Player 2 wins
            return (None, -10000000000000)
        else: # Game is a draw
            return (None, 0)

    if depth == 0:
        return (None, score_position(board, 1 if maximizingPlayer else 2, configuration))

    if maximizingPlayer:
        value = -np.inf
        column = np.random.choice(valid_locations) # Initialize with a random valid move
        for col in valid_locations:
            row = next(r for r in range(configuration.rows - 1, -1, -1) if board[r * configuration.columns + col] == 0)
            b_copy = board.copy()
            b_copy[row * configuration.columns + col] = 1 # Assume player 1 is maximizing
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else: # Minimizing player
        value = np.inf
        column = np.random.choice(valid_locations) # Initialize with a random valid move
        for col in valid_locations:
            row = next(r for r in range(configuration.rows - 1, -1, -1) if board[r * configuration.columns + col] == 0)
            b_copy = board.copy()
            b_copy[row * configuration.columns + col] = 2 # Assume player 2 is minimizing
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

def minimax_agent(observation, configuration):
    """ConnectX agent that uses the Minimax algorithm to choose a move."""
    board = observation.board
    player = observation.mark

    # The minimax function is designed to maximize for player 1.
    # When player 2 is playing, we want to minimize player 1's score,
    # which is what the 'minimizingPlayer' branch in minimax does.
    # So, we just need to pass the correct player to the minimax function
    # and whether they are the maximizing player in the current context.
    # If the current player is 1, they are the maximizing player.
    # If the current player is 2, they are the minimizing player from player 1's perspective,
    # but they are maximizing their own score, which is the negative of player 1's score.
    # The current minimax implementation assumes player 1 is always maximizing their score
    # and player 2 is always minimizing player 1's score. This aligns with the game's zero-sum nature.
    # So, we just need to call minimax with maximizingPlayer=True if the current player is 1,
    # and maximizingPlayer=False if the current player is 2.

    col, minimax_score = minimax(board, 3, -np.inf, np.inf, player == 1, configuration)


    # Ensure the chosen column is valid
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (e.g., depth 0 and no terminal state)
        # or the returned column is somehow invalid, choose the first valid column.
        return valid_locations[0]

    return int(col)
