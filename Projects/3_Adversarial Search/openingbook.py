import pickle, random, time, operator

from isolation import Isolation, DebugState
from collections import defaultdict, Counter

NUM_ROUNDS = 250

def main():
    start = time.time()
    book = build_table()
    end = time.time()
    print("Timer: {}".format(end - start))
    with open("data.pickle", 'wb') as f:
        pickle.dump(book, f)

def build_table(num_rounds=NUM_ROUNDS):
    # Builds a table that maps from game state -> action
    # by choosing the action that accumulates the most
    # wins for the active player. (Note that this uses
    # raw win counts, which are a poor statistic to
    # estimate the value of an action; better statistics
    # exist.)
    book = defaultdict(Counter)
    for _ in range(num_rounds):
        state = Isolation()
        build_tree(state, book)
    result = {}
    for k, v in book.items():
        result[k] = max(v, key=v.get)
    return result

# Depth 4 in specifications of project
def build_tree(state, book, depth=4):
    if (depth == 0):
        debug_board = DebugState.from_state(state)
        print(debug_board)
    #using Open Move Score instead of raw wins
    if depth <= 0 or state.terminal_test():
        return -simulate(state)
    action = alpha_beta_search(state, state.player(), 4)
    reward = build_tree(state.result(action), book, depth - 1)
    book[state.board][action] += reward
    return -reward

def simulate(state):
    while not state.terminal_test():
        state = state.result(random.choice(state.actions()))
    # return the difference in actions between oppponet and agent
    own_loc = state.locs[state.player()]
    opp_loc = state.locs[1 - state.player()]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    return -1 if len(own_liberties) - len(opp_liberties) < 0 else 1

def alpha_beta_search(state, player_id, depth):

    def min_value(state, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(player_id)

        if(depth <= 0 ): return score(state, player_id)

        v = float("inf")
        for a in state.actions():
            v = min(v, max_value(state.result(a), alpha, beta, depth-1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def max_value(state, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(player_id)
        if(depth <= 0 ): return score(state, player_id)


        v = float("-inf")
        for a in state.actions():
            v = max(v, min_value(state.result(a), alpha, beta, depth-1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    alpha = float("-inf")
    beta = float("inf")
    best_score = float("-inf")
    best_move = None
    for a in state.actions():
        v = min_value(state.result(a), alpha, beta, depth-1)
        alpha = max(alpha, v)
        if v >= best_score:
            best_score = v
            best_move = a
    return best_move

def score(state, player_id):
    own_loc = state.locs[player_id]
    opp_loc = state.locs[1 - player_id]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    return len(own_liberties) - len(opp_liberties)


if __name__ == "__main__":
    main()
