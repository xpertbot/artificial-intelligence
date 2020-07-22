import pickle, random, time, operator

from isolation import Isolation, DebugState
from collections import defaultdict, Counter
from _helpers import alpha_beta_search

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

if __name__ == "__main__":
    main()
