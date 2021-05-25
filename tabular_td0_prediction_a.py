import argparse
import matplotlib.pyplot as plt


class State:
    states = [
        ("leaving office, friday at 6", 0, 30),
        ("reach car, raining", 5, 35),
        ("exiting hightway", 20, 15),
        ("2ndary road, behind truck", 30, 10),
        ("entering home street", 40, 3),
        ("arrive home", 43, 0)
    ]

    def __init__(self, idx=0):
        if idx < 0 or idx >= len(self.states):
            raise IndexError("Invalid state")
        self.__index = idx

    @property
    def index(self):
        return self.__index

    def transition(self):
        idx = self.__index + 1
        return State(idx) if idx < len(self.states) - 1 else None, self.states[idx][1] - self.states[self.__index][1]


def mc_evaluate(episodes, values):
    for _ in range(episodes):
        episode = [(State(), None)]
        while True:
            s, _ = episode[-1]
            if s is None:
                break
            episode.append(s.transition())
        g = 0
        for t in reversed(range(len(episode) - 1)):
            g += episode[t + 1][1]
            s, _ = episode[t]
            values[s.index] += g - values[s.index]
    return values


def td_evaluate(episodes, values):
    for _ in range(episodes):
        s = State()
        while True:
            s1, r = s.transition()
            if s1 is None:
                values[s.index] += r - values[s.index]
                break
            else:
                values[s.index] += r + values[s1.index] - values[s.index]
            s = s1
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular TD(0) prediction (Example 6.1).")
    parser.add_argument("--episodes", help="number of episodes (default: 1)", type=int, default=1)
    args = parser.parse_args()

    states, elapsed_time, predicted_time = zip(*State.states)
    mc_values = mc_evaluate(args.episodes, list(predicted_time))
    td_values = td_evaluate(args.episodes, list(predicted_time))

    plt.subplot(2, 1, 1)
    plt.plot(states, [sum(t) for t in zip(elapsed_time, predicted_time)], label="Initial Prediction")
    plt.plot(states, [sum(t) for t in zip(elapsed_time, mc_values)], label="MC Revised Prediction")
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(states, [sum(t) for t in zip(elapsed_time, predicted_time)], label="Initial Prediction")
    plt.plot(states, [sum(t) for t in zip(elapsed_time, td_values)], label="TD(0) Revised Prediction")
    plt.grid(True)
    plt.legend()
    plt.show()
