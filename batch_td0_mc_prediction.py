import argparse


def td_evaluate(batch, epochs, alpha):
    values = [0, 0]
    for _ in range(epochs):
        updates = [0, 0]
        for episode in batch:
            for t, (s, r) in enumerate(episode):
                if t < len(episode) - 1:
                    s1, _ = episode[t + 1]
                    updates[s] += alpha * (r + values[s1] - values[s])
                else:
                    updates[s] += alpha * (r - values[s])
        values = [sum(v) for v in zip(values, updates)]
    return values


def mc_evaluate(batch, epochs, alpha):
    values = [0, 0]
    for _ in range(epochs):
        updates = [0, 0]
        for episode in batch:
            g = 0
            for s, r in reversed(episode):
                g += r
                updates[s] += alpha * (g - values[s])
        values = [sum(v) for v in zip(values, updates)]
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch updating TD(0)/MC prediction (Example 6.4).")
    parser.add_argument("--epochs", help="number of epochs (default: 10000)", type=int, default=10000)
    parser.add_argument("--alpha", help="constant step-size parameter of TD(0) (default: 0.1)", type=float, default=0.1)
    args = parser.parse_args()
    
    batch = [
        [(0, 0), (1, 0)],
        [(1, 1)],
        [(1, 1)],
        [(1, 1)],
        [(1, 1)],
        [(1, 1)],
        [(1, 1)],
        [(1, 0)]
    ]
    print("TD(0) Prediction:", td_evaluate(batch, args.epochs, args.alpha))
    print("MC Prediction:", mc_evaluate(batch, args.epochs, args.alpha))
