change = float(input())
coins = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
i = len(coins) - 1

while abs(change) >= 1e-6 and i >= 0:
    change = round(change * 100) / 100
    n = int((change + 1e-6) / coins[i])
    change -= n * coins[i]

    if n != 0:
        print(f'{coins[i]:5.2f}\t{n}')

    i -= 1
