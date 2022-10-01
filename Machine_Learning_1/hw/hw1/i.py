import sys


def safe_input():
    try:
        return input()
    except EOFError:
        sys.exit(0)


lhs = 1
rhs = 100_000

while rhs - lhs > 1:
    d = (rhs - lhs + 1) // 10
    for i in range(lhs + d, rhs, d):
        print(f'? {i}', flush=True)

    print('+', flush=True)

    ansers = []
    for i in range(10 - 1):
        ansers.append(int(safe_input()))

    i = 0
    while i < 10 - 1 and ansers[i] == 0:
        lhs += d
        i += 1

    rhs = lhs + d

print(f'! {lhs}', flush=True)
