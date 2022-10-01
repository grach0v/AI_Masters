n, m = map(int, input().split())
if m > 0:
    print(eval(' + '.join(
        [str(n) * (i + 1) for i in range(m)]
    )))
else:
    print(0)
