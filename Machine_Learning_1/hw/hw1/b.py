length = int(input())
numbers = [int(n) for n in input().split()]

numbers.sort(key=lambda x: (sum(map(int, str(x))), x))

print(*numbers)
