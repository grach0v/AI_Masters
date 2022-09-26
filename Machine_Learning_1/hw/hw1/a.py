length = int(input())
numbers = [int(n) for n in input().split()]
seen_numbers = set()
count_deleted = 0

for n in numbers:
    if n not in seen_numbers:
        print(n, end=' ')
    else:
        count_deleted += 1

    seen_numbers.add(n)

print('\n', count_deleted, sep='')
