n = int(input())
words = []
for i in range(n):
    words.append(input())

for i in range(n):
    words[i] = words[i], ''.join(sorted(words[i]))

words.sort(key=lambda pair: pair[1])

i = 0
while i < n:
    print(words[i][0], end=' ')
    if i + 1 < n and words[i + 1][1] != words[i][1]:
        print('')

    i += 1
print('')
