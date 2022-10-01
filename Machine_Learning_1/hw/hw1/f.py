line_a = [word.lower() for word in input().split()]
line_b = [word.lower() for word in input().split()]

line_a_counter = {}
line_b_counter = {}

for word in line_a:
    if word in line_a_counter:
        line_a_counter[word] += 1
    else:
        line_a_counter[word] = 1

for word in line_b:
    if word in line_b_counter:
        line_b_counter[word] += 1
    else:
        line_b_counter[word] = 1

is_subline = True
for word, value in line_b_counter.items():
    if word not in line_a_counter or line_a_counter[word] < value:
        is_subline = False
        break

if is_subline:
    print('YES')
else:
    print('NO')
