N = int(input("N: "))
arr = []

for _ in range(N):
    min_map = input(":")
    arr.append(list(map(int, min_map.split())))

directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

result = [[0 for _ in range(N)] for _ in range(N)]

for x in range(N):
    for y in range(N):
        if arr[x][y] == 1:
            result[x][y] = -1
            continue
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and arr[nx][ny] == 1:
                count += 1
        result[x][y] = count

for row in result:
    print(' '.join(map(str, row)).replace("-1", "*"))
