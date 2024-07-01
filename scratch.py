from collections import deque


def window(seq, n=2):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win


window = window([1, 2, 3, 4, 5])

print(list(next(window)))
print(next(window))
print(next(window))
print(next(window))
