__author__ = 'adoug'

def add1(arr):
    for i in range(len(arr)):
        arr[i] += 1
    return arr

b = [[1000, 1000, 1000, 1000, 1000, 1000] for i in range(600)]
a = map(add1, b)
a[1]
print(str(a))
