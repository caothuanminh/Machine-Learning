def f(x):
    return x**2 - 4*x + 5
def df(x):
    return 2*x - 4
x = 5.0
learning_rate = 0.2

print("x ban dau:", x)
print("f(x) ban dau:", f(x))

for i in range(4):
    gradient = df(x)
    x = x - learning_rate * gradient
    print(f"Buoc {i+1}: x = {x:.6f}, f(x) = {f(x):.6f}")
print("Gia tri cuc tieu:", x)
print("f(x) cuc tieu:", f(x))