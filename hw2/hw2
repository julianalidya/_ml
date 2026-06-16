#111210552 林小蓮

x, y, z = 1, 2, 3

a = x * y
f = a + z
print("Forward pass Problem 1:", f)

df_da = 1
df_dz = 1
df_dx = df_da * y
df_dy = df_da * x

print("df/dx =", df_dx)
print("df/dy =", df_dy)
print("df/dz =", df_dz)

t = 4
a2 = x * y
b2 = a2 + z
f2 = b2 * t
print("\nForward pass Problem 2:", f2)

df2_dt = b2
df2_db = t
df2_da = df2_db
df2_dz = df2_db * 1
df2_dx = df2_da * y
df2_dy = df2_da * x

print("df/dt =", df2_dt)
print("df/dx =", df2_dx)
print("df/dy =", df2_dy)
print("df/dz =", df2_dz)
