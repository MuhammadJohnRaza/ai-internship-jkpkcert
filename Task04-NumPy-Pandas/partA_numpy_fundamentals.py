

import numpy as np
import time


print("=" * 60)
print("SECTION 1: Array Creation")
print("=" * 60)


arr_list   = np.array([10, 20, 30, 40, 50])              
arr_range  = np.arange(0, 20, 2)                          
arr_linsp  = np.linspace(0, 1, 6)                         
arr_zeros  = np.zeros(5)                                   
arr_ones   = np.ones(5, dtype=int)                      
arr_rand   = np.random.randint(1, 100, size=5)            

print("\n[1-D Arrays]")
print(f"  From list      : {arr_list}")
print(f"  arange(0,20,2) : {arr_range}")
print(f"  linspace(0,1,6): {arr_linsp}")
print(f"  zeros(5)       : {arr_zeros}")
print(f"  ones(5, int)   : {arr_ones}")
print(f"  randint(1-100) : {arr_rand}")


mat_2d     = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) 
mat_zeros  = np.zeros((3, 4))                             
mat_eye    = np.eye(4)                                    
mat_rand   = np.random.uniform(0, 10, size=(3, 3))        
arr_3d     = np.ones((2, 3, 4), dtype=int)                

print("\n[Multi-dimensional Arrays]")
print(f"  3x3 nested list:\n{mat_2d}")
print(f"  3x4 zeros:\n{mat_zeros}")
print(f"  4x4 identity:\n{mat_eye}")
print(f"  3x3 random floats (0-10):\n{np.round(mat_rand, 2)}")
print(f"  3-D tensor shape (2,3,4): {arr_3d.shape}")



print("\n" + "=" * 60)
print("SECTION 2: Indexing, Slicing, Reshaping & Math Operations")
print("=" * 60)

base = np.arange(1, 13)          # [1 ... 12]
matrix = base.reshape(3, 4)      # reshape to 3x4

print(f"\nBase 1-D array : {base}")
print(f"Reshaped 3x4:\n{matrix}")

# Indexing
print(f"\n[Indexing]")
print(f"  matrix[0]      (row 0) : {matrix[0]}")
print(f"  matrix[1][2]   (r1,c2) : {matrix[1][2]}")
print(f"  matrix[-1]     (last)  : {matrix[-1]}")
print(f"  matrix[0, -1]  (r0,c-1): {matrix[0, -1]}")

# Slicing
print(f"\n[Slicing]")
print(f"  matrix[:2, 1:3] (rows<2, cols 1-2):\n{matrix[:2, 1:3]}")
print(f"  matrix[::2]     (every 2nd row):\n{matrix[::2]}")
print(f"  matrix[:, 0]    (col 0) : {matrix[:, 0]}")

# Boolean indexing
print(f"\n[Boolean Indexing]")
mask = matrix > 6
print(f"  Values > 6 : {matrix[mask]}")

# Fancy indexing
print(f"\n[Fancy Indexing]")
print(f"  Rows [0, 2]:\n{matrix[[0, 2]]}")

# Reshaping & flattening
print(f"\n[Reshaping]")
print(f"  Flatten  : {matrix.flatten()}")
print(f"  Ravel    : {matrix.ravel()}")
print(f"  3x4 -> 2x6:\n{matrix.reshape(2, 6)}")
print(f"  3x4 -> 12x1:\n{matrix.reshape(12, 1).T}")  # transposed for brevity

# Math operations
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])

print(f"\n[Element-wise Math]")
print(f"  a         : {a}")
print(f"  b         : {b}")
print(f"  a + b     : {a + b}")
print(f"  a - b     : {a - b}")
print(f"  a * b     : {a * b}")
print(f"  a / b     : {a / b}")
print(f"  a ** 2    : {a ** 2}")
print(f"  sqrt(a)   : {np.sqrt(a)}")
print(f"  log(a)    : {np.round(np.log(a), 3)}")
print(f"  sum(a)    : {a.sum()}")
print(f"  mean(a)   : {a.mean()}")
print(f"  std(a)    : {a.std()}")
print(f"  min/max   : {a.min()} / {a.max()}")
print(f"  cumsum(a) : {np.cumsum(a)}")


# ─────────────────────────────────────────────────────
# SECTION 3: Broadcasting
# ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Broadcasting")
print("=" * 60)

# Example 1 - scalar broadcast
arr = np.array([1, 2, 3, 4, 5])
print(f"\nEx-1 | arr x 10  (scalar) : {arr * 10}")

# Example 2 - 1-D array broadcast over 2-D matrix
m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
row_bias = np.array([10, 20, 30])        # shape (3,)
result   = m + row_bias                  # row_bias broadcast to (3,3)
print(f"\nEx-2 | m (3x3):\n{m}")
print(f"       row_bias  (3,) : {row_bias}")
print(f"       m + row_bias:\n{result}")

# Example 3 - column vector broadcast
col_scale = np.array([[1], [2], [3]])    # shape (3,1)
scaled    = m * col_scale
print(f"\nEx-3 | m x col_scale [[1],[2],[3]]:\n{scaled}")

# Example 4 - outer product via broadcasting
x = np.arange(1, 4)               # shape (3,)
y = np.arange(1, 4)[:, np.newaxis]# shape (3,1)
outer = x * y
print(f"\nEx-4 | Outer product (1-3 x 1-3):\n{outer}")

print("""
[Broadcasting Advantages]
  [OK] Avoids explicit loops -> faster, more readable code.
  [OK] No extra memory copies; NumPy applies the rule virtually.
  [OK] Enables natural expression of batch maths (e.g. normalising
    each row of a matrix by subtracting the row-mean).
  [OK] Broadcasting rule: dimensions are compatible when they are
    equal OR one of them is 1.
""")


# ─────────────────────────────────────────────────────
# SECTION 4: Vectorized Ops vs Traditional Loops
# ─────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 4: Vectorized Operations vs Traditional Loops")
print("=" * 60)

SIZE = 1_000_000

# --- Traditional Python loop ---
data_list = list(range(SIZE))

start = time.perf_counter()
result_loop = [x ** 2 for x in data_list]
loop_time = time.perf_counter() - start

# --- NumPy vectorized ---
data_np = np.arange(SIZE)

start = time.perf_counter()
result_np = data_np ** 2
np_time = time.perf_counter() - start

speedup = loop_time / np_time

print(f"\nTask : Square {SIZE:,} integers")
print(f"  Loop time     : {loop_time:.4f}s")
print(f"  NumPy time    : {np_time:.6f}s")
print(f"  Speed-up      : {speedup:.1f}x faster")

# --- Second demo: sum of absolute differences ---
a_np = np.random.randn(100_000)
b_np = np.random.randn(100_000)

start = time.perf_counter()
sad_loop = sum(abs(a_np[i] - b_np[i]) for i in range(len(a_np)))
loop2 = time.perf_counter() - start

start = time.perf_counter()
sad_np = np.sum(np.abs(a_np - b_np))
np2 = time.perf_counter() - start

print(f"\nTask : Sum of Absolute Differences (100k floats)")
print(f"  Loop SAD      : {sad_loop:.4f}  in {loop2:.4f}s")
print(f"  NumPy SAD     : {sad_np:.4f}  in {np2:.6f}s")
print(f"  Speed-up      : {loop2 / np2:.1f}x")

print("""
[Why Vectorized?]
  - NumPy operations execute in compiled C/Fortran under the hood.
  - Entire arrays are processed without Python interpreter overhead.
  - Memory layout (contiguous C-order) enables CPU cache efficiency.
  - SIMD (Single Instruction Multiple Data) instructions are used
    automatically for parallel element-wise computation.
""")


# ─────────────────────────────────────────────────────
# SECTION 5: Linear Algebra
# ─────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 5: Linear Algebra Operations")
print("=" * 60)

# ---- Dot product (1-D vectors) ----
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
dot = np.dot(u, v)
print(f"\n[Dot Product]")
print(f"  u = {u},  v = {v}")
print(f"  u - v = {dot}")   # 1x4 + 2x5 + 3x6 = 32

# ---- Matrix multiplication ----
A = np.array([[1, 2], [3, 4], [5, 6]])   # 3x2
B = np.array([[7, 8, 9], [10, 11, 12]])  # 2x3

matmul_result = A @ B    # or np.matmul(A, B)
print(f"\n[Matrix Multiplication  A(3x2) @ B(2x3) = C(3x3)]")
print(f"  A:\n{A}")
print(f"  B:\n{B}")
print(f"  A @ B:\n{matmul_result}")

# ---- Transpose ----
print(f"\n[Transpose]")
print(f"  A (3x2):\n{A}")
print(f"  A.T (2x3):\n{A.T}")

# ---- Determinant & Inverse (square matrix) ----
C = np.array([[2., 1.],
              [5., 3.]])

det = np.linalg.det(C)
print(f"\n[Determinant & Inverse]")
print(f"  C:\n{C}")
print(f"  det(C) = {det:.2f}")

if abs(det) > 1e-10:
    C_inv = np.linalg.inv(C)
    print(f"  inv(C):\n{C_inv}")
    # Verify C @ C_inv ~= Identity
    identity_check = np.round(C @ C_inv, decimals=10)
    print(f"  C @ inv(C) (should be I):\n{identity_check}")
else:
    print("  Matrix is singular -- inverse does not exist.")

# ---- Eigenvalues & Eigenvectors ----
D = np.array([[4, 2],
              [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(D)
print(f"\n[Eigenvalues & Eigenvectors of D:\n{D}]")
print(f"  Eigenvalues  : {np.round(eigenvalues, 4)}")
print(f"  Eigenvectors :\n{np.round(eigenvectors, 4)}")

# ---- Solving a linear system Ax = b ----
A_sys = np.array([[3., 1.], [1., 2.]])
b_sys = np.array([9., 8.])
x_sol = np.linalg.solve(A_sys, b_sys)
print(f"\n[Solving Linear System Ax = b]")
print(f"  A = {A_sys.tolist()},  b = {b_sys.tolist()}")
print(f"  Solution x = {x_sol}")
print(f"  Verification A @ x = {A_sys @ x_sol}")

# ---- Norm ----
vec = np.array([3., 4.])
print(f"\n[Norms]")
print(f"  vector = {vec}")
print(f"  L1 norm  : {np.linalg.norm(vec, 1)}")
print(f"  L2 norm  : {np.linalg.norm(vec, 2)}")
print(f"  Inf norm : {np.linalg.norm(vec, np.inf)}")

print("\n" + "=" * 60)
print("Part A - NumPy Fundamentals: COMPLETE")
print("=" * 60)
