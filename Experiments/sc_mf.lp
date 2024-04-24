Maximize
obj: i1 + i2
Subject to
    r1: i1 - x_r1_r2 - x_r1_r3 = 0
    r2: x_r1_r2 + x_r2_r3 - x_r2_r4 = 0
    r3: i2 + x_r1_r3 - x_r3_r4 - x_r2_r3 = 0
    r4: x_r3_r4 + x_r2_r4 - o1 - o2 = 0
Bounds
    -3 <= x_r1_r2 <= 3
    -10 <= x_r1_r3 <= 10
    -8 <= x_r2_r3 <= 8
    -10 <= x_r2_r4 <= 10
    -10 <= x_r3_r4 <= 10
End