Maximize
obj: x + y

Subject to
\ R1
    x_r1: x - x_r1_r2 - x_r1_r3 = 0
    x_r2: x_r1_r2 + x_r3_r2 - x_r2_r4 = 0
    x_r3: x_r1_r3 - x_r3_r2 - x_r3_r4 = 0
    x_r4: x_r2_r4 + x_r3_r4 - x_o = 0

\ R3
    y_r1: -y_r1_r2 - y_r1_r3 = 0
    y_r2: y_r1_r2 + y_r3_r2 - y_r2_r4 = 0
    y_r3: y + y_r1_r3 - y_r3_r2 - y_r3_r4 = 0
    y_r4: y_r3_r4 + y_r2_r4 - y_o = 0

\ abs X
    x_r1_r3_abs_p: x_r1_r3 - x_r1_r3_abs <= 0
    x_r1_r3_abs_n: x_r1_r3 + x_r1_r3_abs >= 0
    x_r1_r2_abs_p: x_r1_r2 - x_r1_r2_abs <= 0
    x_r1_r2_abs_n: x_r1_r2 + x_r1_r2_abs >= 0
    x_r2_r4_abs_p: x_r2_r4 - x_r2_r4_abs <= 0
    x_r2_r4_abs_n: x_r2_r4 + x_r2_r4_abs >= 0
    x_r3_r2_abs_p: x_r3_r2 - x_r3_r2_abs <= 0
    x_r3_r2_abs_n: x_r3_r2 + x_r3_r2_abs >= 0
    x_r3_r4_abs_p: x_r3_r4 - x_r3_r4_abs <= 0
    x_r3_r4_abs_n: x_r3_r4 + x_r3_r4_abs >= 0

\ abs Y
    y_r1_r3_abs_p: y_r1_r3 - y_r1_r3_abs <= 0
    y_r1_r3_abs_n: y_r1_r3 + y_r1_r3_abs >= 0
    y_r1_r2_abs_p: y_r1_r2 - y_r1_r2_abs <= 0
    y_r1_r2_abs_n: y_r1_r2 + y_r1_r2_abs >= 0
    y_r2_r4_abs_p: y_r2_r4 - y_r2_r4_abs <= 0
    y_r2_r4_abs_n: y_r2_r4 + y_r2_r4_abs >= 0
    y_r3_r2_abs_p: y_r3_r2 - y_r3_r2_abs <= 0
    y_r3_r2_abs_n: y_r3_r2 + y_r3_r2_abs >= 0
    y_r3_r4_abs_p: y_r3_r4 - y_r3_r4_abs <= 0
    y_r3_r4_abs_n: y_r3_r4 + y_r3_r4_abs >= 0

\ abs
    r1_r2: y_r1_r2_abs + x_r1_r2_abs <= 3
    r1_r3: y_r1_r3_abs + x_r1_r3_abs <= 10
    r2_r4: y_r2_r4_abs + x_r2_r4_abs <= 10
    r3_r2: y_r3_r2_abs + x_r3_r2_abs <= 8
    r3_r4: y_r3_r4_abs + x_r3_r4_abs <= 10


Bounds
    x_r1_r2 free
    x_r1_r3 free
    x_r2_r4 free
    x_r3_r2 free
    x_r3_r4 free

    y_r1_r2 free
    y_r1_r3 free
    y_r2_r4 free
    y_r3_r2 free
    y_r3_r4 free

End