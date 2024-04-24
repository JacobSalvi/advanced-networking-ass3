Maximize
obj: i1 + i2
Subject to
    a: i1 - x_ab - x_ae - x_ad = 0
    b: x_ab - x_be - x_bc = 0
    c: x_bc - x_ce - x_cf - o1 = 0
    d: i2 + x_ad - x_de = 0
    e: x_de + x_ae + x_be + x_ce - x_ef = 0
    f: x_ef + x_cf - o2 = 0
Bounds
    -7 <= x_ab <= 7
    -3 <= x_ae <= 3
    -6 <= x_ad <= 6
    -10 <= x_bc <= 10
    -7 <= x_be <= 7
    -10 <= x_ce <= 10
    -10 <= x_cf <= 10
    -5 <= x_de <= 5
    -4 <= x_ef <= 4
End