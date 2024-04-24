Maximize
obj: x + y
\ Flows:
\ X: A,B --> F
\ Y: D --> C
Subject to
\ flow balance constraints: for each flow, for each node
    x_total: x - x1 - x2 = 0
    xa: x1 - x_ab - x_ae - x_ad = 0
    xb: x2 + x_ab - x_be - x_bc = 0
    xc: x_bc - x_ce - x_cf = 0
    xd: x_ad - x_de = 0
    xe: x_de + x_ae + x_be + x_ce - x_ef = 0
    xf: x_ef + x_cf - x = 0
    ya: y_ab + y_ae + y_ad = 0
    yb: y_ab - y_be - y_bc = 0
    yc: y_bc - y_ce - y_cf - y = 0
    yd: y + y_ad - y_de = 0
    ye: y_de + y_ae + y_be + y_ce - y_ef = 0
    yf: y_ef + y_cf = 0
    x_ab_abs_p: x_ab - x_ab_abs <= 0
    x_ab_abs_n: x_ab + x_ab_abs >= 0
    x_ae_abs_p: x_ae - x_ae_abs <= 0
    x_ae_abs_n: x_ae + x_ae_abs >= 0
    x_ad_abs_p: x_ad - x_ad_abs <= 0
    x_ad_abs_n: x_ad + x_ad_abs >= 0
    x_bc_abs_p: x_bc - x_bc_abs <= 0
    x_bc_abs_n: x_bc + x_bc_abs >= 0
    x_be_abs_p: x_be - x_be_abs <= 0
    x_be_abs_n: x_be + x_be_abs >= 0
    x_ce_abs_p: x_ce - x_ce_abs <= 0
    x_ce_abs_n: x_ce + x_ce_abs >= 0
    x_cf_abs_p: x_cf - x_cf_abs <= 0
    x_cf_abs_n: x_cf + x_cf_abs >= 0
    x_de_abs_p: x_de - x_de_abs <= 0
    x_de_abs_n: x_de + x_de_abs >= 0
    x_ef_abs_p: x_ef - x_ef_abs <= 0
    x_ef_abs_n: x_ef + x_ef_abs >= 0
    y_ab_abs_p: y_ab - y_ab_abs <= 0
    y_ab_abs_n: y_ab + y_ab_abs >= 0
    y_ae_abs_p: y_ae - y_ae_abs <= 0
    y_ae_abs_n: y_ae + y_ae_abs >= 0
    y_ad_abs_p: y_ad - y_ad_abs <= 0
    y_ad_abs_n: y_ad + y_ad_abs >= 0
    y_bc_abs_p: y_bc - y_bc_abs <= 0
    y_bc_abs_n: y_bc + y_bc_abs >= 0
    y_be_abs_p: y_be - y_be_abs <= 0
    y_be_abs_n: y_be + y_be_abs >= 0
    y_ce_abs_p: y_ce - y_ce_abs <= 0
    y_ce_abs_n: y_ce + y_ce_abs >= 0
    y_cf_abs_p: y_cf - y_cf_abs <= 0
    y_cf_abs_n: y_cf + y_cf_abs >= 0
    y_de_abs_p: y_de - y_de_abs <= 0
    y_de_abs_n: y_de + y_de_abs >= 0
    y_ef_abs_p: y_ef - y_ef_abs <= 0
    y_ef_abs_n: y_ef + y_ef_abs >= 0
    ab: x_ab_abs + y_ab_abs <= 7
    ae: x_ae_abs + y_ae_abs <= 3
    ad: x_ad_abs + y_ad_abs <= 6
    bc: x_bc_abs + y_bc_abs <= 10
    be: x_be_abs + y_be_abs <= 7
    ce: x_ce_abs + y_ce_abs <= 10
    cf: x_cf_abs + y_cf_abs <= 10
    de: x_de_abs + y_de_abs <= 5
    ef: x_ef_abs + y_ef_abs <= 4
Bounds
\ -infinity <= x_uv <= +infinity
    x_ab free
    x_ae free
    x_ad free
    x_bc free
    x_be free
    x_ce free
    x_cf free
    x_de free
    x_ef free
    y_ab free
    y_ae free
    y_ad free
    y_bc free
    y_be free
    y_ce free
    y_cf free
    y_de free
    y_ef free
End