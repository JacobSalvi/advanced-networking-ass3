Maximize
obj: min
Subject to
    \ min = min {lambda_x, lambda_y}
    x_min: lambda_x - min >= 0
    y_min: lambda_y - min >= 0
    \ flow balance for real-value rate variables
    flow_a_rx: rx_da - rx_ad + rx_ea - rx_ae + rx_ba - rx_ab + lambda_x = 0
    flow_b_rx: rx_ab - rx_ba + rx_eb - rx_be + rx_cb - rx_bc = 0
    flow_c_rx: rx_bc - rx_cb + rx_ec - rx_ce + rx_fc - rx_cf = 0
    flow_d_rx: rx_ad - rx_da + rx_ed - rx_de = 0
    flow_e_rx: rx_de - rx_ed + rx_ae - rx_ea + rx_be - rx_eb + rx_ce - rx_ec = 0
    flow_f_rx: rx_ef - rx_fe + rx_cf - rx_fc - lambda_x = 0
    flow_a_ry: ry_da - ry_ad + ry_ea - ry_ae + ry_ba - ry_ab = 0
    flow_b_ry: ry_ab - ry_ba + ry_eb - ry_be + ry_cb - ry_bc = 0
    flow_c_ry: ry_bc - ry_cb + ry_ec - ry_ce + ry_fc - ry_cf - lambda_y = 0
    flow_d_ry: ry_ad - ry_da + ry_ed - ry_de + lambda_y = 0
    flow_e_ry: ry_de - ry_ed + ry_ae - ry_ea + ry_be - ry_eb + ry_ce - ry_ec = 0
    flow_f_ry: ry_ef - ry_fe + ry_cf - ry_fc = 0

    \ flow balance for indicators, for each flow, for each node
    flow_a_x: x_da - x_ad + x_ea - x_ae + x_ba - x_ab = -1
    flow_b_x: x_ab - x_ba + x_eb - x_be + x_cb - x_bc = 0
    flow_c_x: x_bc - x_cb + x_ec - x_ce + x_fc - x_cf = 0
    flow_d_x: x_ad - x_da + x_ed - x_de = 0
    flow_e_x: x_de - x_ed + x_ae - x_ea + x_be - x_eb + x_ce - x_ec = 0
    flow_f_x: x_ef - x_fe + x_cf - x_fc = 1
    flow_a_y: y_da - y_ad + y_ea - y_ae + y_ba - y_ab = 0
    flow_b_y: y_ab - y_ba + y_eb - y_be + y_cb - y_bc = 0
    flow_c_y: y_bc - y_cb + y_ec - y_ce + y_fc - y_cf = 1
    flow_d_y: y_ad - y_da + y_ed - y_de = -1
    flow_e_y: y_de - y_ed + y_ae - y_ea + y_be - y_eb + y_ce - y_ec = 0
    flow_f_y: y_ef - y_fe + y_cf - y_fc = 0

    \ mutual exclusion of incoming into same node, for each flow, for each node
    in_a_x: x_da + x_ea + x_ba <= 1
    in_b_x: x_ab + x_eb + x_cb <= 1
    in_c_x: x_bc + x_ec + x_fc <= 1
    in_d_x: x_ad + x_ed <= 1
    in_e_x: x_de + x_ae + x_be + x_ce <= 1
    in_f_x: x_ef + x_cf <= 1
    in_a_y: y_da + y_ea + y_ba <= 1
    in_b_y: y_ab + y_eb + y_cb <= 1
    in_c_y: y_bc + y_ec + y_fc <= 1
    in_d_y: y_ad + y_ed <= 1
    in_e_y: y_de + y_ae + y_be + y_ce <= 1
    in_f_y: y_ef + y_cf <= 1

    \ mutual exclusion of outgoing out of same node, for each flow, for each node
    out_a_x: x_ad + x_ae + x_ab <= 1
    out_b_x: x_ba + x_be + x_bc <= 1
    out_c_x: x_cb + x_ce + x_cf <= 1
    out_d_x: x_da + x_de <= 1
    out_e_x: x_ed + x_ea + x_eb + x_ec <= 1
    out_f_x: x_fe + x_fc <= 1
    out_a_y: y_ad + y_ae + y_ab <= 1
    out_b_y: y_ba + y_be + y_bc <= 1
    out_c_y: y_cb + y_ce + y_cf <= 1
    out_d_y: y_da + y_de <= 1
    out_e_y: y_ed + y_ea + y_eb + y_ec <= 1
    out_f_y: y_fe + y_fc <= 1

    \ link capacities, for each link
    ab_c: rx_ab + rx_ba + ry_ab + ry_ba <= 7
    ae_c: rx_ae + rx_ea + ry_ae + ry_ea <= 3
    ad_c: rx_ad + rx_da + ry_ad + ry_da <= 6
    bc_c: rx_bc + rx_cb + ry_bc + ry_cb <= 10
    be_c: rx_be + rx_eb + ry_be + ry_eb <= 7
    ce_c: rx_ce + rx_ec + ry_ce + ry_ec <= 10
    cf_c: rx_cf + rx_fc + ry_cf + ry_fc <= 10
    de_c: rx_de + rx_ed + ry_de + ry_ed <= 5
    ef_c: rx_ef + rx_fe + ry_ef + ry_fe <= 4

    \ control of real-value flow variables by corresponding indicators, for each flow and link
    ab_cx: rx_ab - 7 x_ab <= 0
    ae_cx: rx_ae - 3 x_ae <= 0
    ad_cx: rx_ad - 6 x_ad <= 0
    bc_cx: rx_bc - 10 x_bc <= 0
    be_cx: rx_be - 7 x_be <= 0
    ce_cx: rx_ce - 10 x_ce <= 0
    cf_cx: rx_cf - 10 x_cf <= 0
    de_cx: rx_de - 5 x_de <= 0
    ef_cx: rx_ef - 4 x_ef <= 0
    ba_cx: rx_ba - 7 x_ba <= 0
    ea_cx: rx_ea - 3 x_ea <= 0
    da_cx: rx_da - 6 x_da <= 0
    cb_cx: rx_cb - 10 x_cb <= 0
    eb_cx: rx_eb - 7 x_eb <= 0
    ec_cx: rx_ec - 10 x_ec <= 0
    fc_cx: rx_fc - 10 x_fc <= 0
    ed_cx: rx_ed - 5 x_ed <= 0
    fe_cx: rx_fe - 4 x_fe <= 0
    ab_cy: ry_ab - 7 y_ab <= 0
    ae_cy: ry_ae - 3 y_ae <= 0
    ad_cy: ry_ad - 6 y_ad <= 0
    bc_cy: ry_bc - 10 y_bc <= 0
    be_cy: ry_be - 7 y_be <= 0
    ce_cy: ry_ce - 10 y_ce <= 0
    cf_cy: ry_cf - 10 y_cf <= 0
    de_cy: ry_de - 5 y_de <= 0
    ef_cy: ry_ef - 4 y_ef <= 0
    ba_cy: ry_ba - 7 y_ba <= 0
    ea_cy: ry_ea - 3 y_ea <= 0
    da_cy: ry_da - 6 y_da <= 0
    cb_cy: ry_cb - 10 y_cb <= 0
    eb_cy: ry_eb - 7 y_eb <= 0
    ec_cy: ry_ec - 10 y_ec <= 0
    fc_cy: ry_fc - 10 y_fc <= 0
    ed_cy: ry_ed - 5 y_ed <= 0
    fe_cy: ry_fe - 4 y_fe <= 0
Binary
    x_ab
    x_ba
    x_ae
    x_ea
    x_ad
    x_da
    x_bc
    x_cb
    x_be
    x_eb
    x_ce
    x_ec
    x_cf
    x_fc
    x_de
    x_ed
    x_ef
    x_fe
    y_ab
    y_ba
    y_ae
    y_ea
    y_ad
    y_da
    y_bc
    y_cb
    y_be
    y_eb
    y_ce
    y_ec
    y_cf
    y_fc
    y_de
    y_ed
    y_ef
    y_fe
End