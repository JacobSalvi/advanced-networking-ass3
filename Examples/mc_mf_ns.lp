Minimize
obj: x_ab + x_ba + x_ae + x_ea + x_ad + x_da + x_bc + x_cb + x_be + x_eb + x_ce + x_ec +
x_cf + x_fc + x_de + x_ed + x_ef + x_fe + y_ab + y_ba + y_ae + y_ea + y_ad + y_da +
y_bc + y_cb + y_be + y_eb + y_ce + y_ec + y_cf + y_fc + y_de + y_ed + y_ef + y_fe
Subject to
\ mutual exclusion of flows (directional) on the same link,
    ab_link_x: x_ab + x_ba <= 1
    ae_link_x: x_ae + x_ea <= 1
    ad_link_x: x_ad + x_da <= 1
    bc_link_x: x_bc + x_cb <= 1
    be_link_x: x_be + x_eb <= 1
    ce_link_x: x_ce + x_ec <= 1
    cf_link_x: x_cf + x_fc <= 1
    de_link_x: x_de + x_ed <= 1
    ef_link_x: x_ef + x_fe <= 1

    ab_link_y: y_ab + y_ba <= 1
    ae_link_y: y_ae + y_ea <= 1
    ad_link_y: y_ad + y_da <= 1
    bc_link_y: y_bc + y_cb <= 1
    be_link_y: y_be + y_eb <= 1
    ce_link_y: y_ce + y_ec <= 1
    cf_link_y: y_cf + y_fc <= 1
    de_link_y: y_de + y_ed <= 1
    ef_link_y: y_ef + y_fe <= 1

    \ mutual exclusion of incoming flows, for each flow, node
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
    \ mutual exclusion of outgoing flows, for each flow, node
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
    \ flow-balance constraints (flow x: a --> f, flow y: c --> d)
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
    \ link-capacity constraints (flow demands: x = 4, y = 3)
    ab_c: 4 x_ab + 4 x_ba + 3 y_ab + 3 y_ba <= 7
    ae_c: 4 x_ae + 4 x_ea + 3 y_ae + 3 y_ea <= 3
    ad_c: 4 x_ad + 4 x_da + 3 y_ad + 3 y_da <= 6
    bc_c: 4 x_bc + 4 x_cb + 3 y_bc + 3 y_cb <= 10
    be_c: 4 x_be + 4 x_eb + 3 y_be + 3 y_eb <= 7
    ce_c: 4 x_ce + 4 x_ec + 3 y_ce + 3 y_ec <= 10
    cf_c: 4 x_cf + 4 x_fc + 3 y_cf + 3 y_fc <= 10
    de_c: 4 x_de + 4 x_ed + 3 y_de + 3 y_ed <= 5
    ef_c: 4 x_ef + 4 x_fe + 3 y_ef + 3 y_fe <= 4
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