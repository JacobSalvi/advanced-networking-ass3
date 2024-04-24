Minimize
obj:  x_r1_r2 + x_r2_r1 + x_r1_r3 + x_r3_r1 + x_r3_r2 + x_r2_r3 + x_r2_r4 + x_r4_r2 + x_r3_r4 + x_r4_r3 +
      y_r1_r2 + y_r2_r1 + y_r1_r3 + y_r3_r1 + y_r3_r2 + y_r2_r3 + y_r2_r4 + y_r4_r2 + y_r3_r4 + y_r4_r3

Subject to
    \ mutual exclusion of flows (directional) on the same link
    x_link_r1_r2: x_r1_r2 + x_r2_r1 <= 1
    x_link_r1_r3: x_r1_r3 + x_r3_r1 <= 1
    x_link_r2_r4: x_r2_r4 + x_r4_r2 <= 1
    x_link_r3_r2: x_r3_r2 + x_r2_r3 <= 1
    x_link_r3_r4: x_r3_r4 + x_r4_r3 <= 1

    y_link_r1_r2: y_r1_r2 + y_r2_r1 <= 1
    y_link_r1_r3: y_r1_r3 + y_r3_r1 <= 1
    y_link_r2_r4: y_r2_r4 + y_r4_r2 <= 1
    y_link_r3_r2: y_r3_r2 + y_r2_r3 <= 1
    y_link_r3_r4: y_r3_r4 + y_r4_r3 <= 1

    \ mutual exclusion of incoming flows, for each flow, node
    x_in_r1: x_r2_r1 + x_r3_r1 <= 1
    x_in_r2: x_r1_r2 + x_r3_r2 + x_r4_r2 <= 1
    x_in_r3: x_r1_r3 + x_r2_r3 + x_r4_r3 <= 1
    x_in_r4: x_r2_r4 + x_r3_r4 <= 1

    y_in_r1: y_r2_r1 + y_r3_r1 <= 1
    y_in_r2: y_r1_r2 + y_r3_r2 + y_r4_r2 <= 1
    y_in_r3: y_r1_r3 + y_r2_r3 + y_r4_r3 <= 1
    y_in_r4: y_r2_r4 + y_r3_r4 <= 1

    \ mutual exclusion of outgoing flows, for each flow, node
    x_out_r1: x_r1_r2 + x_r1_r3 <= 1
    x_out_r2: x_r2_r1 + x_r2_r3 + x_r2_r4 <= 1
    x_out_r3: x_r3_r1 + x_r3_r2 + x_r3_r4 <= 1
    x_out_r4: x_r4_r2 + x_r4_r3 <= 1

    y_out_r1: y_r1_r2 + y_r1_r3 <= 1
    y_out_r2: y_r2_r1 + y_r2_r3 + y_r2_r4 <= 1
    y_out_r3: y_r3_r1 + y_r3_r2 + y_r3_r4 <= 1
    y_out_r4: y_r4_r2 + y_r4_r3 <= 1

    \ flow balance constraints
    flow_r1_x: x_r2_r1 - x_r1_r2 + x_r3_r1 - x_r1_r3 = -1
    flow_r2_x: x_r1_r2 - x_r2_r1 + x_r4_r2 - x_r2_r4 + x_r3_r2 - x_r2_r3 = 0
    flow_r3_x: x_r1_r3 - x_r3_r1 + x_r2_r3 - x_r3_r2 + x_r4_r3 - x_r3_r4 = 0
    flow_r4_x: x_r2_r4 - x_r4_r2 + x_r3_r4 - x_r4_r3 = 1

    flow_r1_y: y_r2_r1 - y_r1_r2 + y_r3_r1 - y_r1_r3 = 0
    flow_r2_y: y_r1_r2 - y_r2_r1 + y_r4_r2 - y_r2_r4 + y_r3_r2 - y_r2_r3 = 0
    flow_r3_y: y_r1_r3 - y_r3_r1 + y_r2_r3 - y_r3_r2 + y_r4_r3 - y_r3_r4 = -1
    flow_r4_y: y_r2_r4 - y_r4_r2 + y_r3_r4 - y_r4_r3 = 1

    \ link-capacity constraints (flow demands: x = 10, y = 10)
    r1_r2_c: 10 x_r1_r2 + 10 x_r2_r1 + 10 y_r1_r2 + 10 y_r2_r1 <= 3
    r1_r3_c: 10 x_r1_r3 + 10 x_r3_r1 + 10 y_r1_r3 + 10 y_r3_r1 <= 10
    r2_r4_c: 10 x_r2_r4 + 10 x_r4_r2 + 10 y_r2_r4 + 10 y_r4_r2 <= 10
    r3_r2_c: 10 x_r3_r2 + 10 x_r2_r3 + 10 y_r3_r2 + 10 y_r2_r3 <= 8
    r3_r4_c: 10 x_r3_r4 + 10 x_r4_r3 + 10 y_r3_r4 + 10 y_r4_r3 <= 10

Binary
    x_r1_r2
    x_r2_r1
    x_r1_r3
    x_r3_r1
    x_r3_r2
    x_r2_r3
    x_r2_r4
    x_r4_r2
    x_r3_r4
    x_r4_r3

    y_r1_r2
    y_r2_r1
    y_r1_r3
    y_r3_r1
    y_r3_r2
    y_r2_r3
    y_r2_r4
    y_r4_r2
    y_r3_r4
    y_r4_r3

End