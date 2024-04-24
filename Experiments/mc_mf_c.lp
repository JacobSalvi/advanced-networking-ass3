Maximize
obj:  min

Subject to
    \ min = min {lambda_x, lambda_y}
    x_min: lambda_x - min >= 0
    y_min: lambda_y - min >= 0

    \ flow balance for real-value rate variables
    flow_r1_rx: rx_r2_r1 - rx_r1_r2 + rx_r3_r1 - rx_r1_r3 + lambda_x = 0
    flow_r2_rx: rx_r1_r2 - rx_r2_r1 + rx_r4_r2 - rx_r2_r4 + rx_r3_r2 - rx_r2_r3 = 0
    flow_r3_rx: rx_r1_r3 - rx_r3_r1 + rx_r2_r3 - rx_r3_r2 + rx_r4_r3 - rx_r3_r4 = 0
    flow_r4_rx: rx_r2_r4 - rx_r4_r2 + rx_r3_r4 - rx_r4_r3 - lambda_x = 0

    flow_r1_ry: ry_r2_r1 - ry_r1_r2 + ry_r3_r1 - ry_r1_r3 = 0
    flow_r2_ry: ry_r1_r2 - ry_r2_r1 + ry_r4_r2 - ry_r2_r4 + ry_r3_r2 - ry_r2_r3 = 0
    flow_r3_ry: ry_r1_r3 - ry_r3_r1 + ry_r2_r3 - ry_r3_r2 + ry_r4_r3 - ry_r3_r4 + lambda_y = 0
    flow_r4_ry: ry_r2_r4 - ry_r4_r2 + ry_r3_r4 - ry_r4_r3 - lambda_y = 0

    \ flow balance for indicators, for each flow, for each node
    flow_r1_x: x_r2_r1 - x_r1_r2 + x_r3_r1 - x_r1_r3 = -1
    flow_r2_x: x_r1_r2 - x_r2_r1 + x_r4_r2 - x_r2_r4 + x_r3_r2 - x_r2_r3 = 0
    flow_r3_x: x_r1_r3 - x_r3_r1 + x_r2_r3 - x_r3_r2 + x_r4_r3 - x_r3_r4 = 0
    flow_r4_x: x_r2_r4 - x_r4_r2 + x_r3_r4 - x_r4_r3 = 1

    flow_r1_y: y_r2_r1 - y_r1_r2 + y_r3_r1 - y_r1_r3 = 0
    flow_r2_y: y_r1_r2 - y_r2_r1 + y_r4_r2 - y_r2_r4 + y_r3_r2 - y_r2_r3 = 0
    flow_r3_y: y_r1_r3 - y_r3_r1 + y_r2_r3 - y_r3_r2 + y_r4_r3 - y_r3_r4 = -1
    flow_r4_y: y_r2_r4 - y_r4_r2 + y_r3_r4 - y_r4_r3 = 1

    \ mutual exclusion of incoming flows into same node, for each flow, for each node
    in_r1_x: x_r2_r1 + x_r3_r1 <= 1
    in_r2_x: x_r1_r2 + x_r3_r2 + x_r4_r2 <= 1
    in_r3_x: x_r1_r3 + x_r2_r3 + x_r4_r3 <= 1
    in_r4_x: x_r2_r4 + x_r3_r4 <= 1

    in_r1_y: y_r2_r1 + y_r3_r1 <= 1
    in_r2_y: y_r1_r2 + y_r3_r2 + y_r4_r2 <= 1
    in_r3_y: y_r1_r3 + y_r2_r3 + y_r4_r3 <= 1
    in_r4_y: y_r2_r4 + y_r3_r4 <= 1

    \ mutual exclusion of outgoing out of same node, for each flow, for each node
    out_r1_x: x_r1_r2 + x_r1_r3 <= 1
    out_r2_x: x_r2_r1 + x_r2_r3 + x_r2_r4 <= 1
    out_r3_x: x_r3_r1 + x_r3_r2 + x_r3_r4 <= 1
    out_r4_x: x_r4_r2 + x_r4_r3 <= 1

    out_r1_y: y_r1_r2 + y_r1_r3 <= 1
    out_r2_y: y_r2_r1 + y_r2_r3 + y_r2_r4 <= 1
    out_r3_y: y_r3_r1 + y_r3_r2 + y_r3_r4 <= 1
    out_r4_y: y_r4_r2 + y_r4_r3 <= 1

    \ link capacities, for each link
    r1_r2_c: rx_r1_r2 + rx_r2_r1 + ry_r1_r2 + ry_r2_r1 <= 3
    r1_r3_c: rx_r1_r3 + rx_r3_r1 + ry_r1_r3 + ry_r3_r1 <= 10
    r2_r4_c: rx_r2_r4 + rx_r4_r2 + ry_r2_r4 + ry_r4_r2 <= 10
    r3_r2_c: rx_r3_r2 + rx_r2_r3 + ry_r3_r2 + ry_r2_r3 <= 8
    r3_r4_c: rx_r3_r4 + rx_r4_r3 + ry_r3_r4 + ry_r4_r3 <= 10

    \ control of real-value flow variables by corresponding indicators, for each flow and link
    r1_r2_cx: rx_r1_r2 - 3 x_r1_r2 <= 0
    r1_r3_cx: rx_r1_r3 - 10 x_r1_r3 <= 0
    r2_r3_cx: rx_r2_r3 - 8 x_r2_r3 <= 0
    r2_r4_cx: rx_r2_r4 - 10 x_r2_r4 <= 0
    r3_r4_cx: rx_r3_r4 - 10 x_r3_r4 <= 0

    r2_r1_cx: rx_r2_r1 - 3 x_r2_r1 <= 0
    r3_r1_cx: rx_r3_r1 - 10 x_r3_r1 <= 0
    r3_r2_cx: rx_r3_r2 - 8 x_r3_r2 <= 0
    r4_r2_cx: rx_r4_r2 - 10 x_r4_r2 <= 0
    r4_r3_cx: rx_r4_r3 - 10 x_r4_r3 <= 0

    r1_r2_cy: ry_r1_r2 - 3 y_r1_r2 <= 0
    r1_r3_cy: ry_r1_r3 - 10 y_r1_r3 <= 0
    r2_r3_cy: ry_r2_r3 - 8 y_r2_r3 <= 0
    r2_r4_cy: ry_r2_r4 - 10 y_r2_r4 <= 0
    r3_r4_cy: ry_r3_r4 - 10 y_r3_r4 <= 0

    r2_r1_cy: ry_r2_r1 - 3 y_r2_r1 <= 0
    r3_r1_cy: ry_r3_r1 - 10 y_r3_r1 <= 0
    r3_r2_cy: ry_r3_r2 - 8 y_r3_r2 <= 0
    r4_r2_cy: ry_r4_r2 - 10 y_r4_r2 <= 0
    r4_r3_cy: ry_r4_r3 - 10 y_r4_r3 <= 0



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