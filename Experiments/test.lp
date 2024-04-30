Maximize
obj:  rate_min

Subject to
    \ rate_min is the minimum of all the flow rates
    rate_min_0: rate_0 - rate_min >= 0
    rate_min_1: rate_1 - rate_min >= 0
    ratio_0: lambda_0 - 10 rate_0 = 0
    ratio_1: lambda_1 - 15 rate_1 = 0

    \ flow balance for real-value rate variables
    flow_r1_br0: fbr0_r2_r1 - fbr0_r1_r2 + fbr0_r3_r1 - fbr0_r1_r3 + lambda_0 = 0
    flow_r2_br0: fbr0_r1_r2 - fbr0_r2_r1 + fbr0_r4_r2 - fbr0_r2_r4 + fbr0_r3_r2 - fbr0_r2_r3 = 0
    flow_r3_br0: fbr0_r1_r3 - fbr0_r3_r1 + fbr0_r2_r3 - fbr0_r3_r2 + fbr0_r4_r3 - fbr0_r3_r4 = 0
    flow_r4_br0: fbr0_r2_r4 - fbr0_r4_r2 + fbr0_r3_r4 - fbr0_r4_r3 - lambda_0 = 0

    flow_r1_br1: fbr1_r2_r1 - fbr1_r1_r2 + fbr1_r3_r1 - fbr1_r1_r3 = 0
    flow_r2_br1: fbr1_r1_r2 - fbr1_r2_r1 + fbr1_r4_r2 - fbr1_r2_r4 + fbr1_r3_r2 - fbr1_r2_r3 = 0
    flow_r3_br1: fbr1_r1_r3 - fbr1_r3_r1 + fbr1_r2_r3 - fbr1_r3_r2 + fbr1_r4_r3 - fbr1_r3_r4 + lambda_1 = 0
    flow_r4_br1: fbr1_r2_r4 - fbr1_r4_r2 + fbr1_r3_r4 - fbr1_r4_r3 - lambda_1 = 0

    \ flow balance for indicators, for each flow, for each node
    flow_r1_b0: fbi0_r2_r1 - fbi0_r1_r2 + fbi0_r3_r1 - fbi0_r1_r3 = -1
    flow_r2_b0: fbi0_r1_r2 - fbi0_r2_r1 + fbi0_r4_r2 - fbi0_r2_r4 + fbi0_r3_r2 - fbi0_r2_r3 = 0
    flow_r3_b0: fbi0_r1_r3 - fbi0_r3_r1 + fbi0_r2_r3 - fbi0_r3_r2 + fbi0_r4_r3 - fbi0_r3_r4 = 0
    flow_r4_b0: fbi0_r2_r4 - fbi0_r4_r2 + fbi0_r3_r4 - fbi0_r4_r3 = 1

    flow_r1_b1: fbi1_r2_r1 - fbi1_r1_r2 + fbi1_r3_r1 - fbi1_r1_r3 = 0
    flow_r2_b1: fbi1_r1_r2 - fbi1_r2_r1 + fbi1_r4_r2 - fbi1_r2_r4 + fbi1_r3_r2 - fbi1_r2_r3 = 0
    flow_r3_b1: fbi1_r1_r3 - fbi1_r3_r1 + fbi1_r2_r3 - fbi1_r3_r2 + fbi1_r4_r3 - fbi1_r3_r4 = -1
    flow_r4_b1: fbi1_r2_r4 - fbi1_r4_r2 + fbi1_r3_r4 - fbi1_r4_r3 = 1

    \ mutual exclusion of incoming flows into same node, for each flow, for each node
    in_r1_0: fbi0_r2_r1 + fbi0_r3_r1 <= 1
    in_r2_0: fbi0_r1_r2 + fbi0_r3_r2 + fbi0_r4_r2 <= 1
    in_r3_0: fbi0_r1_r3 + fbi0_r2_r3 + fbi0_r4_r3 <= 1
    in_r4_0: fbi0_r2_r4 + fbi0_r3_r4 <= 1

    in_r1_1: fbi1_r2_r1 + fbi1_r3_r1 <= 1
    in_r2_1: fbi1_r1_r2 + fbi1_r3_r2 + fbi1_r4_r2 <= 1
    in_r3_1: fbi1_r1_r3 + fbi1_r2_r3 + fbi1_r4_r3 <= 1
    in_r4_1: fbi1_r2_r4 + fbi1_r3_r4 <= 1

    \ mutual exclusion of outgoing out of same node, for each flow, for each node
    out_r1_0: fbi0_r1_r2 + fbi0_r1_r3 <= 1
    out_r2_0: fbi0_r2_r1 + fbi0_r2_r3 + fbi0_r2_r4 <= 1
    out_r3_0: fbi0_r3_r1 + fbi0_r3_r2 + fbi0_r3_r4 <= 1
    out_r4_0: fbi0_r4_r2 + fbi0_r4_r3 <= 1

    out_r1_1: fbi1_r1_r2 + fbi1_r1_r3 <= 1
    out_r2_1: fbi1_r2_r1 + fbi1_r2_r3 + fbi1_r2_r4 <= 1
    out_r3_1: fbi1_r3_r1 + fbi1_r3_r2 + fbi1_r3_r4 <= 1
    out_r4_1: fbi1_r4_r2 + fbi1_r4_r3 <= 1

    \ link capacities, for each link
    r1_r2_c: fbr0_r1_r2 + fbr0_r2_r1 + fbr1_r1_r2 + fbr1_r2_r1 <= 3
    r1_r3_c: fbr0_r1_r3 + fbr0_r3_r1 + fbr1_r1_r3 + fbr1_r3_r1 <= 10
    r2_r4_c: fbr0_r2_r4 + fbr0_r4_r2 + fbr1_r2_r4 + fbr1_r4_r2 <= 10
    r3_r2_c: fbr0_r3_r2 + fbr0_r2_r3 + fbr1_r3_r2 + fbr1_r2_r3 <= 8
    r3_r4_c: fbr0_r3_r4 + fbr0_r4_r3 + fbr1_r3_r4 + fbr1_r4_r3 <= 10

    \ control of real-value flow variables by corresponding indicators, for each flow and link
    r1_r2_c0: fbr0_r1_r2 - 3 fbi0_r1_r2 <= 0
    r1_r3_c0: fbr0_r1_r3 - 10 fbi0_r1_r3 <= 0
    r2_r3_c0: fbr0_r2_r3 - 8 fbi0_r2_r3 <= 0
    r2_r4_c0: fbr0_r2_r4 - 10 fbi0_r2_r4 <= 0
    r3_r4_c0: fbr0_r3_r4 - 10 fbi0_r3_r4 <= 0

    r2_r1_c0: fbr0_r2_r1 - 3 fbi0_r2_r1 <= 0
    r3_r1_c0: fbr0_r3_r1 - 10 fbi0_r3_r1 <= 0
    r3_r2_c0: fbr0_r3_r2 - 8 fbi0_r3_r2 <= 0
    r4_r2_c0: fbr0_r4_r2 - 10 fbi0_r4_r2 <= 0
    r4_r3_c0: fbr0_r4_r3 - 10 fbi0_r4_r3 <= 0

    r1_r2_c1: fbr1_r1_r2 - 3 fbi1_r1_r2 <= 0
    r1_r3_c1: fbr1_r1_r3 - 10 fbi1_r1_r3 <= 0
    r2_r3_c1: fbr1_r2_r3 - 8 fbi1_r2_r3 <= 0
    r2_r4_c1: fbr1_r2_r4 - 10 fbi1_r2_r4 <= 0
    r3_r4_c1: fbr1_r3_r4 - 10 fbi1_r3_r4 <= 0

    r2_r1_c1: fbr1_r2_r1 - 3 fbi1_r2_r1 <= 0
    r3_r1_c1: fbr1_r3_r1 - 10 fbi1_r3_r1 <= 0
    r3_r2_c1: fbr1_r3_r2 - 8 fbi1_r3_r2 <= 0
    r4_r2_c1: fbr1_r4_r2 - 10 fbi1_r4_r2 <= 0
    r4_r3_c1: fbr1_r4_r3 - 10 fbi1_r4_r3 <= 0



Binary
    fbi0_r3_r2
    fbi0_r4_r2
    fbi0_r1_r2
    fbi0_r2_r3
    fbi0_r4_r3
    fbi0_r1_r3
    fbi0_r3_r4
    fbi0_r2_r4
    fbi0_r3_r1
    fbi0_r2_r1

    fbi1_r3_r2
    fbi1_r4_r2
    fbi1_r1_r2
    fbi1_r2_r3
    fbi1_r4_r3
    fbi1_r1_r3
    fbi1_r3_r4
    fbi1_r2_r4
    fbi1_r3_r1
    fbi1_r2_r1

End