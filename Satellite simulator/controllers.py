import numpy as np
import attitude_dynamics as att



def PD_controller (q_ob, w_ob_b, q_od, w_od_d, k_p, k_d):

    q_do = att.calculate_inverse_quaternion(q_od)
    q_db = att.T(q_do)@q_ob
    #print("q_db", q_db)
    epsilon_db = q_db[1:]

    R_b_d = att.calculate_rotation_matrix_from_quaternion(q_db)
    w_db_b = w_ob_b - R_b_d @ w_od_d
    
    tau_b_d = -k_p * epsilon_db - k_d * w_db_b 


    return tau_b_d
