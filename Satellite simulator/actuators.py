import numpy as np
import attitude_dynamics as att 



def attitude_control_using_thrusters(tau_d_b, max_thrust, deadzone = 0):

    B = np.array([  [-np.sqrt(2)/5, np.sqrt(2)/5, np.sqrt(2)/5, -np.sqrt(2)/5],
                    [np.sqrt(2)/4, -np.sqrt(2)/4, np.sqrt(2)/4, -np.sqrt(2)/4],
                    [-np.sqrt(2)/4, -np.sqrt(2)/4, np.sqrt(2)/4, np.sqrt(2)/4]
                    ])

    u_d = np.linalg.pinv(B) @ tau_d_b

    u = np.zeros_like(u_d)

    for i in range(len(u_d)):
        if (u_d[i] > deadzone):
            u[i] = max_thrust
        else:
            u[i] = 0

    tau_b_a = B @ u 

    return tau_b_a, u 

def attitude_control_using_reaction_wheels_in_tethrahedron(tau_d_b, w_ib_b, w_bw_b, J_w, max_RPM, B = None): 
    
    max_RPM = max_RPM * 2 * np.pi / 60

    if B is None:
        B = np.array([
            [ np.sqrt(3)/3,-np.sqrt(3)/3,-np.sqrt(3)/3, np.sqrt(3)/3],
            [ np.sqrt(3)/3,-np.sqrt(3)/3, np.sqrt(3)/3,-np.sqrt(3)/3],
            [ np.sqrt(3)/3, np.sqrt(3)/3,-np.sqrt(3)/3,-np.sqrt(3)/3]
        ])

    G = J_w * B

    u = - np.linalg.pinv(G) @ tau_d_b

    for i in range(len(u)):
        if (w_bw_b[i] >= max_RPM) and (u[i] > 0):
            u[i] = 0
        elif (w_bw_b[i] <= -max_RPM) and (u[i] < 0):
            u[i] = 0
        else:
            pass 

    w_bw_b_dot = np.linalg.pinv(G) @ (-att.S(w_ib_b) @ G @ w_bw_b + G @ u)


    tau_b_a = -G @ u

    return tau_b_a, w_bw_b_dot, u 

def attitude_control_using_magnetorquers(tau_d_b, b_b):
    b_b = b_b.reshape(3)
    #print("b_b", b_b)

    I_max = 1 
    N = 500
    A = 1

    m_b = (att.S(b_b) @ tau_d_b) / (np.linalg.norm(b_b)**2)

    I = m_b / (N *A)
    #print("I", I)

    for i in range(len(I)):
        if (I[i] > I_max):
            I[i] = I_max
        elif (I[i] < -I_max):
            I[i] = -I_max
        else:
            pass
    
    m_b_actual = I * N * A

    tau_m = att.S(m_b_actual) @ b_b

    tau_a_b = np.array([np.sign(tau_d_b[0])* np.abs(tau_m[0]), np.sign(tau_d_b[1])* np.abs(tau_m[1]), np.sign(tau_d_b[2])* np.abs(tau_m[2])])

    return tau_a_b 

def reaction_wheels_desaturation(desired_RPM, w_ib_b, w_bw_b, J_w): 

    B = np.array([
        [ np.sqrt(3)/3,-np.sqrt(3)/3,-np.sqrt(3)/3, np.sqrt(3)/3],
        [ np.sqrt(3)/3,-np.sqrt(3)/3, np.sqrt(3)/3,-np.sqrt(3)/3],
        [ np.sqrt(3)/3, np.sqrt(3)/3,-np.sqrt(3)/3,-np.sqrt(3)/3]
    ])

    G = J_w * B

    H_w = J_w * w_bw_b

    
    # Compute the torque exerted by the wheels on the satellite

    print ("H_w", H_w)

    tau_w = G @ H_w # 
    
    # Compute the required torque for desaturation
    tau_desat = -tau_w
    
    print ("tau_desat", tau_desat)
    return tau_desat