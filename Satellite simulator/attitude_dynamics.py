import numpy as np


def S(w): 
    #print(w)
    S_mat = np.array([[0, -w[2], w[1]],
                      [w[2], 0, -w[0]],
                      [-w[1], w[0], 0]])
    return S_mat
                      
def T(q): 
    #print("q"   , q)
    eta = q[0]
    epsilon = np.array([q[1], q[2], q[3]])
    I = np.eye(3)

    top_left = np.array([[eta]])
    top_right = -epsilon.reshape(1,3)
    bottom_left = epsilon.reshape(3,1)
    bottom_right = eta*I + S(epsilon)

    T_mat = np.block([
                    [top_left, top_right], 
                    [bottom_left, bottom_right]
                    ])

    return T_mat 

def calculate_rotation_matrix_from_quaternion(q_ob): 
    #print("q_ob", q_ob)
    """    
    eta = q_ob[0]
    epsilon_x = q_ob[1]
    epsilon_y = q_ob[2]
    epsilon_z = q_ob[3]

    R = np.array([
        [1 - 2 * (epsilon_y**2 + epsilon_z**2), 2 * (epsilon_x * epsilon_y - eta * epsilon_z), 2 * (epsilon_x * epsilon_z + eta * epsilon_y)],
        [2 * (epsilon_x * epsilon_y + eta * epsilon_z), 1 - 2 * (epsilon_x**2 + epsilon_z**2), 2 * (epsilon_y * epsilon_z - eta * epsilon_x)],
        [2 * (epsilon_x * epsilon_z - eta * epsilon_y), 2 * (epsilon_y * epsilon_z + eta * epsilon_x), 1 - 2 * (epsilon_x**2 + epsilon_y**2)]
    ])
    
    return R
    """
    eta = q_ob[0]
    epsilon = np.array([q_ob[1], q_ob[2], q_ob[3]])
    I = np.eye(3)

    R = I + 2 * eta * S(epsilon) +2 * np.dot(S(epsilon), S(epsilon))

    return R

def calculate_inverse_quaternion(q_ob): 
    
    q_bo = np.array([q_ob[0], -q_ob[1], -q_ob[2], -q_ob[3]])
    return q_bo

def quaternion_kinematics(q_ob, w_ob_b): 
    
    q_dot = 0.5 * T(q_ob) @ np.append(0, w_ob_b).T
    
    return q_dot

def attitude_dynamics(J,q_ob, w_ob_b, w_io_i, w_io_i_dot, R_i_o, tau_a_b, tau_p_b): 
    
    R_o_i = R_i_o.T

    q_bo = calculate_inverse_quaternion(q_ob)
    R_b_o = calculate_rotation_matrix_from_quaternion(q_bo)

    R_b_i = R_b_o @ R_o_i

    w_oi_b = - R_b_i @ w_io_i
    w_ib_b = w_oi_b - w_ob_b

    term1 = - S(w_ib_b) @ J @ w_ib_b 
    term2 = tau_a_b + tau_p_b
    term3 = J @ S(w_ib_b) @ R_b_i @ w_io_i
    term4 = J @ R_b_i @ w_io_i_dot

    w_ob_b_dot = np.linalg.inv(J) @ (term1 + term2 - term3 - term4)

    return w_ob_b_dot

def calculate_euler_angles_from_quaternion(q_ab):
    print("q_ab", q_ab)
    q0 = q_ab[0]
    q1 = q_ab[1]
    q2 = q_ab[2]
    q3 = q_ab[3]

    phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    theta = 0 
    if abs(2*(q0*q2 - q1*q3)) >= 1: 
        theta = np.pi/2 * np.sign(2*(q0*q2 - q1*q3))
    else:
        theta = np.arcsin(2*(q0*q2 - q1*q3))

    psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))

    return np.array([phi*180/np.pi, theta*180/np.pi, psi*180/np.pi])

def quaternion_from_vectors(vector_a, vector_b):

    vector_a = vector_a / np.linalg.norm(vector_a)
    vector_b = vector_b / np.linalg.norm(vector_b)

    angle = np.dot(vector_a, vector_b)
    angle = np.arccos(angle)

    axis= np.cross(vector_a, vector_b)

    axis = axis / np.linalg.norm(axis)

    w = np.cos(angle/2)
    x,y,z = axis * np.sin(angle/2)

    quaternion = np.array([w,x,y,z])

    return quaternion

def quaternion_sign_correction(q_new,q_old):
    if np.dot(q_new,q_old) < 0:
        q_new = -q_new
    return q_new

def match_quaternion_sign_to_referance(q_est, q_ref):
    if np.dot(q_est, q_ref) < 0:
        q_est = -q_est
    return q_est