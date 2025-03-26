import numpy as np 
import attitude_dynamics as att

def quest_algorithm(b_b, b_o, s_b, s_o):

    # gains
    a1 = 0.1
    a2 = 0.1 

    B = a1 * np.outer(b_o, b_b) + a2 * np.outer(s_o, s_b)
    
    z = np.array([B[1][2]- B[2][1], B[2][0]- B[0][2], B[0][1]- B[1][0]]).T
    I = np.eye(3)

    C = B + B.T
    eta = np.trace(B)

    top_left = np.array([[eta]])
    top_right = z.reshape(1,3)
    bottom_left = z.reshape(1,3).T
    bottom_right = C - eta*I

    K = np.block([
                [top_left, top_right], 
                [bottom_left, bottom_right]
                ])

    eigenvalues, eigenvectors = np.linalg.eig(K)
    idx_max = np.argmax(eigenvalues)

    q_ob_hat = eigenvectors[:, idx_max]

    q_ob_hat = q_ob_hat / np.linalg.norm(q_ob_hat)

    return q_ob_hat

def measure_sensor(v_b, sensor_accuracy):

    angle_standard_deviation = sensor_accuracy/2

    angle_noise = np.random.normal(0, angle_standard_deviation)

    random_axis_of_rotation = np.random.randn(3)
    k_noise = random_axis_of_rotation / np.linalg.norm(random_axis_of_rotation)

    q_mb = np.array([np.cos(angle_noise/2), k_noise[0]*np.sin(angle_noise/2), k_noise[1]*np.sin(angle_noise/2), k_noise[2]*np.sin(angle_noise/2)])

    R_m_b = att.calculate_rotation_matrix_from_quaternion(q_mb)

    v_b_measurement = R_m_b @ v_b

    return v_b_measurement

def quaternion_sign_correction(q_new,q_old): 
    if (np.dot(q_old, q_new) < 0.0): 
        q_new = -q_new
    return q_new

def match_quaternion_sign_to_reference(q_est, q_ref): 
    if (np.dot(q_est, q_ref) < 0.0): 
        q_est = -q_est
    return q_est



