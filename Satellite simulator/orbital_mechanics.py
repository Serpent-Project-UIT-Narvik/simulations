import numpy as np
import attitude_dynamics as att
import ppigrf

G = 6.669e-11
M_earth = 5.9742e24
MU = G*M_earth




def testFunction():
    print("Hello from orbital_mechanics.py")

def calculate_circular_orbital_speed(radius: float) -> float: 
    
    v = np.sqrt(MU/radius)
    return v


def calculate_satellite_position_in_circular_orbit(angle_phi: float, radius: float) -> np.ndarray:
    
    x = 0 
    y = radius*np.cos(angle_phi)
    z = radius*np.sin(angle_phi)
    return np.array([x, y, z])


def calculate_circular_orbit_velocity(angle_phi: float, radius: float) -> np.ndarray:
    
    V = calculate_circular_orbital_speed(radius)
    varphi_dot = V / radius 

    x_dot = 0
    y_dot = -radius*varphi_dot*np.sin(angle_phi)
    z_dot = radius*varphi_dot*np.cos(angle_phi)

    return np.array([x_dot, y_dot, z_dot])

def calculate_eccentricity(r_apogee: float, r_perigee: float) -> float:
    return (r_apogee - r_perigee)/(r_apogee + r_perigee)

def calculate_semimajor_axis(r_apogee: float, r_perigee: float) -> float:
    return (r_apogee + r_perigee)/2
    
def calculate_mean_motion(semimajor_axis: float) -> float:
    return np.sqrt(MU/semimajor_axis**3)
    
def calculate_orbital_period(mean_motion: float) -> float:
    return 2*np.pi/mean_motion
    
def calculate_eccentric_anomaly(mean_motion: float, t: float, eccentricity: float) -> float:
    mean_anomaly = mean_motion*t
    eccentric_anomaly = mean_anomaly 

    for i in range(25):
        eccentric_anomaly = mean_anomaly + eccentricity*np.sin(eccentric_anomaly)

    return eccentric_anomaly
    
def calculate_true_anomaly(eccentric_anomaly: float, eccentricity: float) -> float:
    true_anomaly = np.arccos((np.cos(eccentric_anomaly) - eccentricity)/(1 - eccentricity*np.cos(eccentric_anomaly)))

    return true_anomaly

def calculate_true_anomaly_derivative(mean_motion: float, true_anomaly: float, eccentricity: float) -> float:

    return (mean_motion * (1 + eccentricity*np.cos(true_anomaly))**2)/(1 - eccentricity**2)**(3/2)
    

def calculate_rotation_matrix_from_inertial_to_pqw(inclination: float, right_ascension: float, argument_of_perigee: float) -> np.ndarray:    
    # for simplicity 
    i = inclination
    o = right_ascension
    w = argument_of_perigee
    
    m11 = np.cos(w)*np.cos(o) - np.cos(i)*np.sin(w)*np.sin(o)
    m12 = np.cos(w)*np.sin(o) + np.sin(w)*np.cos(i)*np.cos(o)
    m13 = np.sin(w)*np.sin(i)
    m21 = -np.sin(w)*np.cos(o) - np.cos(i)*np.sin(o)*np.cos(w)
    m22 = -np.sin(w)*np.sin(o) + np.cos(w)*np.cos(i)*np.cos(o)
    m23 = np.cos(w)*np.sin(i)
    m31 = np.sin(i)*np.sin(o)
    m32 = -np.sin(i)*np.cos(o)
    m33 = np.cos(i)

    return np.array([[m11, m12, m13],
                     [m21, m22, m23],
                     [m31, m32, m33]])


def calculate_rotation_matrix_from_inertial_to_orbit(inclination: float, right_ascension: float, argument_of_perigee: float, true_anomaly: float) -> np.ndarray:
    i = inclination
    o = right_ascension
    w = argument_of_perigee
    t = true_anomaly

    m11 = np.cos(w+t)*np.cos(o) - np.cos(i)*np.sin(w+t)*np.sin(o)
    m12 = np.cos(w+t)*np.sin(o) + np.sin(w+t)*np.cos(i)*np.cos(o)
    m13 = np.sin(w+t)*np.sin(i)
    m21 = -np.sin(w+t)*np.cos(o) - np.cos(i)*np.sin(o)*np.cos(w+t)
    m22 = -np.sin(w+t)*np.sin(o) + np.cos(w+t)*np.cos(i)*np.cos(o)
    m23 = np.cos(w+t)*np.sin(i)
    m31 = np.sin(i)*np.sin(o)
    m32 = -np.sin(i)*np.cos(o)
    m33 = np.cos(i)

    #print(m11, m22, m33)

    return np.array([[m11, m12, m13],
                     [m21, m22, m23],
                     [m31, m32, m33]])

def calculate_angular_velocity_of_orbit_relative_to_inertial_referenced_in_inertial(inclination: float, right_ascension: float, argument_of_perigee: float, eccentric_anomaly: float, semimajor_axis: float, eccentricity: float) -> np.ndarray:
    r_i = calculate_radius_vector_in_inertial(inclination, right_ascension, argument_of_perigee, eccentric_anomaly, semimajor_axis, eccentricity)
    v_i = calculate_velocity_vector_in_inertial(inclination, right_ascension, argument_of_perigee, eccentric_anomaly, semimajor_axis, eccentricity)

    omega_i_i_o = np.cross(r_i, v_i)/(r_i.T @ r_i) 

    return omega_i_i_o

def calculate_angular_acceleration_of_orbit_relative_to_inertial_referenced_in_inertial(inclination: float, right_ascension: float, argument_of_perigee: float, eccentric_anomaly: float, semimajor_axis: float, eccentricity: float) -> np.ndarray:
    r_i = calculate_radius_vector_in_inertial(inclination, right_ascension, argument_of_perigee, eccentric_anomaly, semimajor_axis, eccentricity)
    v_i = calculate_velocity_vector_in_inertial(inclination, right_ascension, argument_of_perigee, eccentric_anomaly, semimajor_axis, eccentricity)
    a_i = calculate_acceleration_vector_in_inertial(inclination, right_ascension, argument_of_perigee, eccentric_anomaly, semimajor_axis, eccentricity)

    omega_dot_i_i_o = ((np.cross(r_i, a_i)@ r_i.T * r_i) - 2*np.cross(r_i, v_i)*v_i.T @ r_i) / ((r_i.T @ r_i)**2)

    return omega_dot_i_i_o
    

def calculate_radius_vector_in_pqw(semimajor_axis: float, eccentricity: float, eccentric_anomaly: float) -> np.ndarray:
    
    x = semimajor_axis * np.cos(eccentric_anomaly) - semimajor_axis * eccentricity
    y = semimajor_axis * np.sin(eccentric_anomaly) * np.sqrt(1 - eccentricity**2)
    z = 0

    return np.array([x, y, z]).T

def calculate_velocity_vector_in_pqw(semimajor_axis: float, eccentricity: float, eccentric_anomaly: float) -> np.ndarray:
    
    mean_motion = calculate_mean_motion(semimajor_axis)

    r_mag = np.linalg.norm(calculate_radius_vector_in_pqw(semimajor_axis, eccentricity, eccentric_anomaly))
    x_dot = -(semimajor_axis**2 * mean_motion * np.sin(eccentric_anomaly))/ r_mag 
    y_dot = ((semimajor_axis**2 * mean_motion) / r_mag) * np.sqrt(1 - eccentricity**2) * np.cos(eccentric_anomaly)
    z_dot = 0

    return np.array([x_dot, y_dot, z_dot]).T

def calculate_acceleration_vector_in_pqw(semimajor_axis: float, eccentricity: float, eccentric_anomaly: float) -> np.ndarray:
    
    mean_motion = calculate_mean_motion(semimajor_axis)

    r_mag = np.linalg.norm(calculate_radius_vector_in_pqw(semimajor_axis, eccentricity, eccentric_anomaly))
    x_ddot = -((semimajor_axis**3 * mean_motion**2)/ r_mag**2)*np.cos(eccentric_anomaly)
    y_ddot = -((semimajor_axis**3 * mean_motion**2)/ r_mag**2)*np.sqrt(1 - eccentricity**2)*np.sin(eccentric_anomaly)
    z_ddot = 0

    return np.array([x_ddot, y_ddot, z_ddot]).T

def calculate_radius_vector_in_inertial(inclination: float, right_ascension: float, argument_of_perigee: float, eccentric_anomaly: float, semimajor_axis: float, eccentricity: float) -> np.ndarray:
    R_i_pqw = calculate_rotation_matrix_from_inertial_to_pqw(inclination, right_ascension, argument_of_perigee).T
    r_pqw = calculate_radius_vector_in_pqw(semimajor_axis, eccentricity, eccentric_anomaly)
    r_i = R_i_pqw @ r_pqw

    return r_i

def calculate_velocity_vector_in_inertial(inclination: float, right_ascension: float, argument_of_perigee: float, eccentric_anomaly: float, semimajor_axis: float, eccentricity: float) -> np.ndarray:
    R_i_pqw = calculate_rotation_matrix_from_inertial_to_pqw(inclination, right_ascension, argument_of_perigee).T
    v_pqw = calculate_velocity_vector_in_pqw(semimajor_axis, eccentricity, eccentric_anomaly)
    v_i = R_i_pqw @ v_pqw

    return v_i

def calculate_acceleration_vector_in_inertial(inclination: float, right_ascension: float, argument_of_perigee: float, eccentric_anomaly: float, semimajor_axis: float, eccentricity: float) -> np.ndarray:
    
    R_i_pqw = calculate_rotation_matrix_from_inertial_to_pqw(inclination, right_ascension, argument_of_perigee).T
    a_pqw = calculate_acceleration_vector_in_pqw(semimajor_axis, eccentricity, eccentric_anomaly)
    a_i = R_i_pqw @ a_pqw

    return a_i

def calculate_quaternion_from_orbital_parameters(arguement_of_perogee, RAAN, inclination, true_anomaly) -> np.ndarray:
    # BIG OMEGA = RAAN 
    # i = inclination
    # omega = argument of perogee
    # theta = true anomaly

    Omega = RAAN
    omega = arguement_of_perogee
    i = inclination
    theta = true_anomaly

    #print (Omega, omega, i, theta)
    
    q_Omega = np.array([np.cos(Omega/2), 0, 0, np.sin(Omega)/2]).T
    q_i = np.array([np.cos(i/2), np.sin(i/2), 0, 0]).T
    q_omega_theta = np.array([np.cos((omega + theta)/2), 0, 0, np.sin((omega + theta)/2)]).T

    #print("q_Omega", q_Omega)
    #print("q_i", q_i)
    #print("q_omega_theta", q_omega_theta)



    q_io = att.T(q_Omega) @ att.T(q_i) @ q_omega_theta

    #print("q_io", q_io)

    return q_io

def calculate_rotation_matrix_from_intertial_to_ecef(t): 
    print("t", t)
    omega = (2*np.pi) / (60*60*24)
    print ("omega", omega)

    R_e_i = np.array([[np.cos(omega * t), -np.sin(omega*t), 0],
                  [np.sin(omega * t), np.cos(omega * t), 0],
                  [0, 0, 1]])
    
    return R_e_i

def ECEF_to_NED(longitude, latitude): 

    mu = latitude 
    l = longitude

    R = np.array([[-np.cos(l)*np.sin(mu), -np.sin(l), -np.cos(l)*np.cos(mu)],
                    [-np.sin(l)*np.sin(mu), np.cos(l), -np.sin(l)*np.cos(mu)],
                    [np.cos(mu), 0, -np.sin(mu)]]).T
    

    return R

def NED_to_ENU(): 
    
    R = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, -1]])

    return R

def calculate_lla_from_ecef(r_e):

    # based on the code provided in the assignment 5 document
    
    a_e = 6378137
    b_e = 6356725
    w_ie = 7.292115e-5
    e_e = 0.0818

    x = r_e[0]
    y = r_e[1]
    z = r_e[2]

    p = np.sqrt(x**2 + y**2)

    mu = np.arctan((z/p)*(1 - e_e**2)**-1)

    mu_old = 10 
    while (np.abs(mu - mu_old) > 1e-10):
        mu_old = mu
        N = a_e **2 / np.sqrt(a_e**2 * np.cos(mu)**2 + b_e**2 *np.sin(mu)**2)

        h = p/np.cos(mu) - N

        mu = np.arctan((z/p)*(1 - e_e**2*(N/(N+h)))**-1)


    l = np.arctan2(y, x)

    latitude = mu
    longitude = l
    altitude = h

    return latitude, longitude, altitude

def calculate_magnetic_field_in_orbit_frame(r_i, R_o_i, date, t, debug = False):
    
    R_i_o = R_o_i.T
    R_e_i = calculate_rotation_matrix_from_intertial_to_ecef(t)
    r_e = R_e_i @ r_i
    latlong = calculate_lla_from_ecef(r_e)
    R_n_e = ECEF_to_NED(latlong[0], latlong[1])
    r_n = R_n_e @ r_e
    R_u_n = NED_to_ENU()
    r_u = R_u_n @ r_n

    alt = latlong[2] / 1000

    B = ppigrf.igrf(latlong[1]*180/np.pi, latlong[0]*180/np.pi, alt, date)

    B_u = np.array(B)

    B_n = R_u_n.T @ B_u
    B_e = R_n_e.T @ B_n
    B_i = R_e_i.T @ B_e
    B_o = R_i_o.T @ B_i

    if debug:
        print("\nB_u = \n", B_u)
        print("\nB_o = \n", B_o)
    
    return B_o * 1e-9