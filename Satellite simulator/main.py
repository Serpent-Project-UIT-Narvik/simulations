import vtk
import pyvista as pv 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import visualization as vis
import orbital_mechanics as orb
import attitude_dynamics as att
import controllers as ctrl
import datetime as dt
import sun_vector_models as sun
import sensors as sens
import attitude_determination as adet
import actuators as act

def satellite_dynamics_loop(t, state, params): 

    true_anomaly = state[0]
    q_ob = np.array([state[1], state[2], state[3], state[4]])
    #print("q_ob", q_ob)
    w_ob_b = np.array([state[5], state[6], state[7]])
    propelant_mass = state[8]
    w_bw_b = np.array([state[9], state[10], state[11], state[12]])
    state_dot = np.zeros_like(state)


    # Orbital dynamics -------------------------------------------------------
    e = orb.calculate_eccentricity(params["r_apogee"], params["r_perigee"])
    a = orb.calculate_semimajor_axis(params["r_apogee"], params["r_perigee"])
    n = orb.calculate_mean_motion(a)
    state_dot[0] = orb.calculate_true_anomaly_derivative(n, true_anomaly, e)
    eccentric_anomaly = orb.calculate_eccentric_anomaly(n, t, e)
    r_i = orb.calculate_radius_vector_in_inertial(params["inclination"], params["RAAN"], params["argumet_of_perigee"], eccentric_anomaly, a, e)

    v_i = orb.calculate_velocity_vector_in_inertial(params["inclination"], params["RAAN"], params["argumet_of_perigee"], eccentric_anomaly, a, e)
    a_i = orb.calculate_acceleration_vector_in_inertial(params["inclination"], params["RAAN"], params["argumet_of_perigee"], eccentric_anomaly, a, e)
    
    R_o_i = orb.calculate_rotation_matrix_from_inertial_to_orbit(params["inclination"], params["RAAN"], params["argumet_of_perigee"], true_anomaly)

    q_io = orb.calculate_quaternion_from_orbital_parameters(params["argumet_of_perigee"], params["RAAN"], params["inclination"], true_anomaly)
    
    R_o_i_2 = att.calculate_rotation_matrix_from_quaternion(q_io)

    q_ib = att.T(q_io)@ q_ob


    w_io_i = orb.calculate_angular_velocity_of_orbit_relative_to_inertial_referenced_in_inertial(params["inclination"], params["RAAN"], params["argumet_of_perigee"], eccentric_anomaly, a,e)
    w_io_i_dot = orb.calculate_angular_acceleration_of_orbit_relative_to_inertial_referenced_in_inertial(params["inclination"], params["RAAN"], params["argumet_of_perigee"], eccentric_anomaly, a, e)

    R_b_o = att.calculate_rotation_matrix_from_quaternion(q_ob)
    R_o_b = R_b_o.T

    # End of orbital dynamics ------------------------------------------------

    # Attitude determination -------------------------------------------------

    s_i = sun.calculate_advanced_sun_model(t, params["datetime"])
    s_o = R_o_i @ s_i
    s_o_unit = s_o / np.linalg.norm(s_o)

    s_b = R_b_o @ s_o
    #s_b_measured = adet.measure_sensor(s_b, 5*np.pi/180) # add noise to the sensor
    s_b_measured = s_b # For faster debugging

    b_o = orb.calculate_magnetic_field_in_orbit_frame(r_i, R_o_i, params["datetime"], t, debug=False)
    b_b = R_b_o @ b_o
    #b_b_measured = adet.measure_sensor(b_b, 0.01*np.pi/180) # add noise to the sensor
    b_b_measured = b_b # For faster simulation

    q_ob_hat = adet.quest_algorithm(b_b_measured, b_o, s_b_measured, s_o)
    q_ob_hat = adet.quaternion_sign_correction(q_ob_hat, params["q_ob_hat_prev"])
    #q_ob_hat = adet.match_quaternion_sign_to_reference(q_ob_hat, q_ob)
    
    params["q_ob_hat_prev"] = q_ob_hat

    orbit_radial_vector = np.array([1,0,0])

    sunwardQuat = att.quaternion_from_vectors(s_o_unit, orbit_radial_vector)
    sunwardQuat = att.quaternion_from_vectors(orbit_radial_vector, s_o_unit)

    q_od = params["q_od"]
    w_ib_b = R_b_o @ w_io_i + w_ob_b
    # End of attitude determination ------------------------------------------
    
    # Attitude control -------------------------------------------------------

    # Desired torque from PD controller
    tau_d_b = ctrl.PD_controller(q_ob, w_ob_b, q_od, params["w_od_d"], k_p=40, k_d=1)
  
    # Attitude control using thrusters
    #tau_a_b, thruster_firings = act.attitude_control_using_thrusters(tau_d_b, max_thrust=0.5, deadzone=0.1)
    #m_dot = np.sum(thruster_firings)/(params["Thruster ISP"]*9.81)
    thruster_firings = np.array([0, 0, 0, 0])
    m_dot = 0

    # Attitude control using reaction wheels
    tau_a_b, w_bw_b_dot, u = act.attitude_control_using_reaction_wheels_in_tethrahedron(tau_d_b, w_ib_b, w_bw_b, J_w=params["J_w"], max_RPM=11000)
    #w_bw_b_dot = np.array([0, 0, 0, 0])
    #u = np.array([0, 0, 0, 0])

    #Desaturation 
    #tau_sat = act.reaction_wheels_desaturation(0, w_ib_b, w_bw_b, params["J_w"])
    #tau_sat = act.attitude_control_using_magnetorquers(tau_sat, b_b)

    # Attitude control using magnetorquers
    #tau_a_b = act.attitude_control_using_magnetorquers(tau_d_b, b_b)
    #print ("tau_sat", tau_sat)
    #print

    #tau_a_b = tau_a_b + tau_sat
    # End of attitude control ------------------------------------------------

    # Attitude dynamics -------------------------------------------------------
    q_ob_b_dot = att.quaternion_kinematics(q_ob, w_ob_b)
    w_ob_b_dot = att.attitude_dynamics(params["J"], q_ob, w_ob_b, w_io_i,  w_io_i_dot, R_o_i, tau_a_b, np.array([0,0,0]))

    # End of attitude dynamics ------------------------------------------------


    # update states -----------------------------------------------------------
    state_dot[1] = q_ob_b_dot[0]
    state_dot[2] = q_ob_b_dot[1]
    state_dot[3] = q_ob_b_dot[2]
    state_dot[4] = q_ob_b_dot[3]
    state_dot[5] = w_ob_b_dot[0]
    state_dot[6] = w_ob_b_dot[1]
    state_dot[7] = w_ob_b_dot[2]
    state_dot[8] = m_dot
    state_dot[9] = w_bw_b_dot[0]
    state_dot[10] = w_bw_b_dot[1]
    state_dot[11] = w_bw_b_dot[2]
    state_dot[12] = w_bw_b_dot[3]
    
    data_entry = {
        "R_o_i": R_o_i, 
        "r_i": r_i,
        "v_i": v_i,
        "a_i": a_i,
        "omega_i_i_o": 0,
        "q_ob": q_ob,
        "w_ob_b": w_ob_b,
        "q_ib": q_ib,
        "q_io": q_io,
        "s_i": s_i,
        "s_o": s_o,
        "b_o": b_o.reshape(3),
        "b_b": b_b.reshape(3),
        "q_ob_hat": q_ob_hat,
        "thruster_firings": thruster_firings,
        "propellant_mass": propelant_mass,
        "w_bw_b": w_bw_b, 
        "wheel_torques": u
    }

    return state_dot, data_entry

def satellite_dynamics_loop_wrapper (t, state, params):
    state_dot, _ = satellite_dynamics_loop(t, state, params)
    return state_dot

# Initial conditions for simulation
t_span = (0, 20) # Simulation timespan
n_steps = 1000   # Number of steps
theta = 0       # Initial true anomaly
q_ob = np.array([0, 1, 0, 0]) # Initial orientation of the satellite (quaternion)
w_ob_b = np.array([0, 0, 0]) # Initial angular velocity of the satellite 
propelant_mass = 0 # Initial mass of the propelant 
w_bw_b = np.array([0, 0, 0, 0])*2*np.pi/60 # Initial angular velocity of the reaction wheels

# Initial state vector
initial_state = np.array([theta, q_ob[0], q_ob[1], q_ob[2], q_ob[3], w_ob_b[0], w_ob_b[1], w_ob_b[2], propelant_mass, w_bw_b[0], w_bw_b[1], w_bw_b[2], w_bw_b[3]])

# Initial orbital parameters
orbital_params = {
    "r_apogee": 6378e3 + 400e3, 
    "r_perigee": 6378e3 + 400e3,
    "t_0": 0,
    "argumet_of_perigee": 0*np.pi/180,
    "RAAN": 0*np.pi/180,
    "inclination": 0*np.pi/180,
    #"inclination": 0,
    "J": np.array([[0.0111, 0, 0], [0, 0.0111, 0], [0, 0, 0.00443]]), #Angular momentum of the satellite 
    "q_od": np.array([1, 0, 0, 0]), # desired orientation of the satellite
    "w_od_d": np.array([0, 0, 0]), # desired angular velocity of the satellite
    "datetime": dt.datetime(2025, 3, 20, 0, 0, 0), 
    "q_ob_hat_prev": q_ob,
    "Thruster ISP": 250, 
    "J_w": 0.0000036 # Angular momentum of the reaction wheels 
}


result = solve_ivp(
    satellite_dynamics_loop_wrapper, 
    t_span, 
    initial_state,
    method="RK45", 
    args=(orbital_params,), 
    t_eval=np.linspace(t_span[0], t_span[1], n_steps),
    rtol=1e-3,  # Relative tolerance
    atol=1e-6
    )

t = result.t
state_vector = result.y

data_log = {
    "R_o_i": [], 
    "r_i": [],
    "v_i": [],
    "a_i": [],
    "omega_i_i_o": [],
    "q_ob": [],
    "w_ob_b": [],
    "q_ib": [],
    "q_io": [],
    "s_i": [],
    "s_o": [],
    "b_o": [],
    "b_b": [],
    "q_ob_hat": [],
    "thruster_firings": [],
    "propellant_mass": [],
    "w_bw_b": [], 
    "tau_a_b": [],
    "wheel_torques": []

}

for i in range(len(t)):
    state = state_vector[:,i]
    _, data_entry = satellite_dynamics_loop(t[i], state, orbital_params)
    for key in data_entry.keys():
        data_log[key].append(data_entry[key])

# plot q_ob TASK 4

plt.figure()

plt.plot(t, data_log["q_ob_hat"], label=["$\eta$", "$\epsilon_x$", "$\epsilon_y$", "$\epsilon_z$"])
plt.xlabel("Time [s]")
plt.ylabel("Quaternion")
plt.title("Quaternion estimate")
plt.legend()
plt.grid()
plt.show()

plt.figure()

plt.plot(t, data_log["q_ob"], label=["$\eta$", "$\epsilon_x$", "$\epsilon_y$", "$\epsilon_z$"])
plt.xlabel("Time [s]")
plt.ylabel("Quaternion")
plt.title("Quaternion truth")
plt.legend()
plt.grid()
plt.show()

plt.figure()

rpm = data_log["w_bw_b"]
rpm = np.array(rpm)*60/(2*np.pi)

plt.plot(t, rpm, label=["$\omega_x$", "$\omega_y$", "$\omega_z$", "$\omega_w$"])
plt.xlabel("Time [s]")
plt.ylabel("RPM")
plt.title("Reaction wheel speeds")


plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(t, data_log["wheel_torques"], label=["$\tau_1$", "$\tau_2$", "$\tau_3$", "$\tau_4"])
plt.xlabel("Time [s]")
plt.ylabel("Nm")
plt.title("Torques")
plt.legend()
plt.grid()

plt.show()



