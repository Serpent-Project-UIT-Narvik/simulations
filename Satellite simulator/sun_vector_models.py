import numpy as np
import datetime as dt


def calculate_rotation_matrix_of_earths_tilt(tilt):

    R = np.array([[1, 0, 0],
                  [0, np.cos(tilt), -np.sin(tilt)],
                  [0, np.sin(tilt), np.cos(tilt)]])
    
    return R

def calculate_rotation_matrix_of_earths_orbit(time):

    omega = 2*np.pi/(24*60*60*365)

    R = np.array([[np.cos(omega*time), -np.sin(omega*time), 0],
                    [np.sin(omega*time), np.cos(omega*time), 0],
                    [0, 0, 1]])
    
    return R
    

def simple_sun_vector_model(time):
     
    S = np.array([1, 0, 0]).T
    
    R_i_s = calculate_rotation_matrix_of_earths_tilt(23.5*np.pi/180) @ calculate_rotation_matrix_of_earths_orbit(time)

    #print("R_i_s", R_i_s)

    S_i = R_i_s @ S

    #print("S_i", S_i)

    return S_i

def calculate_julian_date(simulation_time, initial_datetine):

    timedate = initial_datetine + dt.timedelta(seconds = simulation_time)

    JD = 367*timedate.year - np.floor((7*(timedate.year + np.floor((timedate.month + 9)/12)))/4) 
    JD += np.floor((275*timedate.month)/9) + timedate.day + 1721013.5 + (timedate.hour + (timedate.minute + timedate.second/60)/60)/24

    return JD

def calculate_advanced_sun_model(simulation_time, initial_datetime):
   
    JD = calculate_julian_date(simulation_time, initial_datetime)

    T_ut1 = (JD - 2451545)/36525

    T_tdb = T_ut1 
    
    M = (357.5277233 + 35999.05034 * T_tdb) * np.pi/180
   
    lambda_M = 280.460 + 36000.771 * T_ut1

    lamda_eliptic = (lambda_M + 1.914666471 * np.sin(M) + 0.019994643 * np.sin(2*M)) * np.pi/180
    
    radius = 1.000140612 - 0.016708617 * np.cos(M) - 0.019994643 * np.sin(2*M)
    
    epsilon = (23.439291 - 0.0130042 * T_tdb) * np.pi/180
    
    s_i = radius * np.array([np.cos(lamda_eliptic), np.cos(epsilon)*np.sin(lamda_eliptic), np.sin(epsilon)*np.sin(lamda_eliptic)]).T

    return s_i 




    