import numpy as np
import attitude_dynamics as att
import orbital_mechanics as orb

def calculate_line_sphere_intersection(P1, P2, P3, radius=6370e3): 
    """
        Based on C code by Paul Bourke:
        https://paulbourke.net/geometry/circlesphere/raysphere.c
    """

    planet = []

    a = (P2[0] - P1[0])**2 + (P2[1] - P1[1])**2 + (P2[2] - P1[2])**2
    b = 2*((P2[0]- P1[0])*(P1[0] - P3[0]) + (P2[1]- P1[1])*(P1[1] - P3[1]) + (P2[2]- P1[2])*(P1[2] - P3[2]))
    c = P3[0]**2 + P3[1]**2 + P3[2]**2 
    c += P1[0]**2 + P1[1]**2 + P1[2]**2
    c -= 2*(P3[0]*P1[0] + P3[1]*P1[1] + P3[2]*P1[2])
    c -= radius**2
    print(a, b, c)
    bb4ac = b*b - 4*a*c

    if (abs(a) < 1e-9 or bb4ac < 0):
        does_intersect = False
        point_of_intersection = [0, 0, 0]
    else:
        does_intersect = True
        mu1 = (-b + np.sqrt(bb4ac))/(2*a)
        mu2 = (-b - np.sqrt(bb4ac))/(2*a)

        # seems mu2 is the closest point of intersection, but we still check
        point_of_intersection1 = P1 + mu1*(P2 - P1)
        point_of_intersection2 = P1 + mu2*(P2 - P1)

        los_vector1 = point_of_intersection1 - P1
        los_vector2 = point_of_intersection2 - P1

        if (np.linalg.norm(los_vector1) < np.linalg.norm(los_vector2)):
            point_of_intersection = point_of_intersection1
        else:
            point_of_intersection = point_of_intersection2

    return point_of_intersection, does_intersect

def calculate_raycasting_points(R_i_b, r_i, raycasting_length = 10000e3, field_of_view_half = 30/2*np.pi/180, number_of_raycasting_points = 10): 

    raycasting_points_i = []

    radius = raycasting_length*np.tan(field_of_view_half)

    for theta in np.linspace(0, 2*np.pi, number_of_raycasting_points, endpoint=False):
        points_i = R_i_b @ np.array([raycasting_length, radius*np.cos(theta), radius*np.sin(theta)])
        raycasting_points_i.append(r_i + points_i)

    raycasting_points_i = np.array(raycasting_points_i)

    return raycasting_points_i

def calculate_intersecting_points_in_intertial_frame(r_i, R_i_b, raycasting_length, field_of_view_half_deg, number_of_raycasting_points):

    fov_half_rad = field_of_view_half_deg*np.pi/180
    raycasting_points_i = calculate_raycasting_points(R_i_b, r_i, raycasting_length, fov_half_rad, number_of_raycasting_points)

    P1 = r_i
    P3 = np.array([0, 0, 0])

    intersection_points_i = np.zeros_like(raycasting_points_i)

    for i in range(0, len(raycasting_points_i[:,0])):
        P2 = raycasting_points_i[i]
        closest_point, does_intersect = calculate_line_sphere_intersection(P1, P2, P3,radius=6378e3)
        if does_intersect == True:
            intersection_points_i[i] = closest_point
        else:
            intersection_points_i[i] = raycasting_points_i[i]
    
    # create unit vector pointing in the direction of the cone
    cone_unit_vector = R_i_b @ np.array([1, 0, 0])
    # create unit vectors pointing from the satellite to the intersection points
    point_unit_vectors = np.zeros_like(intersection_points_i)
    for i in range(0, len(intersection_points_i[:,0])):
        point_unit_vectors[i] = (intersection_points_i[i] - r_i)/np.linalg.norm(intersection_points_i[i] - r_i)

    # check if the intersection points are behind the cone, if so reset to raycasting point
    for i in range(0, len(point_unit_vectors[:,0])):
        dot_product = np.dot(cone_unit_vector, point_unit_vectors[i])
        if dot_product < 0:
            intersection_points_i[i] = raycasting_points_i[i]
    

    return intersection_points_i

def measure_sensor (v_b, sensor_accuracy):

    angle_standard_deviation = sensor_accuracy/2

    angle_noise = np.random.normal(0,angle_standard_deviation)

    random_axis_of_rotation = np.random.rand(3)

    k_noise = random_axis_of_rotation / np.linalg.norm(random_axis_of_rotation)

    q_mb = np.array([np.cos(angle_noise/2), k_noise[0]*np.sin(angle_noise/2), k_noise[1]*np.sin(angle_noise/2), k_noise[2]*np.sin(angle_noise/2)])

    R_m_b = att.calculate_rotation_matrix_from_quaternion(q_mb)

    v_b_measurement = R_m_b @ v_b

    return v_b_measurement 