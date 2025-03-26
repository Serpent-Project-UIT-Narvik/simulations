import vtk
import orbital_mechanics as orb
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
import attitude_dynamics as att 
import sensors as sens
import sun_vector_models as sun
import time
import datetime as dt

def create_referance_frame(plotter, labels, scale = 1):
    x_arrow = pv.Arrow(start=(0,0,0),
                            direction=(1,0,0),
                            tip_length=0.25,
                            tip_radius=0.1,
                            tip_resolution=20,
                            shaft_radius=0.05,
                            shaft_resolution=20,
                            scale=scale,
                            )
        
    y_arrow = pv.Arrow(start=(0,0,0),
                        direction=(0,1,0),
                        tip_length=0.25,
                        tip_radius=0.1,
                        tip_resolution=20,
                        shaft_radius=0.05,
                        shaft_resolution=20,
                        scale=scale,
                        )
    
    z_arrow = pv.Arrow(start=(0,0,0),
                        direction=(0,0,1),
                        tip_length=0.25,
                        tip_radius=0.1,
                        tip_resolution=20,
                        shaft_radius=0.05,
                        shaft_resolution=20,
                        scale=scale,
                        )
    referance_frame_mesh = {"scale": scale, 
                            "x": plotter.add_mesh(x_arrow, color='red'),
                            "y": plotter.add_mesh(y_arrow, color='green'),
                            "z": plotter.add_mesh(z_arrow, color='blue'),
                            "x_label": pv.Label(text=labels[0], position = np.array([1, 0, 0])*scale, size=20),
                            "y_label": pv.Label(text=labels[1], position = np.array([0, 1, 0])*scale, size=20),
                            "z_label": pv.Label(text=labels[2], position = np.array([0, 0, 1])*scale, size=20)
                            }
    plotter.add_actor(referance_frame_mesh["x_label"])
    plotter.add_actor(referance_frame_mesh["y_label"])
    plotter.add_actor(referance_frame_mesh["z_label"])

    return referance_frame_mesh


def create_satellite(plotter, size = 0.5): 
    satellite_body_b = pv.Box(bounds= (-size, size, -size, size, -size, size))

    scriptdir = os.path.dirname(__file__)
    satellite_texture = pv.read_texture(os.path.join(scriptdir, "satellite_texture.png"))

    u = np.array([0, 1, 1, 0]* 6)
    v = np.array([0, 0, 1, 1]* 6)
    texture_coordinates = np.c_[u, v]
    satellite_body_b.texture_map_to_plane(inplace=True)

    panel_width = size*2
    panel_height = size/2
    panel_thickness = size/16

    solar_panel_b = pv.Box(bounds= (-size+size/100, -size+panel_thickness/2, -size - panel_width, size + panel_width, -panel_height, panel_height))
    solar_panel_b . texture_map_to_plane (origin =(- size - panel_thickness /2, 0, 0), point_u =(- size - panel_thickness /2, panel_width /2, 0), point_v =(-size - panel_thickness /2, 0, panel_height /2) ,inplace =True)
    solar_panel_texture = pv.read_texture(os.path.join(scriptdir, "high_quality_solar_panel_texture.png"))

    scientific_instrument_b = pv.Cone(center=(size,0,0), direction=(-1,0,0),height=0.5*size, radius=0.5*size,resolution=50)
    scientific_instrument_b.texture_map_to_sphere(inplace=True)
    scientific_instrument_texture = pv.read_texture(os.path.join(scriptdir, "camera_texture.png"))

    satellite_mesh = {"body": plotter.add_mesh(satellite_body_b, color='white', texture=satellite_texture),
                    "solar_panel": plotter.add_mesh(solar_panel_b, color='blue', texture=solar_panel_texture),
                    "scientific_instrument": plotter.add_mesh(scientific_instrument_b, color='red', texture = scientific_instrument_texture)
                    }
    
    return satellite_mesh

def update_satellite_pose(satellite_mesh, r_i, theta_ib):

    satellite_mesh["body"].SetPosition(r_i)
    satellite_mesh["solar_panel"].SetPosition(r_i)
    satellite_mesh["scientific_instrument"].SetPosition(r_i)

    satellite_mesh["body"].SetOrientation(theta_ib)
    satellite_mesh["solar_panel"].SetOrientation(theta_ib)
    satellite_mesh["scientific_instrument"].SetOrientation(theta_ib)

def create_earth(plotter, radius = 1):
    earth = pv.examples.planets.load_earth(radius=radius)
    earth_texture = pv.examples.load_globe_texture()
    earth_mesh = plotter.add_mesh(earth, texture=earth_texture, smooth_shading=True)

    return earth_mesh

def update_earth_orientation(earth_mesh, time):
    earth_angular_velocity = 2*np.pi/(24*60*60)*180/np.pi
    theta = np.array([0, 0, earth_angular_velocity*time])
    earth_mesh.SetOrientation(theta)

def update_referance_frame_pose(referance_frame, r_i, theta_ib):
    scale = 1 * referance_frame["scale"]
    R_i_b = pyvista_rotation_matrix_from_euler(theta_ib)

    for label in referance_frame:
        if label in ["x", "y", "z"]:
            referance_frame[label].SetPosition(r_i)
            referance_frame[label].SetOrientation(theta_ib)
        elif label == "x_label":
            referance_frame[label].position = r_i + R_i_b @ np.array([scale, 0, 0])
        elif label == "y_label":
            referance_frame[label].position = r_i + R_i_b @ np.array([0, scale, 0])
        elif label == "z_label":
            referance_frame[label].position = r_i + R_i_b @ np.array([0, 0, scale])

def update_ecef_frame_orientation(referance_frame, t):
    earth_angular_velocity = 2*np.pi/(24*60*60)*180/np.pi

    theta = np.array([0, 0, earth_angular_velocity*t])
    
    update_referance_frame_pose(referance_frame, np.array([0, 0, 0]), theta)


def rotation_x(angle): 
    return np.array([[1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]])

def rotation_y(angle): 
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]])

def rotation_z(angle): 
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])

def pyvista_rotation_matrix_from_euler(orientation_euler):

    phi = orientation_euler[0]*np.pi/180
    theta = orientation_euler[1]*np.pi/180
    psi = orientation_euler[2]*np.pi/180

    matrix = rotation_z(psi) @ rotation_x(phi) @ rotation_y(theta)
    return matrix
    
def visualise_scene(plotter):

    earth_radius = 6371*10**3
    earth = create_earth(plotter, earth_radius)
    ECEF = create_referance_frame(plotter, [r"$X_e$", r"$Y_e$", r"$Z_e$"], earth_radius*1.5)

    ECI = create_referance_frame(plotter, [r"$X_i$ ", r"$Y_i$", r"$Z_i$"], earth_radius*2)

    satellite = create_satellite(plotter, earth_radius*0.1)
    satellite_body_frame = create_referance_frame(plotter,[r"$x_{b}$", r"$y_{b}$", r"$z_{b}$"], earth_radius*0.5)

    satellite_orientation = np.array([90, 45, 0])

    gif_path = os.path.join("LaTeX","Week 1", "Figures", "Day1 Satellite Animations.gif")
    plotter.open_gif(gif_path)

    n_frames = 100

    simulation_time = 24*60*60
    time_step = 1 # 1 second
    frame_interval = simulation_time // n_frames


    orbit_angle = 0
    orbit_radious = earth_radius*3

    # for circular orbit these will be constant
    V = orb.calculate_circular_orbital_speed(orbit_radious)
    varphi_dot = V / orbit_radious

    for t in np.arange(0, simulation_time, time_step):
        # extract radius vector
        r_vec = orb.calculate_satellite_position_in_circular_orbit(orbit_angle, orbit_radious)

        # apply euler forward integration to orbit angle
        orbit_angle += varphi_dot*time_step

        # update gif on each frame interval 
        if t % frame_interval == 0:

            update_satellite_pose(satellite, r_vec, satellite_orientation)
            update_referance_frame_pose(satellite_body_frame, r_vec, satellite_orientation)

            update_earth_orientation(earth, t)
            update_ecef_frame_orientation(ECEF, t)

            plotter.write_frame()
            

def create_sensor_cone(plotter, r_i, R_i_b, raycasting_length, field_of_view_half_deg, number_of_raycasting_points):
    intersection_points_i = sens.calculate_intersecting_points_in_intertial_frame(r_i, R_i_b, raycasting_length, field_of_view_half_deg, number_of_raycasting_points)

    triangles = []
    #print(len(intersection_points_i))
    for i in range(0, len(intersection_points_i[:,0])-1):
        print(i)
        triangles.append([r_i, intersection_points_i[i+1], intersection_points_i[i]])

    triangles.append([r_i, intersection_points_i[0], intersection_points_i[-1]])

    cone_mesh = pv.PolyData()
    for tri in triangles:
        cone_mesh += pv.Triangle(tri)

    plotter.add_mesh(cone_mesh, color='green', opacity=0.25)

    return cone_mesh

def update_sensor_cone_points(plotter, line_actor, sensor_cone_mesh, r_i, R_i_b, raycasting_length, field_of_view_half_deg, number_of_raycasting_points):

    intersection_points_i = sens.calculate_intersecting_points_in_intertial_frame(r_i, R_i_b, raycasting_length, field_of_view_half_deg, number_of_raycasting_points)

    new_points = np.vstack((r_i, intersection_points_i))

    sensor_cone_mesh.points = new_points

    circle_lines = []
    for i in range(1, len(intersection_points_i) -1 ):
        circle_lines.append([intersection_points_i[i], intersection_points_i[i+1]])
    
    circle_lines.append([intersection_points_i[-1], intersection_points_i[0]])

    line_points = np.array(circle_lines).reshape(-1, 3)

    if line_actor is not None:
        plotter.remove_actor(line_actor)

    line_actor = plotter.add_lines(line_points, color='green', width=2)

    return line_actor

def create_sun(plotter):
    mesh = pv.examples.planets.load_sun(radius=10000e3)
    texture = pv.examples.planets.download_sun_surface(texture=True)
    image_path = pv.examples.planets.download_stars_sky_background()
    
    sun_mesh = plotter.add_mesh(mesh, texture=texture)
    #plotter.add_background_image(image_path)

    return sun_mesh

def update_sun_pose(sun_mesh, s_i):
    AU = 149597870.7e3
    AU = 1000000e3
    sun_mesh.SetPosition(s_i*AU)

def animate_eliptical_orbit(t, data_log, date = dt.datetime(2000, 1, 1, 12, 0, 0)): 
    plotter = pv.Plotter(off_screen=False)

    ECEF_ECI_syncdate = dt.datetime(2000, 1, 1, 12, 0, 0)

    #star = create_sun(plotter)

    earth_radius = 6371*10**3
    earth = create_earth(plotter, earth_radius)
    ECEF = create_referance_frame(plotter, [r"$X_e$", r"$Y_e$", r"$Z_e$"], earth_radius*1.5)

    ECI = create_referance_frame(plotter, [r"$X_i$ ", r"$Y_i$", r"$Z_i$"], earth_radius*2)

    satellite = create_satellite(plotter, earth_radius*0.1)
    satellite_body_frame = create_referance_frame(plotter,[r"$x_{b}$", r"$y_{b}$", r"$z_{b}$"], earth_radius*1)

    orbit_frame = create_referance_frame(plotter, [r"$x_{o}$", r"$y_{o}$", r"$z_{o}$"], earth_radius*1)
    
    satellite_orientation = att.calculate_euler_angles_from_quaternion(data_log["q_ib"][0])
    orbit_orientation = att.calculate_euler_angles_from_quaternion(data_log["q_io"][0])

    gif_path = os.path.join("LaTeX","Week 1", "Figures", "Day6-4 Satellite Animations.gif")
    plotter.open_gif(gif_path)

    n_frames = 1000
    frame_interval = len(t) // n_frames

    r_i_array = np.array(data_log["r_i"])

    raycasting_length = 10000e3
    field_of_view_half_deg = 15/2
    number_of_raycasting_points = 100
    R_i_b = pyvista_rotation_matrix_from_euler(satellite_orientation)
    sensor_cone_mesh = create_sensor_cone(plotter, r_i_array[0], R_i_b, raycasting_length, field_of_view_half_deg, number_of_raycasting_points)

    line_actor = None

    line_actor = update_sensor_cone_points(plotter, line_actor, sensor_cone_mesh, r_i_array[0], R_i_b, raycasting_length, field_of_view_half_deg, number_of_raycasting_points)
    
    advanced_sun_vector_arrow = pv.Arrow(start=(0,0,0),
        direction=sun.calculate_advanced_sun_model(t[0], date),
            tip_length=0.25,
            tip_radius=0.1,
            tip_resolution=20,
            shaft_radius=0.05,
            shaft_resolution=20,
            scale=earth_radius*3,
            )
    
    advanced_sun_vector_arrow_orbit = pv.Arrow(start=r_i_array[0],
        direction=sun.calculate_advanced_sun_model(t[0], date),
            tip_length=0.25,
            tip_radius=0.1,
            tip_resolution=20,
            shaft_radius=0.05,
            shaft_resolution=20,
            scale=earth_radius*3,
            )

    advanced_sun_vector_mesh = plotter.add_mesh(advanced_sun_vector_arrow, color='orange')
    advanced_sun_vector_mesh_orbit = plotter.add_mesh(advanced_sun_vector_arrow_orbit, color='orange')

    for i in range(len(t)): 
        
        if i % frame_interval == 0:
            
            ECEFtime = (date + dt.timedelta(seconds=t[i]) - ECEF_ECI_syncdate).total_seconds()

            satellite_orientation = att.calculate_euler_angles_from_quaternion(data_log["q_ib"][i])
            orbit_orientation = att.calculate_euler_angles_from_quaternion(data_log["q_io"][i])
            #print(satellite_orientation)

            #update_sun_pose(star, sun.calculate_advanced_sun_model(t[i], date))

            update_satellite_pose(satellite, r_i_array[i], satellite_orientation)
            update_referance_frame_pose(satellite_body_frame, r_i_array[i], satellite_orientation)

            update_earth_orientation(earth, ECEFtime)
            update_ecef_frame_orientation(ECEF, ECEFtime)

            update_referance_frame_pose(orbit_frame, r_i_array[i], orbit_orientation)

            R_i_b = pyvista_rotation_matrix_from_euler(satellite_orientation)

            line_actor = update_sensor_cone_points(plotter, line_actor, sensor_cone_mesh, r_i_array[i], R_i_b, raycasting_length, field_of_view_half_deg, number_of_raycasting_points)

            plotter.remove_actor(advanced_sun_vector_mesh)
            plotter.remove_actor(advanced_sun_vector_mesh_orbit)

            advanced_sun_vector_arrow = pv.Arrow(start=(0,0,0),
                direction=sun.calculate_advanced_sun_model(t[i], date),
                    tip_length=0.25,
                    tip_radius=0.1,
                    tip_resolution=20,
                    shaft_radius=0.05,
                    shaft_resolution=20,
                    scale=earth_radius*3,
                    )
            
            advanced_sun_vector_arrow_orbit = pv.Arrow(start=r_i_array[i],
                direction=sun.calculate_advanced_sun_model(t[i], date),
                    tip_length=0.25,
                    tip_radius=0.1,
                    tip_resolution=20,
                    shaft_radius=0.05,
                    shaft_resolution=20,
                    scale=earth_radius*1.5,
                    )
            
            advanced_sun_vector_mesh = plotter.add_mesh(advanced_sun_vector_arrow, color='orange')
            advanced_sun_vector_mesh_orbit = plotter.add_mesh(advanced_sun_vector_arrow_orbit, color='orange')

            plotter.write_frame()

    plotter.close()    


def animate_simple_sun_vector(t):
    plotter = pv.Plotter(off_screen=False)

    earth_radius = 6371*10**3
    earth = create_earth(plotter, earth_radius)
    ECEF = create_referance_frame(plotter, [r"$X_e$", r"$Y_e$", r"$Z_e$"], earth_radius*1.5)

    ECI = create_referance_frame(plotter, [r"$X_i$ ", r"$Y_i$", r"$Z_i$"], earth_radius*2)

    sun_vector_arrow = pv.Arrow(start=(0,0,0),
                            direction=(1,0,0),
                            tip_length=0.25,
                            tip_radius=0.1,
                            tip_resolution=20,
                            shaft_radius=0.05,
                            shaft_resolution=20,
                            scale=earth_radius*3,
                            )
    
    sun_vector_mesh = plotter.add_mesh(sun_vector_arrow, color='yellow')


    gif_path = os.path.join("LaTeX","Week 1", "Figures", "Day6 sunvector animations.gif")
    plotter.open_gif(gif_path)

    n_frames = 100
    frame_interval = len(t) // n_frames

    text = plotter.add_text("", position='upper_edge', font_size=30, color='black')

    for i in range(len(t)): 
        
        if i % frame_interval == 0:
            
            days = t[i] // (24*60*60)
            hours = (t[i] - days*24*60*60) // 3600
            minutes = (t[i] - days*24*60*60 - hours*3600) // 60
            seconds = t[i] - days*24*60*60 - hours*3600 - minutes*60

            text.SetText(2, f"Day: {days:03}")


            update_earth_orientation(earth, t[i])
            update_ecef_frame_orientation(ECEF, t[i])

            # remove old sun vector
            plotter.remove_actor(sun_vector_mesh)

            sun_vector_arrow = pv.Arrow(start=(0,0,0),
                            direction=sun.simple_sun_vector_model(t[i]),
                            tip_length=0.25,
                            tip_radius=0.1,
                            tip_resolution=20,
                            shaft_radius=0.05,
                            shaft_resolution=20,
                            scale=earth_radius*3,
                            )

            sun_vector_mesh = plotter.add_mesh(sun_vector_arrow, color='yellow')    
            
            plotter.write_frame()

    plotter.close() 



def animate_sun_vector_comparison(t, date):
    plotter = pv.Plotter(off_screen=False)

    earth_radius = 6371*10**3
    earth = create_earth(plotter, earth_radius)
    ECEF = create_referance_frame(plotter, [r"$X_e$", r"$Y_e$", r"$Z_e$"], earth_radius*1.5)

    ECI = create_referance_frame(plotter, [r"$X_i$ ", r"$Y_i$", r"$Z_i$"], earth_radius*2)

    sun_vector_arrow = pv.Arrow(start=(0,0,0),
                            direction=(1,0,0),
                            tip_length=0.25,
                            tip_radius=0.1,
                            tip_resolution=20,
                            shaft_radius=0.05,
                            shaft_resolution=20,
                            scale=earth_radius*3,
                            )
    
    
    simple_sun_vector_mesh = plotter.add_mesh(sun_vector_arrow, color='yellow')

    advanced_sun_vector_mesh = plotter.add_mesh(sun_vector_arrow, color='orange')

    gif_path = os.path.join("LaTeX","Week 1", "Figures", "Day6-1 sunvector animations.gif")
    plotter.open_gif(gif_path)

    n_frames = 100
    frame_interval = len(t) // n_frames

    text = plotter.add_text("", position='upper_edge', font_size=30, color='black')

    for i in range(len(t)): 
        
        if i % frame_interval == 0:
            
            days = t[i] // (24*60*60)
            hours = (t[i] - days*24*60*60) // 3600
            minutes = (t[i] - days*24*60*60 - hours*3600) // 60
            seconds = t[i] - days*24*60*60 - hours*3600 - minutes*60

            text.SetText(2, f"Day: {days:03}")

            update_earth_orientation(earth, t[i])
            update_ecef_frame_orientation(ECEF, t[i])

            # remove old sun vector
            plotter.remove_actor(simple_sun_vector_mesh)
            plotter.remove_actor(advanced_sun_vector_mesh)

            sun_vector_arrow = pv.Arrow(start=(0,0,0),
                            direction=sun.simple_sun_vector_model(t[i]),
                            tip_length=0.25,
                            tip_radius=0.1,
                            tip_resolution=20,
                            shaft_radius=0.05,
                            shaft_resolution=20,
                            scale=earth_radius*3,
                            )
            
            advanced_vector_arrow = pv.Arrow(start=(0,0,0),
                            direction=sun.calculate_advanced_sun_model(t[i], date),
                            tip_length=0.25,
                            tip_radius=0.1,
                            tip_resolution=20,
                            shaft_radius=0.05,
                            shaft_resolution=20,
                            scale=earth_radius*3,
                            )
            

            simple_sun_vector_mesh = plotter.add_mesh(sun_vector_arrow, color='yellow')
            advanced_sun_vector_mesh = plotter.add_mesh(advanced_vector_arrow, color='orange') 
        

            plotter.write_frame()

            time.sleep(0.1)

    plotter.close() 