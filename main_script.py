# Import libraries
import sys
sys.setrecursionlimit(5000)
import numpy as np
import cv2 as cv
from tdmclient import ClientAsync
from Local_Nav import Local_Nav as LN
from Global_Nav import Global_Nav 
import time


# Thymio Parametres----------------------------------------------------------------------------------------------------------------------------------------------
thymio_size     = 120 #size of thymio q
r               = 2 # radius of the wheel (cm)q
L               = 9.5 # distance from wheel to wheel (cm)
v_max           = 20 # cm/s
cruising        = 300 
motor_speed_max = 500
omega_to_motor  = (motor_speed_max*r)/v_max

pixel_to_cm     = 1/12

# PD Controller Parametres---------------------------------------------------------------------------------------------------------------------------------------
Kp             = 0.7   # Proporcional gain
Ki             = 0  # Integral gain
Kd             = 0.05  # Derivative gain
error_sum      = 0     # Integral gain
previous_error = 0     # Derivative error

ts = 0.1  # time step

# Kalman Filter Parametres---------------------------------------------------------------------------------------------------------------------------------------

P0 = Global_Nav.covariance(1000, 1000, 1000, 1000, 1000, 1000)       # state covariance matrix

Q  = Global_Nav.covariance(0.0615, 0.004, 0.0615, 0.004, 0.001, 0.01)  # process noise covariance matrix

R_oscar = Global_Nav.covariance(0.1, 0.1, 0.1, 0.1, 0.01, 0.001)          # measurement covariance matrix

R_cam = Global_Nav.covariance(0.0615, 0.025, 0.0615, 0.025, 0.0615, 0.004)     # measurement covariance matrix

client = ClientAsync()
node   = client.aw(client.wait_for_node())

# Define global_nav object with class attributes
GN = Global_Nav(client, node, cruising, ts, Kp, Ki, Kd, error_sum, previous_error, r, L, omega_to_motor, pixel_to_cm, Q, P0)

# Find shortest Path---------------------------------------------------------------------------------------------------------------------------------------------
path_pix, mid_point_back, img = GN.get_shortest_path(thymio_size)
print(path_pix)
i=0
path = np.zeros((len(path_pix),2))
path_cm = np.zeros((len(path_pix),2))
for point in path_pix:
    path[i,0] = point[1]
    path[i,1] = point[0]
    i+=1

i=0
for point in path_pix:
    path_cm[i,0] = path[i,0]*pixel_to_cm
    path_cm[i,1] = path[i,1]*pixel_to_cm
    i+=1
for i in range(len(path_pix)):
    img = cv.circle(img, (int(path[i,0]), int(path[i,1])), radius=5, color=(0, 255, 0), thickness=-1)
cv.imshow('a',img)
cv.waitKey(0)

# Inicial states-------------------------------------------------------------------------------------------------------------------------------------------------
thymio_pix = np.array([path[0,0], path[0,1]])
thymio_pose = np.array([path_cm[0,0], path_cm[0,1], 0.0])

plot_path = []
plot_path.append(thymio_pix.tolist())

distThymio = np.sqrt((thymio_pix[0]-mid_point_back[0])**2+(thymio_pix[1]-mid_point_back[1])**2)

# Inicial states
X = np.array([      [0.0],     # x_dot0
        [thymio_pose[0]],       # x0
                     [0.0],    # y_dot0
        [thymio_pose[1]],    # y0
                     [0.0],    # phi_dot0
        [thymio_pose[2]]])   # phi0 

vid = cv.VideoCapture(1+cv.CAP_DSHOW)
vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920*0.7)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080*0.7)

pre_kalman = []
pos_kalman = []
pre_kalman.append([thymio_pose[0], thymio_pose[1]])
pos_kalman.append([thymio_pose[0], thymio_pose[1]])
abs_pos = []
sensor_vals = []
sample_count = 0

motor_left = 0
motor_right = 0

next_point = 1

while(True):
    isTrue, img = vid.read()      
    while (np.abs(X[3]-path_cm[-1,1]) > 1.5) or (np.abs(X[1]-path_cm[-1,0]) > 1.5):
        
        # define current goal
        goal_cm = np.array([path_cm[next_point,0],path_cm[next_point,1]])
        goal = np.array([path[next_point,0],path[next_point,1]])
        print(goal)
        while (np.abs(X[3]-goal_cm[1]) > 1.5) or (np.abs(X[1]-goal_cm[0]) > 1.5):
            isTrue, img = vid.read()
            img = img[48:640,158:1110]
            if isTrue:
                start = time.time()
                node = client.aw(client.wait_for_node()) 
                client.aw(node.lock_node())

                #Activate Obstacle avoidance if necessary
                local_nav = LN(client, node, int(motor_left), int(motor_right))
                flag_obs = local_nav.analyse_data()
                if flag_obs == 1:
                    task = local_nav.obstacle_avoidance() #trigger task depending on if flag detected
                else:
                    # Find the error in the orientation--------------------------------------------------------------------------------------------------------------
                    # vector from the mid back point of the thymio to the current goal
                    thymio2goal = goal - mid_point_back 

                    # vector from the mid back point of the thymio to the centre of the thymio
                    orientation_vec = thymio_pix - mid_point_back
                    
                    # Find angle through dot product
                    norm_1 = np.linalg.norm(orientation_vec)
                    norm_2 = np.linalg.norm(thymio2goal)
                    dot_product = np.dot(orientation_vec , thymio2goal)
                    error = np.arccos(dot_product/(norm_1*norm_2))

                    # Find whether the angle is positive or negative
                    d = (thymio_pix[0]-mid_point_back[0])*(goal[1]-mid_point_back[1])-(thymio_pix[1]-mid_point_back[1])*(goal[0]-mid_point_back[0])
                    if d > 0 :
                        error = -error
                    
                    # Control action to make the Thymio follow the reference--------------------------------------------------------------------------------------------
                    vr, vl = GN.PIDcontroller(error)
                    
                    motor_left = vl*omega_to_motor
                    motor_right = vr*omega_to_motor                
                
                    adjust_speed = GN.motors(int(motor_left), int(motor_right))
                    node.send_set_variables(adjust_speed)
                node.flush()    
                
                # Update states---------------------------------------------------------------------------------------------------------------------------------------
                motor_sensor = GN.get_motor_speed()
                sensor_vals.append(local_nav.get_sensor_data())
                speed_sensor = motor_sensor/GN.omega_to_motor      # cm/s
                _,thymio_pose,_,flag,_ = GN.find_thymio(img)
                Y = np.zeros_like(X)
                if flag==1: # if the camera can find the thymio
                    stop = time.time()
                    ts = stop-start 
                    Y[1] = thymio_pose[0]                                          # x
                    Y[3] = thymio_pose[1]                                          # y
                    Y[5] = thymio_pose[2]                                          # phi
                    Y[4] = (r/L)*(speed_sensor[1]-speed_sensor[0])                 # phi_dot
                    Y[0] = (r/2)*(speed_sensor[0]+speed_sensor[1])*np.cos(Y[5])    # x_dot
                    Y[2] = (r/2)*(speed_sensor[0]+speed_sensor[1])*np.sin(Y[5])    # y_dot
                    R = R_cam
                else:
                    stop = time.time()
                    ts = stop-start 
                    Y[4] = (r/L)*(speed_sensor[0]-speed_sensor[1])                # phi_dot
                    Y[5] = X[5] + Y[4]*ts                                         # phi
                    Y[0] = (r/2)*(speed_sensor[0]+speed_sensor[1])*np.cos(Y[5])   # x_dot
                    Y[2] = (r/2)*(speed_sensor[0]+speed_sensor[1])*np.sin(Y[5])   # y_dot
                    Y[1] = X[1] + Y[0]*ts                                         # x
                    Y[3] = X[3] + Y[2]*ts                                         # y
                    R = R_oscar                

                A = np.array([[1, 0, 0, 0, 0, 0],     # xdot                       # state transition matrix
                                [ts, 1, 0, 0, 0, 0],    # x
                                [0, 0, 1, 0, 0, 0],     # ydot
                                [0, 0, ts, 1, 0, 0],    # y
                                [0, 0, 0, 0, 1, 0],     # phidot
                                [0, 0, 0, 0, ts, 1]])   # phi
                pre_kalman.append([Y[1][0], Y[3][0]])
                X = GN.kalman_filter(X, Y, A, R)
                thymio_pix[0] = X[1]/pixel_to_cm
                thymio_pix[1] = X[3]/pixel_to_cm
                mid_point_back[0] = thymio_pix[0]-distThymio*np.cos(X[5]) 
                mid_point_back[1] = thymio_pix[1]-distThymio*np.sin(X[5]) 
                plot_path.append(thymio_pix.tolist())
                pos_kalman.append([X[1][0], X[3][0]])

                if sample_count%12 == 0:
                    abs_pos.append([X[1][0], X[3][0], X[5][0]])
                sample_count += 1

                # Show thymio and goal position -------------------------------------------------------------------------------------------------------------------------
                img = cv.circle(img, (int(thymio_pix[0]), int(thymio_pix[1])), radius=10, color=(0, 0, 255), thickness=-1)
                img = cv.circle(img, (int(goal[0]), int(goal[1])), radius=10, color=(255, 0, 0), thickness=-1)
                img = cv.circle(img, (int(mid_point_back[0]), int(mid_point_back[1])), radius=10, color=(200, 0, 200), thickness=-1)
                for k in range(len(plot_path)):
                    img = cv.circle(img, (int(plot_path[k][0]), int(plot_path[k][1])), radius=5, color=(200, 200, 80), thickness=-1)
                
                cv.imshow("Video", img)
                # the 'q' button is set as the quitting button you may use any desired button of your choice
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Go to next point in path-----------------------------------------------------------------------------------------------------------------------------------------
        next_point += 1
        cv.imshow("Video", img)
        # the 'q' button is set as the quitting button you may use any desired button of your choice
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Stop the thymio when it reaches the end-------------------------------------------------------------------------------------------------------------------------------
    motor_left = 0
    motor_right = 0
    adjust_speed = GN.motors(motor_left, motor_right)
    node.send_set_variables(adjust_speed)
    node.flush()
    img = img[48:640,158:1110]
    img = cv.circle(img, (int(thymio_pix[0]), int(thymio_pix[1])), radius=10, color=(0, 0, 255), thickness=-1)
    img = cv.circle(img, (int(goal[0]), int(goal[1])), radius=10, color=(255, 0, 0), thickness=-1)
    img = cv.circle(img, (int(mid_point_back[0]), int(mid_point_back[1])), radius=10, color=(200, 0, 200), thickness=-1)
    for k in range(len(plot_path)):
        img = cv.circle(img, (int(plot_path[k][0]), int(plot_path[k][1])), radius=5, color=(200, 200, 80), thickness=-1)
    cv.imshow("Video", img)
    # the 'q' button is set as the quitting button you may use any desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()

