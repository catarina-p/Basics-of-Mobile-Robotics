from tdmclient import ClientAsync
import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode
import Polygon as poly
import video as vd
import extremitypathfinder_master.extremitypathfinder as EXTRE


class Global_Nav():

    def __init__(self, client, node, cruising, time_step, Kp, Ki, Kd, error_sum, previous_error, r, L, omega_to_motor, pixel_to_cm, Q, P0):
        self.motor_cruising = cruising
        self.K = np.array([Kp, Ki, Kd])
        self.r = r
        self.L = L
        self.omega_to_motor = omega_to_motor
        self.Q = Q
        self.P_est = P0
        self.client = client
        self.ts = time_step
        self.previous_error = previous_error
        self.error_sum = error_sum
        self.pixel_to_cm = pixel_to_cm
        self.node = node 
        self.sat_speed = 300


    def get_shortest_path(self, Thymio_size):
        #---------------------Get Thymio
        vid = cv.VideoCapture(1 + cv.CAP_DSHOW)
        vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920*0.7)
        vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080*0.7)
        count = 0
        while True:
            isTrue, img = vid.read()
            count += 1
            if isTrue and count>50:
                img = img[48:640,158:1110]
                source,_,qr_loc,flag,back_point = self.find_thymio(img)
                if flag==1:
                    break
        vid.release()  
        source = [source[1], source[0]] 
        boxSource = []
        for i in range(len(qr_loc)):
            boxSource.append([qr_loc[i].x, qr_loc[i].y])
        #cut off the image to just show the enviroment
        maxX,maxY,_ = img.shape
        # ---------------------Draw vitual enviroment 
        boxSource = poly.augment_polygons([boxSource],maxX,maxY,50)
        virtual_image = vd.draw_objects_in_Virtual_Env(img)
        cv.imwrite('virtual_image.jpg',virtual_image)
        #------------------ Erase the source from virtual enviroments
        pts = np.array(boxSource, np.int32)
        pts = pts.reshape((-1,1,2))
        cv.fillPoly(virtual_image, [pts], color=(255,255,255))
        #-----------------Get polygons of virtual enviroment
        sink, poly_list = poly.get_polygons_from_image(virtual_image,True,True)
        #sink = [sink[1], sink[0]] 
        #----------------- Augment Polygons 
        augmented_poly_list = poly.augment_polygons(poly_list,maxX,maxY,Thymio_size)
        #----------------- see polygons out of bound
        new_poly_list = poly.poly_out_of_bound(augmented_poly_list,maxX,maxY, Thymio_size)
        #----------------- Get shortest path
        environment = EXTRE.PolygonEnvironment()

        boundary_coordinates = [(0, 0), (maxX, 0), (maxX, maxY), (0, maxY)]

        environment.store(boundary_coordinates,new_poly_list,validate = False)
        environment.prepare()
        path,_ = environment.find_shortest_path(source,sink)
        
        return path, back_point, img


    def PIDcontroller(self, error):# phi_d, phi):
        """ PID controller """
        self.error_sum += error
        error_dif = error - self.previous_error
        self.previous_error = error

        omega = self.K[0]*error + self.K[1]*self.error_sum - self.K[2]*error_dif

        v = (self.motor_cruising)/(self.omega_to_motor*self.r)

        vr = (2*v+omega*self.L)/(2*self.r) # angular velocity of the right wheel
        vl = (2*v-omega*self.L)/(2*self.r) # angular velocity of the left wheel
        return vr, vl


    def get_motor_speed(self):
        self.client.aw(self.node.wait_for_variables({"motor.left.speed"}))
        left_motor_speed = self.node.v.motor.left.speed
        self.client.aw(self.node.wait_for_variables({"motor.right.speed"}))
        right_motor_speed = self.node.v.motor.right.speed
        return  np.array([left_motor_speed, right_motor_speed])


    def covariance(sigmax, sigmaxdot, sigmay, sigmaydot, sigmatheta, sigmathetadot):
        """creates diagonal covariance matrix of errors"""
        cov_matrix = np.array([[sigmax ** 2, 0, 0, 0, 0 ,0],
                               [0, sigmaxdot ** 2, 0, 0, 0, 0],
                               [0, 0, sigmay ** 2, 0, 0, 0],
                               [0, 0, 0, sigmaydot ** 2, 0, 0],
                               [0, 0, 0, 0, sigmatheta ** 2, 0],
                               [0, 0, 0, 0, 0, sigmathetadot ** 2]])
        return cov_matrix


    def kalman_filter(self, X, Y, A, R):
        """takes the previous state (position and velocity in x and y) and covariance matrices  and uses the current sensor\
        data to update the state and covariance matrices i.e. get the robots current location"""
        # Use the previous state to predict the new state
        X_est_a_priori = np.dot(A, X)

        # Use the previous covariance to predict the new covariance
        P_est_a_priori = np.dot(A, np.dot(self.P_est, A.T))
        P_est_a_priori = P_est_a_priori + self.Q if type(self.Q) != type(None) else P_est_a_priori

        # Define Y and H for the posteriori update
        # Y = np.array([[xdot], [x], [ydot], [y], [phi_dot], [phi]])
        H = np.identity(6)

        # innovation / measurement residual
        i = Y - np.dot(H, X_est_a_priori)
        S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R

        # Calculate the Kalman gain
        K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))

        # Update the state and covariance matrices with the current sensor data
        X_new = X_est_a_priori + np.dot(K, i)
        self.P_est = P_est_a_priori - np.dot(K, np.dot(H, P_est_a_priori))
        return X_new


    def find_thymio(self, img):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
        # create a mask based on the threshold
        thresh = 150
        while thresh>=80:
            binary_mask = img_gray > thresh
            bin_img = np.zeros_like(img)
            bin_img[binary_mask] = img[binary_mask]
            binary_mask = bin_img != 0
            white = np.ones_like(img)*255
            bin_img[binary_mask] = white[binary_mask]
            thresh = thresh - 10
            result = decode(bin_img)
            if len(result)>=1:
                break
        if len(result)>=1:
            qr_loc = result[0][3]
            thymio_xy_pix = np.array([int((qr_loc[0].x+qr_loc[1].x+qr_loc[2].x+qr_loc[3].x)/4), 
                                    int((qr_loc[0].y+qr_loc[1].y+qr_loc[2].y+qr_loc[3].y)/4)]) #in pixels
            thymio_xy = thymio_xy_pix*self.pixel_to_cm # cm

            # Orientation
            if str(result[0][4])=='ZBarOrientation.RIGHT':
                if np.abs(qr_loc[3].y-qr_loc[0].y) <= np.abs(qr_loc[1].y-qr_loc[0].y):
                    mid_point_back = np.array([int((qr_loc[0].x+qr_loc[1].x)/2), int((qr_loc[0].y+qr_loc[1].y)/2)])
                else:
                    mid_point_back = np.array([int((qr_loc[0].x+qr_loc[3].x)/2), int((qr_loc[0].y+qr_loc[3].y)/2)])
            
            elif str(result[0][4])=='ZBarOrientation.DOWN':
                if np.abs(qr_loc[0].y-qr_loc[3].y) > np.abs(qr_loc[1].y-qr_loc[0].y):
                    mid_point_back = np.array([int((qr_loc[2].x+qr_loc[3].x)/2), int((qr_loc[2].y+qr_loc[3].y)/2)])
                else:
                    mid_point_back = np.array([int((qr_loc[0].x+qr_loc[3].x)/2), int((qr_loc[0].y+qr_loc[3].y)/2)])

            elif str(result[0][4])=='ZBarOrientation.LEFT':
                if np.abs(qr_loc[0].y-qr_loc[3].y) > np.abs(qr_loc[1].y-qr_loc[0].y):
                    mid_point_back = np.array([int((qr_loc[2].x+qr_loc[1].x)/2), int((qr_loc[2].y+qr_loc[1].y)/2)])
                else:
                    mid_point_back = np.array([int((qr_loc[2].x+qr_loc[3].x)/2), int((qr_loc[2].y+qr_loc[3].y)/2)])      
            
            elif str(result[0][4])=='ZBarOrientation.UP':
                if np.abs(qr_loc[3].y-qr_loc[0].y) >= np.abs(qr_loc[1].y-qr_loc[0].y):
                    mid_point_back = np.array([int((qr_loc[0].x+qr_loc[1].x)/2), int((qr_loc[0].y+qr_loc[1].y)/2)])
                else:
                    mid_point_back = np.array([int((qr_loc[2].x+qr_loc[1].x)/2), int((qr_loc[2].y+qr_loc[1].y)/2)])

            thymio_vec = thymio_xy_pix - mid_point_back
            phi = np.arctan2(thymio_vec[1],thymio_vec[0])
            thymio_pose = np.array([thymio_xy[0], thymio_xy[1], phi])
            flag = 1
        else:
            thymio_xy_pix = np.zeros((1,2))
            thymio_pose = np.zeros((1,3))
            qr_loc = np.zeros((1,3))
            flag = 0
            mid_point_back = np.zeros((1,2))
        return thymio_xy_pix, thymio_pose, qr_loc, flag, mid_point_back


    def motors(self, motor_speed_left, motor_speed_right):
        if motor_speed_left > self.sat_speed:
            motor_speed_left = self.sat_speed

        if motor_speed_right > self.sat_speed:
            motor_speed_right = self.sat_speed

        if motor_speed_left < -self.sat_speed:
            motor_speed_left = - self.sat_speed

        if motor_speed_right < -self.sat_speed:
            motor_speed_right = -self.sat_speed
        
        return {
            "motor.left.target": [motor_speed_left],
            "motor.right.target": [motor_speed_right],
        }