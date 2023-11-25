import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Local_Nav:

    def __init__(self,client,node, motor_speed_left,motor_speed_right):
        self.motor_speed_left = motor_speed_left
        self.motor_speed_right = motor_speed_right
        self.client = client
        self.node = node
        self.sat_speed = 500
        self.lower_threshold = 2800 #lower threshold for proximity sensor values
        self.middle_threshold = 3500
        self.upper_threshold = 4000

        #The measures were obtained empirically...
        self.sensor_distances = np.array([15, 9.6, 9, 8.5, 7.5, 7, 6, 5, 4, 3, 2.5, 2, 1.5, 0.5, 0.2, 0])
        self.sensor_measurements = np.array(
            [0, 1186, 1420, 1510, 1715, 1830, 2069, 2250, 2468, 2850, 3155, 3468, 3940, 4355, 4432, 5000])
        self.center_offset = np.array([5.5, 5.5])
        self.sensor_pos_from_center = np.array(
            [[0.9, 9.4], [3.1, 10.5], [5.5, 11.0], [8.0, 10.4], [10.2, 9.3], [2.5, 0], [8.5, 0]]) - self.center_offset
        self.sensor_angles = np.array([120, 105, 90, 75, 60, -90, -90]) * math.pi / 180
        self.thymio_coords = np.array([[0, 0], [11, 0], [11, 8.5], [10.2, 9.3],
                                  [8, 10.4], [5.5, 11], [3.1, 10.5],
                                  [0.9, 9.4], [0, 8.5], [0, 0]]) - self.center_offset

#######################################################################################################################
########################################   Local Navigation Section  ##################################################
#######################################################################################################################

    def get_sensor_data(self):
        self.client.aw(self.node.wait_for_variables({"prox.horizontal"}))
        prox_sens = self.node.v.prox.horizontal
        return  list(prox_sens)

    def analyse_data(self):
        prox_sens_data = self.get_sensor_data()
        check = [i for i in prox_sens_data if i >= self.lower_threshold]
        if any(check):
            flag = 1
            return flag
        else :
            flag = 0
            return flag

    def obstacle_avoidance(self):
        ''' If a flag is detected, then obstacle avoidance will be activated. The flag returns 1 if any of the sensors
        measured something higher that the self.lower_threshold value. The latter was iteratively tuned to adjust to the
        constraining environment conditions, and could be made smaller if not in a confined space (with white walls...)
        for better results.'''

        flag = self.analyse_data()
        obstSpeedGain = [6, 4, 2]
        prox_sens_data = self.get_sensor_data()
        extermity_sens = [prox_sens_data[0], prox_sens_data[4]]
        mid_extermity_sens = [prox_sens_data[1], prox_sens_data[3]]
        middle_sensor = prox_sens_data[2]
        gain_low = [(mid_extermity_sens[0] + extermity_sens[0]) * obstSpeedGain[2] // 200,
                    (mid_extermity_sens[1] + extermity_sens[1]) * obstSpeedGain[2] // 200]
        gain_mid = [(mid_extermity_sens[0] + extermity_sens[0]) * obstSpeedGain[1] // 200,
                (mid_extermity_sens[1] + extermity_sens[1]) * obstSpeedGain[1] // 200]
        gain_high = [(mid_extermity_sens[0] + extermity_sens[0]) * obstSpeedGain[0] // 200,
                (mid_extermity_sens[1] + extermity_sens[1]) * obstSpeedGain[0] // 200]
        if flag == 1:
            if extermity_sens[0] or mid_extermity_sens[0] >= self.lower_threshold:
                adjust_speed = self.motors(self.motor_speed_left + gain_low[0],
                                           -self.motor_speed_right - gain_low[1])
                self.node.send_set_variables(adjust_speed)

            elif extermity_sens[1] or mid_extermity_sens[1] >= self.lower_threshold:
                adjust_speed = self.motors(-self.motor_speed_left - gain_low[0],
                                           +self.motor_speed_right + gain_low[1])
                self.node.send_set_variables(adjust_speed)

            elif self.lower_threshold <= middle_sensor <= self.middle_threshold:
                if mid_extermity_sens[0] <= mid_extermity_sens[1]:
                    adjust_speed = self.motors(-self.motor_speed_left - gain_mid[0],
                                               self.motor_speed_right + gain_mid[1])
                else:
                    adjust_speed = self.motors(self.motor_speed_left + gain_mid[0],
                                                -self.motor_speed_right - gain_mid[1])
                self.node.send_set_variables(adjust_speed)

            elif middle_sensor >= self.lower_threshold and mid_extermity_sens[0] >= self.lower_threshold:
                    adjust_speed = self.motors(-self.motor_speed_left - gain_mid[0],
                                               -self.motor_speed_right - gain_mid[1])
                    if mid_extermity_sens[0] >= self.upper_threshold :
                        adjust_speed = self.motors(-self.motor_speed_left - gain_high[0],
                                                   -self.motor_speed_right - gain_high[1])
                    self.node.send_set_variables(adjust_speed)

            elif middle_sensor >= self.lower_threshold and mid_extermity_sens[1] >= self.lower_threshold:
                    adjust_speed = self.motors(-self.motor_speed_left - gain_mid[0],
                                               -self.motor_speed_right - gain_mid[1])
                    if mid_extermity_sens[0] >= self.upper_threshold :
                        adjust_speed = self.motors(-self.motor_speed_left - gain_high[0],
                                                   - self.motor_speed_right -  gain_high[1])
                    self.node.send_set_variables(adjust_speed)

    def motors(self, motor_speed_left, motor_speed_right):
        if motor_speed_left >= 500:
            return {
                "motor.left.target": [self.sat_speed],
                "motor.right.target": [motor_speed_right],
            }
        elif motor_speed_right >= 500:
            return {
                "motor.left.target": [motor_speed_left],
                "motor.right.target": [self.sat_speed],
            }
        elif motor_speed_left <= -500:
            return {
                "motor.left.target": [-self.sat_speed],
                "motor.right.target": [motor_speed_right],
            }
        elif motor_speed_right <= -500:
            return {
                "motor.left.target": [motor_speed_left],
                "motor.right.target": [-self.sat_speed],
            }
        else:
            return {
                "motor.left.target": [motor_speed_left],
                "motor.right.target": [motor_speed_right],
            }

#######################################################################################################################
####################################   Local Occupancy Graph Generation ###############################################
#######################################################################################################################
    def sensor_val_to_cm_dist(self, sens_val):
        """
        Returns the distance corresponding to the sensor value based
        on the sensor characteristics
        :param val: the sensor value that you want to convert to a distance
        :return: corresponding distance in cm
        """
        if sens_val == 0:
            return np.inf

        f = interp1d(self.sensor_measurements, self.sensor_distances)
        return f(sens_val).item()

    def obstacles_pos_from_sensor_vals(self, sensor_vals):
        """
        Returns a list containing the position of the obstacles
        w.r.t the center of the Thymio robot.
        :param sensor_vals: sensor values provided clockwise starting from the top left sensor.
        :return: numpy.array() that contains the position of the different obstacles
        """
        dist_to_sensor = [self.sensor_val_to_cm_dist(x) for x in sensor_vals]
        dx_from_sensor = [d * math.cos(alpha) for (d, alpha) in zip(dist_to_sensor, self.sensor_angles)]
        dy_from_sensor = [d * math.sin(alpha) for (d, alpha) in zip(dist_to_sensor, self.sensor_angles)]
        obstacles_pos = [[x[0] + dx, x[1] + dy] for (x, dx, dy) in
                         zip(self.sensor_pos_from_center, dx_from_sensor, dy_from_sensor)]
        return np.array(obstacles_pos)

    def rotate(self, angle, coords):
        """
        Rotates the coordinates of a matrix by the desired angle
        :param angle: angle in radians by which we want to rotate
        :return: numpy.array() that contains rotated coordinates
        """
        R = np.array(((np.cos(angle), -np.sin(angle)),
                      (np.sin(angle), np.cos(angle))))
        return R.dot(coords.transpose()).transpose()

    def local_occupancy(self, abs_pos, sensor_vals):
        '''
        abs_pos: List of lists that will contain the absolute x,y,theta coordinates of the robot
        sensor_vals:  List of lists that will contain the 7 sensor values
        '''

        abs_pos = np.array(abs_pos)

        local_occupancy_grids = [self.obstacles_pos_from_sensor_vals(x) for x in sensor_vals]

        global_map, overall_thymio_coords = [], []

        for (local_grid, pos) in zip(local_occupancy_grids, abs_pos):
            # Rotate the local occupancy grid
            rotated_grid = self.rotate(pos[2] - math.pi / 2, local_grid)
            rotated_thymio_coords = self.rotate(pos[2] - math.pi / 2, self.thymio_coords)

            # Translate the grid at the position of the Thymio
            obstacles_pos = rotated_grid + np.array([pos[0], pos[1]])
            abs_Thymio_coords = rotated_thymio_coords + np.array([pos[0], pos[1]])

            # Store position of the obstacles and Thymio in the global map
            global_map.append(obstacles_pos)
            overall_thymio_coords.append(abs_Thymio_coords)

        global_map = np.array(np.vstack(global_map))

        plt.figure(figsize=(10, 10))
        plt.plot(abs_pos[:, 0], abs_pos[:, 1])
        plt.scatter(global_map[:, 0], global_map[:, 1], color="r", s=10)

        for coords in overall_thymio_coords:
            plt.plot(coords[:, 0], coords[:, 1], color="g")

        plt.axis("equal")
        plt.gca().invert_yaxis()

        plt.show()