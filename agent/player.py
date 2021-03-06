import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from .cubic_spline_planner import *
from scipy import interpolate

from .models import *
from torchvision import transforms
import torch


def to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

class HockeyPlayer:

    def __init__(self, player_id = 0):

        self.kart = 'konqi'
        self.puck_ahead_thresh = 60
        self.goal_ahead_thresh = 90
        self.puck_goal_deg_thresh = 10
        self.puck_close_thresh = 2
        self.kart_stuck_thresh = 0.5
        self.kart_stuck_counter = 0
        self.puck_super_close_thresh = 0.1
        self.extra_counter = 0
        self.rev_counter = 0

        self.history = defaultdict(list)
        self.history['puck_x'].append(0)
        self.history['puck_y'].append(0)
        self.framestep = 0

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.CNN_Classifier = load_model('cnn_classifier').to(self.device).eval()
        self.CNN_Regressor = load_model('cnn_regressor').to(self.device).eval()
        self.transform = transforms.ToTensor()
        self.puck_x, self.puck_y = 0, 0
        
        print(self.device)

    def puck_is_close(self): 
        # Check if puck is close based on heuristic
        return np.sqrt(self.puck_x**2 + self.puck_y**2) < self.puck_close_thresh

    def puck_is_super_close(self):
        return self.puck_is_ahead() and np.sqrt(self.puck_x**2 + self.puck_y**2) < self.puck_super_close_thresh

    def puck_is_ahead(self):
        # Check if puck is ahead within heuristic range
        self.logit = self.CNN_Classifier(self.transform(self.image)[None].to(self.device))
        self.result = self.logit.argmax(1)
        self.proba = torch.max(torch.sigmoid(self.logit))
        return  (self.result == 1.0) and (self.proba > 0.7)

    def goal_is_ahead(self):
        # Check if puck is ahead within heuristic range
        return np.arccos(self.kart_goal_dp)*180/np.pi < self.goal_ahead_thresh

    def kart_is_stuck(self):
        if len(self.history['kart_location']) < 20:
            return False
        else: 
            return np.linalg.norm(np.array(self.history['kart_location'][-1]) - \
                np.array(self.history['kart_location'][-20])) < self.kart_stuck_thresh

    def reverse_drive(self):
        # Reset the threshold the the base heuristic
        self.puck_ahead_thresh = 20

        # Angle to steering towards puck, corrected with 4.5 as a heuristic I used in HW#5
        puck_steer = 4.5*self.puck_x

        # Compute steering direction and magnitude 
        steer_direction = -np.sign(self.history['puck_x'][-1])
        steer_magnitude = abs(puck_steer) if abs(puck_steer) >= 0.5 else 0.5 # heuristic to make sure we do not reverse without actually steering in some direction
        action = {'acceleration': 0, 'brake': True, 'drift': False, 'nitro': False, 'rescue': False, 'steer': steer_direction*steer_magnitude}

        return action

    def spline_drive(self): 
        # Reset the threshold the the base heuristic
        self.puck_ahead_thresh = 60
        x = [0.0, self.puck_x, self.goal_x]
        y = [0.0, self.puck_y, self.goal_y]
        ds = 0.01
        sp = Spline2D(x, y)
        s = np.arange(0, sp.s[-1], ds)
        rx, ry = [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)

        x = rx[np.argmin([abs(i-self.puck_y/2) for i in ry])]*(30 if self.puck_is_close() else 30)
        acceleration = 0.5 if self.current_vel < 15 else 0.0
        acceleration = acceleration if np.abs(self.goal_x - self.puck_x) > 0.1 else 1.0
        acceleration = acceleration if self.goal_is_ahead else (0.1 if self.current_vel < 5 else 0.0)

        action = {'acceleration': acceleration, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': x}

        return action

    def chase_drive(self): 
        # Reset the threshold the the base heuristic
        self.puck_ahead_thresh = 60

        x = -self.goal_x*(3 if self.goal_is_ahead else 6)
        acceleration = 0.5 if (self.goal_is_ahead and self.current_vel < 15)  else 0.1
        action = {'acceleration': acceleration, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': x}

        return action

    def release_stuck(self):
        # print('Stuck')
        self.kart_stuck_counter += 1

        last_action = self.history['action'][-1]

        if last_action == 'Forward':
            return {'acceleration': 0, 'brake': True, 'drift': False, 'nitro': False, 'rescue': False, 'steer': self.history['puck_x'][-1]}
        else: 
            return {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': self.history['puck_x'][-1]}

    def act(self, image, player_info):
        self.player_info = player_info
        self.image = image


        if self.framestep == 0:
            self.goal_line_1 = [10.46, 0.07, -np.sign(player_info.kart.location[2])*64.5]
            self.goal_line_2 = [-10.46, 0.07, -np.sign(player_info.kart.location[2])*64.5]
            self.goal_line = [0.0, 0.07, -np.sign(player_info.kart.location[2])*64.5]

            self.own_goal_line_1 = [-10.46, 0.07, np.sign(player_info.kart.location[2])*64.5]
            self.own_goal_line_2 = [10.46, 0.07, np.sign(player_info.kart.location[2])*64.5]
            self.own_goal_line = [0.0, 0.07, np.sign(player_info.kart.location[2])*64.5]

        self.proj = np.array(player_info.camera.projection).T
        self.view = np.array(player_info.camera.view).T
        self.goal_x, self.goal_y = (to_image(self.goal_line_1, self.proj, self.view) + 
                         to_image(self.goal_line_2, self.proj, self.view)) / 2
        self.own_goal_x, self.own_goal_y = (to_image(self.own_goal_line_1, self.proj, self.view) + 
                         to_image(self.own_goal_line_2, self.proj, self.view)) / 2
        self.current_vel = np.linalg.norm(player_info.kart.velocity)
        
        if self.puck_is_ahead():
            self.puck_x, self.puck_y = self.CNN_Regressor(self.transform(self.image)[None].to(self.device)).to(torch.device("cpu")).data.numpy()[0]
            self.history['puck_x'].append(self.puck_x)
            self.history['puck_y'].append(self.puck_y)

        self.history['kart_location'].append(player_info.kart.location)
        self.framestep += 1

        # Perform 3D vecotr manipulation
        self.kart_front_vec = np.array(player_info.kart.front) - np.array(player_info.kart.location)
        self.kart_front_vec_norm = self.kart_front_vec / np.linalg.norm(self.kart_front_vec)
        self.kart_goal_vec = np.array(self.goal_line) - np.array(player_info.kart.location)
        self.kart_goal_vec_norm = self.kart_goal_vec / np.linalg.norm(self.kart_goal_vec)
        self.kart_goal_dp = self.kart_front_vec_norm.dot(self.kart_goal_vec_norm)
        self.own_goal_kart_vec = np.array(player_info.kart.front) - np.array(self.own_goal_line)
        self.dist_to_own_goal = np.linalg.norm(self.own_goal_kart_vec)

        if self.kart_is_stuck():
            self.kart_stuck_thresh = 1
            action = self.release_stuck()
            return action
        elif self.kart_stuck_counter > 40:
            return {'acceleration': 0.0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': True, 'steer': 0.0}

        if self.puck_is_ahead():
            self.history['action'].append('Forward')
            if self.goal_is_ahead():

                if self.puck_is_super_close():
                    action =  self.chase_drive()

                else:
                    action =  self.spline_drive()
            else:
                self.history['action'].append('Forward')
                self.goal_y = -0.5
                self.goal_x = np.sign(self.own_goal_x)*1

                if self.puck_is_super_close():
                    action =  self.chase_drive()

                else:
                    action =  self.spline_drive()
                    action['acceleration'] = 0.0 if self.current_vel > (15 if self.dist_to_own_goal > 15 else 5) else 0.5

                self.puck_ahead_thresh = 90

        elif self.proba > 0.95 and self.extra_counter == 0:
            # print('Reverse')
            self.history['action'].append('Reverse')
            action =  self.reverse_drive()

        else:
            puck_x_arr = np.array(self.history['puck_x'][-5:])
            ii = np.arange(0, len(puck_x_arr))
            fx = interpolate.interp1d(ii, puck_x_arr, fill_value='extrapolate')
            self.puck_x = fx(5)

            puck_y_arr = np.array(self.history['puck_y'][-5:])
            ii = np.arange(0, len(puck_y_arr))
            fy = interpolate.interp1d(ii, puck_y_arr, fill_value='extrapolate')
            self.puck_y = fy(5)

            if np.abs(self.puck_x) < 1 and self.puck_y < 0 and self.extra_counter < 15:
                # print('Extra Spline')
                self.extra_counter += 1
                self.history['action'].append('Forward')
                action =  self.spline_drive()
                action['acceleration'] = 0.0 if self.current_vel > 7.5 else 0.75
            else:
                # print('Extra Reverse')
                self.history['action'].append('Reverse')
                action =  self.reverse_drive()
                self.rev_counter += 1
                if self.rev_counter >= 15 and self.extra_counter >=15:
                    self.extra_counter = 0
                    self.rev_counter = 0

        self.kart_stuck_counter = 0
        self.kart_stuck_thresh = 0.5

        return action
