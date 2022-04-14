#!/usr/bin/env python

import numpy as np
from matplotlib import animation

from scripts.helper_func import *
from scripts.simulator import *
from scripts.SnakeStaticValidation import *

class PointCloudValidation(SnakeStaticValidation):
    def __init__(self, n):
        super().__init__()
        self.__pt_eye_alpha_left = np.empty((3))*np.nan
        self.__pt_eye_alpha_right = np.empty((3))*np.nan
        self.__pt_eye_beta_left = np.empty((3))*np.nan
        self.__pt_eye_beta_right = np.empty((3))*np.nan
        self.__time = 0
        self.nb_frames = n
        
        # snake parts
        self.head_parts_l = generate_snake_head()
        self.head_parts_r = generate_snake_head()
        self.body_parts = generate_snake_body()
        self.tail_parts = generate_snake_tail()
        self.rattle_parts = generate_snake_rattle()
        self.eye_parts_1 = generate_snake_eye()
        self.eye_parts_2 = generate_snake_eye()
        self.eye_parts_3 = generate_snake_eye()
        self.eye_parts_4 = generate_snake_eye()

        # prepare handles for animation
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
        self.ploly_snake_body = draw_snake_body(self.ax, self.body_parts, np.eye(3))
        self.ploly_snake_headl, self.ploly_snake_neckl = draw_snake_head(self.ax, self.head_parts_l, np.eye(3))
        self.ploly_snake_headr, self.ploly_snake_neckr = draw_snake_head(self.ax, self.head_parts_r, np.eye(3))

        self.ploly_snake_tail1, self.ploly_snake_tail2 = draw_snake_tail(self.ax, self.tail_parts, np.eye(3))
        self.ploly_snake_rattle = draw_snake_rattle(self.ax, self.rattle_parts, np.eye(3))

        self.ploly_snake_eye1 = draw_snake_eye(self.ax, self.eye_parts_1, np.eye(3))
        self.ploly_snake_eye2 = draw_snake_eye(self.ax, self.eye_parts_2, np.eye(3))
        self.ploly_snake_eye3 = draw_snake_eye(self.ax, self.eye_parts_3, np.eye(3))
        self.ploly_snake_eye4 = draw_snake_eye(self.ax, self.eye_parts_4, np.eye(3))

        self.map_scatter = self.ax.scatter([], [], c="tab:red", s=4)
        self.point_cloud = np.array([]).reshape(0,3)
        
        # list of all plotting data
        self.all_body = []
        self.all_headl = []
        self.all_neckl = []
        self.all_headr = []
        self.all_neckr = []

        self.all_tail1 = []
        self.all_tail2 = []
        self.all_rattle = []

        self.all_eye1 = []
        self.all_eye2 = []
        self.all_eye3 = []
        self.all_eye4 = []
        
        self.all_pt_eye_alpha_left = []
        self.all_pt_eye_alpha_right = []
        self.all_pt_eye_beta_left = []
        self.all_pt_eye_beta_right = []

    def valid_point(self, point):
        assert(point.shape[0] == 3 and point.ndim == 1), "The point must have 3 elements (homogeneous coordiantes)!"
        return point
    
    def set_global_point_alpha_left(self, point): 
        self.__pt_eye_alpha_left = self.valid_point(point)
    
    def set_global_point_alpha_right(self, point):
        self.__pt_eye_alpha_right = self.valid_point(point)
        
    def set_global_point_beta_left(self, point):
        self.__pt_eye_beta_left = self.valid_point(point)
        
    def set_global_point_beta_right(self, point):
        self.__pt_eye_beta_right = self.valid_point(point)

    def save_animation_frame(self):
        self.all_body.append((self._SnakeStaticValidation__body @ self.body_parts[1])[0:2,:].T)
        self.all_headl.append((self._SnakeStaticValidation__head_alpha @ self.head_parts_l[1])[0:2,:].T)
        self.all_neckl.append((self._SnakeStaticValidation__head_alpha @ self.head_parts_l[2])[0:2,:].T)
        self.all_headr.append((self._SnakeStaticValidation__head_beta @ self.head_parts_r[1])[0:2,:].T)
        self.all_neckr.append((self._SnakeStaticValidation__head_beta @ self.head_parts_r[2])[0:2,:].T)

        self.all_tail1.append((self._SnakeStaticValidation__tail @ self.tail_parts[1])[0:2,:].T)
        self.all_tail2.append((self._SnakeStaticValidation__tail @ self.tail_parts[2])[0:2,:].T)
        self.all_rattle.append((self._SnakeStaticValidation__rattle @ self.rattle_parts[1])[0:2,:].T)

        self.all_eye1.append((self._SnakeStaticValidation__eye_alpha_left @ self.eye_parts_1[1])[0:2,:].T)
        self.all_eye2.append((self._SnakeStaticValidation__eye_alpha_right @ self.eye_parts_2[1])[0:2,:].T)
        self.all_eye3.append((self._SnakeStaticValidation__eye_beta_left @ self.eye_parts_3[1])[0:2,:].T)
        self.all_eye4.append((self._SnakeStaticValidation__eye_beta_right @ self.eye_parts_4[1])[0:2,:].T)
        
        self.all_pt_eye_alpha_left.append(self.__pt_eye_alpha_left)
        self.all_pt_eye_alpha_right.append(self.__pt_eye_alpha_right)
        self.all_pt_eye_beta_left.append(self.__pt_eye_beta_left)
        self.all_pt_eye_beta_right.append(self.__pt_eye_beta_right)
        
    def draw_frame(self, t):
        self.ploly_snake_body.set_xy(self.all_body[t])
        self.ploly_snake_headl.set_xy(self.all_headl[t])
        self.ploly_snake_neckl.set_xy(self.all_neckl[t])
        self.ploly_snake_headr.set_xy(self.all_headr[t])
        self.ploly_snake_neckr.set_xy(self.all_neckr[t])

        self.ploly_snake_tail1.set_xy(self.all_tail1[t])
        self.ploly_snake_tail2.set_xy(self.all_tail2[t])
        self.ploly_snake_rattle.set_xy(self.all_rattle[t])

        self.ploly_snake_eye1.set_xy(self.all_eye1[t])
        self.ploly_snake_eye2.set_xy(self.all_eye2[t])
        self.ploly_snake_eye3.set_xy(self.all_eye3[t])
        self.ploly_snake_eye4.set_xy(self.all_eye4[t])
        
        self.point_cloud = np.vstack([self.point_cloud, self.all_pt_eye_alpha_left[t], self.all_pt_eye_alpha_right[t],
                                      self.all_pt_eye_beta_right[t], self.all_pt_eye_beta_left[t]])
        
        self.map_scatter.set_offsets(self.point_cloud[:,0:2])
            
        self.ax.set_xlim((-50, 50))
        self.ax.set_ylim((-50, 50))
        
        return (self.ploly_snake_body,
                self.ploly_snake_headl, 
                self.ploly_snake_neckl, 
                self.ploly_snake_headr, 
                self.ploly_snake_neckr,
                self.ploly_snake_tail1,
                self.ploly_snake_tail2,
                self.ploly_snake_rattle,
                self.ploly_snake_eye1,
                self.ploly_snake_eye2,
                self.ploly_snake_eye4
               )
    
    def display_all_frames(self):
        for t in np.arange(self.nb_frames):
            self.draw_frame(t)
        self.point_cloud = np.array([]).reshape(0,3)
        
    def display_animation(self):
        anim = animation.FuncAnimation(self.fig, self.draw_frame,
                               frames=np.arange(self.nb_frames), interval=40, 
                               blit=True, repeat=False)
        plt.close(anim._fig)
        return anim