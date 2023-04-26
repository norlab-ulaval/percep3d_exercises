#!/usr/bin/env python

import numpy as np
from scripts.helper_func import *
from scripts.simulator import *

class SnakeStaticValidation:
    def __init__(self):
        # Note: those are very bad names for transformation variables but
        # the goal is to have students come up with their proper variable names.
        # So, if you're doing the assignment and looking here for some clues, there are none.
        self.__body = np.empty((3,3))*np.nan
        self.__head_alpha = np.empty((3,3))*np.nan
        self.__head_beta = np.empty((3,3))*np.nan

        self.__tail = np.empty((3,3))*np.nan
        self.__rattle = np.empty((3,3))*np.nan

        self.__eye_alpha_left = np.empty((3,3))*np.nan
        self.__eye_alpha_right = np.empty((3,3))*np.nan
        self.__eye_beta_left = np.empty((3,3))*np.nan
        self.__eye_beta_right = np.empty((3,3))*np.nan
        
    def valid_trans(self, T):
        assert(T.shape[0] == 3 and T.shape[1] == 3), "The matrix must be 3 by 3!"
        return T
    
    def set_global_body(self, T): self.__body = self.valid_trans(T)
    def set_global_head_alpha(self, T): self.__head_alpha = self.valid_trans(T)
    def set_global_head_beta(self, T): self.__head_beta = self.valid_trans(T)
        
    def set_global_tail(self, T): self.__tail = self.valid_trans(T)
    def set_global_rattle(self, T): self.__rattle = self.valid_trans(T)
        
    def set_global_eye_alpha_left(self, T): self.__eye_alpha_left = self.valid_trans(T)
    def set_global_eye_alpha_right(self, T): self.__eye_alpha_right = self.valid_trans(T)
    def set_global_eye_beta_left(self, T): self.__eye_beta_left = self.valid_trans(T)
    def set_global_eye_beta_right(self, T): self.__eye_beta_right = self.valid_trans(T)
    
    

    def draw_snake(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
        if np.isfinite(self.__body).all():
            draw_snake_body(ax, generate_snake_body(), self.__body)
        if np.isfinite(self.__head_alpha).all():
            draw_snake_head(ax, generate_snake_head(), self.__head_alpha)
        if np.isfinite(self.__head_beta).all():
            draw_snake_head(ax, generate_snake_head(), self.__head_beta)
            
        if np.isfinite(self.__tail).all():
            draw_snake_tail(ax, generate_snake_tail(), self.__tail)
        if np.isfinite(self.__rattle).all():
            draw_snake_rattle(ax, generate_snake_rattle(), self.__rattle)
            
        if np.isfinite(self.__eye_alpha_left).all():
            draw_snake_eye(ax, generate_snake_eye(), self.__eye_alpha_left)
        if np.isfinite(self.__eye_alpha_right).all():
            draw_snake_eye(ax, generate_snake_eye(), self.__eye_alpha_right)
        if np.isfinite(self.__eye_beta_left).all():
            draw_snake_eye(ax, generate_snake_eye(), self.__eye_beta_left)
        if np.isfinite(self.__eye_beta_right).all():
            draw_snake_eye(ax, generate_snake_eye(), self.__eye_beta_right)
        ax.set_title("Everything will be fine");
        ax.set_xlabel("X [m]");
        ax.set_ylabel("Y [m]");
        return ax