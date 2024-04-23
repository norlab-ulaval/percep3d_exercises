#!/usr/bin/env python

import numpy as np
from helper_func import *

S_points = np.array([[0., 3, 5, 5, 1.,  1,  3,  5,  5,  2,  0, 0, 4, 4, 2, 0, 0],
                     [0., 0, 2, 5, 9., 10, 12, 12, 13, 13, 11, 8, 4, 3, 1, 1, 0],
                     [1., 1, 1, 1, 1.,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1]])

E_points = np.array([[0., 5, 5, 1, 1., 3, 3, 1,  1,  5,  5,  0, 0],
                     [0., 0, 1, 1, 6., 6, 7, 7, 12, 12, 13, 13, 0],
                     [1., 1, 1, 1, 1., 1, 1, 1,  1,  1,  1,  1, 1]])

two_points = np.array([[0., 5, 5, 1, 5.,  5,  3,  0,  0,  2,  4, 4, 0, 0],
                       [0., 0, 1, 1, 5., 11, 13, 13, 12, 12, 10, 6, 2, 0],
                       [1., 1, 1, 1, 1.,  1,  1,  1,  1,  1,  1, 1, 1, 1]])

square_points = np.array([[1, 1, -1, -1, 1],
                          [-1, 1, 1, -1, -1],
                          [1, 1, 1, 1, 1]])

def ray_tracing_single_shape(T, array_points, sigma=0., max_range=np.inf):
    L = np.linalg.inv(T) @ array_points
    
    #ax.plot(L[0], L[1]);
    
    intersec = np.empty((L.shape[0],L.shape[1]-1))
    intersec[2,:] = np.ones(L.shape[1]-1)

    for i in np.arange(L.shape[1]-1):

        dx = L[0,i] - L[0,i+1]
        dy = L[1,i] - L[1,i+1]
        if(dy != 0):
            intersec[0,i] = L[0,i] - (dx/dy)*L[1,i] + np.random.normal(scale=sigma)
        else:
            intersec[0,i] = float("inf")
        intersec[1,i] = 0
        
        # is the point within the segment
        bound_max = np.max([L[0,i], L[0,i+1]])
        bound_min = np.min([L[0,i], L[0,i+1]])

        outside = not((intersec[0,i] <= bound_max) and 
                  (intersec[0,i] >= bound_min))

        behind = (intersec[0,i] < 0)
        too_far = (intersec[0,i] > max_range)
        
        if(outside or behind or too_far):
            intersec[0:2,i] = [np.nan, np.nan]
        
    closest_id = 0
    if(np.isfinite(intersec[0,:]).any()):
        closest_id = np.nanargmin(intersec[0,:])
    
    if(np.isfinite(intersec[:,closest_id]).all()):
        closest_hit = T @ intersec[:,closest_id]
    else:
        closest_hit = intersec[:,closest_id]
    
    return closest_hit

def ray_tracing_multi_shape(T, list_shapes, sigma=0., max_range=np.inf):
    dist = np.inf
    closest_hit =[np.nan, np.nan, 1.]
    for shape in list_shapes:
        current_shape_hit = ray_tracing_single_shape(T, shape, sigma, max_range)
        current_dist = np.linalg.norm(current_shape_hit-T[:,2])
        if(np.isfinite(current_dist) and current_dist < dist):
            dist = current_dist
            closest_hit = current_shape_hit
            
    return closest_hit

def generate_snake_head():
    p = np.array([-1., 0., 1.])
    snake_head = np.vstack([p, 
                            rigid_transformation((0, 0, np.pi/3.)) @ p, 
                            rigid_transformation((0, 0, (2*np.pi)/3.)) @ p, 
                            rigid_transformation((0, 0, np.pi)) @ p,
                            [1., 1., 1.],
                            [0.25, 2.5, 1.],
                            [-0.25, 2.5, 1.],
                            [-1, 1, 1.]
                           ]).T
    snake_head = rigid_transformation((2., 0., -np.pi/2)) @ snake_head

    snake_hex = np.empty((0,3))

    for theta in np.linspace(0, 2.*np.pi, 7): 
        snake_hex = np.vstack([snake_hex, rigid_transformation((0,0,theta)) @ p])

    snake_hex = snake_hex.T
    snake_hex = rigid_transformation((0., 0., -np.pi/2)) @ snake_hex
    snake_head_o = np.array([0,0,1.])
    
    return snake_head_o, snake_hex, snake_head

def draw_snake_head(ax, head_parts, T = rigid_transformation((0,0,0)), draw_frame=False):
    snake_head_o, snake_hex, snake_head = head_parts
    
    poly1 = T @ snake_head
    poly2 = T @ snake_hex
    o     = T @ snake_head_o
    
    polygone1 = plt.Polygon(poly1[0:2,:].T, color='white', alpha=0.2)
    ax.add_patch(polygone1)
    
    polygone2 = plt.Polygon(poly2[0:2,:].T, color='white', alpha=0.2)
    ax.add_patch(polygone2)
    
    if(draw_frame):
        draw_frame(ax, origin=o[0:2], text_x=r"$\vec{\mathscr{x}}$", color = "tab:blue")
    all_points = np.hstack([poly1, poly2, o[:,None]])
    return polygone1, polygone2


# generate body
def generate_snake_body():
    p = np.array([-1., 0., 1.])

    half_hex = np.vstack([p, 
                        rigid_transformation((0, 0, np.pi/3.)) @ p, 
                        rigid_transformation((0, 0, (2*np.pi)/3.)) @ p, 
                        rigid_transformation((0, 0, np.pi)) @ p
                         ]).T

    snake_body = np.vstack([half_hex.T,
                            [1., 1., 1.],
                            [3., 4., 1.],
                            (rigid_transformation((2,5,np.pi)) @ half_hex).T,
                            [1., 5., 1.],
                            [1., 4., 1.],
                            [0., 3., 1.],
                            [-1., 4., 1.],
                            [-1., 5., 1.],
                            (rigid_transformation((-2,5,np.pi)) @ half_hex).T,
                            [-3., 4., 1.],
                            [-1., 1., 1.],
                           ]).T

    snake_body = rigid_transformation((0., 0., -np.pi/2)) @ snake_body
    snake_body_o = np.array([0,0,1.])
    return snake_body_o, snake_body

def draw_snake_body(ax, body_parts, T = rigid_transformation((0,0,0)), draw_frame=False):
    snake_body_o, snake_body = body_parts
    
    poly1 = T @ snake_body
    o     = T @ snake_body_o
    polygone = plt.Polygon(poly1[0:2,:].T, color='white', alpha=0.2)
    ax.add_patch(polygone)
    if(draw_frame):
        draw_frame(ax, origin=o[0:2], text_x=r"$\vec{\mathscr{x}}$", color = "tab:blue")
    all_points = np.hstack([poly1, o[:,None]])
    return polygone

# Snake tail
def generate_snake_tail():

    p = np.array([-1., 0., 1.])

    snake_tail1 = np.empty((0,3))
    for theta in np.linspace(0, 2.*np.pi, 7): 
        snake_tail1 = np.vstack([snake_tail1, rigid_transformation((0,0,theta)) @ p])

    snake_tail1 = snake_tail1.T

    snake_tail1 = rigid_transformation((0., 0., -np.pi/2)) @ snake_tail1
    snake_tail2 = rigid_transformation((-2., 0., 0.)) @ snake_tail1
    snake_tail_o = np.array([0,0,1.])
    return snake_tail_o, snake_tail1, snake_tail2

def draw_snake_tail(ax, tail_parts, T = rigid_transformation((0,0,0)), draw_frame=False):
    snake_tail_o, snake_tail1, snake_tail2 = tail_parts
    
    poly1 = T @ snake_tail1
    poly2 = T @ snake_tail2
    o     = T @ snake_tail_o
    
    polygone1 = plt.Polygon(poly1[0:2,:].T, color='white', alpha=0.2)
    ax.add_patch(polygone1)
    polygone2 = plt.Polygon(poly2[0:2,:].T, color='white', alpha=0.2)
    ax.add_patch(polygone2)
    
    if(draw_frame):
        draw_frame(ax, origin=o[0:2], text_x=r"$\vec{\mathscr{x}}$", color = "tab:blue")
    all_points = np.hstack([poly1, poly2, o[:,None]])
    return polygone1, polygone2


# generate rattle
def generate_snake_rattle():
    p = np.array([-1., 0., 1.])

    snake_rattle = np.vstack([p, 
                          rigid_transformation((0, 0, np.pi/3.)) @ p, 
                          rigid_transformation((0, 0, (2*np.pi)/3.)) @ p, 
                          rigid_transformation((0, 0, np.pi)) @ p,
                          [1.,1.,1.],
                          [0.,3.,1.],
                          [-1.,1.,1.]
                         ]).T

    snake_rattle = rigid_transformation((0., 0., np.pi/2)) @ snake_rattle
    
    snake_rattle_o = np.array([0,0,1.])
    
    return snake_rattle_o, snake_rattle

def draw_snake_rattle(ax, rattle_parts, T = rigid_transformation((0,0,0)), draw_frame=False):
    snake_rattle_o, snake_rattle = rattle_parts
    
    poly1 = T @ snake_rattle
    o     = T @ snake_rattle_o
    
    polygone = plt.Polygon(poly1[0:2,:].T, color='white', alpha=0.2)
    ax.add_patch(polygone)
    if(draw_frame):
        draw_frame(ax, origin=o[0:2], text_x=r"$\vec{\mathscr{x}}$", color = "tab:blue")
    all_points = np.hstack([poly1, o[:,None]])
    return polygone

def generate_snake_eye():
    snake_eye = np.vstack([[0.,-2.,1.],
                          [1.,0.,1.],
                          [0.,2.,1.]
                         ]).T
    s = 0.3
    snake_eye = scale_transformation((s,s)) @ snake_eye
    
    snake_eye_o = np.array([0,0,1.])
    
    return snake_eye_o, snake_eye

def draw_snake_eye(ax, eye_parts, T=rigid_transformation((0,0,0)), draw_frame=False):
    snake_eye_o, snake_eye = eye_parts
    
    poly1 = T @ snake_eye
    o     = T @ snake_eye_o
    
    polygone = plt.Polygon(poly1[0:2,:].T, color='red', alpha=0.4)
    ax.add_patch(polygone)
    if(draw_frame):
        draw_frame(ax, origin=o[0:2], text_x=r"$\vec{\mathscr{x}}$", color = "tab:blue")
    all_points = np.hstack([poly1, o[:,None]])
    return polygone


    
class SnakeDynamicValidation:
    def __init__(self, n):
        self.__x = np.zeros(n)
        self.__y = np.zeros(n)
        self.__time = 0
        
    def set_position(self, x, y):
        self.__x[self.__time] = x
        self.__y[self.__time] = y
        self.__time += 1
        
    def draw_relative_position(self):
        t = np.arange(len(self.__x))
        fig = plt.figure(figsize=(12, 6))
        grid = plt.GridSpec(2, 2, hspace=0.1, wspace=0.2)
        
        ax_xy = fig.add_subplot(grid[:, 0])
        ax_xy.scatter(self.__x, self.__y, alpha=0.4, c=np.arange(len(self.__x)))
        ax_xy.set_xlabel("X [m]")
        ax_xy.set_ylabel("Y [m]")
        
        ax_xt = fig.add_subplot(grid[0, 1])
        ax_xt.set_ylabel("X [m]")
        ax_xt.scatter(t, self.__x, alpha=0.4, c=t, s=5)
        ax_xt.plot(t, self.__x, alpha=0.4)
        plt.setp(ax_xt.get_xticklabels(), visible=False)
        
        ax_yt = fig.add_subplot(grid[1, 1:], sharex=ax_xt)
        ax_yt.scatter(t, self.__y, alpha=0.4, c=t, s=5)
        ax_yt.plot(t, self.__y, alpha=0.4)
        ax_yt.set_ylabel("Y [m]")
        ax_yt.set_xlabel("time [s]")
        
