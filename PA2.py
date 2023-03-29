#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Control in Human-Robot Interaction Assignment 2a: teleoperation & tele-impedance
-------------------------------------------------------------------------------
DESCRIPTION:
2-DOF planar robot arm model with shoulder and elbow joints. The code includes
simulation environment and visualisation of the robot.

Assume that the robot is a torque-controlled robot and its dynamics is
compensated internally, therefore the program operates at the endpoint in
Cartesian space. The input to the robot is the desired endpoint force and
the output from the robot is the measured endpoint position.

Important variables:
pm[0] -> mouse x position
pm[1] -> mouse y position
pr[0] -> reference endpoint x position
pr[1] -> reference endpoint y position
p[0] -> actual endpoint x position
p[1] -> actual endpoint y position
dp[0] -> endpoint x velocity
dp[1] -> endpoint y velocity
F[0] -> endpoint x force
F[1] -> endpoint y force

TASK:
Design an impedance controller for the remote robot, whose reference position
is controlled with a computer mouse. Use keyboard keys as a tele-impedance
interface to increase or decrease the stiffness of the robot in different axes.
Create a virtual wall perpendicular to x-axis and a periodic force perturbation
along y-axis. Display force manipulability of the robot.

NOTE: Keep the mouse position inside the robot workspace before pressing 'e'
to start and then maintain it within the workspace during the operation,
in order for the inverse kinematics calculation to work properly.
-------------------------------------------------------------------------------


INSTURCTOR: Luka Peternel
e-mail: l.peternel@tudelft.nl

"""



import numpy as np
import math
import matplotlib.pyplot as plt
import pygame
from scipy.spatial import distance



# def rectRotated(self, surface, rect, color, rotation):
#      """
#      Draws a rotated Rect.
#      surface: pygame.Surface
#      rect: pygame.Rect
#      color: pygame.Color
#      rotation: float (degrees)
#      return: np.ndarray (vertices)
#      """
#      # calculate the rotation in radians
#      rot_radians = -rotation * np.pi / 180
#      #rot_radians = -rotation
#      # calculate the points around the center of the rectangle, taking width and height into account
#      angle = math.atan2(rect.height / 2, rect.width / 2)
#      angles = [angle, -angle + np.pi, angle + np.pi, -angle]
#      radius = math.sqrt((rect.height / 2)**2 + (rect.width / 2)**2)

#      # create a numpy array of the points
#      points = np.array([
#          [rect.x + radius * math.cos(angle + rot_radians), rect.y + radius * math.sin(angle + rot_radians)]
#          for angle in angles
#      ])

#      # draw the polygon
#      pygame.draw.polygon(surface, color, points)

#      # return the vertices of the rectangle
#      return points

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    distance_ = distance.cdist([node], nodes).min()
    return [nodes[closest_index],distance_] #gives the coordinates of the line which is the closest to the mouse & the distance between the two

'''ROBOT MODEL'''

class robot_arm_2dof:
    def __init__(self, l):
        self.l = l # link length
    
    
    
    # arm Jacobian matrix
    def Jacobian(self, q):
        J = np.array([[-self.l[0]*np.sin(q[0]) - self.l[1]*np.sin(q[0] + q[1]),
                     -self.l[1]*np.sin(q[0] + q[1])],
                    [self.l[0]*np.cos(q[0]) + self.l[1]*np.cos(q[0] + q[1]),
                     self.l[1]*np.cos(q[0] + q[1])]])
        return J
    
    
    # inverse kinematics
    def IK(self, p):
        q = np.zeros([2])
        r = np.sqrt(p[0]**2+p[1]**2)
        q[1] = np.pi - math.acos((self.l[0]**2+self.l[1]**2-r**2)/(2*self.l[0]*self.l[1]))
        q[0] = math.atan2(p[1],p[0]) - math.acos((self.l[0]**2-self.l[1]**2+r**2)/(2*self.l[0]*r))
        
        return q


'''SIMULATION'''

# SIMULATION PARAMETERS
dt = 0.01 # intergration step timedt = 0.01 # integration step time
dts = dt*1 # desired simulation step time (NOTE: it may not be achieved)



# ROBOT PARAMETERS
x0 = 0.0 # base x position
y0 = 0.0 # base y position
l1 = 0.33 # link 1 length
l2 = 0.33 # link 2 length (includes hand)
l = [l1, l2] # link length



# IMPEDANCE CONTROLLER PARAMETERS
K = np.diag([1000,100]) # stiffness matrix N/m
stiffness_increment = 100 # for tele-impedance


# SIMULATOR
# initialise robot model class
model = robot_arm_2dof(l)


# initialise real-time plot with pygame

#initializing the window and background
width, height = 800, 600
pygame.init() # start pygame
window = pygame.display.set_mode((width, height)) # create a window (size in pixels)
screenPatient = pygame.Surface((width, height))
window.fill((255,255,255)) # white background
screenPatient.fill((255,255,255)) # white background
xc, yc = window.get_rect().center # window center

body_im = pygame.image.load('body.png')
body_im = pygame.transform.scale(body_im,(width,height))


# initialise tool image - draw scalpel
scalpel_im = pygame.image.load('scalpel.png')
scalpel_scale = (50,50)
scalpel_im = pygame.transform.scale(scalpel_im, scalpel_scale)
scalpel = pygame.Rect(*screenPatient.get_rect().center, 0, 0).inflate(scalpel_scale[0], scalpel_scale[1])
CUT = False # are we cutting?
pc_arr = [];
xs = np.array(scalpel.center) # scalpel center

# initialise tool image 2 - draw needle
needle_im = pygame.image.load('needle.png')
needle_scale = (50,50)
needle_im = pygame.transform.scale(needle_im, needle_scale)
needle = pygame.Rect(*screenPatient.get_rect().center, 0, 0).inflate(needle_scale[0], needle_scale[1])
xn = np.array(needle.center) # scalpel center

#performance feedback visualization - 
color_acc = (255,255,255)


# draw the reference cut line
ref_cut = np.array([[180,450],[300,400]]) #coordinates of the reference line
line_coor = []
a, b = coefficients = np.polyfit([ref_cut[0,0],ref_cut[1,0]], [ref_cut[0,1],ref_cut[1,1]], 1)
 
for x in range(ref_cut[0,0], ref_cut[1,0]+1): #calculates all the points on the reference line
    y = a*x + b
    line_coor.append([x,y])
perf_pm = [] 
p_line_close_pm = []    #point on the line which is the closest to the scalpel during cutting
dis_line_pm = []        #distance between those points = accuracy
mean_dis = 0            #mean distance between those points
draw_time = []          #time how long it takes to make the cut
draw_time2 = [0] 

# fonts and captions
pygame.display.set_caption('Patient body')
font = pygame.font.Font('freesansbold.ttf', 12) # printing text font and font size
text = font.render('Patient body', True, (0, 0, 0), (255, 255, 255)) # printing text object
textRect = text.get_rect()
textRect.topleft = (600,10) # printing text position with respect to the top-left corner of the window

clock = pygame.time.Clock() # initialise clock
FPS = int(1/dts) # refresh rate

# initial conditions
t = 0.0 # time
pm = np.zeros(2) # mouse position
pr = np.zeros(2) # reference endpoint position
p = np.array([0.1,0.1]) # actual endpoint position
dp = np.zeros(2) # actual endpoint velocity
F = np.zeros(2) # endpoint force
q = np.zeros(2) # joint position
p_prev = np.zeros(2) # previous endpoint position
m = 0.5 # endpoint mass -- scalpel mass 22g  + robot's end effector
i = 0 # loop counter
state = [] # state vector

# scaling
window_scale = 800 # conversion from meters to pixles

# drawing cut line
drawing = False
pm_arr = []
last = None
moving = False

# wait until the start button is pressed
run = True
while run:
    for event in pygame.event.get(): # interrupt function
        if event.type == pygame.KEYUP:
            if event.key == ord('e'): # enter the main loop after 'e' is pressed
                run = False


# MAIN LOOP
i = 0
run = True
while run:
    for event in pygame.event.get(): # interrupt function
        if event.type == pygame.QUIT: # force quit with closing the window
            run = False

        else:
            if event.type == pygame.KEYUP:
                if event.key == ord('q'): # force quit with q button
                    run = False
            else:
                if event.type == pygame.MOUSEMOTION:
                    moving = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    CUT = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    CUT = False
                else:
                    moving = False


            '''*********** Student should fill in ***********'''
            # tele-impedance interface / switch controllers
            '''*********** Student should fill in ***********'''
    
    
    # making backgrounds and represent a tool -- scalpel and needle
    pm = np.array(pygame.mouse.get_pos())  # in VRenv frame
    scalpel.center = pm
    screenPatient.fill((255, 255, 255))
    screenPatient.blit(body_im, (0, 0))
    screenPatient.blit(scalpel_im, (scalpel.topleft[0],scalpel.topleft[1]))

    needle.center = pm
    #screenPatient.blit(needle_im,(needle.topleft[0],needle.topleft[1]))
     
    #code to draw a continuous cut
    if moving == True:
        if drawing == True:
            mouse_pos = pygame.mouse.get_pos()
            if last is not None:
                pc_arr.append(last)          #safe 'last' data points for all cuts
                pm_arr.append(mouse_pos)     #safe 'mouse' data points all cuts
                draw_time.append(t)
            last = mouse_pos
        if CUT == False:
            drawing = False
            last = None
            mouse_pos = (0,0)
            #draw_time.append(None)           #so you can see the difference between time taken for different lines?
        elif CUT == True:
            drawing = True
    
    # draw the reference line for the cut
    pygame.draw.line(screenPatient,(0,0,0),ref_cut[0], ref_cut[1], 2)
    
    
    #the code to actually draw the line
    if len(pc_arr) != 0:
        for i in range(len(pc_arr)):
            k = i-1
            pygame.draw.line(screenPatient,(255,0,0),pc_arr[k], pm_arr[k], 2)
    
    #find the position and rotation of the reference line to define
    #margins necessary for the performance metric
    #alpha = np.arctan((ref_cut[0,1]-ref_cut[1,1])/(ref_cut[1,0]-ref_cut[0,0])) #calculate angle of line
    #l_cut = (ref_cut[1,0]-ref_cut[0,0])/np.cos(alpha) #length of the cut
    #c_cut = [ref_cut[1,0]-(l_cut/2*(ref_cut[1,0]-ref_cut[0,0]))/l_cut,(l_cut/2*(ref_cut[0,1]-ref_cut[1,1])/l_cut)+ref_cut[1,1]]#center coordinate of cut
    #height = 10
    #width = l_cut+20
    color = (0,0,0)
    #rect = pygame.Rect(c_cut[0],c_cut[1],width,height)
    #rotation = float(alpha*180/np.pi)
    #surface = screenPatient
    #points = rectRotated(rect, surface, rect, color, rotation)
    #pygame.draw.polygon(screenPatient,(0,0,0),points) #a polygon which can be used as a margin
    #pygame.draw.circle(screenPatient,(0,0,0),(points[0,0],points[0,1]),3) #visualize the middle of the line
   
    
    #performance metric
    if pm_arr != []:
        z = pm_arr[-1] #get the mouse position of the line drawing
        perf_pm = [z[0],z[1]] #place it in an array
        z = closest_node(perf_pm, line_coor) #use the function to calculate the point closest to the mouse during cutting and the distance between them
        p_line_close_pm = z[0] #point on line which is closest to the pm during cutting
        dis_line_pm.append(z[1]) #distance between those points
        pygame.draw.circle(screenPatient,(0,0,0),(p_line_close_pm[0],p_line_close_pm[1]),3) #visualize the point on the line which is closest to the mouse
        #calculate performance?
        mean_dis = sum(dis_line_pm)/len(dis_line_pm) #mean distance = accuracy?
        #speed = draw_time/t #--> gives the time it took to draw a line relative to the total time?
        draw_time2 = draw_time
        
        #visualize how well the person does it?
        if dis_line_pm[-1] <= 3: #think of some good measure?
            color_acc = (0,255,0) #green if good
        elif dis_line_pm[-1] > 3 and dis_line_pm[-1] <= 6:
            color_acc = (255,100,10)
        else:
            color_acc = (255,0,0) #red if bad lol
        
    pygame.draw.rect(screenPatient,color_acc,[0,0,100,100],0)

	# previous endpoint position for velocity calculation
    p_prev = pm.copy()

    # log states for analysis
    state.append([t, pr[0], pr[1], p[0], p[1], dp[0], dp[1], F[0], F[1], K[0,0], K[1,1]])
    
    # integration
    ddp = F/m
    dp += ddp*dt
    p += dp*dt
    t += dt
    
    '''*********** Student should fill in ***********'''
    # simulate a wall
    '''*********** Student should fill in ***********'''

    # increase loop counter
    i = i + 1


    # update individual link position
    q = model.IK(p)
    x1 = l1*np.cos(q[0])
    y1 = l1*np.sin(q[0])
    x2 = x1+l2*np.cos(q[0]+q[1])
    y2 = y1+l2*np.sin(q[0]+q[1])
    

    # print data
    #text = font.render("FPS = " + str( round( clock.get_fps() ) ) + "   K = " + str( [K[0,0],K[1,1]] ) + " N/m" + "   xh = " + str( np.round(scalpel.center,3) ) + " m" + "   F = " + str( np.round(F,0) ) + " N", True, (0, 0, 0), (255, 255, 255))
    text = font.render("Time = " + str( round( draw_time2[-1] ) ) +        "      Accuracy = " + str((round( mean_dis))), True, (0, 0, 0), (255, 255, 255))
    
    window.blit(screenPatient, (0,0))
    window.blit(text, textRect)


    pygame.display.flip() # update display
    
    
    # try to keep it real time with the desired step time
    clock.tick(FPS)
    
    if run == False:
        break

pygame.quit() # stop pygame







