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
pygame.init() # start pygame
window = pygame.display.set_mode((800, 600)) # create a window (size in pixels)
window.fill((255,255,255)) # white background


# body = pygame.image.load('body.png')
screenPatient = pygame.Surface((800,600))
screenPatient.blit(pygame.image.load('body.png'), (0,0))
window.blit(screenPatient, (0, 0))

xc, yc = window.get_rect().center # window center
pygame.display.set_caption('Patient body')

font = pygame.font.Font('freesansbold.ttf', 12) # printing text font and font size
text = font.render('Patient body', True, (0, 0, 0), (255, 255, 255)) # printing text object
textRect = text.get_rect()
textRect.topleft = (10, 10) # printing text position with respect to the top-left corner of the window

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

# draw scalpel
hscalpel = pygame.image.load('scalpel.png')
hscalpel = pygame.transform.scale(hscalpel, (0.5, 0.5))
scalpel  = pygame.Rect(*screenPatient.get_rect().center, 0, 0).inflate(5, 5)
cursor  = pygame.Rect(0, 0, 5, 5)

xh = np.array(scalpel.center)

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
        elif event.type == pygame.KEYUP:
            if event.key == ord('q'): # force quit with q button
                run = False
            '''*********** Student should fill in ***********'''
            # tele-impedance interface / switch controllers
            '''*********** Student should fill in ***********'''
    
    
    
    '''*********** Student should fill in ***********'''
    # main control code
    pm = np.array(pygame.mouse.get_pos())  # in VRenv frame
    scalpel.center = pm

	# previous endpoint position for velocity calculation
    p_prev = p.copy()

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
    
    '''*********** Student should fill in ***********'''
    # calculate force manipulability
    '''*********** Student should fill in ***********'''
    
    # real-time plotting
    window.fill((255,255,255)) # clear window
    # screenPatient.blit(hscalpel, (scalpel.topleft[0], scalpel.topleft[1]))
    ### scalpel visualisation
    screenPatient.blit(hscalpel, (scalpel.topleft[0], scalpel.topleft[1]))
    window.blit(screenPatient, (0, 0))
    '''*********** Student should fill in ***********'''
    # pygame.draw.circle(window, (0, 255, 0), (pm[0], pm[1]), 5) # draw reference position
    # pygame.draw.lines(window, (0, 0, 255), False, [(window_scale*x0+xc,-window_scale*y0+yc), (window_scale*x1+xc,-window_scale*y1+yc), (window_scale*x2+xc,-window_scale*y2+yc)], 6) # draw links
    # pygame.draw.circle(window, (0, 0, 0), (window_scale*x0+xc,-window_scale*y0+yc), 9) # draw shoulder / base
    # pygame.draw.circle(window, (0, 0, 0), (window_scale*x1+xc,-window_scale*y1+yc), 9) # draw elbow
    # pygame.draw.circle(window, (255, 0, 0), (window_scale*x2+xc,-window_scale*y2+yc), 5) # draw hand / endpoint
    #
    # force_scale = 50/(window_scale*(l1*l1)) # scale for displaying force vector
    # pygame.draw.line(window, (0, 255, 255), (window_scale*x2+xc,-window_scale*y2+yc), ((window_scale*x2+xc)+F[0]*force_scale,(-window_scale*y2+yc-F[1]*force_scale)), 2) # draw endpoint force vector
    #
    '''*********** Student should fill in ***********'''
    # visualise manipulability
    '''*********** Student should fill in ***********'''
    
    # print data
    text = font.render("FPS = " + str( round( clock.get_fps() ) ) + "   K = " + str( [K[0,0],K[1,1]] ) + " N/m" + "   xh = " + str( np.round(scalpel.center,3) ) + " m" + "   F = " + str( np.round(F,0) ) + " N", True, (0, 0, 0), (255, 255, 255))
    window.blit(text, textRect)
    
    pygame.display.flip() # update display
    
    
    
    # try to keep it real time with the desired step time
    clock.tick(FPS)
    
    if run == False:
        break

pygame.quit() # stop pygame












'''ANALYSIS'''

state = np.array(state)


plt.figure(3)
plt.subplot(411)
plt.title("VARIABLES")
plt.plot(state[:,0],state[:,1],"b",label="x")
plt.plot(state[:,0],state[:,2],"r",label="y")
plt.legend()
plt.ylabel("pr [m]")

plt.subplot(412)
plt.plot(state[:,0],state[:,3],"b")
plt.plot(state[:,0],state[:,4],"r")
plt.ylabel("p [m]")

plt.subplot(413)
plt.plot(state[:,0],state[:,7],"b")
plt.plot(state[:,0],state[:,8],"r")
plt.ylabel("F [N]")

plt.subplot(414)
plt.plot(state[:,0],state[:,9],"c")
plt.plot(state[:,0],state[:,10],"m")
plt.ylabel("K [N/m]")
plt.xlabel("t [s]")

plt.tight_layout()




plt.figure(4)
plt.title("ENDPOINT BEHAVIOUR")
plt.plot(0,0,"ok",label="shoulder")
plt.plot(state[:,1],state[:,2],"lime",label="reference")
plt.plot(state[:,3],state[:,4],"r",label="actual")
plt.axis('equal')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()

plt.tight_layout()




