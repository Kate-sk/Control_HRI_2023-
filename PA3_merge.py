# -*- coding: utf-8 -*-
'''
Name student: Fleur Kleene
Student number: 5637104
PA1a
'''

"""
Control in Human-Robot Interaction Assignment 3
-------------------------------------------------------------------------------
DESCRIPTION:
Creates a simulated haptic device (left) and VR environment (right)

The forces on the virtual haptic device are displayed using pseudo-haptics. The 
code uses the mouse as a reference point to simulate the "position" in the 
user's mind and couples with the virtual haptic device via a spring. the 
dynamics of the haptic device is a pure damper, subjected to perturbations 
from the VR environment. 

IMPORTANT VARIABLES
xc, yc -> x and y coordinates of the center of the haptic device and of the VR
xm -> x and y coordinates of the mouse cursor 
xh -> x and y coordinates of the haptic device (shared between real and virtual panels)
fe -> x and y components of the force fedback to the haptic device from the virtual impedances

TASKS:
1- Implement the impedance control of the haptic device
2- Implement an elastic element in the simulated environment
3- Implement a position dependent potential field that simulates a bump and a hole
4- Implement the collision with a 300x300 square in the bottom right corner 
5- Implement the god-object approach and compute the reaction forces from the wall

REVISIONS:
Initial release MW - 14/01/2021
Added 2 screens and Potential field -  21/01/2021
Added Collision and compressibility (LW, MW) - 25/01/2021
Added Haptic device Robot (LW) - 08/02/2022

INSTRUCTORS: Michael Wiertlewski & Laurence Willemet & Mostafa Attala
e-mail: {m.wiertlewski,l.willemet,m.a.a.atalla}@tudelft.nl
"""

import pygame
import numpy as np
import math
import matplotlib.pyplot as plt
#from pantograph import Pantograph
from pyhapi import Board, Device, Mechanisms
from pshape import PShape
import sys, serial, glob
from serial.tools import list_ports
import time
from scipy.spatial import distance
from numpy import ones,vstack
from numpy.linalg import lstsq
from tabulate import tabulate
import pandas as pd

#use this if you want to save the data in a specific path
#else just type the file name as for example "Sample.csv"
path1 = "/Users/pltangkau/Desktop/Python/Results/PP1/Con2_perturbation/Trial1.csv"


def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    distance_ = distance.cdist([node], nodes).min()
    return [nodes[closest_index],distance_,closest_index] #gives the coordinates of the line which is the closest to the mouse & the distance between the two

class Vector: #creates vectors
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    def norm(self):
        return self.dot(self)**0.5
    def normalized(self):
        norm = self.norm()
        return Vector(self.x / norm, self.y / norm)
    def perp(self):
        return Vector(1, -self.x / self.y)
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    def __str__(self):
        #return f'({self.x}, {self.y})'
        return np.array(self.x,self.y)


##################### General Pygame Init #####################
##initialize pygame window
pygame.init()
window = pygame.display.set_mode((800, 600))   ##twice 600x400 for haptic and VR
pygame.display.set_caption('Virtual Haptic Device')

screenPatient = pygame.Surface((800,600)) # was screenHaptic in PA1
#screenVR = pygame.Surface((600,400))

body_im = pygame.image.load('body.png')
body_im = pygame.transform.scale(body_im,(800, 600))


##add nice icon from google
icon = pygame.image.load('scalpel.png')
pygame.display.set_icon(icon)

##add text on top to debugToggle the timing and forces
font = pygame.font.Font('freesansbold.ttf', 18)

pygame.mouse.set_visible(True)     ##Hide cursor by default. 'm' toggles it
 
##set up the on-screen debugToggle
text = font.render('Virtual Haptic Device', True, (0, 0, 0),(255, 255, 255))
textRect = text.get_rect()
textRect.topleft = (10, 10)


xc,yc = screenPatient.get_rect().center ##center of the screen


##initialize "real-time" clock
clock = pygame.time.Clock()
FPS = 100   #in Hertz

##define some colors
cWhite = (255,255,255)
cDarkblue = (36,90,190)
cLightpurple = (147,112,219)
cRed = (255,0,0)
cOrange = (255,127,80)
cYellow = (255,255,0)
cBlack = (0,0,0)


##################### Init Simulated haptic device #####################

# SIMULATION PARAMETERS (from PA2)
dt = 0.01 # intergration step timedt = 0.01 # integration step time
dts = dt*1 # desired simulation step time (NOTE: it may not be achieved)

####Virtual environment -  Wall/height map.. etc.
T = np.arange(0.1, 8, 100)
F_env =(T/10)**2*9.81/(2*np.pi)
plt.plot(T, (T/10)**2*9.81/(2*np.pi))
plt.show()


####Pseudo-haptics dynamic parameters, k/b needs to be <1
K1 = .5       ##Stiffness between cursor and haptic display
b = .8       ##Viscous of the pseudohaptic display

##################### Define sprites #####################

##define sprites
scalpel_im = pygame.image.load('scalpel.png') # was hhandle in PA1
#haptic  = pygame.Rect(*screenPatient.get_rect().center, 0, 0).inflate(48, 48)
cursor  = pygame.Rect(0, 0, 5, 5)
colorHaptic = cOrange ##color of the wall

#from Kate
scalpel_scale = (50,50)
scalpel_im = pygame.transform.scale(scalpel_im, scalpel_scale)
haptic = pygame.Rect(*screenPatient.get_rect().center, 0, 0).inflate(scalpel_scale[0], scalpel_scale[1])
CUT = False # are we cutting?
pc_arr = [];
xs = np.array(haptic.center) # scalpel center

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

# draw reference lines for the sewing
line_s = Vector(ref_cut[0,0],ref_cut[0,1]) #line start coordinate
line_e = Vector(ref_cut[1,0],ref_cut[1,1]) 
line_v = line_e-line_s #vector of the line
perp_normed = line_v.perp().normalized() #normalized perpendicular vector from the line
p_dist = 10 #distance of the points from the reference line
step_point = round(121/6)

p1 = line_e + perp_normed*p_dist
p2 = line_e - perp_normed*p_dist
p3 = Vector(line_coor[121-step_point][0],line_coor[121-step_point][1]) + perp_normed*p_dist
p4 = Vector(line_coor[121-step_point][0],line_coor[121-step_point][1]) - perp_normed*p_dist
p5 = Vector(line_coor[121-step_point*2][0],line_coor[121-step_point*2][1]) + perp_normed*p_dist
p6 = Vector(line_coor[121-step_point*2][0],line_coor[121-step_point*2][1]) - perp_normed*p_dist
p7 = Vector(line_coor[121-step_point*3][0],line_coor[121-step_point*3][1]) + perp_normed*p_dist
p8 = Vector(line_coor[121-step_point*3][0],line_coor[121-step_point*3][1]) - perp_normed*p_dist
p9 = Vector(line_coor[121-step_point*4][0],line_coor[121-step_point*4][1]) + perp_normed*p_dist
p10 = Vector(line_coor[121-step_point*4][0],line_coor[121-step_point*4][1]) - perp_normed*p_dist
p11 = Vector(line_coor[121-step_point*5][0],line_coor[121-step_point*5][1]) + perp_normed*p_dist
p12 = Vector(line_coor[121-step_point*5][0],line_coor[121-step_point*5][1]) - perp_normed*p_dist
p13 = Vector(line_coor[121-121][0],line_coor[121-121][1]) + perp_normed*p_dist
p14 = Vector(line_coor[121-121][0],line_coor[121-121][1]) - perp_normed*p_dist
points = np.array([[p2.x,p2.y],[p3.x,p3.y],[p4.x,p4.y],[p5.x,p5.y],[p6.x,p6.y],
                   [p7.x,p7.y],[p8.x,p8.y],[p9.x,p9.y],[p10.x,p10.y],[p11.x,p11.y],
                   [p12.x,p12.y],[p13.x,p13.y]]) #make an array with the coordinates of all the points for the performance metric


def poly(points3,points3x,points3y):
    x_coords, y_coords = zip(*points3)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    polynomial = np.poly1d([m,c])
    x_axis = np.linspace(points3x,points3y,20)
    y_axis = polynomial(x_axis)
    line_coor2 = np.transpose(np.vstack((x_axis, y_axis)))
    return line_coor2


#calculate reference lines for in between the points
points_13_12 = [(p13.x,p13.y),(p12.x,p12.y)]
points_11_10 = [(p11.x,p11.y),(p10.x,p10.y)]
points_9_8 = [(p9.x,p9.y),(p8.x,p8.y)]
points_7_6 = [(p7.x,p7.y),(p6.x,p6.y)]
points_5_4 = [(p5.x,p5.y),(p4.x,p4.y)]
points_3_2 = [(p3.x,p3.y),(p2.x,p2.y)]

p13_line_coor = poly(points_13_12,p13.x,p12.x)
p11_line_coor = poly(points_11_10,p11.x,p10.x)
p9_line_coor = poly(points_9_8,p9.x,p8.x)
p7_line_coor = poly(points_7_6,p7.x,p6.x)
p5_line_coor = poly(points_5_4,p5.x,p4.x)
p3_line_coor = poly(points_3_2,p3.x,p2.x)

xh = np.array(haptic.center)

##Set the old value to 0 to avoid jumps at init
xhold = 0
xmold = 0


##################### Init Virtual env. #####################


##################### Detect and Connect Physical device #####################
# USB serial microcontroller program id data:
def serial_ports():
    """ Lists serial port names """
    ports = list(serial.tools.list_ports.comports())

    result = []
    for p in ports:
        try:
            port = p.device
            s = serial.Serial(port)
            s.close()
            if p.description[0:12] == "Arduino Zero":
                result.append(port)
                print(p.description[0:12])
        except (OSError, serial.SerialException):
            pass
    return result


CW = 0
CCW = 1

haplyBoard = Board
device = Device
SimpleActuatorMech = Mechanisms
#pantograph = Pantograph
robot = PShape
   

#########Open the connection with the arduino board#########
port = serial_ports()   ##port contains the communication port or False if no device
if port:
    print("Board found on port %s"%port[0])
    haplyBoard = Board("test", port[0], 0)
    device = Device(5, haplyBoard)
    #pantograph = Pantograph()
    #device.set_mechanism(pantograph)
    
    device.add_actuator(1, CCW, 2)
    device.add_actuator(2, CW, 1)
    
    device.add_encoder(1, CCW, 241, 10752, 2)
    device.add_encoder(2, CW, -61, 10752, 1)
    
    device.device_set_parameters()
else:
    print("No compatible device found. Running virtual environnement...")
    #sys.exit(1)
    

# initial conditions
t = 0.0 # time
pm = np.zeros(2) # mouse position
pr = np.zeros(2) # reference endpoint position
p = np.array([0.1,0.1]) # actual endpoint position
dxh = np.zeros(2) # actual endpoint velocity
fe = np.zeros(2) # endpoint force
#q = np.zeros(2) # joint position
#p_prev = np.zeros(2) # previous endpoint position
m = 0.5 # endpoint mass -- scalpel mass 22g  + robot's end effector
i = 0 # loop counter
state = [] # state vector


 # conversion from meters to pixels
window_scale = 3 
''' 800 ??\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\'''


#first task -- drawing cut line 
drawing = False
pm_arr = []
last = None
moving = False
time_task1 = []
pm_task1_array = []

#second task
task2 = False
i2 = 0
dis_line_task2 = []
time_task2 = []
time_task22 = []
tot_t = []
inter = []
pm_task2_array = []


# wait until the start button is pressed
run = True
while run:
    for event in pygame.event.get(): # interrupt function
        if event.type == pygame.KEYUP:
            if event.key == ord('e'): # enter the main loop after 'e' is pressed
                run = False

##################### Main Loop #####################
##Run the main loop


#run = True
#ongoingCollision = False
fieldToggle = True
robotToggle = True

debugToggle = False

run = True

while run:
        
    #########Process events  (Mouse, Keyboard etc...)#########
    for event in pygame.event.get():
        ##If the window is close then quit 
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYUP:
            if event.key == ord('m'):   ##Change the visibility of the mouse
                pygame.mouse.set_visible(not pygame.mouse.get_visible())  
            if event.key == ord('q'):   ##Force to quit
                run = False            
            if event.key == ord('d'):
                debugToggle = not debugToggle
            if event.key == ord('r'):
                robotToggle = not robotToggle
            if event.key == ord('s'):
                task2 = True
        else:
            if event.type == pygame.MOUSEMOTION:
                moving = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                CUT = True
            elif event.type == pygame.MOUSEBUTTONUP:
                CUT = False
            else:
                moving = False
                
                    # Add toggle for turning environmental force on/off?

    ######### Read position (Haply and/or Mouse)  #########
    
    ##Get endpoint position xh
    if port and haplyBoard.data_available():    ##If Haply is present
        #Waiting for the device to be available
        #########Read the motorangles from the board#########
        device.device_read_data()
        motorAngle = device.get_device_angles()
        
        #########Convert it into position#########
        device_position = device.get_device_position(motorAngle)
        xh = np.array(device_position)*1e3*window_scale
        #xh[0] = np.round(-xh[0]+300)
        #xh[1] = np.round(xh[1]-60)
        
        #xh[0] = np.round(-xh[0]+300)
        #xh[1] = np.round(xh[1])
        xm = xh     ##Mouse position is not used
         
    else:
        ##Compute distances and forces between blocks
        xh = np.clip(np.array(haptic.center),0,799)
        xh = np.round(xh)
        
        ##Get mouse position
        cursor.center = pygame.mouse.get_pos()
        xm = np.clip(np.array(cursor.center),0,799)
    
    
    ######### Compute forces ########
    K = 100
    d = 1.5                  # Damping (Ns/m)
    
    fe[0] = K* (xm[0] - xh[0]) - d*dxh[0]  # Compute interaction force (F (N))
    fe[1] = K * (xm[1] - xh[1]) - d*dxh[1]  # Compute interaction force (F (N))
   
    F_env = np.zeros(2)         # Forces caused by ocean etc, maybe change it so the 'body' moves?
    F_env[0] = np.sin(2*t+1)**3-np.cos(5*t)**2 - np.sin(3*t - 1)**5+np.cos(3*t)/2
    F_env[1] = 2*np.cos(3*t-1)**2 + np.sin(t-2) - 2* np.sin(2*t-2)**2 - np.cos(t/3)**2 /2 

    fe += 300*F_env #do it with and without
    
    tot_t.append(t)
    
    # for event in pygame.event.get():
    #     if event.type == pygame.MOUSEMOTION:
    #         if any(event.buttons):
    #             last = (event.pos[0] - event.rel[0], event.pos[1] - event.rel[1])
    #             pygame.draw.line(screenPatient,(255,0,0),last, event.pos, 10)
    
    ##Update old samples for velocity computation
    xhold = xh
    xmold = xm
    dxhold = dxh
    
    state.append([t, xm[0], xm[1], xh[0], xh[1], dxh[0], dxh[1], fe[0], fe[1], K])
    

    ######### Send forces to the device #########
    if port:
        fe[1] = -fe[1]  ##Flips the force on the Y=axis 
        
        ##Update the forces of the device
        device.set_device_torques(fe)
        device.device_write_torques()
        #pause for 1 millisecond
        time.sleep(0.001)
    else:
        ######### Update the positions according to the forces ########
        ##Compute simulation (here there is no inertia)
        ##If the haply is connected xm=xh and dxh = 0

        ddxh = fe/m
        dxh = dxh + ddxh*dt
        xh = xh + dxh*dt
        t += dt
    
    #I tried to get the mouse and haptic on the same position while in rest
    xh[0] = xh[0] - 7
    xh[1] = xh[1] + 8
    
    haptic.bottomleft = xh 
    
    ######### Graphical output #########
    ##Render the haptic surface
    screenPatient.fill(cWhite)
    
    
    ### Body and scalpel visualisation
    screenPatient.blit(body_im, (0, 0))
    if task2 == False:
        screenPatient.blit(scalpel_im,(haptic.topleft[0],haptic.topleft[1]))
    else: 
        screenPatient.blit(needle_im,(needle.topleft[0],needle.topleft[1]))

    pygame.draw.line(screenPatient, cDarkblue, (haptic.bottomleft),(xm))#2*k*(xm-xh)))
    
    needle.bottomleft = xh
    #screenPatient.blit(needle_im,(needle.topleft[0],needle.topleft[1]))
    
    #code to draw a continuous cut
    if moving == True:
        if drawing == True:
            mouse_pos = xh                   #use endpoint position for the drawing
            if last is not None:
                pc_arr.append(last)          #safe 'last' data points for all cuts
                pm_arr.append(mouse_pos)     #safe 'virtual endpoint' data points all cuts
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
    
    # draw the reference line for the cut
    if task2 == True:
        #pygame.draw.circle(screenPatient,(0,0,0),(p1.x,p1.y),3)
        pygame.draw.circle(screenPatient,(0,0,0),(p2.x,p2.y),3)
        pygame.draw.circle(screenPatient,(0,0,0),(p3.x,p3.y),3)
        pygame.draw.circle(screenPatient,(0,0,0),(p4.x,p4.y),3)
        pygame.draw.circle(screenPatient,(0,0,0),(p5.x,p5.y),3)
        pygame.draw.circle(screenPatient,(0,0,0),(p6.x,p6.y),3)
        pygame.draw.circle(screenPatient,(0,0,0),(p7.x,p7.y),3)
        pygame.draw.circle(screenPatient,(0,0,0),(p8.x,p8.y),3)    
        pygame.draw.circle(screenPatient,(0,0,0),(p9.x,p9.y),3)
        pygame.draw.circle(screenPatient,(0,0,0),(p10.x,p10.y),3)
        pygame.draw.circle(screenPatient,(0,0,0),(p11.x,p11.y),3)
        pygame.draw.circle(screenPatient,(0,0,0),(p12.x,p12.y),3)
        pygame.draw.circle(screenPatient,(0,0,0),(p13.x,p13.y),3)
        #pygame.draw.circle(screenPatient,(0,0,0),(p14.x,p14.y),3)
        
        #can be turned on to see 'perfect' sewing lines!
        #pygame.draw.line(screenPatient,(255,0,0),p13_line_coor[0], p13_line_coor[-1], 2)
        #pygame.draw.line(screenPatient,(255,0,0),p11_line_coor[0], p11_line_coor[-1], 2)
        #pygame.draw.line(screenPatient,(255,0,0),p9_line_coor[0], p9_line_coor[-1], 2)
        #pygame.draw.line(screenPatient,(255,0,0),p7_line_coor[0], p7_line_coor[-1], 2)
        #pygame.draw.line(screenPatient,(255,0,0),p5_line_coor[0], p5_line_coor[-1], 2)
        #pygame.draw.line(screenPatient,(255,0,0),p3_line_coor[0], p3_line_coor[-1], 2)

    
    #the code to actually draw the line
    if len(pm_arr) != 0:
        for i in range(len(pm_arr)):
            pygame.draw.line(screenPatient,(255,0,0),pc_arr[i], pm_arr[i], 2)
    
    
    #performance of task 1
    if pm_arr != [] and task2 == False and drawing == True:
        #z = pm_arr[-1] #get the mouse position of the line drawing
        pm_task1_array.append(xh)
        z = xh
        perf_pm = [z[0],z[1]] #place it in an array
        z = closest_node(perf_pm, line_coor) #use the function to calculate the point closest to the mouse during cutting and the distance between them
        p_line_close_pm = z[0] #point on line which is closest to the pm during cutting
        dis_line_pm.append(z[1]) #distance between those points
        #pygame.draw.circle(screenPatient,(0,0,0),(p_line_close_pm[0],p_line_close_pm[1]),3) #visualize the point on the line which is closest to the mouse
        #calculate performance?
        mean_dis = sum(dis_line_pm)/len(dis_line_pm) #mean distance = accuracy?
        #speed = draw_time/t #--> gives the time it took to draw a line relative to the total time?
        
        #visualize how well the person does it?
        if dis_line_pm[-1] <= 3: #think of some good measure?
            color_acc = (0,255,0) #green if good
        elif dis_line_pm[-1] > 3 and dis_line_pm[-1] <= 6:
            color_acc = (255,100,10)
        else:
            color_acc = (255,0,0) #red if bad lol
        pygame.draw.rect(screenPatient,color_acc,[0,0,100,100],0)
        time_task1.append(t)

    #performance of task 2
    if pm_arr != [] and task2 == True:
        time_task2.append(t)
        if drawing == True:
            #inter.append([pm_arr[-1][0],pm_arr[-1][1]])
            inter.append(xh)
        elif drawing == False:
            inter = []

    if pm_arr != [] and task2 == True and drawing == True:
        time_task22.append(t)
        #pm_task2 = pm_arr[-1][0],pm_arr[-1][1]    #haptic position
        pm_task2 = xh
        pm_task2_array.append(xh)
        #find out 
        z = closest_node(inter[0], points) #use the function to calculate the point closest to the mouse during cutting and the distance between them
        #right_point = z[2] #0 = p2, 11 = p13        #so which sew point is the closest to the mouse, use this to calculate performance towards the right line
        z3 = z[2]       
        #line1 = p13-p12
        if z3 == 10 or z3 == 11: #check if the haptic is closest to either one of the endpoints of the first line
            z2 = closest_node(pm_task2, p13_line_coor) #calculate the distance between haptic and the first line
            dis_line_task2.append(z2[1]) #add the distance to the performance line
        #line2 = p11-p10
        if z3 == 8 or z3 == 9: #check if the mouse is closest to either one of the endpoints of the second line
            z2 = closest_node(pm_task2, p11_line_coor)
            dis_line_task2.append(z2[1])
        #line3 = p9-p8
        if z3 == 6 or z3 == 7:
            z2 = closest_node(pm_task2, p9_line_coor)
            dis_line_task2.append(z2[1])
        #line4 = p7-p6
        if z3 == 4 or z3 == 5:
            z2 = closest_node(pm_task2, p7_line_coor)
            dis_line_task2.append(z2[1])    
        #line5 = p5-p4
        if z3 == 2 or z3 == 3:
            z2 = closest_node(pm_task2, p5_line_coor)
            dis_line_task2.append(z2[1])
        #line6 = p9-p8
        if z3 == 0 or z3 == 1:
            z2 = closest_node(pm_task2, p3_line_coor)
            dis_line_task2.append(z2[1])
            
    ##Fuse it back together
    window.blit(screenPatient, (0,0))
    #window.blit(screenVR, (600,0))

    ##Print status in  overlay
    if debugToggle: 
        
        text = font.render("FPS = " + str(round(clock.get_fps())) + \
                            "  xm = " + str(np.round(10*xm)/10) +\
                            "  xh = " + str(np.round(10*xh)/10) +\
                            "  fe = " + str(np.round(10*fe)/10) \
                            , True, (0, 0, 0), (255, 255, 255))
        window.blit(text, textRect)


    pygame.display.flip()    
    ##Slow down the loop to match FPS
    clock.tick(FPS)

pygame.display.quit()
pygame.quit()


'''Performance Analysis'''
#t1 = time start drawing until you stop drawing
#t2 = time start drawing until the last point you drew

#cut the time to stop at the appropriate time
#get the index of the last data point you need
cut_t = np.where(np.array(time_task2) == time_task22[-1])

 
t1 = time_task1[-1]-time_task1[0]
t2 = time_task2[-1]-time_task2[0]

#distance to the 'perfect' line in pixels
perf1 = np.array(dis_line_pm)
perf2 = np.array(dis_line_task2)

#mean distance to 'perfect' line
mean_perf1 = np.mean(dis_line_pm)
mean_perf2 = np.mean(dis_line_task2)

#std
std_perf1 = np.std(dis_line_pm)
std_perf2 = np.std(dis_line_task2)

#rms
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

zeros_task1 = np.zeros((len(dis_line_pm),1))
zeros_task2 = np.zeros((len(dis_line_task2),1))

rms_perf1 = rmse(zeros_task1,dis_line_pm)
rms_perf2 = rmse(zeros_task2,dis_line_task2)

#array with the results
#mean - std - rms - time
#results = np.array([[mean_perf1,mean_perf2],[std_perf1,std_perf2],[rms_perf1,rms_perf2],[t1,t2]])

#table with results for a quick review
results_table2 = [['','Task 1','Task 2'],
                 ['Mean',mean_perf1,mean_perf2],
                 ['Std',std_perf1,std_perf2],
                 ['RMS',rms_perf1,rms_perf2],
                 ['Time',t1,t2]]

print(tabulate(results_table2))

#make a table for the average values for an individual trial
results_table = {'':['Mean','Std','RMS','Time'],
                  'Task 1': [mean_perf1,std_perf1,rms_perf1,t1],
                  'Task 2': [mean_perf2,std_perf2,rms_perf2,t2]}

#make a table for the performance 
results_perf = {'Perf 1': perf1,
                'Perf 2': perf2}


#table with the first table and second table in one
results_tot = {'':['Mean','Std','RMS','Time'],
                  'Task 1': np.array([mean_perf1,std_perf1,rms_perf1,t1]),
                  'Task 2': np.array([mean_perf2,std_perf2,rms_perf2,t2]),
                  'Perf 1': perf1,
                  'Perf 2': perf2,
                  'Time 1': time_task1,
                  'Time 2': time_task2,
                  'Trajectory 1x': np.array(pm_task1_array)[:,0],
                  'Trajectory 1y': np.array(pm_task1_array)[:,1],
                  'Trajectory 2x': np.array(pm_task2_array)[:,0],
                  'Trajectory 2y': np.array(pm_task2_array)[:,1]}

#make the arrays fitting to dataframe, since their size vary
df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in results_tot.items()]))
#save the data in a csv file in the path1 of choice
df.to_csv(path1, index = False, header = True, sep=';', decimal=",")

#plot together the performance - accuracy against time
plt.plot(time_task1,perf1, label="task 1")
plt.plot(time_task22,perf2, label="task 2")
plt.ylabel("Accuracy [pixels]")
plt.xlabel("t [s]")
plt.title("The performance for task 1 and 2 against time")
plt.legend(loc="upper right")
plt.show()

#plot the trajectories next to the 'perfect lines'
plt.plot(ref_cut[:,0],ref_cut[:,1], color = 'k', label="target")
plt.plot(np.array(pm_task1_array)[:,0],np.array(pm_task1_array)[:,1], label="performance")
plt.ylabel("Y [pixels]")
plt.xlabel("X [pixels]")
plt.title("Trajectory task 1")
plt.legend(loc="upper right")
plt.show()

plt.plot([p13.x,p12.x],[p13.y,p12.y], color = 'k', label="target")
plt.plot([p11.x,p10.x],[p11.y,p10.y], color = 'k')
plt.plot([p9.x,p8.x],[p9.y,p8.y], color = 'k')
plt.plot([p7.x,p6.x],[p7.y,p6.y], color = 'k')
plt.plot([p5.x,p4.x],[p5.y,p4.y], color = 'k')
plt.plot([p3.x,p2.x],[p3.y,p2.y], color = 'k')
plt.plot(np.array(pm_task2_array)[:,0],np.array(pm_task2_array)[:,1], label="performance")
plt.ylabel("Y [pixels]")
plt.xlabel("X [pixels]")
plt.title("Trajectory task 2")
plt.legend(loc="upper right")
plt.show()

'''ANALYSIS'''

state = np.array(state)

plt.figure(3)
plt.subplot(411)
plt.title("VARIABLES")
plt.plot(state[:,0],state[:,1],"b",label="x")
plt.plot(state[:,0],state[:,2],"r",label="y")
plt.legend()
plt.ylabel("xm [m]")

plt.subplot(412)
plt.plot(state[:,0],state[:,3],"b")
plt.plot(state[:,0],state[:,4],"r")
plt.ylabel("xh [m]")

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

