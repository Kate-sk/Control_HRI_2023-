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

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    distance_ = distance.cdist([node], nodes).min()
    return [nodes[closest_index],distance_] #gives the coordinates of the line which is the closest to the mouse & the distance between the two


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
draw_time2 = [0] 


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
        xh[0] = np.round(-xh[0]+300)
        xh[1] = np.round(xh[1]-60)
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

    fe += 300*F_env

    
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
        
    haptic.bottomleft = xh 
    
    ######### Graphical output #########
    ##Render the haptic surface
    screenPatient.fill(cWhite)
    
    
    ### Body and scalpel visualisation
    screenPatient.blit(body_im, (0, 0))
    screenPatient.blit(scalpel_im,(haptic.topleft[0],haptic.topleft[1]))
    pygame.draw.line(screenPatient, (0, 0, 0), (haptic.bottomleft),(xm))#2*k*(xm-xh)))
    
    needle.bottomleft = xh
    #screenPatient.blit(needle_im,(needle.topleft[0],needle.topleft[1]))
    
    
    #code to draw a continuous cut
    if moving == True:
        if drawing == True:
            mouse_pos = xh
            if last is not None:
                print('last = ', last)
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
    if len(pm_arr) != 0:
        for i in range(len(pm_arr)):
            '''
            print('\ni = ', i)

            #print('\npc_arr = ', pc_arr)
            print('\nlength pc_arr = ', len(pc_arr))
            print('pc_arr[i] = ', pc_arr[i])

            print('\nlength pm_arr = ', len(pm_arr))
            print('pm_arr[i] = ', pm_arr[i])
            '''
            #k = i-1
            pygame.draw.line(screenPatient,(255,0,0),pc_arr[i], pm_arr[i], 2)
    
    


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

