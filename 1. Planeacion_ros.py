#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries.
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import time
import random
from IPython.display import clear_output
from PIL import Image
import rospy
from std_msgs.msg import String
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
import roslib



get_ipython().run_line_magic('matplotlib', 'inline')


# ### Ambiente

# In[2]:


class GridEnvironment():
    
    def __init__(self, fin_x, fin_y, route_img):
        self.fin_x = fin_x
        self.fin_y = fin_y
        
        
        # RECOMPENSA
        self.rw = -1 # Living (Movement) Penalty
        
        # CARGA IMAGEN
        im = Image.open(route_img, 'r')
        
        # DIMENSIONES
        self.columns = im.size[0]
        self.rows = im.size[1]
        
        # INICIALIZACION
        pix_val = list(im.getdata())
        self.new_pix = [x[0] for x in pix_val]
        self.walls_and_paths = np.ones([self.rows,self.columns])
        for i in range(self.rows):
            for j in range(self.columns):
                if self.new_pix[i*self.columns+j] == 0:
                    self.walls_and_paths[i,j] = 0
        self.rewards = self.rw*np.ones([self.rows,self.columns])
        
        # Cambiar el valor
        self.value_state_table = np.zeros([self.rows,self.columns])
        self.value_state_table[fin_x][fin_y]=1000
        
        
        

    def dibujo_ruta (self,array_pos):
        matrixMap = np.ones([self.rows,self.columns])
        RGBMap = []

        for i in range(self.rows):
            for j in range(self.columns):
                matrixMap[i,j] = self.walls_and_paths[i,j]

        for i in range(len(array_pos[:,0])):
            matrixMap[array_pos[i,1],array_pos[i,0]] = 2

        for i in range(self.rows):
            file = []
            for j in range(self.columns):
                if matrixMap[i,j] == 1:
                    RGB = [255,255,255,255]                     #RGBA de color blanco
                elif matrixMap[i,j] == 0:
                    RGB = [0,0,0,255]                           #RGBA de color negro
                else:
                    RGB = [255,0,0,255]                         #RGBA de color rojo
                file.append(RGB)
            RGBMap.append(file)

        RGBMap = np.array(RGBMap).astype(np.uint8)    #El módulo Image necesita los array definidos de esta forma.
        routeMap = Image.fromarray(RGBMap, "RGBA")
        routeMap.save("MapaRuta.png")
        
        return routeMap
    
    
    def getStateValue(self, position):
        return self.value_state_table[position[0]][position[1]]
    
    def getValue_state_table(self):
        return self.value_state_table
    
    def reset(self, fin_x, fin_y):
        self.value_state_table = np.zeros([self.rows,self.columns])
        self.value_state_table[fin_x][fin_y]=100


# ### Agente

# In[3]:


class valueBasedAgent():
    
    def __init__(self, environment, policy, discount_factor):
        self.pos = [0,0]
        self.total_reward = 0
        self.environment = environment
        self.discount_factor = discount_factor
        self.actions = ["up", "down", "left", "right"]
        
        # Start with a random policy. 0.25 chance of moving to any direction.
        self.policy = policy   
            
    def forwardState(self, pos, action):
        
        # New position array.
        tam_y = len(self.environment.value_state_table[0])-1
        tam_x = len(self.environment.value_state_table)-1
        new_position = pos
        
        # Compute new position based on action taken.
        if(action == "up" and pos[1] < tam_y ):
            if(self.environment.walls_and_paths[pos[0]][pos[1] + 1]) == 1:
                new_position = [pos[0], pos[1] + 1]

        elif(action == "down" and pos[1] > 0):
            if(self.environment.walls_and_paths[pos[0]][pos[1] - 1]) == 1:
                new_position = [pos[0], pos[1] - 1]
                
        elif(action == "left" and pos[0] > 0):
            if(self.environment.walls_and_paths[pos[0] - 1][pos[1]]) == 1:
                new_position = [pos[0] - 1, pos[1]]

        elif(action == "right" and pos[0] < tam_x):
            if(self.environment.walls_and_paths[pos[0] + 1][pos[1]]) == 1:
                new_position = [pos[0] + 1, pos[1]]
        return new_position
        
        
    def valueFunction(self):
            
        # Initialize variable.
        new_state_value = 0
    
        # Random movement! - Cuando aun no se ha inicializado 
        if self.policy[self.pos[0]][self.pos[1]] == "r":
            for action in self.actions:        
                forward_state = self.forwardState(self.pos, action)
                
                # Simplified version of Q-value. BELLMANS EQUATION
                q_value = (self.environment.rewards[forward_state[0]][forward_state[1]] 
                                    + self.discount_factor * self.environment.value_state_table[forward_state[0]][forward_state[1]])
                new_state_value += q_value * 0.25 # Probabilidad de 0.25 para cada una de las acciones
            return new_state_value
        
        # Not random movement!
        else: 
            action = self.policy[self.pos[0]][self.pos[1]]
            forward_state = self.forwardState(self.pos, action)
            
            # Simplified version of Q-value.
            q_value = (self.environment.rewards[forward_state[0]][forward_state[1]] 
                                    + self.discount_factor * self.environment.value_state_table[forward_state[0]][forward_state[1]])
            new_state_value += q_value # Probabilidad de 1
            return new_state_value
        
    def getPosition(self):
        return self.pos
    
    def getReward(self):
        return self.total_reward
    
    def setPosition(self, x, y):
        self.pos = [x, y]
        
    def updateValueStateTable(self):
        new_state_value = self.valueFunction()
        self.environment.value_state_table[self.pos[0]][self.pos[1]] = new_state_value
        
    def selectBestAction(self):
        
        # Compute new possible states.
        go_up = self.forwardState(self.pos, "up")
        go_down = self.forwardState(self.pos, "down")
        go_left = self.forwardState(self.pos, "left")
        go_right = self.forwardState(self.pos, "right")
        
        # Q values (simplified version).
        up_value = (self.environment.rewards[go_up[0]][go_up[1]] + 
                    self.discount_factor * self.environment.value_state_table[go_up[0]][go_up[1]])
        down_value = (self.environment.rewards[go_down[0]][go_down[1]] + 
                      self.discount_factor * self.environment.value_state_table[go_down[0]][go_down[1]])
        left_value = (self.environment.rewards[go_left[0]][go_left[1]] + 
                        self.discount_factor * self.environment.value_state_table[go_left[0]][go_left[1]])
        right_value = (self.environment.rewards[go_right[0]][go_right[1]] + 
                       self.discount_factor * self.environment.value_state_table[go_right[0]][go_right[1]])
        
        # Array of Q-values.
        values = [up_value, down_value, left_value, right_value]
        
        best_action = self.actions[values.index(max(values))] 
        return best_action       
            
    def move(self):
    
        # Select action according to policy.
        action = self.policy[self.pos[0]][self.pos[1]]

        # Move to new position according to action taken.
        self.pos = self.forwardState(self.pos, action)
        


# ### Politica

# In[4]:


class PolicyEvaluation():
    
    def __init__(self,x_fin,y_fin, environment, agent, iterations = 3):
        
        self.x_fin = x_fin
        self.y_fin = y_fin
        self.environment = environment       
        self.agent = agent
        self.iterations = iterations
    
    def evaluate(self, plot_grid = True):
        self.DP_policy_evaluation(self.iterations, plot_grid)
        
    def DP_policy_evaluation(self, iterations, plot_grid):
        
        for k in range(0, iterations):
            print("Iteracion #",k)
            for i in range(0, len(self.environment.value_state_table)):
                for j in range(0, len(self.environment.value_state_table[0])):

                    if self.environment.walls_and_paths[i][j] == 1 and self.canChangeStateValue(i, j):

                        # Set agent position.
                        self.agent.setPosition(i, j)
                        self.agent.updateValueStateTable()
                            
    

    def canChangeStateValue(self, x, y):
        # Posicion que no se puede modificar
        cant_modify = bool((x == self.x_fin and y == self.y_fin)) # or (x == 4 and y == 4))
        
        grid = self.environment.walls_and_paths
        coords = list()
        
        # Get walls.
        for i in range(0, len(grid)):
            for j in range(0, len(grid[0])):
                if grid[i][j] == 0:
                    coords.append([i, j])
        for c in coords: 
            if c == [x, y]:
                cant_modify = True
                break
                
        return not cant_modify
    
    def updatePolicy(self):
        
         for i in range(0, len(self.environment.value_state_table)):
                for j in range(0, len(self.environment.value_state_table[0])):
                    if self.environment.walls_and_paths[i][j] == 1:
                        
                        # Set agent position.
                        self.agent.setPosition(i, j)
                        best_action = self.agent.selectBestAction()
                        self.agent.policy[i][j] = best_action


# In[5]:


class Game():
    
    def __init__(self,x_fin,y_fin, environment, agent):
        
        self.environment = environment       
        self.agent = agent             
        print("GridWorld Initialize!")
                
    def update(self, secs):
        
        route = []
        pos = self.agent.getPosition()
        while not ((self.agent.pos[0] == x_fin and self.agent.pos[1] == y_fin)):
            
            self.agent.move()
            pos = self.agent.getPosition()
            
            route.append(pos)
        
        print("array_pos", route)
        return route


# ### Parte Principal

# In[14]:




# Initialize environment and agent.
x_fin = 0
y_fin = 0

x_ini = 0
y_ini = 0

x_len, y_len = 0,0
discount_factor = 1
ruta_imagen="imagen1.png"

centinela_inicio_fin= False


def Inicio_Fin(coordenadas):
    x_ini = coordenadas.inicio.x    #toca crear el mensaje
    y_ini = coordenadas.inicio.y    #toca crear el mensaje
    x_fin = coordenadas.final.x     #toca crear el mensaje
    y_fin = coordenadas.final.y     #toca crear el mensaje
    #cambiar las coordenadas de metros a pixeles
    
    centinela_inicio_fin = True
    

def publisher_ROS(array_pos):
    rospy.init_node('node', anonymous=True)
    rospy.Publisher("algo", queue_size=0)

# Generate the random policy.
def run_policy(size):
    policy = list()
    for i in range(0, size[0]):
        column = list()
        for j in range(0, size[1]):
            column.append("r")
        policy.append(column)
    return policy

def correct_policy(policy):
    politica = list()
    for i in range(len(policy)):
        row = list()
        for j in range(len(policy[0])):
            if policy[i,j] == 'u':
                row.append("up")
            if policy[i,j] == 'd':
                row.append("down")
            if policy[i,j] == 'r':
                row.append("right")
            if policy[i,j] == 'l':
                row.append("left")
        politica.append(row)
    return politica




        
#function that debugs the position array and returns only the desired coordinates
def coordenates(route, route_img, x_len, y_len):  
    im = Image.open(route_img, 'r')
    x_center = im.size[1] /2
    y_center = im.size[0] / 2
    x_scale = x_len  / im.size[1] 
    y_scale = y_len  / im.size[0]
    final_coordenates=[]   
    for i in range (1,len (route) -1):      
        if route [i][0] == route [i-1][0] :
            if route [i][0] != route [i+1][0] :
                new_coord = []
                new_coord.append ( (route [i][0] - x_center) * x_scale )
                new_coord.append (( route [i][1] - y_center) * y_scale )
                final_coordenates.append (new_coord)  
                
        elif route [i][1] == route [i-1][1] :
            if route [i][1] != route [i+1][1] :
                new_coord = []
                new_coord.append ( (route [i][0] - x_center) * x_scale )
                new_coord.append (( route [i][1] - y_center) * y_scale )
                final_coordenates.append (new_coord)  
        elif route [i][0]-1 == route [i-1][0] and route [i][1]-1 == route [i-1][1]:  
            if route [i][0]+1 != route [i+1][0] or  route [i][1]+1 != route [i+1][1]:
                new_coord = []
                new_coord.append ( (route [i][0] - x_center) * x_scale )
                new_coord.append (( route [i][1] - y_center) * y_scale )
                final_coordenates.append (new_coord)              
            
                
    final_coordenates.append ([( (route [len (route)-1][0] - x_center) * x_scale ) , (( route [len (route)-1][1] - y_center) * y_scale )])   
    
    return final_coordenates



def _deserialize_numpy(self, str):
    """
    wrapper for factory-generated class that passes numpy module into deserialize
    """
    # pass in numpy module reference to prevent import in auto-generated code
    self.deserialize_numpy(str, np) #si no funciona poner numpy
    dims=map(lambda x:x.size, self.layout.dim)
    self.data = self.data.reshape(dims)
    return self

## Use this function to generate message instances using numpy array types for numerical arrays. 
## @msg_type Message class: call this functioning on the message type that you pass
## into a Publisher or Subscriber call. 
## @returns Message class
def numpy_nd_msg(msg_type):
    classdict = { 'slots': msg_type.slots, '_slot_types': msg_type._slot_types,
                  '_md5sum': msg_type._md5sum, '_type': msg_type._type,
                  '_has_header': msg_type._has_header, '_full_text': msg_type._full_text,
                  'serialize': _serialize_numpy, 'deserialize': _deserialize_numpy,
                  'serialize_numpy': msg_type.serialize_numpy,
                  'deserialize_numpy': msg_type.deserialize_numpy
                  }

    # create the numpy message type
    msg_typename = "Numpy%s"%msg_type._type.replace('/', '__')
    return type(msg_type_name,(msg_type,),classdict)




def planeacion_nodo ():
    ruta =Ruta()
    
    rospy.init_node('Planeacion', anonymous=True)  # Inicia el nodo teleop
    pub = rospy.Publisher('/robocol/ruta', numpy_nd_msg(Float32MultiArray), queue_size=1)    #toca modificar los mensajes 
    rospy.Subscriber('/Robocol/Inicio_fin', Twist, Inicio_Fin, queue_size=1)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown() and centinela_inicio_fin:
        
        environment = GridEnvironment(x_fin,y_fin, ruta_imagen)
        size = environment.value_state_table.shape
        policy = run_policy(size)
        agent = valueBasedAgent(environment, np.array(policy), discount_factor)
        # Initialize policy evaluation class.
        policy_evaluation = PolicyEvaluation(x_fin,y_fin,environment, agent, iterations = 15)
        policy_evaluation.evaluate(plot_grid = False)
        policy_evaluation.updatePolicy()
        # Run and train the agent
        agent.pos = [x_ini,y_ini] #13, 3
        game = Game(x_fin,y_fin, environment, agent)
        array_pos = np.array(game.update(0.3))
        final_coordenates = np.array(coordenates(array_pos, ruta_imagen, x_len, y_len))
        environment.dibujo_ruta(array_pos)
        if ruta.callback == True:
            ans = ruta.navegacion()
            a = np.array(ans, dtype=np.float32)   #Si no funciona, probar con np = numpy
            print("sending\n", a)
            pub.publish(data=a)
            ruta.callback = False
            
        centinela_inicio_fin = False
        rate.sleep()


if __name__ == '__main__':
    try:
        planeacion_nodo()
    except rospy.ROSInterrupyException:
        rospy.loginfo("node terminated")


