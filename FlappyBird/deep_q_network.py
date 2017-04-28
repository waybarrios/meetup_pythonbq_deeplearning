#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

first_training=True #primera vez se entrena. Falso si no es la primera vez
GAME = 'bird' # Nombe de los logs del juego
ACTIONS = 2 # Numero de acciones validas: No hacer nada y subir
GAMMA = 0.99 # decay rate de las observaciones pasadas
OBSERVE = 100000. # timesteps para observar antes del entrenamiento
EXPLORE = 2000000. # frames over para el entrenamiento
FINAL_EPSILON = 0.0001 # valor final del epsilon
INITIAL_EPSILON = 0.0001 # valor inicial del epsilon
REPLAY_MEMORY = 50000 # Tamaño para las Antiguas transiciones (conocimiento pasado)
BATCH = 32 # tamaño minibatch
FRAME_PER_ACTION = 1 # solo una acción es permitida por frame.
if first_training: 
    OBSERVE = 10000
    EXPLORE = 3000000
    FINAL_EPSILON = 0.0001
    INITIAL_EPSILON = 0.1
    
"""
Dimensionality
From what we've learned so far, how can we calculate the number of neurons of each layer in our CNN?

Given:

our input layer has a width of W and a height of H
our convolutional layer has a filter size F
we have a stride of S
a padding of P
and the number of filters K,
the following formula gives us the width of the next layer: W_out = (W−F+2P)/S+1.

The output height would be H_out = (H-F+2P)/S + 1.

And the output depth would be equal to the number of filters D_out = K.

The output volume would be W_out * H_out * D_out.

Shape de Salida de cada capa convolucional 
new_height = (input_height - filter_height + 2 * P)/S + 1
new_width = (input_width - filter_width + 2 * P)/S + 1

"""
def weight_variable(shape):
    """ 
    F(X)=W*x + bias ; corresponde a W= weights.
    Esta funcion contiene los pesos. 
    Se debe crear usando tensorflow. 
    Hint: se debe usar tf.truncated_normal con stddev=0.01
    junto con tf.variable
    Input: Shape: Tamaño del array. (Se tiene en cuenta con el tamaño de la red)
    Return Varible en tf con los pesos
    """
    #TODO
    initial = None
    return tf.Variable(initial)

def bias_variable(shape):
    """
    Funcion que conteine la constante de bias de la red W*x + bias.
    Se puede inicialar con el valor de 0.01
    Return: tf.variable
    """
    #TODO
    initial = None
    return None

def conv2d(x, W, stride):
    """
    Capa convolucional respetado la forma W*x donde:
    W: Weights
    x: inputs(datos)
    strides: Tamaño de la Ventana o Kernel para cada dimesion de la entrada.
    HINT: Se usa padding='Same'
    """
    #TODO
    return None

def max_pool_2x2(x):
    """
    Operacion de Max pooling:
    Pooling con los datos de mayor resultado 
    Se tiene en cuenta los siguientes parametros:
    x: Datos para hacerle maxpooling
    HINT: 
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1], padding = "SAME"


    """
    #TODO
    return None

def createNetwork():
    #TODO: Construir la red usando tensorflow basados en la arquitectura presentada
    # Pesos: Primera capa convolucional 
    W_conv1 = None
    b_conv1 = None

    #Pesos: Segunda capa convolucional
    W_conv2 = None
    b_conv2 = None

    #Pesos: Tercera capa convolucional
    W_conv3 = None
    b_conv3 = None

    #Pesos: Primera capa FullyConnected 
    W_fc1 = None
    b_fc1 = None

    #Pesos: Segunda capa FullyConnected 
    #HINT: La dimesion es data por D*ACTIONS:

    """
    Recuerda: La Salidad de la red siempre estara determinada
    en el numero de acciones que se realice. 
    Ya que nuestra salida corresponde al respuesta del joystick  
    """
    
    W_fc2 = None
    b_fc2 = None

    # input layer
    #HINT: tf.placeholder
    s = None

    # hidden layers
    #Hint: Cada capa convolucional tiene su activacion: relu. 
    # tf.nn.relu()
    h_conv1 = None    
    h_pool1 = None
    
    #Hidden layer 2
    h_conv2 = None
    #Hidden Layer 3
    h_conv3 = None

    #HINT: Reshape para poder ser entrada de la fullyconnected.
    #tf.reshape 
    #size= [-1, 1600]
    h_conv3_flat = None
    #Primera FullyConnected 
    #Con Activacion Relu
    h_fc1 = None

    # readout layer: Ultima fullyconnected sin activacion
    readout = None

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # Funcion de costo 
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    #Abrir un estado de juego para comunicarse con el emulador
    game_state = game.GameState()

    # observaciones previas a  replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    #Obtener el primer estado sin hacer nada y preprocesar la imagen a 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # Save  networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # Comenzar entrenamiento
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":

        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # decay epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # conseguir frames y preprocesamiento
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # guardar cada transicion
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # Solo se entrena si la observacion termino
        if t > OBSERVE:
            #  minibatch para training
            minibatch = random.sample(D, BATCH)

            # batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # pasos del gradiente
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # valores viejos 
        s_t = s_t1
        t += 1

        # salvar cada 10000 iters
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
    

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
