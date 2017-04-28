import tensorflow as tf
import numpy as np
import random
import gym
import math



def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters",[33600*2,3])
        state = tf.placeholder("float",[None,33600*2])
        actions = tf.placeholder("float",[None,3])
        advantages = tf.placeholder("float",[None,1])
        linear = tf.matmul(state,params)
        probabilities = tf.nn.softmax(linear)
        good_probabilities = tf.reduce_sum(tf.matmul(probabilities, actions),reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
        return probabilities, state, actions, advantages, optimizer

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None,33600*2])
        newvals = tf.placeholder("float",[None,1])
        w1 = tf.get_variable("w1",[33600*2,1000])
        b1 = tf.get_variable("b1",[1000])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[1000,1])
        b2 = tf.get_variable("b2",[1])
        calculated = tf.matmul(h1,w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []
    prevBuffer = []
    #prevFeatures
    t = 0

    while True:
        # calculate policy
        if render: env.render()
        if t < 10:
            prevBuffer.append(np.zeros((33600), dtype=np.float))
            prevFeatures = prevBuffer[0]
        else:
            prevFeatures = prevBuffer.pop(0)

        obs_vector = np.expand_dims(observation, axis=0)
        obs_vector = np.mean(obs_vector, axis=3)
        obs_vector = np.reshape(obs_vector, (33600)) / 255.0

        if t >= 10:
            prevBuffer.append(obs_vector)
        obs_vector = np.concatenate((obs_vector, prevFeatures), axis=0)
        obs_vector = np.reshape(obs_vector, (-1, 33600*2))

        probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
        randPerc = random.uniform(0,1) - probs[0][0]
        action = 1
        while randPerc > 0 and action < 3:
            randPerc -= probs[0][action]
            action += 1
        action -= 1

        # record the transition
        obs2 = np.reshape(obs_vector, (33600*2))
        states.append(obs2)
        actionblank = np.zeros(3)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation

        transAct = 0
        if action == 0:
            transAction = 1
        elif action == 1:
            transAction = 4
        elif action == 2:
            transAction = 5

        observation, reward, done, info = env.step(transAction)
        transitions.append((old_observation, action, reward, prevFeatures))
        totalreward += reward
        if done:
            break
        t += 1
    return totalreward

if __name__ == '__main__':
    episode_number = 0
    render = True #True for visualization 
    env = gym.make('SpaceInvaders-v0')
    policy_grad = policy_gradient()
    value_grad = value_gradient()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    while True:
        episode_number+=1
        reward = run_episode(env, policy_grad, value_grad, sess)
        print "Reward: ", reward
        print "Episodio %d" % episode_number

