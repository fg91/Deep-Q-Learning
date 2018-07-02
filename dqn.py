
# coding: utf-8

# In[13]:


import gym
import tensorflow as tf
import random
import numpy as np
import os
import imageio
from skimage.transform import resize

import timeit


# In[14]:


PATH = "dqn/animations/"
os.makedirs(PATH,exist_ok=True)


# In[15]:


env = gym.make('PongDeterministic-v3')


# In[16]:


def generate_gif(sess, frame_number, frames_for_gif):
    
    for idx,frame_idx in enumerate(frames_for_gif): 
        frames_for_gif[idx] = resize(frame_idx,(420,320,3),preserve_range=True, order=0).astype(np.uint8)
        
    imageio.mimsave(f'{PATH}{"pong_frame_{0}.gif".format(frame_number)}', frames_for_gif, duration=1/30)


# In[17]:


class processFrame():
    def __init__(self):
        self.frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
        self.processed = tf.image.resize_images(self.processed, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    def process(self, sess, frame):
        """
        Args:
            sess: A Tensorflow session object
            frame: A (210, 160, 3) frame of an Atari game in RGB
        Returns:
            A processed (84, 84, 1) frame in grayscale
        """
        return sess.run(self.processed, feed_dict={ self.frame:frame})


# In[18]:


class dqn():
    def __init__(self, hidden=512, learning_rate=0.00005):
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.input = tf.placeholder(shape=[None,84,84,4], dtype=tf.float32)
        self.inputscaled = self.input/255
        self.conv1 = tf.layers.conv2d(
            inputs=self.inputscaled, filters=32, kernel_size=[8,8], strides=4,
            padding="valid", activation=tf.nn.relu, use_bias=False)
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1, filters=64, kernel_size=[4,4], strides=2, 
            padding="valid", activation=tf.nn.relu, use_bias=False)
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2, filters=64, kernel_size=[3,3], strides=1, 
            padding="valid", activation=tf.nn.relu, use_bias=False)
        self.conv4 = tf.layers.conv2d(
            inputs=self.conv3, filters=hidden, kernel_size=[7,7], strides=1, 
            padding="valid", activation=tf.nn.relu, use_bias=False)
        self.valuestream, self.advantagestream = tf.split(self.conv4,2,3)
        self.valuestream = tf.layers.flatten(self.valuestream)
        self.advantagestream = tf.layers.flatten(self.advantagestream)
        self.Advantage = tf.layers.dense(
            inputs=self.advantagestream,units=env.action_space.n,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.Value = tf.layers.dense(
            inputs=self.valuestream,units=1,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.Qvalues = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keepdims=True))
        self.bestaction = tf.argmax(self.Qvalues,1)
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action = tf.placeholder(shape=[None],dtype=tf.int32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qvalues, tf.one_hot(self.action, env.action_space.n, dtype=tf.float32)), axis=1)
        
        self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.targetQ, predictions=self.Q))
        #self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)#0.0001
        self.update = self.optimizer.minimize(self.loss)


# In[60]:


def getaction(frame_number,state, inference=False):
    if frame_number < replay_start_size:
        e = exploration_initial
    if frame_number >= replay_start_size and frame_number < replay_start_size + exploration_decay_frames:
        e = m*frame_number + b
    if frame_number >= replay_start_size + exploration_decay_frames:
        e = m2*frame_number + b2
    if inference:
        e = exploration_inference
    if np.random.rand(1) < e:
        return np.random.randint(0, env.action_space.n)
    else:
        return sess.run(mainDQN.bestaction, feed_dict={mainDQN.input:[state]})[0]       


# In[20]:


class MemoryBuffer:
    def __init__(self, size = 1000000, frame_height=84, frame_width=84, agent_history_length = 4, batch_size = 32):
        
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height,self.frame_width), dtype=np.uint8)
        self.donestates = np.empty(self.size, dtype=np.bool)
        
        # Pre-allocate memory for the States and newStates in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width), dtype=np.uint8)
        self.newstates = np.empty((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)
        
    def addExperience(self, action, frame, reward, done):
        """ 
        Adds an experience to the replay memory. Convention: array frames contains the newstates after the action
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current,...] = frame
        self.rewards[self.current] = reward
        self.donestates[self.current] = done
        
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
        
            
    def getState(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1,...]
        
    def getValidIndices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.donestates[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index
            
    def getMinibatch(self):
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')
        self.getValidIndices()
            
        for i, idx in enumerate(self.indices):
            self.states[i] = self.getState(idx - 1)
            self.newstates[i] = self.getState(idx)
        return np.transpose(self.states,axes=(0,2,3,1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.newstates,axes=(0,2,3,1)), self.donestates[self.indices]
                


# In[21]:


def updateTargetVars_full(mainDQN_vars, targetDQN_vars):
    update_ops = []
    for i, var in enumerate(mainDQN_vars):
        op = targetDQN_vars[i].assign(var.value())
        update_ops.append(op)
    return update_ops


# In[22]:


def learn():
    states, actions, rewards, newstates, dones = myMemoryBuffer.getMinibatch()    
    argQmax = sess.run(mainDQN.bestaction, feed_dict={mainDQN.input:newstates})
    Qvals = sess.run(targetDQN.Qvalues, feed_dict={targetDQN.input:newstates})
    
    done_mult = (1-dones)
    doubleQ = Qvals[range(bs), argQmax]
    targetQ = rewards + (gamma*doubleQ * done_mult)
    _ = sess.run(mainDQN.update,feed_dict={mainDQN.input:states,mainDQN.targetQ:targetQ, mainDQN.action:actions})


# In[58]:


# Control parameter
max_episode_len = 18000
bs = 32
target_network_update_freq = 10000
gamma = 0.99
exploration_initial = 1
exploration_final = 0.1
exploration_inference = 0.01
exploration_decay_frames = 1000000#1000000
replay_start_size = 5000
max_frames = 4000000
memory_size = 1000000 #1000000
hidden = 512
#interpol_factor = 0.001
no_op_steps = 20
gif_freq = 50
learning_rate = 0.00005

m = -(exploration_initial - exploration_final)/exploration_decay_frames
b = exploration_initial - m*replay_start_size
m2 = -(exploration_final - exploration_inference)/(max_frames - exploration_decay_frames - replay_start_size)
b2 = exploration_final - m2*(replay_start_size + exploration_decay_frames)


# In[24]:


tf.reset_default_graph()

myMemoryBuffer = MemoryBuffer(size=memory_size, batch_size=bs)
mainDQN = dqn(hidden, learning_rate)
targetDQN = dqn(hidden)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
restore = False


variables = tf.trainable_variables()
mainDQN_vars = variables[0:len(variables)//2]
targetDQN_vars = variables[len(variables)//2:]
updateTargetVars = updateTargetVars_full(mainDQN_vars, targetDQN_vars)
frameprocessor = processFrame()

start_time = timeit.default_timer()
with tf.Session() as sess:
    sess.run(init)
    if restore == True:
        saver.restore(sess,tf.train.latest_checkpoint(PATH))

    frame_number = 0
    episode_number=0
    rewards = []
    while frame_number < max_frames:
        if episode_number % gif_freq == 0: 
            frames_for_gif = []
        frame = env.reset()
        done = False
        for _ in range(random.randint(1, no_op_steps)):
            frame, _, _, _ = env.step(0)
        processed_frame = frameprocessor.process(sess,frame)
        state = np.repeat(processed_frame,4, axis=2)
        ep_reward_sum = 0
        for j in range(max_episode_len):
            action = getaction(frame_number,state)
            newframe, reward, done, _ = env.step(action)
            
            if episode_number % gif_freq == 0: 
                frames_for_gif.append(newframe)
                        
            processed_newframe = frameprocessor.process(sess,newframe)
            newstate = np.append(state[:,:,1:],processed_newframe,axis=2)

            frame_number += 1
            
            myMemoryBuffer.addExperience(action=action, frame=processed_newframe[:,:,0], reward=reward, done=done)
            
            if frame_number > replay_start_size:
                learn()
            
            
            if frame_number % target_network_update_freq == 0 and frame_number > replay_start_size:
                update_ops = updateTargetVars
                for op in update_ops:
                    sess.run(op)
            
            
            ep_reward_sum += reward
            state = newstate
            if done == True:
                break
        rewards.append(ep_reward_sum)
        if episode_number % gif_freq == 0: 
            generate_gif(sess, frame_number, frames_for_gif)
        if episode_number % 50 == 0:
            saver.save(sess,PATH+'/my_model',global_step=frame_number)
        if episode_number % 10 == 0:
            print(episode_number, frame_number,np.mean(rewards[-100:]), j)
            with open('rewards.dat','a') as f:
                print(episode_number, frame_number,np.mean(rewards[-100:]), j,file=f)
        episode_number += 1
elapsed = timeit.default_timer() - start_time
print(elapsed/60)


# # Inference

# In[ ]:


init = tf.global_variables_initializer()
frameprocessor = processFrame()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,tf.train.latest_checkpoint(PATH))
    frames_for_gif = []
    done = False
    frame = env.reset()
    processed_frame = frameprocessor.process(sess,frame)
    state = np.repeat(processed_frame,4, axis=2)
    ep_reward_sum = 0
    
    while not done:
        action = getaction(frame_number,state,inference=True)
        newframe, reward, done, _ = env.step(action)
            
        frames_for_gif.append(newframe)
                        
        processed_newframe = frameprocessor.process(sess,newframe)
        newstate = np.append(state[:,:,1:],processed_newframe,axis=2)


        ep_reward_sum += reward
        state = newstate
    print("Total reward: %s" % ep_reward_sum)
    generate_gif(sess,0, frames_for_gif)

