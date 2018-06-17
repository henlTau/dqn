"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
PRINT_TIMES = False


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": [],
    "running_times": []
}

def dqn_learing(
    env,
    q_func,
    optimizer_spec,
    exploration,
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000
    ):


    print("running new version")

    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    """

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
    num_actions = env.action_space.n

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    # Initialize target q function and q function, i.e. build the model.
    
    """ ---------------------------- OUR CODE ---------------------------- """
    Q = q_func(input_arg,num_actions) # The parameters are random
    Qtag = q_func(input_arg,num_actions)
    if (USE_CUDA):
        Q.cuda()
        Qtag.cuda()
    Qtag.load_state_dict(Q.state_dict())

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)
    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
    """ ------------------------------------------------------------------ """
    
    
    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    reward = None
    done = None
    info = None
    LOG_EVERY_N_STEPS = 10000
    
    startTime = time.time();
    
    for t in count():
        """ Tsuf: ---- Stuff for debigging times for various places --- """
        T1=0
        t1Tmp=0
        T2=0
        t2Tmp=0
        T3=0
        t3Tmp=0
        T4=0
        t4Tmp=0
        T5=0
        t5Tmp=0
        T6=0
        t6Tmp=0
        T7=0
        t7Tmp=0
        T8=0
        t8Tmp=0
        """ ----------------------------------------------------------- """
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            break

		if (t>1000000):
            break
        ### 2. Step the env and store the transition
        
        # At this point, "last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Note that you cannot use "last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)
        
        """ -------------------------- OUR CODE -------------------------- """


        #store last_obs, and get latest obs's as the input for the n.n
        t1Tmp=time.time()
        cur_idx = replay_buffer.store_frame(last_obs)
        next_input = replay_buffer.encode_recent_observation()
        T1+=time.time()-t1Tmp



        #take random action or use the net
        t2Tmp=time.time()
        action = select_epilson_greedy_action(Q,next_input,t) #the returned action is on the CPU
        T2+=time.time()-t2Tmp


        #see what happens after we take that action
        t3Tmp=time.time()
        last_obs, reward, done, info = env.step(action) #the returned parameters are on the CPU
        T3+=time.time()-t3Tmp



   #     print(t)
       # env.render()
        #store the results on the replay buffer
        replay_buffer.store_effect(cur_idx,action,reward,done) #on the CPU
        
        #if the simulation is done, reset the environment
        if (done):
            last_obs = env.reset()
        """ -------------------------------------------------------------- """

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # Note: Move the variables to the GPU if avialable
            # 3.b: fill in your own code to compute the Bellman error. This requires
            # evaluating the current and next Q-values and constructing the corresponding error.
            # Note: don't forget to clip the error between [-1,1], multiply is by -1 (since pytorch minimizes) and
            #       maskout post terminal status Q-values (see ReplayBuffer code).
            # 3.c: train the model. To do this, use the bellman error you calculated perviously.
            # Pytorch will differentiate this error for you, to backward the error use the following API:
            #       current.backward(d_error.data.unsqueeze(1))
            # Where "current" is the variable holding current Q Values and d_error is the clipped bellman error.
            # Your code should produce one scalar-valued tensor.
            # Note: don't forget to call optimizer.zero_grad() before the backward call and
            #       optimizer.step() after the backward call.
            # 3.d: periodically update the target network by loading the current Q network weights into the
            #      target_Q network. see state_dict() and load_state_dict() methods.
            #      you should update every target_update_freq steps, and you may find the
            #      variable num_param_updates useful for this (it was initialized to 0)
            """ ------------------------ OUR CODE ------------------------ """


            #sample a batch of history samples
            t4Tmp=time.time()
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size) #on CPU
            
            obs_batch = torch.from_numpy(obs_batch).type(dtype) / 255.0 # When available, move the samples batch to GPU
            next_obs_batch = torch.from_numpy(next_obs_batch).type(dtype) / 255.0 #GPU
            T4+=time.time()-t4Tmp



            #see which Q values the current network gives, for all obs's
            t5Tmp=time.time()
            inter_Qs = Q(Variable(obs_batch)) #input is on GPU, output is on GPU
            inter_Qs_chosen = Variable(torch.zeros(batch_size).type(dtype)) #GPU
            #take the action that was chosen before
            for i in range(batch_size):
                inter_Qs_chosen[i] = inter_Qs[i,act_batch[i]]
            #take only the intermediate (non-terminal) obs's
            inter_idx = np.where(done_mask==False)[0] #CPU
            inter_next_obs_batch = next_obs_batch[inter_idx,:,:,:]
            T5+=time.time()-t5Tmp



            #see what the "target" (backuped) network says for the intermediate ones
            t6Tmp=time.time()
            inter_next_Qs = Qtag(Variable(inter_next_obs_batch, volatile=True)).data.max(1)[0] #All on GPU
            T6+=time.time()-t6Tmp



            #calculate the bellman errors
            t7Tmp=time.time()
            #for final obs's, the target is just the reward
            targets = torch.from_numpy(rew_batch).type(dtype) #Moved rew_batch to GPU (as 'targets')
            for (i,idx) in enumerate(inter_idx):
                targets[idx] += gamma*inter_next_Qs[i] #The bellman item
            # errors = -(inter_Qs_chosen.data - targets)**2 #EQUATION COULD BE WRONG!!   [on GPU]
            # for i in range(len(errors)):
            #     if errors[i]<-1:
            #         errors[i] = -1
            #     elif errors[i]>1:
            #         errors[i] = 1
            errors = inter_Qs_chosen.data - targets
            errors.clamp(-1, 1)
            T7+=time.time()-t7Tmp



            #train the network! (:
            t8Tmp=time.time()
            optimizer.zero_grad()
            inter_Qs_chosen.backward(errors) #COULD BE WRONG WAY!!    [Everything is on GPU (: ]
            optimizer.step()
            T8+=time.time()-t8Tmp



            num_param_updates += 1
            if (num_param_updates % target_update_freq == 0):
                Qtag.load_state_dict(Q.state_dict())
            
            
            """ ---------------------------------------------------------- """

        ### 4. Log progress and keep track of statistics
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        Statistic["mean_episode_rewards"].append(mean_episode_reward)
        Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)
        Statistic["running_times"].append(int(time.time()-startTime))

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            if (PRINT_TIMES):
                print("-----------------------")
                print(T1)
                print(T2)
                print(T3)
                print(T4)
                print(T5)
                print(T6)
                print(T7)
                print(T8)
                print("-----------------------")
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            # Dump statistics to pickle
            with open('statistics.pkl', 'wb') as f:
                pickle.dump(Statistic, f)
                print("Saved to %s" % 'statistics.pkl')
