"""
Game is solved when agent consistently gets 900+ points. Track is random every episode.
"""

import numpy as np
import gym
import time, tqdm
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

import cma
import multiprocessing as mp

from train_VAE import load_vae

_EMBEDDING_SIZE = 32
_NUM_PREDICTIONS = 2
_NUM_ACTIONS = 3
_NUM_PARAMS = _NUM_PREDICTIONS * _EMBEDDING_SIZE + _NUM_PREDICTIONS


class DQNAgent(object):
    ''' Train the agent with VAE and QN'''
    def __init__(self):
        self.env = CarRacing()
    def normalize_observation(self, observation):
        return observation.astype('float32') / 255.

    def get_weights_bias(self, params):
        weights = params[:_NUM_PARAMS - _NUM_PREDICTIONS]
        bias = params[-_NUM_PREDICTIONS:]
        weights = np.reshape(weights, [_EMBEDDING_SIZE, _NUM_PREDICTIONS])
        return weights, bias

    def decide_action(self, sess, network, observation, params):
        ### TODO ###
        #Convert here to deep Q network
        observation = self.normalize_observation(observation)
        embedding = sess.run(network.z, feed_dict={network.image: observation[None, :,  :,  :]})
        weights, bias = self.get_weights_bias(params)

        action = np.zeros(_NUM_ACTIONS)
        prediction = np.matmul(np.squeeze(embedding), weights) + bias
        prediction = np.tanh(prediction)

        action[0] = prediction[0]
        if prediction[1] < 0:
            action[1] = np.abs(prediction[1])
            action[2] = 0
        else:
            action[2] = prediction[1]
            action[1] = 0

        return action



    def play(self, params, render=True, verbose=False):
        ### TODO ###
        #Convert here to Deep Q Network
        sess, network = load_vae()
        _NUM_TRIALS = 1
        agent_reward = 0
        for trial in range(_NUM_TRIALS):
            observation = self.env.reset()
            # Little hack to make the Car start at random positions in the race-track
            np.random.seed(int(str(time.time()*1000000)[10:13]))
            position = np.random.randint(len(self.env.track))
            self.env.car = Car(self.env.world, *self.env.track[position][1:4])

            total_reward = 0.0
            steps = 0
            while True:
                if render:
                    self.env.render()
                ##TODO##
                #change this place with DQN
                action = self.decide_action(sess, network, observation, params)
                #end change
                observation, r, done, info = self.env.step(action)
                total_reward += r
                # NB: done is not True after 1000 steps when using the hack above for
                # 	  random init of position
                if verbose and (steps % 200 == 0 or steps == 999):
                    print("\naction " + str(["{:+0.2f}".format(x) for x in action]))
                    print("step {} total_reward {:+0.2f}".format(steps, total_reward))

                steps += 1
                if steps == 999:
                    break

            # If reward is out of scale, clip it
            total_reward = np.maximum(-100, total_reward)
            agent_reward += total_reward

        return - (agent_reward / _NUM_TRIALS)

    def train(self):
        es = cma.CMAEvolutionStrategy(_NUM_PARAMS * [0], 0.1, {'popsize': 16})
        rewards_through_gens = []
        generation = 1
        try:
            while not es.stop():
                solutions = es.ask()
                with mp.Pool(mp.cpu_count()) as p:
                    rewards = list(tqdm.tqdm(p.imap(self.play, list(solutions)), total=len(solutions)))

                es.tell(solutions, rewards)

                rewards = np.array(rewards) *(-1.)
                print("\n**************")
                print("Generation: {}".format(generation))
                print("Min reward: {:.3f}\nMax reward: {:.3f}".format(np.min(rewards), np.max(rewards)))
                print("Avg reward: {:.3f}".format(np.mean(rewards)))
                print("**************\n")

                generation += 1
                rewards_through_gens.append(rewards)
                np.save('rewards', rewards_through_gens)

        except (KeyboardInterrupt, SystemExit):
            print("Manual Interrupt")
        except Exception as e:
            print("Exception: {}".format(e))
        return es

if __name__ == '__main__':
    dqnagent = DQNAgent()
    es = dqnagent.train()
    np.save('best_params', es.best.get()[0])
    input("Press enter to play... ")
    RENDER = True
    score = dqnagent.play(es.best.get()[0], render=RENDER, verbose=True)
    print("Final Score: {}".format(-score))