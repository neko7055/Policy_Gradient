from multiprocessing import Pool, freeze_support
import gym
import numpy as np
import matplotlib.pyplot as plt
from model_RL import PolicyContinuous


# a single simulation
def sample(model, problem):
    env = gym.make(problem)
    prev_state, info = env.reset()
    reward_list = []
    state_list = []
    action_list = []
    while True:
        action = model.sample_action(np.array([prev_state]))
        state, reward, terminated, truncated, info = env.step(action)
        reward_list.append(reward)
        state_list.append(prev_state)
        action_list.append(action)
        if terminated or truncated:
            return np.array(state_list), np.array(action_list), np.array(reward_list)
        prev_state = state


if __name__ == "__main__":
    # Some setting

    # Parallel setting
    freeze_support()
    pool = Pool(12)

    # OpenAI GYM environment setting
    problem = "Ant-v4"
    env = gym.make(problem, render_mode="human")

    # Get the dimension of state , dimension of action , the upper bound and the lower bound of action
    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    upper_bound = env.action_space.high
    lower_bound = env.action_space.low
    print("Size of Action Space ->  {}".format(num_actions))
    print()

    # Declare the neural network model
    model = PolicyContinuous(num_states, num_actions, upper_bound, learning_rate=0.0001)

    # Training
    total_episodes = 2000
    avg_reward_list = []
    for ep in range(total_episodes):
        # Simulate 128 times
        pool_buffer = pool.starmap_async(sample, [(model, problem)] * 128)
        S, A, R, avg_R = [], [], [], []
        for s, a, r in pool_buffer.get():
            S.append(s)
            A.append(a)
            # Calculate reward sum of each simulation for display
            avg_R.append(np.sum(r))
            # Calculate the TD learning state value with the discount factor gamma
            cumulative = 0.
            gamma = 0.999
            discount_rewards = np.zeros_like(r)
            for i in reversed(range(len(r))):
                cumulative = cumulative * gamma + r[i]
                discount_rewards[i] = cumulative

            # Standardize the TD learning state value for controlling the variance of policy gradient
            discount_rewards -= np.mean(discount_rewards)
            discount_rewards /= np.std(discount_rewards)
            R.append(discount_rewards)

        # Apply policy gradient to update the model's parameters
        model.update(S, A, R)

        # Display the average of reward sum of these 128 times simulation
        reward_sum = np.mean(avg_R)
        print("Episode * {} * Avg Reward is ==> {}".format(ep + 1, reward_sum))
        avg_reward_list.append(reward_sum)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()

    # Close Parallel
    pool.close()
    pool.terminate()

    # Testing
    prev_state = env.reset()

    for i in range(10):
        reward_sum = 0
        prev_state, info = env.reset()
        while True:
            env.render()
            action = model.predict(np.array([prev_state]))[0]
            state, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            if terminated or truncated:
                break
            prev_state = state
        print("eps:", i, "total rewards:", reward_sum)
