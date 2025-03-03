## Run
Training:
```
python train.py
```

## Reward function
```py
def custom_reward(self, state, prev_state):
    prev_obj_pos = prev_state[2:4]
    obj_pos = state[2:4]
    goal_pos = state[4:6]
    
    prev_obj_to_goal = np.linalg.norm(prev_obj_pos - goal_pos)
    obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
    diff = prev_obj_to_goal - obj_to_goal

    prev_ee_to_obj = np.linalg.norm(prev_state[:2] - prev_obj_pos)
    ee_to_obj = np.linalg.norm(state[:2] - obj_pos)
    ee_diff = prev_ee_to_obj - ee_to_obj

    return diff * 2 + ee_diff * 0.5 - 0.001
```

## State
* 6 - current high level state 
* 4 - previous ee and obj locations

## Config 1:
* 10x64x8 network
* update_frequency = 8
* target_update_frequency = 200
* lr = 0.002
* epsilon = 0.8
* epsilon_decay = 0.995
* epsilon_min = 0.1
* batch_size=64
* gamma=0.95
* buffer_size=1000000

![image](https://github.com/user-attachments/assets/b3b8a15b-be3a-4020-9e59-bf043755247b)


## Config 2: 
* 10x32x32x8
* update_frequency = 8
* target_update_frequency = 200
* lr = 0.001
* epsilon = 0.8
* epsilon_decay = 0.998
* epsilon_min = 0.1
* batch_size=64
* gamma=0.95
* buffer_size=1000000

![image](https://github.com/user-attachments/assets/c98c70e9-a7f0-4755-8193-3ac11ecc16da)
