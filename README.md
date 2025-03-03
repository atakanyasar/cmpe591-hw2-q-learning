## Examples
https://github.com/user-attachments/assets/c8b78d69-9113-4db7-a81e-93b7d8872999


https://github.com/user-attachments/assets/86785d21-764c-47cc-b903-380cbc85ae29


https://github.com/user-attachments/assets/94c52998-ae1b-45cd-b6a6-9308888e11a4



## Run
* For training:
```
python train.py
```
* To test: run `test.ipynb`

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

![fig](https://github.com/user-attachments/assets/74c3b32d-f9e3-4188-99f0-e4d6aecc13de)

## Config 2:
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

## Conclusion
- Config 2 did not perform well, indicating that it failed to learn effectively. As a result, Config 1 was chosen for further training.
- The reward plot shows an upward trend, confirming that the model is learning over time.
- The model was trained for approximately 10 hours, but additional training time is needed to achieve better performance.









