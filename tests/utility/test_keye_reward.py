from keye_reward import keye_compute_reward
from numpy import array


if __name__ == "__main__":
    kwargs={'swift_reward_type': 'model_base', 'prompt_type': 'instruct', 'messages': array([{'content': '\nQuestion:\nWithin quadrilateral ABCD, with midpoints E and F on sides AB and AD respectively, and EF = 6, BC = 13, and CD = 5, what is the area of triangle DBC?\nChoices:\nA: 60\nB: 30\nC: 48\nD: 65', 'role': 'user'}], dtype=object)}
    reward = keye_compute_reward("model_base", "Answer:B", "$B$", **kwargs)
    print(reward)
