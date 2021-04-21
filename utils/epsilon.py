def standard_epsilon(min_epsilon):
    return min_epsilon

def decaying_epsilon(min_epsilon, t):
    return min(1 / t, min_epsilon)

def exponential_decay_epsilon(min_epsilon, init_epsilon, alpha, t):
    return min(init_epsilon * (alpha**t), min_epsilon)
    