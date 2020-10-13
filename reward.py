import numpy as np
import itertools as it

# reward parameters
a = -0.510066
b = 0.760666
c = -0.35663
d = -0.184483


# test_observation = np.asarray([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#                                 [0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
#                                 [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
#                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                                 [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
#                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


test_observation = np.asarray([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

def custom_reward(new_observation):

    for x in range(0, 5):
        new_observation = np.delete(new_observation, 0, 0)

    aggregate_height = compute_aggregate_height(new_observation)
    complete_lines = compute_complete_lines(new_observation)
    holes = compute_holes(new_observation)
    bumpiness = compute_bumpiness(new_observation)

    print(aggregate_height, complete_lines, holes, bumpiness)

    return a * aggregate_height + b * complete_lines + c * holes + d * bumpiness



def compute_aggregate_height(observation):

    aggregate_height = 0

    for column in observation.T:
        aggregate_height += compute_column_height(column)

    return aggregate_height

def compute_complete_lines(observation):
    return  (observation.sum(axis=1) == 10).sum()


def compute_holes(observation):
    
    holes = 0

    for column in observation.T:
        prev_point = 0
        for point in column:
            if prev_point == 1 and point == 0:
                holes += 1
            prev_point = point 
    
    return holes

def compute_bumpiness(observation):

    bumpiness = 0

    prev_height = None
    for column in observation.T:
        column_height = compute_column_height(column)
        if prev_height != None:
            bumpiness += abs(column_height - prev_height)
        prev_height = column_height

    return bumpiness

def compute_column_height(column):
    
    height = 0

    found_top = False
    for point in column:
        if not found_top:
            found_top = point == 1        
        if found_top:
            height += 1

    return height


print(custom_reward(test_observation))

# print(test_observation)
# print(np.delete(test_observation, 0, 0))