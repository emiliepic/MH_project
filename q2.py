from data import read_data, PROJECT_PATH
import os
from heuristics import greedy_heuristic

q2_path = os.path.join(PROJECT_PATH, 'q2')
nb_instances, instances_example = read_data("exemple1.txt", q2_path)
instance_example = instances_example[0]

# Run the greedy heuristic on the example instance using criteria 2 with 'max' as problem type
print(greedy_heuristic(instance_example, criteria=2, pb="max"))