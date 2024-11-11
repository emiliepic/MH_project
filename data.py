import os
import numpy as np

OPTIMAL_VALUES = {
  'gap1': [336, 327, 339, 341, 326],
  'gap2': [434, 436, 420, 419, 428],
  'gap3': [580, 564, 573, 570, 564],
  'gap4': [656, 644, 673, 647, 664],
  'gap5': [563, 558, 564, 568, 559],
  'gap6': [761, 759, 758, 752, 747],
  'gap7': [942, 949, 968, 945, 951],
  'gap8': [1133, 1134, 1141, 1117, 1127],
  'gap9': [709, 717, 712, 723, 706],
  'gap10': [958, 963, 960, 947, 947],
  'gap11': [1139, 1178, 1195, 1171, 1171],
  'gap12': [1451, 1449, 1433, 1447, 1446],

  'gapa': [1698, 1360, 1158, 3235, 2623, 2339],
  'gapb': [1843, 1407, 1166, 3552, 2827, 2339],
  'gapc': [1931, 1402, 1243, 3456, 2806, 2391],
  'gapd': [6353, 6347, 6185, 12742, 12430, 12241]
}


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')


def read_data(file, path=DATA_PATH):
  with open(os.path.join(path, file), 'r') as f:
    lines = f.readlines()  # Read all lines from the file

  nb_instances = int(lines[0].strip())  # First line specifies the number of instances

  instances = []  # Initialize an empty list to store instances
  idx = 1  # Start index for reading instance data

  for _ in range(nb_instances):
    # Read the values of m and t for the instance
    m, t = map(int, lines[idx].strip().split())
    idx += 1

    # Read the next m lines to construct array c
    c = np.array([list(map(int, lines[idx + i].strip().split())) for i in range(m)])
    idx += m

    # Read the next m lines to construct array r
    r = np.array([list(map(int, lines[idx + i].strip().split())) for i in range(m)])
    idx += m

    # Read line for the array b
    b = list(map(int, lines[idx].strip().split()))
    idx += 1

    # Append the current instance (m, t, c, r, b) to instances list
    instances.append((m, t, c, r, b))

  # Return the number of instances and the list of instances
  return nb_instances, instances