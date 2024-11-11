import numpy as np
import random
from neighborhood_search import switches_gains, affectation_matrix_after_switch


# the tabou list is composed of the tasks that cant be switch or the agents

def taboo_search(affectation_matrix, instance, nb_max_iterations=10000, cycle_size=3, search_type="random", nb_values=5, pb="max", tabou_type="tasks", tabou_size=3, aspiration=False):
  best_affectation_matrix = affectation_matrix.copy()
  current_affectation_matrix = affectation_matrix.copy()
  current_additional_gain = 0
  best_additional_gain = 0
  tabou_list = [[] for _ in range(tabou_size)]
  possible_switches, additional_gains = switches_gains(current_affectation_matrix, instance, cycle_size, tabou_search=True, tabou_list=tabou_list, tabou_type=tabou_type, aspiration=aspiration)

  for k in range(nb_max_iterations):
    if not possible_switches:
      return current_affectation_matrix, current_additional_gain
    if search_type == "random":  # Choose a completely random switch
      ind = random.randint(0, len(possible_switches) - 1)
    elif search_type == "best":  # Choose the best neighbor
      if pb == "max":
        ind = np.argmax(additional_gains)
      else:  # pb == "min"
        ind = np.argmin(additional_gains)
      if (pb == "max" and additional_gains[ind] <= 0) or (pb == "min" and additional_gains[ind] >= 0):
        print("No more switch at iteration", k)
        break
    elif search_type == "random_nb":  # Choose among the best 'nb_values' neighbors
      if pb == "max":
        indices_max = np.argsort(additional_gains)[-nb_values:][::-1]
      else:  # pb == "min"
        indices_max = np.argsort(additional_gains)[:nb_values]
      ind = indices_max[random.randint(0, len(indices_max) - 1)]
    else:
      print("Invalid search type.")
      return None

    tasks, agents = possible_switches[ind]
    if tabou_type == "tasks":
      elements_to_add = tasks
    elif tabou_type == "agents":
      elements_to_add = agents
    else:
      print("Invalid tabou type.")
      return None

    # delete first element of the tabou list
    tabou_list.pop(0)
    tabou_list.append(elements_to_add)

    additional_gain = additional_gains[ind]
    m, t, c, r, b = instance
    b_residual = b - (current_affectation_matrix * r).sum(axis=1)
    current_affectation_matrix = affectation_matrix_after_switch(current_affectation_matrix, tasks, agents)
    current_additional_gain += additional_gain

    if (pb == "max" and current_additional_gain > best_additional_gain) or (pb == "min" and current_additional_gain < best_additional_gain):
      best_additional_gain = current_additional_gain
      best_affectation_matrix = current_affectation_matrix.copy()

    possible_switches, additional_gains = switches_gains(current_affectation_matrix, instance, cycle_size, tabou_search=True, tabou_list=tabou_list, tabou_type=tabou_type, aspiration=aspiration)

  return best_affectation_matrix, best_additional_gain
