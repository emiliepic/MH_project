import numpy as np
import itertools
import random


def elems_in_tabou_list(elem_list, tabou_list, tabou_type, aspiration=False, profit=0):
    if tabou_list:
        for tabou in tabou_list:
            for tabou_elem in tabou:
                if tabou_elem in elem_list:
                    if aspiration:
                        if profit <= 0:
                            return True
                    return True
    return False


# Neighborhood 0 : switching task from one agent to the other
def find_task_to_switch(affectation_matrix, instance, tabou_search=False, tabou_list=[], tabou_type="tasks", aspiration=False):
  m, t, c, r, b = instance
  possible_switches = []
  additional_gains = []
  b_residual = b - (affectation_matrix * r).sum(axis=1)
  for task in range(t):
    agent_1 = np.where(affectation_matrix[:, task] == 1)[0][0]
    for agent in range(m):
      if agent != agent_1 and b_residual[agent] >= r[agent][task]:
        profit = -c[agent_1][task] + c[agent][task]
        ok_to_add = True
        if tabou_search:
            if tabou_type == "tasks":
                if elems_in_tabou_list([task], tabou_list, tabou_type, aspiration, profit):
                    ok_to_add = False
            elif tabou_type == "agents":
                if elems_in_tabou_list([agent_1, agent], tabou_list, tabou_type, aspiration, profit):
                    ok_to_add = False
        if ok_to_add:
          possible_switches.append([[task], [agent_1, agent]])
          additional_gains += [- c[agent_1][task]+ c[agent][task]]
  return possible_switches, additional_gains


 # Neighborhood 1 : switching two tasks between two agents
def find_pair_tasks_to_switch(affectation_matrix, instance, tabou_search=False, tabou_list=[], tabou_type="tasks", aspiration=False):
  m, t, c, r, b = instance
  possible_switches = []
  additional_gains = []
  b_residual = b - (affectation_matrix * r).sum(axis=1)
  for task1 in range(t):
    for task2 in range(task1 + 1, t):
      agent_1 = np.where(affectation_matrix[:, task1] == 1)[0][0]
      agent_2 = np.where(affectation_matrix[:, task2] == 1)[0][0]

      # Check if tasks can be swapped between agents
      if agent_1 != agent_2:
        if ok_to_switch_for_agent(agent_1, task1, task2, b_residual, r) and ok_to_switch_for_agent(agent_2, task2, task1, b_residual, r):
          additional_gain = -c[agent_1][task1] - c[agent_2][task2] + c[agent_1][task2] + c[agent_2][task1]
          # check if agents or tasks are not in the tabou list
          ok_to_add = True
          if tabou_search:
            if tabou_type == "agents":
                if elems_in_tabou_list([agent_1, agent_2], tabou_list, tabou_type, aspiration, additional_gain):
                    ok_to_add = False
            elif tabou_type == "tasks":
                if elems_in_tabou_list([task1, task2], tabou_list, tabou_type, aspiration, additional_gain):
                    ok_to_add = False
          if ok_to_add:
            possible_switches.append([[task1, task2], [agent_1, agent_2]])
            additional_gains += [- c[agent_1][task1] - c[agent_2][task2] + c[agent_1][task2] + c[agent_2][task1]]
  return possible_switches, additional_gains

def ok_to_switch_for_agent(agent, task, new_task, b_residual, r):
  return b_residual[agent] +  r[agent][task] >= r[agent][new_task]


# Neighborhood >= 2 : switching N tasks between N agents
def find_tasks_to_switch_cyclic(affectation_matrix, instance, nb_tasks_to_switch=3, tabou_search=False, tabou_list=[], tabou_type="tasks", aspiration=False):
  m, t, c, r, b = instance
  possible_switches = []
  additional_gains = []
  b_residual = b - (affectation_matrix * r).sum(axis=1)

  # Generate all possible combinations of "nb_tasks_to_switch" tasks among "t" tasks
  task_combinations = [comb for comb in itertools.permutations(range(t), nb_tasks_to_switch) if len(set(comb)) == len(comb)]

  # Evaluate each combination for potential task-agent swaps
  for tasks in task_combinations:
    agents = [np.where(affectation_matrix[:, task] == 1)[0][0] for task in tasks]
    if len(set(agents)) != len(agents):
      continue

    # Verify if each agent can switch their task with the next task in the cycle
    can_switch = True
    for i in range(nb_tasks_to_switch):
      current_agent = agents[i]
      current_task = tasks[i]
      next_task = tasks[i + 1] if i < len(tasks) - 1 else tasks[0]
      if not ok_to_switch_for_agent(current_agent, current_task, next_task, b_residual, r):
        can_switch = False
        break

    # If all swaps in the cycle are possible, calculate the gain and add to results
    if can_switch:
      ok_to_add = True
      profit = 0
      for i in range(nb_tasks_to_switch):
        current_agent = agents[i]
        current_task = tasks[i]
        next_task = tasks[i + 1] if i < len(tasks) - 1 else tasks[0]
        profit += -c[current_agent][current_task] + c[current_agent][next_task]

      if tabou_search:
        if tabou_type == "agents":
            if elems_in_tabou_list(agents, tabou_list, tabou_type, aspiration, profit):
                ok_to_add = False
        elif tabou_type == "tasks":
            if elems_in_tabou_list(tasks, tabou_list, tabou_type, aspiration, profit):
                ok_to_add = False
      if ok_to_add:
        possible_switches.append([list(tasks), list(agents)])
        additional_gains.append(profit)
  return possible_switches, additional_gains



# Update the affectation matrix by switching tasks between agents
def affectation_matrix_after_switch(affectation_matrix, tasks, agents):
  new_affectation_matrix = affectation_matrix.copy()
  for k in range(len(tasks)):
    task = tasks[k]
    agent_1 = agents[k]
    agent_2 = agents[k - 1] if k > 0 else agents[-1]
    new_affectation_matrix[agent_1][task] = 0
    new_affectation_matrix[agent_2][task] = 1
  return new_affectation_matrix


# Select neighbors based on the chosen neighbor type
def switches_gains(affectation_matrix, instance, neighbour_type, tabou_search=False, tabou_list=[], tabou_type="tasks", aspiration=False):
  if neighbour_type == 1:
    return find_pair_tasks_to_switch(affectation_matrix, instance, tabou_search=tabou_search, tabou_list=tabou_list, tabou_type=tabou_type, aspiration=aspiration)
  elif neighbour_type == 0:
    return find_task_to_switch(affectation_matrix, instance, tabou_search=tabou_search, tabou_list=tabou_list, tabou_type=tabou_type, aspiration=aspiration)
  elif neighbour_type >= 2:
    return find_tasks_to_switch_cyclic(affectation_matrix, instance, nb_tasks_to_switch=neighbour_type, tabou_search=tabou_search, tabou_list=tabou_list, tabou_type=tabou_type, aspiration=aspiration)
  else:
    print("Invalid neighbour type.")
    return None
  

  # Perform neighborhood search to find improved solutions
def neighborhood_search(affectation_matrix, instance, nb_max_iterations=10000, neighbour_type=0, search_type="random", nb_values=5, pb="max"):
  best_affectation_matrix = affectation_matrix.copy()
  current_affectation_matrix = affectation_matrix.copy()
  current_additional_gain = 0
  best_additional_gain = 0
  possible_switches, additional_gains = switches_gains(current_affectation_matrix, instance, neighbour_type)

  for k in range(nb_max_iterations):

    # Check if there are any possible switches before proceeding
    if not possible_switches:
      print("No more possible switches at iteration", k)
      break  # Exit the loop if no switches are found

    if search_type == "random":  # Choose a completely random switch
      ind = random.randint(0, len(possible_switches) - 1)
    
    elif search_type == "best":  # Choose the best neighbor
      if pb == "max":
        if len(additional_gains) > 0:
          ind = np.argmax(additional_gains)
        else:
          print("No more switch at iteration", k)
          break
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

      # Ensure the random index is within the bounds of indices_max
      nb_choices = min(nb_values, len(indices_max)) # Limit nb_choices to the actual number of available choices
      ind = indices_max[random.randint(0, nb_choices - 1)]

    else:
      print("Invalid search type.")
      return None

    tasks, agents = possible_switches[ind]
    additional_gain = additional_gains[ind]
    m, t, c, r, b = instance
    b_residual = b - (current_affectation_matrix * r).sum(axis=1)
    current_affectation_matrix = affectation_matrix_after_switch(current_affectation_matrix, tasks, agents)

    current_additional_gain += additional_gain

    if (pb == "max" and current_additional_gain > best_additional_gain) or (pb == "min" and current_additional_gain < best_additional_gain):
      best_additional_gain = current_additional_gain
      best_affectation_matrix = current_affectation_matrix.copy()

    possible_switches, additional_gains = switches_gains(current_affectation_matrix, instance, neighbour_type)

  return best_affectation_matrix, best_additional_gain