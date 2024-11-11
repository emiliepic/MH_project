import numpy as np
import evaluation

# Heuristic affectation function for criteria 1
def best_agent_available_for_task_min_necessary_ressource(task, b_residual, r):
  m = len(b_residual)
  list_available_agents = [agent for agent in range(m) if b_residual[agent] >= r[agent][task]]
  if len(list_available_agents) == 0:
    return None
  # Calculate required resources for each available agent and return the one with the minimum requirement
  necessary_ressources = np.array([r[agent][task] for agent in list_available_agents])
  return list_available_agents[np.argmin(necessary_ressources)]


# Heuristic affectation function for criteria 2
def best_agent_available_for_task_max_ratio_profit_ressource(task, b_residual, r, c, pb="max"):
  m = len(b_residual)
  list_available_agents = [agent for agent in range(m) if b_residual[agent] >= r[agent][task]]
  if len(list_available_agents) == 0:
    return None

  if pb == "min":
    # Calculate the product of profit and resource requirement for available agents and choose minimum
    ratio = np.array([c[agent][task]*r[agent][task] for agent in list_available_agents])
    return list_available_agents[np.argmin(ratio)]
  else:
    # Calculate profit-to-resource ratio for available agents and choose maximum
    ratio = np.array([c[agent][task]/r[agent][task] for agent in list_available_agents])
    return list_available_agents[np.argmax(ratio)]



def greedy_heuristic(instance, criteria=1, pb="max"):
  m, t, c, r, b = instance
  affectation_matrix = np.zeros((m, t))
  b_residual = b.copy()
  
  for task in range(t):
    if criteria == 1:
      # Use minimum necessary resource criteria to select agent
      agent = best_agent_available_for_task_min_necessary_ressource(task, b_residual, r)
    elif criteria == 2:
      # Use maximum profit-to-resource ratio criteria to select agent
      agent = best_agent_available_for_task_max_ratio_profit_ressource(task, b_residual, r, c, pb=pb)
    else:
      print("Invalid criteria.")
      return None

    if agent is not None:
      b_residual[agent] -= r[agent][task]
      affectation_matrix[agent][task] = 1
    else:
      print("The heuristic does not provide a feasible solution.")
      return None

  return affectation_matrix, evaluation.gain(affectation_matrix, instance)

