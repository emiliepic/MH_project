import numpy as np
import signal
from neighborhood_search import switches_gains, affectation_matrix_after_switch


# Function to handle the timeout
def timeout_handler(signum, frame):
    raise TimeoutError("The execution time has exceeded the allowed limit.")


# We move to the best closest improving neighbor. If none is found, we move on to the next
# neighborhood structure.
def variable_neighborhood_search(affectation_matrix, instance, nb_max_iterations=10000, max_cycle_size=3, time_limit=600, pb="max"):
  # Set up an alarm signal and a handler for the timeout
  signal.signal(signal.SIGALRM, timeout_handler)
  signal.alarm(time_limit)  # Set the alarm for 10 minutes

  try:
    best_affectation_matrix = affectation_matrix.copy()
    current_affectation_matrix = affectation_matrix.copy()
    current_additional_gain = 0
    best_additional_gain = 0

    for k in range(nb_max_iterations):
      cycle_size = 0
      possible_switches, additional_gains = switches_gains(current_affectation_matrix, instance, cycle_size)

      # Find the first improving neighbor or increase the cycle size
      while not possible_switches:
        cycle_size += 1
        if cycle_size > max_cycle_size:
          return best_affectation_matrix, best_additional_gain
        possible_switches, additional_gains = switches_gains(current_affectation_matrix, instance, cycle_size)

      # Select the best neighbor according to the objective (maximization or minimization)
      if pb == "max":
        ind = np.argmax(additional_gains)
        improving_neighbor = additional_gains[ind] > 0
      else:  # pb == "min"
        ind = np.argmin(additional_gains)
        improving_neighbor = additional_gains[ind] < 0

      # Increase cycle size if no improving neighbor is found
      while not improving_neighbor:
        cycle_size += 1
        if cycle_size > max_cycle_size:
          return best_affectation_matrix, best_additional_gain
        possible_switches, additional_gains = switches_gains(current_affectation_matrix, instance, cycle_size)

        if possible_switches:
          if pb == "max":
            ind = np.argmax(additional_gains)
            improving_neighbor = additional_gains[ind] > 0
          else:  # pb == "min"
            ind = np.argmin(additional_gains)
            improving_neighbor = additional_gains[ind] < 0

      tasks, agents = possible_switches[ind]
      additional_gain = additional_gains[ind]
      current_affectation_matrix = affectation_matrix_after_switch(current_affectation_matrix, tasks, agents)
      current_additional_gain += additional_gain

      # Update the best solution found according to the objective
      if (pb == "max" and current_additional_gain > best_additional_gain) or (pb == "min" and current_additional_gain < best_additional_gain):
        best_additional_gain = current_additional_gain
        best_affectation_matrix = current_affectation_matrix.copy()

    return best_affectation_matrix, best_additional_gain

  except TimeoutError:
    # Convert the time limit into minutes and seconds
    minutes = time_limit // 60
    seconds = time_limit % 60
    print(f"The local search stopped after reaching the {minutes} minute(s) and {seconds} second(s) time limit.")
    return best_affectation_matrix, best_additional_gain

  finally:
    # Disable the alarm after execution
    signal.alarm(0)



# Move to the first improving neighbor.
# Check each potential neighbor as soon as it's found in the list of neighbors, and move to it immediately if it improves the solution.
# This will make the local search faster as it won't need to explore all neighbors before choosing a move.

def variable_neighborhood_search_first_neighbor(affectation_matrix, instance, nb_max_iterations=10000, max_cycle_size=3, time_limit=600, pb="max"):
  # Define the alarm signal and handler for timeout
  signal.signal(signal.SIGALRM, timeout_handler)
  signal.alarm(time_limit)  # Set the alarm to 10 minutes

  try:
    best_affectation_matrix = affectation_matrix.copy()
    current_affectation_matrix = affectation_matrix.copy()
    current_additional_gain = 0
    best_additional_gain = 0

    for k in range(nb_max_iterations):
      cycle_size = 0
      found_improving_neighbor = False

      while cycle_size <= max_cycle_size and not found_improving_neighbor:
        possible_switches, additional_gains = switches_gains(current_affectation_matrix, instance, cycle_size)

        # Traverse neighbors to find the first improving neighbor
        for ind, gain in enumerate(additional_gains):
          if (pb == "max" and gain > 0) or (pb == "min" and gain < 0):
            # Apply the first improving neighbor found
            tasks, agents = possible_switches[ind]
            additional_gain = gain
            current_affectation_matrix = affectation_matrix_after_switch(current_affectation_matrix, tasks, agents)
            current_additional_gain += additional_gain
            found_improving_neighbor = True

            # Update the best solution found if it improves
            if (pb == "max" and current_additional_gain > best_additional_gain) or (pb == "min" and current_additional_gain < best_additional_gain):
              best_additional_gain = current_additional_gain
              best_affectation_matrix = current_affectation_matrix.copy()
            break  # Exit the loop once the first improving neighbor is applied

        # Increase the cycle size if no improving neighbor is found
        cycle_size += 1

      # Stop if no improving neighbor was found in all neighborhood cycles
      if not found_improving_neighbor:
        break

    return best_affectation_matrix, best_additional_gain

  except TimeoutError:
    # Convert the time limit into minutes and seconds
    minutes = time_limit // 60
    seconds = time_limit % 60
    print(f"The local search stopped after reaching the {minutes} minute(s) and {seconds} second(s) time limit.")
    return best_affectation_matrix, best_additional_gain

  finally:
    # Disable the alarm after execution
    signal.alarm(0)