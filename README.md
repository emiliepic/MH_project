# Metaheuristic Project


## Data
read_data(file, path=DATA_PATH): reads the data

## Evaluation
gain(affectation_matrix, instance): compute the total gain for an affectation_matrix
is_affectation_matrix_ok(affectation_matrix, instance): check if the affectation_matrix is a feasible solution

## Heuristics
greedy_heuristic(instance, criteria=1, pb="max"): returns the affectation matrix determined using different criteria:
- criteria 1: choose the agent that needs the minimum ressource needed for a task
- criteria 2: choose the agent that needs the maximum ration profit/ressource needed for a task

## q2
Runs the greedy_heuristic on the example instance of question 2.

## Neighborhood_search
neighborhood_search(affectation_matrix, instance, nb_max_iterations=10000, neighbour_type=0, search_type="random", nb_values=5, pb="max"):
- neighbor_type:
    - 0: switching task from one agent to the other
    - 1: switching two tasks between two agents
    - N >= 2: switching N tasks between N agents
- search_type:
    - "best": Choose the best switch
    - "random": Choose a completely random switch
    - "nb_random": Choose a random switch between the nb_values best switchess

## Variable_neighborhood_search
variable_neighborhood_search(affectation_matrix, instance, nb_max_iterations=10000, max_cycle_size=3, time_limit=600, pb="max"): We move to the best closest improving neighbor. If none is found, we move on to the next neighborhood structure not exceeding max_cycle_size.
variable_neighborhood_search_first_neighbor(affectation_matrix, instance, nb_max_iterations=10000, max_cycle_size=3, time_limit=600, pb="max"): Move to the first improving neighbor. Check each potential neighbor as soon as it's found in the list of neighbors, and move to it immediately if it improves the solution. This will make the local search faster as it won't need to explore all neighbors before choosing a move.


## taboo_search
taboo_search(affectation_matrix, instance, nb_max_iterations=10000, cycle_size=3, search_type="random", nb_values=5, pb="max", tabou_type="tasks", tabou_size=3, aspiration=False): The tabou list is composed of the tasks that cant be switch or the agents

## Examples
Provide an example using all the previously defined functions on an instance of maximization and an instance of minimization
