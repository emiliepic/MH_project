from data import read_data
from heuristics import greedy_heuristic
from evaluation import gain, is_affectation_matrix_ok
from neighborhood_search import neighborhood_search
from variable_neighborhood_search import variable_neighborhood_search, variable_neighborhood_search_first_neighbor
from taboo_search import taboo_search


print("\n Maximisation example")
# Read data from the specified file
instances = read_data("gap1.txt")
instance_1 = instances[1][0]  # Access the first element of the first instance

print("\n Greedy heuristic")
affectation_matrix_criteria_1, gain_criteria_1 = greedy_heuristic(instance_1, criteria=1)
affectation_matrix_criteria_2, gain_criteria_2 = greedy_heuristic(instance_1, criteria=2)

print("The gain of the solution for criteria 1 is ", gain_criteria_1)
print("Is the solution feasible for criteria 1 ?", is_affectation_matrix_ok(affectation_matrix_criteria_1, instance_1))
print("The gain of the solution for criteria 2 is ", gain_criteria_2)
print("Is the solution feasible for criteria 2 ?", is_affectation_matrix_ok(affectation_matrix_criteria_2, instance_1))

print("\n Neighborhood search from the greedy solution using criteria 1")   
best_affectation_matrix, best_additional_gain = neighborhood_search(affectation_matrix_criteria_1, instance_1, nb_max_iterations=20, search_type="random_nb", nb_values=10, neighbour_type=0)
print("Is the solution feasible ?", is_affectation_matrix_ok(best_affectation_matrix, instance_1))
print("The gain of the solution is ", gain(best_affectation_matrix, instance_1))

print("\n Variable Neighborhood search from the greedy solution using criteria 2")
best_affectation_matrix, best_additional_gain = variable_neighborhood_search(affectation_matrix_criteria_2, instance_1, nb_max_iterations=100, max_cycle_size=4)
print("Is the solution feasible ?", is_affectation_matrix_ok(best_affectation_matrix, instance_1))
print("The gain of the solution is ", gain(best_affectation_matrix, instance_1))

print("\n Variable Neighborhood search using the first improving neighbor from the greedy solution using criteria 2")
best_affectation_matrix, best_additional_gain = variable_neighborhood_search_first_neighbor(affectation_matrix_criteria_2, instance_1, nb_max_iterations=100, max_cycle_size=4)
print("Is the solution feasible ?", is_affectation_matrix_ok(best_affectation_matrix, instance_1))
print("The gain of the solution is ", gain(best_affectation_matrix, instance_1))

print("\n Tabou search using the greedy solution using criteria 2")
best_affectation_matrix, best_additional_gain = taboo_search(affectation_matrix_criteria_2, instance_1, nb_max_iterations=5, cycle_size=1, search_type="random", nb_values=5, pb="max", tabou_type="tasks", tabou_size=2, aspiration=True)
print("Is the solution feasible ?", is_affectation_matrix_ok(best_affectation_matrix, instance_1))
print("The gain of the solution is ", gain(best_affectation_matrix, instance_1))


print("\n Minimisation example")
# Read data from the specified file (minimization)
instances_min = read_data("gapa.txt")
instance_1_min = instances_min[1][0]  # Access the first element of the first instance

print("\n Greedy heuristic")
affectation_matrix_criteria_1, gain_criteria_1 = greedy_heuristic(instance_1_min, criteria=1, pb="min")
print("The gain of the solution for criteria 1 is ", gain_criteria_1)

print("\n Neighborhood search from the greedy solution using criteria 1")
best_affectation_matrix, best_additional_gain = neighborhood_search(affectation_matrix_criteria_1, instance_1_min, nb_max_iterations=20, search_type="random_nb", nb_values=10, neighbour_type=0, pb="min")
print("Is the solution feasible ?", is_affectation_matrix_ok(best_affectation_matrix, instance_1_min))
print("The gain of the solution is ", gain(best_affectation_matrix, instance_1_min))

print("\n Variable Neighborhood search from the greedy solution using criteria 1")
best_affectation_matrix, best_additional_gain = variable_neighborhood_search(affectation_matrix_criteria_1, instance_1_min, nb_max_iterations=100, max_cycle_size=4, pb="min")
print("Is the solution feasible ?", is_affectation_matrix_ok(best_affectation_matrix, instance_1_min))
print("The gain of the solution is ", gain(best_affectation_matrix, instance_1_min))

print("\n Variable Neighborhood search using the first improving neighbor from the greedy solution using criteria 1")
best_affectation_matrix, best_additional_gain = variable_neighborhood_search_first_neighbor(affectation_matrix_criteria_1, instance_1_min, nb_max_iterations=100, max_cycle_size=4, pb="min")
print("Is the solution feasible ?", is_affectation_matrix_ok(best_affectation_matrix, instance_1_min))
print("The gain of the solution is ", gain(best_affectation_matrix, instance_1_min))

print("\n Tabou search using the greedy solution using criteria 1")
best_affectation_matrix, best_additional_gain = taboo_search(affectation_matrix_criteria_1, instance_1_min, nb_max_iterations=5, cycle_size=1, search_type="random", nb_values=5, pb="min", tabou_type="tasks", tabou_size=2, aspiration=True)
print("Is the solution feasible ?", is_affectation_matrix_ok(best_affectation_matrix, instance_1_min))