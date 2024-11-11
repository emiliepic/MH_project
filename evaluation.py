import numpy as np

def gain(affectation_matrix, instance):
  m, t, c, r, b = instance
  return np.sum(affectation_matrix * c)  # Sum of profits for all assigned tasks


def is_affectation_matrix_ok(affectation_matrix, instance):
  m, t, c, r, b = instance
  b_residual = b - (affectation_matrix * r).sum(axis=1)
  return all(x >= 0 for x in b_residual)  # True if all agents have sufficient resources
