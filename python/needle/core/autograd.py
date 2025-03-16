from .base import Value
from ..type import *

def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Value, List[Value]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
    # print_node_list(reverse_topo_order)
    ### BEGIN YOUR SOLUTION
    for node in reverse_topo_order:
      node.grad = sum_node_list(node_to_output_grads_list[node])
      if node.op is None: # leaf node
        continue
      gradients = node.op.gradient_as_tuple(node.grad, node)
      n_inputs = len(node.inputs)
      for i in range(n_inputs):
        if node.inputs[i] not in node_to_output_grads_list.keys():
          node_to_output_grads_list[node.inputs[i]] = []
        node_to_output_grads_list[node.inputs[i]].append(gradients[i])
    ### END YOUR SOLUTION


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    visited = set()
    topo_order = []
    for node in node_list:
      topo_sort_dfs(node, visited, topo_order)
    return topo_order
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    if id(node) in visited:
      return
    if node.is_leaf():
      visited.add(id(node))
      topo_order.append(node)
      return
    for ipt in node.inputs:
        topo_sort_dfs(ipt, visited, topo_order)
    visited.add(id(node))
    topo_order.append(node)
    ### END YOUR SOLUTION


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)

def print_node_list(node_list: List[Value]):
    for node in node_list:
        print(10*'=')
        print(f'node.cached_data: {node.cached_data}')
        print(f'node.op: {node.op}')
        print(f'node.inputs: {node.inputs}')