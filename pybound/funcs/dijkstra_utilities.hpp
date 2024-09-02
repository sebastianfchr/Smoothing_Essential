#pragma once 
#include <vector>

// take a predecessor array, and print the path to a 
// preds has indices from 0
// Note: This works IFF Dijkstra founds a path [from, ..., to], else it will hang!
template<typename T>
std::vector<T> path_from_pred(std::vector<T> preds, std::size_t from, std::size_t to) {

  assert(from < preds.size() && to < preds.size()); // be within bounds, but that's not all

  std::vector<T> v;
  v.push_back(to);

  while(to != from) {
    to = preds[to];
    v.insert(v.begin(), to); // really inefficient, maybe redo
  }

  return v;

}
