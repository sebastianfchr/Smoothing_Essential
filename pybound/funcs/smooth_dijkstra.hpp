#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include "SType.hpp"
#include "dijkstra_utilities.hpp"


template<typename T>
std::tuple<std::vector<SType<T>>, std::vector<int>> Dijkstra(std::vector<std::vector<SType<T>>>& cost, int from) {

  std::size_t N = cost.size();

  // std::vector<SType<T>> distance; for(int i=0; i<N; i++) distance.push_back(0); // but this is somehow equivalent...?
  std::vector<SType<T>> distance(N);

  SType<T> mindistance; 
  
  std::vector<int> pred(N);
  int visited[N], count, nextnode, i, j;

  for (i = 0; i < N; i++) {
    distance[i] = cost[from][i]; // Note that distance==INF means "not discovered yet" (if link exists)
    pred[i] = from;
    visited[i] = 0;
  }
  

  distance[from] = 0;
  visited[from] = 1;
  count = 1;

  // SEB: This is where the actual Algorithm begins. 
  while (count < N - 1) {
    mindistance = INFINITY; // inf < inf => disc (0)

    // check for the unvisited node with the shortest path. Will discover from there!
    for (i = 0; i < N; i++){
      if (discrete(distance[i] < mindistance && !visited[i])) {  // a) smoothing here means to consider more distant alternatives earlier [invalidates guarantees of algorithm?]           
        mindistance = distance[i];
        nextnode = i;
      }
    }

    // mark node as visited and check whether path through this node shortens distances of other unvisited nodes
    visited[nextnode] = 1;
    for (i = 0; i < N; i++)
      if (!visited[i]) {
        IF (mindistance + cost[nextnode][i] < distance[i]) {    // c) smoothing here means to split control-flow if distance is "close" 
          distance[i] = mindistance + cost[nextnode][i];        // c.1
          pred[i] = nextnode;                                   // c.2
        }
      }

    count++;
  } 

  return std::make_tuple(distance, pred); // The idea would be to have this as an SType-return, increment an SOutType from it...
}


template<typename T, std::size_t N>
using square_matrix = typename std::array<std::array<T, N>, N>;

template<typename T> requires is_numeric_v<T>
auto dijkstra_full(std::vector<std::vector<T>> cost, std::size_t from, std::size_t to) {
  // TODO: Check bounds of from, to!
  std::size_t N = cost.size();
  if( ! ((0 <= from && from < N) && (0 <= to && to < N)) ) { throw new std::logic_error("from and to have to be within bounds of 'cost' matrix" ); }

  // Filtering cost matrix 0 => inf
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) 
      if (cost[i][j]== 0) cost[i][j] = std::numeric_limits<double>::infinity();
      


  auto tape_ptr = tape::make_smart_tape_ptr();


  // fill SType cost matrix
  std::vector<std::vector<SType<T>>> cost_matrix;
  // because of our (potentially stupid) init-system, all of these have to be pushed back manually
  for(int i=0; i<N; i++) {
    std::vector<SType<T>> rowi;
    for(int j=0; j<N; j++) {
      rowi.push_back(SType<T>(cost[i][j]));
    }
    cost_matrix.push_back(rowi);
  }

  SOutType o; // for smoothing result and backprop

  std::vector<std::vector<int>> paths;
  std::vector<T> contributions;
  std::vector<std::vector<T>> distances_per_iteration;

  SMOOTHING();
    
    // run dijkstra and take the distance
    auto&& [distances, pred] = Dijkstra(cost_matrix, from);
    o = distances[to];


    // path for each branching-scenario
    // (path will be an empty vector unless 'to' can be reached from 'from')
    std::vector<int> path;
    if (value(distances[to]) != std::numeric_limits<double>::infinity() ) path = path_from_pred(pred, from, to);
    paths.push_back(path);

    // log this path's contribution
    contributions.push_back(tape_ptr->get_contribution());

    // 
    std::vector<T> distances_out; 
    for(auto&& ds : distances) distances_out.push_back(value(ds)); 
    distances_per_iteration.push_back(distances_out);



  SMOOTHING_END();

  o.seed(1.);


  // final result, incremented step-wise
  std::vector<std::vector<T>> cost_matrix_deriv(N, std::vector<T>(N, 0));

  // save the increments for further inspection
  std::vector<std::vector<std::vector<T>>> cost_matrix_deriv_increments; 

  BACKPROP_BEGIN();

    // harvest derivative increment per iteration
    std::vector<std::vector<T>> cost_matrix_deriv_increment(N, std::vector<T>(N));
    for(int i=0; i<N; i++){
      for(int j=0; j<N; j++){
        cost_matrix_deriv_increment[i][j] = derivative(cost_matrix[i][j])-cost_matrix_deriv[i][j];
        cost_matrix_deriv[i][j] += cost_matrix_deriv_increment[i][j];
      }
    }
    cost_matrix_deriv_increments.push_back(cost_matrix_deriv_increment);


  BACKPROP_END();
  // std::cout << o.get_value() << std::endl;
  // //  END REMOVE



  // returns: do/dcost_matrix, paths, distances
  return std::make_tuple(cost_matrix_deriv, cost_matrix_deriv_increments, distances_per_iteration, paths, contributions);

}

