#include <array>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

#ifndef param_call_utils_h
#define param_call_utils_h
// parameters to array
// toArray(a,b,c,d) == {{a,b,c,d}};
template <typename T, typename... Ts> // array from parameters
std::array<T, sizeof...(Ts)> toArray(const Ts &... x) {
  std::array<T, sizeof...(Ts)> arr{{(x)...}};
  return arr;
}

// ==========================================================================
//          Call a function array, whose entries become parameters
// ==========================================================================
// helper for call_cfunction
// Agrument of integer-sequence makes template deduce indices of Ts
template <typename T, std::size_t... Ts, typename... Argtypes>
T call_function_helper(std::integer_sequence<std::size_t, Ts...>,
                       T (*function)(const Argtypes &...),
                       std::array<T, sizeof...(Ts)> arr) {

  return function(arr[Ts]...);
}

// call_function_helper(func, arr) => func(arr[0], arr[1], ..., arr[n])
template <std::size_t I, typename T, typename... Argtypes>
T call_function(T (*function)(const Argtypes &...),
                std::array<T, I> array_args) {
  return call_function_helper(std::make_index_sequence<sizeof...(Argtypes)>{},
                              function, array_args);
}

// ==========================================================================
//                    Array assigner: array to parameters
// ==========================================================================
template <size_t I, typename T>
void arrayAssignerH(const std::array<T, I> &a, const size_t &, T &b) {
  b = a[I - 1];
}

template <size_t I, typename T, typename... Argtypes>
void arrayAssignerH(const std::array<T, I> &a, const size_t &currInd, T &b,
                    Argtypes &... bs) {
  b = a[currInd];
  arrayAssignerH<I, Argtypes...>(a, currInd + 1, bs...);
}
// arrayAssigner({{1,2,3,4}}, a, b, c, d) => a==1; b==2; c==3; d==4;
template <size_t I, typename T, typename... Argtypes>
void arrayAssigner(const std::array<T, I> &a, Argtypes &... bs) {
  arrayAssignerH(a, 0, bs...);
}

#else
#endif