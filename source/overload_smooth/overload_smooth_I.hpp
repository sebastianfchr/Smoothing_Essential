// This is included in "overload_smooth.hpp"

// #include "overload_smooth.hpp" // for readability, uncomment. Unnecessary,
//                                // since we're included

#include "param_call_utils.hpp" // toArray, call_function

template <typename T, typename... ArgTypes>
T all_tape_calls(SType<T> (*function)(const SType<ArgTypes> &...),
                 const SType<ArgTypes> &... x) {
  /*
   * receives T function(T...), and casts it to SType<T> function(SType<T>...)
   * Calls function(x...), Records contribution bt.c
      and multiplies them. Add those products up
   * As often as bt.next_round() returns true (so long as markers in tree).
   */

  global_branchtape<T>::new_tape(); // assign fresh tape
  T y = 0.;
  // call tape until all markers are removed from tree
  do {
    y += function(x...).get_value() * global_branchtape<T>::contribution();
  } while (global_branchtape<T>::next_round());
  return y;
}

// same with contribution (for contribution-counting. No identification of
// contributions!)
template <typename T, typename... ArgTypes>
std::tuple<T, std::vector<T>> all_tape_calls_with_contributions(
    SType<T> (*function)(const SType<ArgTypes> &...),
    const SType<ArgTypes> &... x) {

  global_branchtape<T>::new_tape();

  T y = 0;

  std::vector<T> ys_separate;
  std::vector<T> contributions;
  // std::vector<std::vector<bool>> path_identifiers;
  do {

    // (1) Go down the current path, adding the val*cont (as usual)
    y += function(x...).get_value() * global_branchtape<T>::contribution();

    // (2) record contribution of this path
    contributions.push_back(global_branchtape<T>::contribution());

  } while (global_branchtape<T>::next_round());

  return std::make_tuple(y, contributions);
}

template <typename T, typename... ArgTypes>
std::tuple<T, int>
all_tape_calls_cont_count(SType<T> (*function)(const SType<ArgTypes> &...),
                          const SType<ArgTypes> &... x) {

  global_branchtape<T>::new_tape();

  T y = 0;
  int num_evals = 0;
  std::vector<T> ys_separate;
  do {
    // (1) Go down the current path, adding the val*cont (as usual)
    y += function(x...).get_value() * global_branchtape<T>::contribution();
    // (2) record how often we evaluated
    num_evals++;

  } while (global_branchtape<T>::next_round());

  return std::make_tuple(y, num_evals);
}

template <typename T, typename... Ts>
std::array<T, sizeof...(Ts)>
gradient_adjoint(const T &upstream_gradient,
                 DCO_T<T> (*function)(const DCO_T<Ts> &...), const Ts &... xs) {
  //  std::array<I, T>
  std::array<T, sizeof...(Ts)> xs_arr = toArray<T>(xs...);

  DCO_M<T>::global_tape = DCO_TT<T>::create();
  DCO_M<T>::global_tape->sparse_interpret() = true;
  std::array<DCO_T<T>, sizeof...(Ts)> xs_dco_arr;
  // fill passive values
  for (int i = 0; i < sizeof...(Ts); i++) {
    xs_dco_arr[i] = xs_arr[i];
    DCO_M<T>::global_tape->register_variable(xs_dco_arr[i]);
  }

  DCO_T<T> y = call_function(function, xs_dco_arr);
  // explicitly:
  // DCO_T<T> y = call_function<sizeof...(Ts), DCO_T<T>, DCO_T<Ts>...>(function,
  //                                                                   xs_dco_arr);

  // set upstream-gradient and interpret
  dco::derivative(y) = upstream_gradient;
  DCO_M<T>::global_tape->interpret_adjoint();

  std::array<T, sizeof...(Ts)> ret;
  for (int i = 0; i < sizeof...(Ts); i++)
    ret[i] = dco::derivative(xs_dco_arr[i]);
  return ret;
};

template <typename T, typename... Ts>
std::array<T, sizeof...(Ts)>
gradient_tangent(const T &upstream_gradient,
                 DCO_TanType<T> (*function)(const DCO_TanType<Ts> &...),
                 const Ts &... xs) {

  // here, what happens is (double) SType<T> => gives T
  std::array<T, sizeof...(Ts)> xs_arr = toArray<T>(xs...);

  // DCO_M<T>::global_tape = DCO_TT<T>::create();
  std::array<DCO_TanType<T>, sizeof...(Ts)> xs_dco_arr;
  // fill passive values
  for (int i = 0; i < sizeof...(Ts); i++) {
    xs_dco_arr[i] = xs_arr[i];
  }

  std::array<T, sizeof...(Ts)> ret;
  for (int i = 0; i < sizeof...(Ts); i++) {
    dco::derivative(xs_dco_arr[i]) = 1.;
    DCO_TanType<T> y = call_function(function, xs_dco_arr);
    dco::derivative(xs_dco_arr[i]) = 0.;
    ret[i] = dco::derivative(y) *
             upstream_gradient; // multiply with upstream, it's vector * scalar
  }
  return ret;
};

// give a function, turn out its smothed gradient (tangent-mode)
template <typename T, typename... Argtypes>
std::array<T, sizeof...(Argtypes)> smoothed_gradient(
    const T &upstream_gradient,
    DCO_TanType<SType<T>> (*func)(const DCO_TanType<SType<Argtypes>> &... xs),
    const Argtypes &... xs) {

  std::array<DCO_TanType<SType<T>>, sizeof...(Argtypes)> xs_dco_stype =
      toArray<DCO_TanType<SType<T>>>(SType<T>(xs)...);

  std::array<T, sizeof...(Argtypes)> grad = {0.}; // only 0s

  for (int i = 0; i < sizeof...(Argtypes); i++) {
    // Tangent-mode: run once for each of the gradient entries
    dco::derivative(xs_dco_stype[i]) = SType<T>(1.);
    global_branchtape<T>::new_tape();

    // all tape calls! We cannot use the actual all_tape_calls,
    // because would sum the derivatives of y too (needed separately)
    do {
      DCO_TanType<SType<T>> y = call_function(func, xs_dco_stype);
      grad[i] +=
          dco::derivative(y).value * global_branchtape<T>::contribution();

    } while (global_branchtape<T>::next_round());

    dco::derivative(xs_dco_stype[i]) = SType<T>(0.);
  }
  // Multiply it with
  // the upstream gradient (assuming the upstream gradient is of shape (1,1) )
  for (auto &g : grad)
    g *= upstream_gradient;

  return grad;
}
