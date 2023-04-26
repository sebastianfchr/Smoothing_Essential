#include "overload_smooth.hpp"
#include <array>
#include <tuple>
#include <utility>

// #include <bits/stdc++.h>

template <typename T> STypeResult<T>::STypeResult(bool absolute, T tendency) {
  this->absolute = absolute;
  this->tendency = tendency;
}

template <typename T> const bool STypeResult<T>::get_absolute() const {
  return this->absolute;
}

template <typename T> const T &STypeResult<T>::get_tendency() const {
  return this->tendency;
}

// abs_flips_contribution: flip the contribution if abs_a == false (0)
template <typename T> T a_f_c(const T &contribution_a, const bool &abs_a) {
  // std::cout << contribution_a << "!!!!!!" << std::endl;
  return (!abs_a) + pow(-1, !abs_a) * contribution_a;
  // return abs_a ? contribution_a : 1 - contribution_a;
  //(!abs_a) + pow(-1, !abs_a) * contribution_a;
}

template <typename T>
std::tuple<bool, T>
and_absolute_and_tendential(const T &cont_a, const T &cont_b, const bool &abs_a,
                            const bool &abs_b) {

  bool absolute = abs_a && abs_b;
  // note that a_f_c(cont_a, abs_a) * a_f_c(cont_b, abs_b)
  T tendency_if_absolute_true = a_f_c(cont_a, abs_a) * a_f_c(cont_b, abs_b);
  T tendency = a_f_c(tendency_if_absolute_true, absolute);
  return std::make_tuple(absolute, tendency);
};

template <typename T>
STypeResult<T> operator&&(const STypeResult<T> &a, const STypeResult<T> &b) {
  // auto [new_absolute, new_tendency]
  std::tuple<bool, T> new_absolute_new_tendency = and_absolute_and_tendential(
      a.get_tendency(), b.get_tendency(), a.get_absolute(), b.get_absolute());
  bool new_absolute = std::get<0>(new_absolute_new_tendency);
  T new_tendency = std::get<1>(new_absolute_new_tendency);
  return STypeResult<T>(new_absolute, new_tendency);
};

template <typename T>
STypeResult<T> operator&&(const STypeResult<T> &a, const bool &b) {
  // TODO: Is this correct? or do we need the general method?
  bool new_absolute = a.get_absolute() && b;
  T b_tendency = (T)b;
  T new_tendency = a.get_tendency() * b_tendency;
  return STypeResult<T>(new_absolute, new_tendency);
};

template <typename T>
STypeResult<T> operator&&(const bool &b, const STypeResult<T> &a) {
  return a && b;
};

template <typename T>
std::tuple<bool, T> or_absolute_and_tendential(T cont_a, T cont_b, bool abs_a,
                                               bool abs_b) {

  bool absolute = abs_a || abs_b;
  T tendency_if_absolute_true =
      a_f_c(cont_a, abs_a) * a_f_c(cont_b, abs_b) +
      a_f_c((T)(1. - cont_a), abs_a) * a_f_c(cont_b, abs_b) +
      a_f_c(cont_a, abs_a) * a_f_c((T)(1. - cont_b), abs_b);
  T tendency = a_f_c(tendency_if_absolute_true, absolute);

  return std::make_tuple(absolute, tendency);
};

template <typename T>
STypeResult<T> operator||(const STypeResult<T> &a, const STypeResult<T> &b) {
  // bool, T
  // auto [new_absolute, new_tendency]
  std::tuple<bool, T> new_absolute_new_tendency = or_absolute_and_tendential(
      a.get_tendency(), b.get_tendency(), a.get_absolute(), b.get_absolute());
  bool new_absolute = std::get<0>(new_absolute_new_tendency);
  T new_tendency = std::get<1>(new_absolute_new_tendency);
  return STypeResult<T>(new_absolute, new_tendency);
};

template <typename T>
STypeResult<T> operator||(const STypeResult<T> &a, const bool &b) {
  const bool new_absolute = a.get_absolute() || b;
  T b_tendency = (T)b;
  // TODO: Is this correct? b_tendency always 0 or 1
  T new_tendency =
      a.get_tendency() + b_tendency - a.get_tendency() * b_tendency;
  return STypeResult<T>(new_absolute, new_tendency);
};

template <typename T>
STypeResult<T> operator||(const bool &b, const STypeResult<T> &a) {
  return a || b;
};

template <typename T> STypeResult<T> operator!(const STypeResult<T> &a) {
  bool new_absolute = !(a.get_absolute());
  // why does tendency not need to be changed? Because the algorithm checks the
  // truth-value and then decides whether to look for complement
  T new_tendency = a.get_tendency();
  return STypeResult<T>(new_absolute, new_tendency);
};

template <typename T> STypeResult<T>::operator bool() const {
  // std::cout << "cast as bool!" << std::endl;
  // casting this as a bool should only happen inside if(...),
  // ... and will trigger writing into the tape
  return global_branchtape<T>::check(this->absolute, this->tendency);
}

// per preprocessor macro
#define EXPLICITLY_INSTANTIATE(T)                                              \
  template class STypeResult<T>;                                               \
  template STypeResult<T> operator||(const STypeResult<T> &a, const bool &b);  \
  template STypeResult<T> operator||(const STypeResult<T> &a,                  \
                                     const STypeResult<T> &b);                 \
  template STypeResult<T> operator&&(const STypeResult<T> &a, const bool &b);  \
  template STypeResult<T> operator&&(const STypeResult<T> &a,                  \
                                     const STypeResult<T> &b);                 \
  template STypeResult<T> operator!(const STypeResult<T> &a);                  \

// EXPLICITLY_INSTANTIATE(DCO_T<SType<double>>);
EXPLICITLY_INSTANTIATE(DCO_TanType<double>);
EXPLICITLY_INSTANTIATE(DCO_T<double>);
EXPLICITLY_INSTANTIATE(double);

// EXPLICITLY_INSTANTIATE(DCO_T<SType<float>>);
EXPLICITLY_INSTANTIATE(DCO_TanType<float>);
EXPLICITLY_INSTANTIATE(DCO_T<float>);
EXPLICITLY_INSTANTIATE(float);