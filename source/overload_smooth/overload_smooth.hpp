#ifndef ODERLOAD_SMOOTH_H
#define ODERLOAD_SMOOTH_H
// integrates: branchTape.cpp, stype.cpp

#include "../dco_shorthands.h"
#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

/* ===================================================================
 * Arithmetic type for smoothing (SType) for type-generic functions
 * All used mathematical operators are overloaded for this type. In-out!
 * When it is compared, its contribution changes
 * ===================================================================
 */

template <typename T> class SType; // forward-declare

// // // go somewhere else! (?)
// template <typename T> struct nested_s { static constexpr bool val = false; };
// template <typename T> struct nested_s<SType<T>> {
//   static constexpr bool val = true;
// };

class smoothing_params { // 'static class' to adapt the smooting_factor from the
                         // outside
public:
  static double smfactor;
  static void set_smfactor(double smfactor);
  template <typename U> static const U get_smfactor();

private:
  smoothing_params();
};

struct marker {
  int counterPosition;
  bool notFalsified; // this is false => falsified!
};

template <typename T>
class branchTape { // instance of what we have globally too!
public:
  T contribution; // contribution. Accumulated downwards, reset to 1 in
                  // next run!
  std::vector<marker>
      markers; // MARKERS: position of branches with 'close' T/F-value
  int markerIndexCounter; // marker-markerIndexCounter (array-pos of marker): at
                          // which #comparison are we
  int last_falsified_marker_pos; // the last falsified marker is *after* where
                                 // we record new markers

  branchTape();

  // Prepare the next round:
  // * consume sequence of bottom falsified nodes
  // * falsify upmost true node
  // ... or give back false, if upmost falsified node is consumed
  bool next_round();

  // check performed by overloaded operators.
  // In: result and tendency given by operator. Out: T/F decision based on
  // operator-outcome and tape
  bool check(const bool &absolute, const T &tendency);
};

template <typename T> class global_branchtape {
public:
  static bool next_round();
  static bool check(const bool &absolute,
                    const T &tendency); // Check comparison
  static void new_tape();
  static T &contribution();

private:
  static branchTape<T> bt; // underlying branchTape provides functionality
  global_branchtape();     // static class. Hide default initializer!
};

/*
  * Here, predeclare. Declared in the overload_smoothI.h-file!
  * function is called on different paths depending on the
    markers set by the branchTape, until all markers are consumed
  * As function will be T f(T,...), Argtypes are a series of Ts
 */
template <typename T, typename... ArgTypes>
T all_tape_calls(SType<T> (*function)(const SType<ArgTypes> &...),
                 const SType<ArgTypes> &... x);

/*
 * Result of comparisons involving an SType, or operators on STypeResult
 */
template <typename T> class STypeResult {
public:
  STypeResult(bool absolute, T tendential);
  operator bool() const; // casting will trigger evaluation. DIRTY TRICK!
                         // because we know that if() always casts inside!
  const bool get_absolute() const; // (leading const not necessary, since bool)
  const T &get_tendency() const;

private:
  bool absolute;
  T tendency;
};

template <typename T>
STypeResult<T> operator||(const STypeResult<T> &a, const STypeResult<T> &b);

template <typename T>
STypeResult<T> operator||(const STypeResult<T> &a, const bool &b);

template <typename T>
STypeResult<T> operator||(const bool &b, const STypeResult<T> &a);

template <typename T>
STypeResult<T> operator&&(const STypeResult<T> &a, const STypeResult<T> &b);

template <typename T>
STypeResult<T> operator&&(const STypeResult<T> &a, const bool &b);

template <typename T>
STypeResult<T> operator&&(const bool &b, const STypeResult<T> &a);

template <typename T> STypeResult<T> operator!(const STypeResult<T> &a);

template <typename T> bool eval(const STypeResult<T> &res);

template <typename T> class SType {
  /* The overloaded 'wrapper' containing the datatype.
  Overloaded operations will trigger the tape-calls in appropriate cases */
public:
  T get_value() const { return value; }

  SType();
  SType(const T &value);
  // template <typename U> SType(const U &value);
  // SType(const double &value);
  // SType(const T &value);
  // template<typename U> SType(const U &value);
  template <typename U> SType<T> &operator=(const U &b);
  SType<T> &operator+=(const SType<T> &b);
  SType<T> &operator+=(const T &b);
  SType<T> &operator-=(const SType<T> &b);
  SType<T> &operator-=(const T &b);
  SType<T> &operator*=(const SType<T> &b);
  SType<T> &operator*=(const T &b);
  // Cast dco-type to containing generic T gets the underlying value
  operator T() const { return (T)dco::passive_value(value); } // cast
  T value;
};

template <typename T> T value_type(const SType<T> &x) { return x.get_value(); }
template <typename T> T value_type(const T &x) { return x; }

// Choose the dco-type of the two, if either is a dco-type
// To represent explicitly that DCO_T<float> * double = DCO_T<float>
// Not perfect, but ok
template <typename T1, typename T2> struct either_dco_type {
  static constexpr bool bt1 = dco::mode<T1>::is_dco_type;
  typedef typename std::conditional<bt1, T1, T2>::type type;
};
// ... and abbreviate
template <typename T1, typename T2>
using eitherD = typename either_dco_type<T1, T2>::type;

template <typename T, typename U>
auto operator+(const SType<T> &a, const SType<U> &b) {
  // std::cout << "aaa" << std::endl;
  eitherD<T, U> res = a.get_value() + b.get_value();
  // return SType<typename either_dco_type<T, U>::type>(res);
  return SType<eitherD<T, U>>(res);
};

// a is SType, b isn't
template <typename T, typename U>
auto operator+(const SType<T> &a, const U &b) {
  // std::cout << "bbb" << std::endl;
  return SType<eitherD<T, U>>(a.get_value() + b);
};

// a is SType, b isn't
template <typename T, typename U>
auto operator+(const T &b, const SType<U> &a) {
  // std::cout << "ccc" << std::endl;
  return SType<eitherD<T, U>>(a.get_value() + b);
};

template <typename T, typename U>
auto operator-(const SType<T> &a, const SType<U> &b) {
  return SType<eitherD<T, U>>(a.get_value() - b.get_value());
};

template <typename T, typename U>
auto operator-(const SType<T> &a, const U &b) {
  return SType<eitherD<T, U>>(a.get_value() - b);
};

template <typename T, typename U> // note the sequence!
auto operator-(const T &b, const SType<U> &a) {
  return SType<eitherD<T, U>>(b - a.get_value());
};

template <typename T> SType<T> operator-(const SType<T> &a) {
  return SType<T>(-a.get_value());
};

// // rather use a is_same<U,T>, and figure out taking
template <typename T, typename U>
auto operator*(const SType<T> &a, const SType<U> &b) {
  // std::cout << "ooo" << std::endl;
  return SType<eitherD<T, U>>(a.get_value() * b.get_value());
};

template <typename T, typename U>
auto operator*(const SType<T> &a, const U &b) {
  return SType<eitherD<T, U>>(a.get_value() * b);
};

template <typename T, typename U>
auto operator*(const U &b, const SType<T> &a) {
  return SType<eitherD<T, U>>(
      a.get_value() * b); // value_type, in case the other is also a SType!
};

template <typename T, typename U> // not ideal! get the dominant one of U,T
auto operator/(const SType<T> &a, const SType<U> &b) {
  return SType<eitherD<T, U>>(a.get_value() / b.get_value());
};
template <typename T, typename U>
auto operator/(const SType<T> &a, const U &b) {
  return SType<eitherD<T, U>>(a.get_value() / b);
};
template <typename T, typename U>
auto operator/(const T &b, const SType<U> &a) {
  return SType<eitherD<T, U>>(b / a.get_value());
};

// assuming all the arithmetic operators are properly overloaded, we can do
// this:
template <typename T, typename U> auto sigmoid_righthigh(T border, U x) {
  return 1. / (1. + exp(-smoothing_params::smfactor * (x - border)));
};

template <typename T, typename U> auto sigmoid_lefthigh(T border, U x) {
  return 1. - 1. / (1. + exp(-smoothing_params::smfactor * (x - border)));
};

template <typename T, typename U>
auto operator<(const SType<T> &a, const U &b) {
  bool left = a.get_value() < b;
  eitherD<T, U> tendency = (int)left * sigmoid_lefthigh(b, a.get_value()) +
                           (int)!left * sigmoid_righthigh(b, a.get_value());
  return STypeResult<eitherD<T, U>>(left,
                                    tendency); // calls on the tape as well!
};

template <typename T, typename U>
STypeResult<eitherD<T, U>> operator<(const SType<T> &b, const SType<U> &a) {
  return a - b < (eitherD<T, U>)0.;
};

template <typename T, typename U>
STypeResult<eitherD<T, U>> operator<(const T &a, const SType<U> &b) {
  return b >= a; // logical reverse
};

template <typename T, typename U>
STypeResult<eitherD<T, U>> operator<=(const SType<T> &a,
                                      const U &b) { // x < float
  // bool left = a.get_value() <= b;
  // eitherD<T, U> tendency = (int)left * sigmoid_lefthigh(b, a.get_value()) +
  //                          (int)!left * sigmoid_righthigh(b, a.get_value());
  // return STypeResult<eitherD<T, U>>(left,
  //                                   tendency); // calls on the tape as well!
  return a < b; // in smoothing terms, it's equivalent!
};

template <typename T, typename U>
STypeResult<eitherD<T, U>>
operator<=(const SType<T> &a, const SType<U> &b) { // SType this < SType b
  return a - b <= (eitherD<T, U>)0.;
};

template <typename T, typename U>
STypeResult<eitherD<T, U>> operator<=(const T &a, const SType<U> &b) {
  return b >= a;
};

template <typename T, typename U>
STypeResult<eitherD<T, U>> operator>(const SType<T> &a,
                                     const U &b) { // x < float

  bool right = a.value >= b;
  eitherD<T, U> tendency = (int)right * sigmoid_righthigh(b, a.value) +
                           (int)!right * sigmoid_lefthigh(b, a.value);
  return STypeResult<eitherD<T, U>>(right,
                                    tendency); // calls on the tape as well!
};

template <typename T, typename U>
STypeResult<eitherD<T, U>>
operator>(const SType<T> &a,
          const SType<U> &b) { // SType this < SType b
  // std::cout << a.value << " < " << b.value << std::endl;
  return a - b > (eitherD<T, U>)0;
  // TODO!!!!!!!!!!!!!!!!!!
  // TODO!!!!!!!!!!!!!!!!!!
  // TODO!!!!!!!!!!!!!!!!!! WHY DOES THIS WORK SUDDEN
  // TODO!!!!!!!!!!!!!!!!!!
  // TODO!!!!!!!!!!!!!!!!!!
  // TODO!!!!!!!!!!!!!!!!!!
  // TODO!!!!!!!!!!!!!!!!!!
};

template <typename T, typename U>
STypeResult<eitherD<T, U>> operator>(const T &a, const SType<U> &b) {
  return b <= a; // logical reverse
};

template <typename T, typename U>
STypeResult<eitherD<T, U>> operator>=(const SType<T> &a,
                                      const U &b) { // x < float
  // bool right = a.get_value() > b;
  // T tendency = (int)right * sigmoid_righthigh(b, a.get_value()) +
  //              (int)!right * sigmoid_lefthigh(b, a.get_value());
  // return STypeResult<eitherD<T, U>>(right,
  //                                   tendency); // calls on the tape as well!
  return a > b; // in smoothing terms, it's equal
};

template <typename T, typename U>
auto operator>=(const SType<T> &a,
                const SType<U> &b) { // SType this < SType b
  return a - b >= (eitherD<T, U>)0.;
};

template <typename T, typename U>
STypeResult<eitherD<T, U>> operator>=(const T &a, const SType<U> &b) {
  return b < a; // equiv
};

template <typename T, typename U>
auto operator==(const SType<T> &a, const SType<U> &b) {
  return (a - b > (eitherD<T, U>) 0.) && (a - b < (eitherD<T, U>) 0.);
};

template <typename T, typename U>
auto operator==(const SType<T> &a, const U &b) {
  return (a - b > (eitherD<T, U>)0.) && (a - b < (eitherD<T, U>)0.);
};

template <typename T, typename U>
auto operator==(const U &a, const SType<T> &b) {
  return a == b;
};
  // //

  // template <typename T, typename U>
  // STypeResult<T> operator<(const SType<T> &a, const SType<U> &b);
  // template <typename T, typename U>
  // STypeResult<T> operator<(const SType<T> &a, const U &b);
  // template <typename T, typename U>
  // STypeResult<T> operator<(const T &a, const SType<U> &b);

  // template <typename T, typename U>
  // STypeResult<T> operator>(const SType<T> &a, const U &b);
  // template <typename T, typename U>
  // STypeResult<T> operator>(const T &a, const SType<U> &b);
  // template <typename T, typename U>
  // STypeResult<T> operator>(const SType<T> &a, const SType<U> &b);

  // template <typename T, typename U>
  // STypeResult<T> operator<=(const SType<T> &a, const U &b);
  // template <typename T, typename U>
  // STypeResult<T> operator<=(const T &a, const SType<U> &b);
  // template <typename T, typename U>
  // STypeResult<T> operator<=(const SType<T> &a, const SType<U> &b);

  // template <typename T, typename U>
  // STypeResult<T> operator>=(const SType<T> &a, const SType<U> &b);
  // template <typename T, typename U>
  // STypeResult<T> operator>=(const SType<T> &a, const U &b);
  // template <typename T, typename U>
  // STypeResult<T> operator>=(const T &a, const SType<U> &b);

  // template <typename T, typename R>
  // STypeResult<T> operator==(const SType<T> &a, const SType<R> &b);
  // template <typename T, typename R>
  // STypeResult<T> operator==(const SType<T> &a, const R &b);
  // template <typename T, typename R>
  // STypeResult<T> operator==(const T &a, const SType<R> &b);

#include "overload_smooth_I.hpp" // variadic templates for flexible inputs

#include "overload_math.hpp" // overload all used math-functions for the SType

#endif // #ifndef ODERLOAD_SMOOTH_H