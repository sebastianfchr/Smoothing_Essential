#ifndef FUNCTIONS_HPP_H
#define FUNCTIONS_HPP_H

#include "../overload_smooth/overload_smooth.hpp"
#include <tuple>
#include <utility>

template <typename T> T spiral(const T &x1, const T &x2) {

  T f1 = pow(x1 - sqrt(pow(x1, 2) + pow(x2, 2)) *
                      cos(sqrt(pow(x1, 2) + pow(x2, 2))),
             2) +
         0.005 * (pow(x1, 2) + pow(x2, 2));
  T f2 = pow(x2 - sqrt(pow(x1, 2) + pow(x2, 2)) *
                      sin(sqrt(pow(x1, 2) + pow(x2, 2))),
             2) +
         0.005 * (pow(x1, 2) + pow(x2, 2));
  if (f1 > f2) {
    return f1;
  } else {
    return f2;
  }
}
template <typename T> T spiral_smooth(const T &x1, const T &x2) {
  return all_tape_calls<T>(&spiral, SType<T>(x1), SType<T>(x2));
}
template <typename T> std::array<T, 2> spiral_grad(const T &x1, const T &x2) {
  return gradient_adjoint(1., &spiral, x1, x2);
}

template <typename T>
std::array<T, 2> spiral_smooth_grad(const T &x1, const T &x2) {
  return gradient_adjoint(1., &spiral_smooth, x1, x2);
}

template <typename T> T ftest(const T &x1, const T &x2) {
  if (1. > x1) {
    return x1 * x2;
  } else {
    return x2 * x2;
  }
}
template <typename T> T ftest_smooth(const T &x1, const T &x2) {
  return all_tape_calls<T>(&ftest, SType<T>(x1), SType<T>(x2));
}

template <typename T> std::array<T, 2> ftest_grad(const T &x1, const T &x2) {
  return gradient_adjoint(1., &ftest, x1, x2);
}

template <typename T>
std::array<T, 2> ftest_smooth_grad(const T &x1, const T &x2) {
  return gradient_adjoint(1., &ftest_smooth, x1, x2);
}

template <typename T> T cb2(const T &x1, const T &x2) {
  T r;
  // smoothing_vars<double>::set_smfactor(50.);
  T f1 = x1 * x1 + x2 * x2 * x2 * x2;
  T f2 = pow(2. - x1, 2) + pow(2. - x2, 2);
  T f3 = 2. * exp(-x1 + x2);
  return maximum<T>(f1, f2, f3);
}

template <typename T> T simple_2d_curve(const T &x1, const T &x2) {

  T y = (T)0.;

  if (x2 > x1 * x1 || x2 < -x1 * x1) {
    y = (T)1.;
  } else {
    y = (T)0.;
  }
  // std::cout << dco::tape_index(y) << std::endl;
  return y;
}

// smooth versions

template <typename T> T cb2_smooth(const T &x1, const T &x2) {
  SType<T> x_1 = x1;
  SType<T> x_2 = x2;
  return all_tape_calls<T>(&cb2, x_1, x_2);
}

template <typename T> T simple_2d_curve_smooth(const T &x1, const T &x2) {
  SType<T> x_1 = x1;
  SType<T> x_2 = x2;
  return all_tape_calls<T>(&simple_2d_curve, x_1, x_2);
}

// grads

template <typename T> std::array<T, 2> cb2_grad(const T &x1, const T &x2) {
  // return {{0., 0.}};
  return gradient_adjoint(1., &cb2, x1, x2);
}

template <typename T>
std::array<T, 2> simple_2d_curve_grad(const T &x1, const T &x2) {
  // return {{0., 0.}};
  return gradient_adjoint(1., &simple_2d_curve, x1, x2);
}

// grads of smoothed functions

template <typename T>
std::array<T, 2> cb2_smooth_grad(const T &x1, const T &x2) {
  return gradient_adjoint(1., &cb2_smooth, x1, x2);
}

template <typename T>
std::array<T, 2> simple_2d_curve_smooth_grad(const T &x1, const T &x2) {
  return gradient_adjoint(1., &simple_2d_curve_smooth, x1, x2);
}

template <typename T>
std::array<T, 2> cb2_grad_smooth(const T &x1, const T &x2) {
  return smoothed_gradient<T>(1., &cb2, x1, x2);
}

template <typename T>
std::array<T, 2> simple_2d_curve_grad_smooth(const T &x1, const T &x2) {
  return smoothed_gradient<T>(1., &simple_2d_curve, x1, x2);
}

#endif // FUNCTIONS_HPP_H !