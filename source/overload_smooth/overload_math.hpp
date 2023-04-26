// excerpt dropped into in overload_smooth.hpp
// all mathematical operations are those of the libraries used in
// overload_smooth.hpp
#include <cmath>

template <typename T, typename... Args> T maximum(const Args &... xs) {
  std::array<T, sizeof...(Args)> arr = toArray<T>(xs...);
  T x_curr = arr[0];
  for (int i = 1; i < sizeof...(Args); i++) {
    if (arr[i] > x_curr)
      x_curr = arr[i];
  }
  return x_curr;
}

template <typename T> SType<T> sqrt(const SType<T> &x) {
  return SType<T>(sqrt(x.value));
}

template <typename T> SType<T> abs(const SType<T> &x) {
  if (x < 0)
    return -x;
  else
    return x;
  // return SType<T>(abs(x.value));
}

template <typename T1, typename T2> SType<T1> pow(SType<T1> x1, T2 x2) {
  return SType<T1>(pow(x1.value, x2));
}
template <typename T> SType<T> exp(SType<T> x1) {
  return SType<T>(exp(x1.value));
}
template <typename T> SType<T> sin(SType<T> x1) {
  return SType<T>(sin(x1.value));
}

template <typename T> SType<T> cos(SType<T> x1) {
  return SType<T>(cos(x1.value));
}