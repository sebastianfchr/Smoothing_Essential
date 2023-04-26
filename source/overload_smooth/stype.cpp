#include "overload_smooth.hpp"
#include <locale>

// smfactor
double smoothing_params::smfactor = 5.;
void smoothing_params::set_smfactor(double smfactor) {
  smoothing_params::smfactor = smfactor;
}

template <typename U> const U smoothing_params::get_smfactor() {
  return (U)smoothing_params::smfactor;
}

template <typename T> SType<T>::SType(){};

template <typename T> SType<T>::SType(const T &value) { this->value = value; };
// template <typename T> template <typename U> SType<T>::SType(const U &value) {
//   this->value = value;
// };

template <typename T>
template <typename U>
SType<T> &SType<T>::operator=(const U &b) {
  // std::cout << "ci=yi" << std::endl;
  this->value = b;
  return *this;
};

template <typename T> SType<T> &SType<T>::operator+=(const SType<T> &b) {
  this->value += b.get_value();
  return *this;
};

template <typename T> SType<T> &SType<T>::operator+=(const T &b) {
  this->value += b;
  return *this;
};

template <typename T> SType<T> &SType<T>::operator-=(const SType<T> &b) {
  this->value -= b.get_value();
  return *this;
};

// does this cause problems if SType<DCO_T<T>> - T?
template <typename T> SType<T> &SType<T>::operator-=(const T &b) {
  this->value -= b;
  return *this;
};

template <typename T> SType<T> &SType<T>::operator*=(const SType<T> &b) {
  this->value *= b.get_value();
  return *this;
};

template <typename T> SType<T> &SType<T>::operator*=(const T &b) {
  this->value *= b;
  return *this;
};

  // per preprocessor macro

// R that can be assigned to T!
#define OPS_INSTANTIATE(T, U)                                                  \
  template SType<T> &SType<T>::operator=(const U &);                           \

#define VALUETYPE_INSTANTIATE(T)                                               \
  OPS_INSTANTIATE(T, T);                                                       \
  template class SType<T>;                                                     \
  template class STypeResult<T>;                                               \

#define VALUETYPE_INSTANTIATE_REC(T, R) OPS_INSTANTIATE(T<R>, R);

// VALUETYPE_INSTANTIATE(DCO_TanType<double>); // not relevant for lab
VALUETYPE_INSTANTIATE(DCO_T<double>);
VALUETYPE_INSTANTIATE(double);

// INSTANTIATE_ASYM(SType<DCO_T<double>>, double);

// VALUETYPE_INSTANTIATE(DCO_TanType<float>); // not relevant for lab
VALUETYPE_INSTANTIATE(DCO_T<float>);
VALUETYPE_INSTANTIATE(float);