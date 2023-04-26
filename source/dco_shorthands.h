#define DCO_DISABLE_AVX2_WARNING
#define DCO_DISABLE_AUTO_WARNING
#define DCO_AUTO_SUPPORT
#include <dco.hpp>

// adjoint mode
template <typename T> using DCO_M = typename dco::ga1s<T>;
template <typename T> using DCO_T = typename DCO_M<T>::type;
template <typename T> using DCO_TT = typename DCO_M<T>::tape_t;

// tangent mode
template <typename T> using DCO_TanM = dco::gt1s<T>;
template <typename T> using DCO_TanType = typename DCO_TanM<T>::type;
