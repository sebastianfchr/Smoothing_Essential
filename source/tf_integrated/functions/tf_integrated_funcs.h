#ifndef ALL_FUNCS_H
#define ALL_FUNCS_H

#include <math.h>
#define EIGEN_USE_THREADS
#define DCO_EXT_EIGEN_IGNORE_VERSION
#define DCO_EXT_EIGEN_IGNORE_NO_INTERMEDIATES
#define DCO_DISABLE_AUTO_WARNING
#define DCO_DISABLE_AVX2_WARNING
#include "../../overload_smooth/overload_smooth.hpp"
// #include <Eigen/Dense>
//  I WANT CLANG TO GET IT THAT I DON'T NEED THAT WHOLE PATH!
#include </home/seb/.local/lib/python3.8/site-packages/tensorflow/include/Eigen/Dense>
#include </home/seb/.local/lib/python3.8/site-packages/tensorflow/include/third_party/eigen3/unsupported/Eigen/CXX11/Tensor>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <dco.hpp>
#include <tuple>
#include <utility>
template <typename T> using DCO_M = typename dco::ga1s<T>;
template <typename T> using DCO_T = typename DCO_M<T>::type;
template <typename T> using DCO_TT = typename DCO_M<T>::tape_t;

// #include "../../overload_smooth/overload_smooth.hpp"
// #include "tf_integrated_param_calls.h"
#include <functional>

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice; // never used in CPU version!

template <typename T> T cb2(const T &x1, const T &x2);
template <typename T> T crescent(const T &x1, const T &x2);

#endif // ALL_FUNCS_H
