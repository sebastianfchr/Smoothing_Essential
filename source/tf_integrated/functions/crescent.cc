#include "tf_integrated_funcs.h"

#include <iterator>
#include <math.h>

using namespace tensorflow;

template <typename T> T crescent(const T &x1, const T &x2) {
  T r;


  // crescent
  T r1 = x1 * x1 + (x2 - (T)1.) * (x2 - 1.) + x2 - (T)1.;
  T r2 = -x1 * x1 - (x2 - (T)1.) * (x2 - (T)1.) + x2 + (T)1.; //- (T)(x2 - 1.);



  if (r1 > r2)
    r = r1;
  else
    r = r2;
  return r;
}

// ============= gradient, smooth version, and smooth gradient =============
template <typename T> 
std::array<T, 2> crescent_gradient(const T &downstream_gradient, const T &x1,
                                   const T &x2) {
  return gradient_adjoint(downstream_gradient, &crescent, x1, x2);
};

template <typename T> T crescent_smooth(const T &x1, const T &x2) {
  return all_tape_calls<T>(&crescent, SType<T>(x1), SType<T>(x2));
  // (SType<T>, SType<T>) -> T
};

template <typename T>
std::array<T, 2> crescent_smooth_gradient(const T &downstream_gradient,
                                          const T &x1, const T &x2) {

  return gradient_adjoint(downstream_gradient, &crescent_smooth, x1, x2);
};


// ====================== Registration, number of args ======================
// Register the CPU kernels.
#define REGISTER_CPU(T, Fname, OpName)                                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(Fname).Device(DEVICE_CPU).TypeConstraint<T>("T"),                   \
      OpName<CPUDevice, T>);

#define REGISTER_OP_T(NameStr)                                                 \
  REGISTER_OP(NameStr)                                                         \
      .Attr("T: numbertype")                                                   \
      .Input("input_x1: T")                                                    \
      .Input("input_x2: T")                                                    \
      .Output("output: T");

#define REGISTER_BACKPROP_T(NameStr)                                           \
  REGISTER_OP(NameStr)                                                         \
      .Attr("T: numbertype")                                                   \
      .Input("input_x1: T")                                                    \
      .Input("input_x2: T")                                                    \
      .Input("smoothing_factor: T")                                            \
      .Output("output_x1_a: T")                                                \
      .Output("output_x2_a: T");

template <typename Device, typename T> class CrescentOp : public OpKernel {
private:
  // T smoothingFactor;

public:
  // take non-tensor inputs
  explicit CrescentOp(OpKernelConstruction *context) : OpKernel(context) {
    // OP_REQUIRES_OK(context,
    //                context->GetAttr("smoothing_factor", &smoothingFactor));
  }
  void Compute(OpKernelContext *context) override {
    // Grab the input tensor

    // assert(context->num_inputs() == 3);
    const Tensor &input_tens_x1 = context->input(0);
    const Tensor &input_tens_x2 = context->input(1);

    // std::cout << "aaa" << *smoothingFactor << std::endl;
    OP_REQUIRES(context,
                (input_tens_x1.IsSameSize(input_tens_x2) &&
                 input_tens_x1.dims() == input_tens_x2.dims()),
                errors::InvalidArgument(
                    "All value-input tensors must have the same shape!"));

    // assign output tensor
    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {input_tens_x1.shape()},
                                                     &output_tensor));

    // Or add both numEls?
    OP_REQUIRES(context, input_tens_x1.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    T *output = output_tensor->flat<T>().data();

    // casting const to normal T! It actually is a const-pointer!
    T *input_x1 = (T *)input_tens_x1.flat<T>().data();
    T *input_x2 = (T *)input_tens_x2.flat<T>().data(); // in reality, const
    const int num_elements = input_tens_x1.NumElements();

    for (int i = 0; i < num_elements; i++)
      output[i] = crescent(input_x1[i], input_x2[i]);
  }
};

template <typename Device, typename T>
class CrescentGradientOp : public OpKernel {
private:
  // T smoothingFactor;

public:
  // take non-tensor inputs
  explicit CrescentGradientOp(OpKernelConstruction *context)
      : OpKernel(context) {
    // OP_REQUIRES_OK(context,
    //                context->GetAttr("smoothing_factor", &smoothingFactor));
  }
  void Compute(OpKernelContext *context) override {

    const Tensor &upstream_gradient_tens = context->input(0);
    const Tensor &input_tens_x1 = context->input(1);
    const Tensor &input_tens_x2 = context->input(2);

    // std::cout << "aaa" << *smoothingFactor << std::endl;
    OP_REQUIRES(context,
                (input_tens_x1.IsSameSize(input_tens_x2) &&
                 input_tens_x1.dims() == input_tens_x2.dims()),
                errors::InvalidArgument(
                    "All value-input tensors must have the same shape!"));

    // assign output tensor
    Tensor *output_tensor_x1_a = NULL;
    Tensor *output_tensor_x2_a = NULL;
    // there will be two adjoints, one for each input
    OP_REQUIRES_OK(context, context->allocate_output(0, {input_tens_x1.shape()},
                                                     &output_tensor_x1_a));
    OP_REQUIRES_OK(context, context->allocate_output(1, {input_tens_x2.shape()},
                                                     &output_tensor_x2_a));

    // Or add both numEls?
    OP_REQUIRES(context, input_tens_x1.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    const T *upstream_gradient =
        (const T *)upstream_gradient_tens.flat<T>().data();
    const T *input_x1 = (const T *)input_tens_x1.flat<T>().data();
    const T *input_x2 =
        (const T *)input_tens_x2.flat<T>().data(); // in reality, const
    const int num_elements = input_tens_x1.NumElements();

    // map on outputs!
    T *x1_a = (T *)output_tensor_x1_a->flat<T>().data();
    T *x2_a = (T *)output_tensor_x2_a->flat<T>().data();

    for (int i = 0; i < num_elements; i++) {
      arrayAssigner(
          crescent_gradient(upstream_gradient[i], input_x1[i], input_x2[i]),
          x1_a[i], x2_a[i]);
    }
  }
};

template <typename Device, typename T>
class CrescentSmoothOp : public OpKernel {
private:
  // T smoothingFactor;

public:
  // take non-tensor inputs
  explicit CrescentSmoothOp(OpKernelConstruction *context) : OpKernel(context) {
    // OP_REQUIRES_OK(context,
    //                context->GetAttr("smoothing_factor", &smoothingFactor));
  }
  void Compute(OpKernelContext *context) override {
    // Grab the input tensor

    // T *smoothingFactor =
    //     (T *)context->input(2).data(); // simple, since one element!

    const Tensor &input_tens_x1 = context->input(0);
    const Tensor &input_tens_x2 = context->input(1);

    // std::cout << "aaa" << *smoothingFactor << std::endl;
    OP_REQUIRES(context,
                (input_tens_x1.IsSameSize(input_tens_x2) &&
                 input_tens_x1.dims() == input_tens_x2.dims()),
                errors::InvalidArgument(
                    "All value-input tensors must have the same shape!"));

    // assign output tensor
    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {input_tens_x1.shape()},
                                                     &output_tensor));

    // Or add both numEls?
    OP_REQUIRES(context, input_tens_x1.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    T *output = output_tensor->flat<T>().data();
    // casting const to normal T! It actually is a const-pointer!
    const T *input_x1 = input_tens_x1.flat<T>().data();
    const T *input_x2 = input_tens_x2.flat<T>().data(); // in reality, const
    const int num_elements = input_tens_x1.NumElements();

    for (int i = 0; i < num_elements; i++)
      output[i] = crescent_smooth(input_x1[i], input_x2[i]);
  }
};

template <typename Device, typename T>
class CrescentSmoothGradientOp : public OpKernel {
private:
  // T smoothingFactor;

public:
  // take non-tensor inputs
  explicit CrescentSmoothGradientOp(OpKernelConstruction *context)
      : OpKernel(context) {
    // OP_REQUIRES_OK(context,
    //                context->GetAttr("smoothing_factor", &smoothingFactor));
  }
  void Compute(OpKernelContext *context) override {

    const Tensor &upstream_gradient_tens = context->input(0);
    const Tensor &input_tens_x1 = context->input(1);
    const Tensor &input_tens_x2 = context->input(2);

    // std::cout << "aaa" << *smoothingFactor << std::endl;
    OP_REQUIRES(context,
                (input_tens_x1.IsSameSize(input_tens_x2) &&
                 input_tens_x1.dims() == input_tens_x2.dims()),
                errors::InvalidArgument(
                    "All value-input tensors must have the same shape!"));

    // assign output tensor
    Tensor *output_tensor_x1_a = NULL;
    Tensor *output_tensor_x2_a = NULL;
    // there will be two adjoints, one for each input
    OP_REQUIRES_OK(context, context->allocate_output(0, {input_tens_x1.shape()},
                                                     &output_tensor_x1_a));
    OP_REQUIRES_OK(context, context->allocate_output(1, {input_tens_x2.shape()},
                                                     &output_tensor_x2_a));

    // Or add both numEls?
    OP_REQUIRES(context, input_tens_x1.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    const T *upstream_gradient =
        (const T *)upstream_gradient_tens.flat<T>().data();
    const T *input_x1 = (const T *)input_tens_x1.flat<T>().data();
    const T *input_x2 =
        (const T *)input_tens_x2.flat<T>().data(); // in reality, const
    const int num_elements = input_tens_x1.NumElements();

    // map on outputs!
    T *x1_a = (T *)output_tensor_x1_a->flat<T>().data();
    T *x2_a = (T *)output_tensor_x2_a->flat<T>().data();

    for (int i = 0; i < num_elements; i++) {
      std::array<T, 2> res = crescent_smooth_gradient(upstream_gradient[i],
                                                      input_x1[i], input_x2[i]);
      x1_a[i] = std::get<0>(res);
      x2_a[i] = std::get<1>(res);

    }
  }
};

// REGISTRATION TO TENSORFLOW AND TO CPU!
REGISTER_OP_T("Crescent");
REGISTER_CPU(double, "Crescent", CrescentOp);
REGISTER_CPU(float, "Crescent", CrescentOp);

REGISTER_BACKPROP_T("CrescentGradient")
REGISTER_CPU(double, "CrescentGradient", CrescentGradientOp);
REGISTER_CPU(float, "CrescentGradient", CrescentGradientOp);

REGISTER_OP_T("CrescentSmooth");
REGISTER_CPU(double, "CrescentSmooth", CrescentSmoothOp);
REGISTER_CPU(float, "CrescentSmooth", CrescentSmoothOp);

REGISTER_BACKPROP_T("CrescentSmoothGradient")
REGISTER_CPU(double, "CrescentSmoothGradient", CrescentSmoothGradientOp);
REGISTER_CPU(float, "CrescentSmoothGradient", CrescentSmoothGradientOp);
