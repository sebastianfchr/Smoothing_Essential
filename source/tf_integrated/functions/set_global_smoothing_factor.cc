#include "tf_integrated_funcs.h"

#include <iterator>
#include <math.h>

using namespace tensorflow;

// ====================== Registration, number of args ======================
// Register the CPU kernels.
#define REGISTER_CPU(T, Fname, OpName)                                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(Fname).Device(DEVICE_CPU).TypeConstraint<T>("T"),                   \
      OpName<CPUDevice, T>);

template <typename Device, typename T>
class SetGlobalSmoothingFactorOp : public OpKernel {
private:
public:
  // take non-tensor inputs
  explicit SetGlobalSmoothingFactorOp(OpKernelConstruction *context)
      : OpKernel(context) {
    // T smoothing_factor;
    // OP_REQUIRES_OK(context,
    //                context->GetAttr("smoothing_factor", &smoothing_factor));
  }
  void Compute(OpKernelContext *context) override {
    // Grab the input tensor

    assert(context->num_inputs() == 1);
    const Tensor &global_smoothing_factor = context->input(0);
    const T *smfactor = global_smoothing_factor.flat<T>().data();
    // std::cout << "till here" << std::endl;
    // std::cout << *smfactor << std::endl;
    smoothing_params::set_smfactor(*smfactor);
  }
};

// REGISTRATION TO TENSORFLOW AND TO CPU!
// REGISTER_OP_T("Ql");
REGISTER_OP("SetGlobalSmoothingFactor")
    .Attr("T: numbertype")
    .Input("global_smoothing_factor: T");

REGISTER_CPU(double, "SetGlobalSmoothingFactor", SetGlobalSmoothingFactorOp);
REGISTER_CPU(float, "SetGlobalSmoothingFactor", SetGlobalSmoothingFactorOp);

// .Input("input_x2: T")
// .Input("smoothing_factor: T")
// .Output("output: T");
