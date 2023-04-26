
#include "smoothing_examples.hpp"
#include <math.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(Smoothing, m) {

  m.def("set_smfactor", &smoothing_params::set_smfactor,
        R"pbdoc(
      	Set the smoothing factor for the sigmoids GLOBALLY! double -> void
      )pbdoc");
  // define spiral!
  m.def("spiral", &spiral<double>, "doc...");
  m.def("spiral_grad", &spiral_grad<double>, "doc...");
  m.def("spiral_smooth", &spiral_smooth<double>, "doc...");
  m.def("spiral_smooth_grad", &spiral_smooth_grad<double>, "doc...");

  m.def("ftest", &ftest<double>,
        R"pbdoc(
      	f2d_smooth_with_contributions. in: float, float, out: float
      )pbdoc");
  m.def("ftest_smooth", &ftest_smooth<double>,
        R"pbdoc(
      	f2d_smooth_with_contributions. in: float, float, out: float
      )pbdoc");
  m.def("ftest_grad", &ftest_grad<double>,
        R"pbdoc(
      	f2d_smooth_with_contributions. in: float, float, out: float
      )pbdoc");
  m.def("ftest_smooth_grad", &ftest_smooth_grad<double>,
        R"pbdoc(
      	f2d_smooth_with_contributions. in: float, float, out: float
      )pbdoc");

  // simple 2d curve definition
  m.def("simple_2d_curve", &simple_2d_curve<double>,
        R"pbdoc(
      	f2d_smooth_with_contributions. in: float, float, out: float
      )pbdoc");
  m.def("simple_2d_curve_smooth", &simple_2d_curve_smooth<double>,
        R"pbdoc(
      	f2d_smooth_with_contributions. in: float, float, out: float
      )pbdoc");
  m.def("simple_2d_curve_grad", &simple_2d_curve_grad<double>,
        R"pbdoc(
      	simple_2d_curve_grad. in: float, float, out: [float, float]
      )pbdoc");
  m.def("simple_2d_curve_smooth_grad", &simple_2d_curve_smooth_grad<double>,
        R"pbdoc(
      	simple_2d_curve_grad. in: float, float, out: [float, float]
      )pbdoc");
  m.def("simple_2d_curve_grad_smooth", &simple_2d_curve_grad_smooth<double>,
        R"pbdoc(
      	simple_2d_curve_grad. in: float, float, out: [float, float]
      )pbdoc");

  // cb2-definition
  m.def("cb2", &cb2<double>,
        R"pbdoc(
      	f2d_smooth_with_contributions. in: float, float, out: float
      )pbdoc");
  m.def("cb2_smooth", &cb2_smooth<double>,
        R"pbdoc(
      	f2d_smooth_with_contributions. in: float, float, out: float
      )pbdoc");
  m.def("cb2_grad", &cb2_grad<double>,
        R"pbdoc(
      	cb2_grad. in: float, float, out: [float, float]
      )pbdoc");
  m.def("cb2_smooth_grad", &cb2_smooth_grad<double>,
        R"pbdoc(
      	cb2_grad. in: float, float, out: [float, float]
      )pbdoc");
  m.def("cb2_grad_smooth", &cb2_grad_smooth<double>,
        R"pbdoc(
      	cb2_grad. in: float, float, out: [float, float]
      )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}