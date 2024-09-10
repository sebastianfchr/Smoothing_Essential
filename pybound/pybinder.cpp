#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SType.hpp"
#include <tuple>
#include "funcs/smooth_dijkstra.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


// std::tuple<double, double, double> ff(double x1i, double x2i) {
std::tuple<double, double, double> ff(double x1i, double x2i) {

    auto tape_ptr = tape::make_smart_tape_ptr();

    SType<double> x1 = x1i;
    SType<double> x2 = x2i;
    SOutType o;

    auto vglambda = [&](){

        SType cond1 = tape::get_tape_ptr()->make_split(x1*x1+x2*x2 < 2*2); 
        if(cond1.valptr->discrete) {
            o = x1+x2;
        } else {
            o = x1*x2;
        }


    };

  o.seed(1);

  do {

    tape::get_tape_ptr()->prepare_valtape();
    vglambda();
    tape::get_tape_ptr()->forward_backward_souttypes();    
    tape::get_tape_ptr()->backprop();

	} while(tape::get_tape_ptr()->tape_next_round());

  return std::make_tuple(o.get_value().value, x1.get_value().partial, x2.get_value().partial);

}

std::tuple<double, double, double> ff2(double x1i, double x2i) {

    auto tape_ptr = tape::make_smart_tape_ptr();
    
	// short way of writing it!
	SType<double> x1 = x1i;
	SType<double> x2 = x2i;
	SOutType o;

	SMOOTHING();

		IF(x1*x1+x2*x2 < 2 || x1*x1+x2*x2 > 3*3){
			o = x1; // out_op doesn't want an lvalue. Or even downstream-funcs?
		} ELSE() {
			o = 1; 
		}

	SMOOTHING_END();

	o.seed(1);

	// double discrete_val = o.get_value().value; // 

	BACKPROP();

	// return std::make_tuple(o.get_value().value, deriv(x1.get_value()), deriv(x2.get_value()));
	return std::make_tuple(o.get_value().value, deriv(x1.get_value()), deriv(x2.get_value()));

}


std::tuple<double, double, double> ff_for_opt(double x1i, double x2i) {

    auto tape_ptr = tape::make_smart_tape_ptr();
    
	// short way of writing it!
	SType<double> x1 = x1i;
	SType<double> x2 = x2i;
	SOutType o;

	SMOOTHING();

        SType<double> res = 2;
		IF(x1*x1+x2*x2 < 2){
			res = res - 1; // out_op doesn't want an lvalue. Or even downstream-funcs?
		} 

        IF(x1 < x2) {
            res = res - 1;
        }        
        
        o = res;

	SMOOTHING_END();

	o.seed(1);

	BACKPROP();

	return std::make_tuple(o.get_value().value, deriv(x1.get_value()), deriv(x2.get_value()));

}

// ff2, but condition is inversed!
std::tuple<double, double, double> ff3(double x1i, double x2i) {

    auto tape_ptr = tape::make_smart_tape_ptr();

    SType<double> x1 = x1i;
    SType<double> x2 = x2i;
    SOutType o;

    auto vglambda = [&](){

        SType cond1 = tape::get_tape_ptr()->make_split(!(x1*x1+x2*x2 < 2 || x1*x1+x2*x2 > 3*3)); 
        if(cond1.valptr->discrete) {
            o = x1; // out_op doesn't want an lvalue. Or even downstream-funcs?
        } else {
            o = 1; // x2 alone doesn't work, why?
        }

    };
    // TODO: Probably reorganize like this to obtain first pass through discrete path!

  o.seed(1);

    tape::get_tape_ptr()->prepare_valtape();
    vglambda();
    tape::get_tape_ptr()->forward_backward_souttypes();
    tape::get_tape_ptr()->backprop();
  while(tape::get_tape_ptr()->tape_next_round()) {
    tape::get_tape_ptr()->prepare_valtape();
    vglambda();
    tape::get_tape_ptr()->forward_backward_souttypes();
    tape::get_tape_ptr()->backprop();

  } ;

  return std::make_tuple(o.get_value().value, x1.get_value().partial, x2.get_value().partial);

}

std::tuple<double, double> f1d(double x1i) {

    auto tape_ptr = tape::make_smart_tape_ptr();
    
	// short way of writing it!
	SType<double> x1 = x1i;
	// SType<double> x2 = 3;
	SOutType o;

	SMOOTHING();

        IF(x1 < 1.5) {
            o = sin(x1);

        } ELSE() {
            o = cos(x1); //-x1*1;

        }
		// IF(x1 < x2){
		// 	o = x1*x1; // out_op doesn't want an lvalue. Or even downstream-funcs?
		// } ELSE() {
		// 	o = x1; 
		// }

	SMOOTHING_END();

	o.seed(1);

	// double discrete_val = o.get_value().value; // 

	// BACKPROP_BEGIN();
	// BACKPROP_END();

    BACKPROP();

    return std::make_tuple(o.get_value().value, x1.get_value().partial);

//     auto tape_ptr = tape::make_smart_tape_ptr();

//     SOutType o;
//     SType<double> x1 = x1i;

//     auto vglambda = [&](){

//         // SType cond1 = tape::get_tape_ptr()->make_split(x1 > 3 && x1 < 5);  // even 2 unexpected spikes 
//         // SType cond1 = tape::get_tape_ptr()->make_split(x1 < 3 && x1 < 5 ); 
//         // SType cond1 = tape::get_tape_ptr()->make_split(x1 > 3 && x1 < 5 );  // TODO: unexpected spike at 3
//         SType cond1 = tape::get_tape_ptr()->make_split(x1 < 3  ); 
//         // TODO: DOES IS HAVE SOMETHING TO DO WITH ACTUAL VALUE ARRIVING AT SPLIT_OP?
//         if(cond1.valptr->discrete) {
//             if((tape::get_tape_ptr()->make_split(x1 < 1  )).valptr->discrete)
//                 o = 0;
//             else {
//                 o = 2;// 2*x1;
//             }
//         } else {                    // # ELSE&& x1 < 5
//             o = -3;//*x1;
//         }

//     };

//   o.seed(1);

//   do {

//     tape::get_tape_ptr()->prepare_valtape();
//     vglambda();
//     tape::get_tape_ptr()->forward_backward_souttypes();    
//     tape::get_tape_ptr()->backprop();

//   } while(tape::get_tape_ptr()->tape_next_round());



//   return std::make_tuple(o.get_value().value, x1.get_value().partial);

}



PYBIND11_MODULE(smfuncs, m) {


    m.def("set_smoothing_factor", &set_smfactor,
        R"pbdoc(
            test func
        )pbdoc");

    m.def("smooth_dijkstra", &dijkstra_full<double>, // this one fixes datatype double!
        R"pbdoc(
            test func
        )pbdoc");

    m.def("ff", &ff,
        R"pbdoc(
            test func
        )pbdoc");

    m.def("ff2", &ff2,
        R"pbdoc(
            test func
        )pbdoc");

    m.def("ff3", &ff3,
        R"pbdoc(
            test func
        )pbdoc");

    m.def("f1d", &f1d,
        R"pbdoc(
            test func
        )pbdoc");

    m.def("ff_for_opt", &ff_for_opt,
        R"pbdoc(
            test func
        )pbdoc");


#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}