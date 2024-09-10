#pragma once
#include "general_types.hpp"
#include "type_helpers.hpp"
#include <cmath>
// #include "STape.hpp"




// TODO: Concepts: Arithmetic, Logical for those plugins!

// =================  Arithmetic Binary =================

struct Add {
    static decltype(auto) eval(auto&& x1, auto&& x2) { return x1+x2; }
    static decltype(auto) partial_x1(auto&& x1, auto&& x2, auto&& adjoint) { return adjoint; }
    static decltype(auto) partial_x2(auto&& x1, auto&& x2, auto&& adjoint) { return adjoint; }
};
struct Multiply  {
    static decltype(auto) eval(auto&& x1, auto&& x2) { return x1*x2; }
    static decltype(auto) partial_x1(auto&& x1, auto&& x2, auto&& adjoint) { return x2*adjoint; }
    static decltype(auto) partial_x2(auto&& x1, auto&& x2, auto&& adjoint) { return x1*adjoint; }
};
struct Subtract  {
    static decltype(auto) eval(auto&& x1, auto&& x2) { return x1-x2; }
    static decltype(auto) partial_x1(auto&& x1, auto&& x2, auto&& adjoint) { return adjoint; }
    static decltype(auto) partial_x2(auto&& x1, auto&& x2, auto&& adjoint) { return -1*adjoint; }
};
struct Divide {
    static decltype(auto) eval(auto&& x1, auto&& x2) { return x1/x2; }
    static decltype(auto) partial_x1(auto&& x1, auto&& x2, auto&& adjoint) { return 1./x2*adjoint; }
    static decltype(auto) partial_x2(auto&& x1, auto&& x2, auto&& adjoint) { return -x1/(x2*x2)*adjoint; }
};


// =================  Arithmetic Unary =================
struct Negative {
    static decltype(auto) eval(auto&& x1) { return -x1; }
    static decltype(auto) partial(auto&& x1, auto&& adjoint) { return -adjoint; }
};

struct Sin {
    static decltype(auto) eval(auto&& x1) { return sin(x1); }
    static decltype(auto) partial(auto&& x1, auto&& adjoint) { return cos(x1)*adjoint; }
};

struct Cos {
    static decltype(auto) eval(auto&& x1) { return cos(x1); }
    static decltype(auto) partial(auto&& x1, auto&& adjoint) { return -1*sin(x1)*adjoint; }
};


static long double smfactor = 1;
void set_smfactor(long double f) { smfactor = f; }

// cast smfactor to prevent promotion to long double
static decltype(auto) sigmoid_righthigh(auto&& x) { return 1./(1+exp(static_cast<std::remove_cvref_t<decltype(x)>>(smfactor) * - x)); }
static decltype(auto) sigmoid_lefthigh(auto&& x) { return 1- sigmoid_righthigh(x); }

// note that it doesn't matter if we use lefthigh or righthigh. It's always going to be 
static decltype(auto) sigmoid_righthigh_deriv(auto&& x) { return smfactor * sigmoid_righthigh(x)*(1-sigmoid_righthigh(x)); }
static decltype(auto) sigmoid_lefthigh_deriv(auto&& x) { return - sigmoid_righthigh_deriv(x); }


// ================= Comparison =================
// note: all of these have to be flipped manually inside the op_t-s, if they're inverted!
// inverting here means: result: 1-y, and derivative: -1* derivative 
struct Lt {
    static decltype(auto) eval_discrete(auto&& x1, auto&& x2) { return x1 < x2; }
    static decltype(auto) eval_tendency(auto&& x1, auto&& x2) { return sigmoid_lefthigh(x1-x2); }
    static decltype(auto) partial_x1(auto&& x1, auto&& x2, auto&& y, auto&& adjoint) { return sigmoid_lefthigh_deriv(x1-x2) * adjoint ; }
    static decltype(auto) partial_x2(auto&& x1, auto&& x2, auto&& y, auto&& adjoint) { return -partial_x1(x1, x2, y, adjoint); }  
};
struct Le {
    static decltype(auto) eval_discrete(auto&& x1, auto&& x2) { return x1 <= x2; }
    static decltype(auto) eval_tendency(auto&& x1, auto&& x2) { return sigmoid_lefthigh(x1-x2); }
    static decltype(auto) partial_x1(auto&& x1, auto&& x2, auto&& y, auto&& adjoint) { return sigmoid_lefthigh_deriv(x1-x2) * adjoint; }
    static decltype(auto) partial_x2(auto&& x1, auto&& x2, auto&& y, auto&& adjoint) { return -partial_x1(x1, x2, y, adjoint); }  
};
struct Gt {
    static decltype(auto) eval_discrete(auto&& x1, auto&& x2) { return x1 > x2; }
    static decltype(auto) eval_tendency(auto&& x1, auto&& x2) { return sigmoid_righthigh(x1-x2); }
    static decltype(auto) partial_x1(auto&& x1, auto&& x2, auto&& y, auto&& adjoint) { return sigmoid_righthigh_deriv(x1-x2) * adjoint; }
    static decltype(auto) partial_x2(auto&& x1, auto&& x2, auto&& y, auto&& adjoint) { return -partial_x1(x1, x2, y, adjoint); }  
};
struct Ge {
    static decltype(auto) eval_discrete(auto&& x1, auto&& x2) { return x1 >= x2; }
    static decltype(auto) eval_tendency(auto&& x1, auto&& x2) { return sigmoid_righthigh(x1-x2); }
    static decltype(auto) partial_x1(auto&& x1, auto&& x2, auto&& y, auto&& adjoint) { return sigmoid_righthigh_deriv(x1-x2) * adjoint; }
    static decltype(auto) partial_x2(auto&& x1, auto&& x2, auto&& y, auto&& adjoint) { return -partial_x1(x1, x2, y, adjoint); }  
};


// ================= Logical Binary =================
struct And {
    static decltype(auto) eval_discrete(auto&& x1, auto&& x2) { return x1 && x2; }
    static decltype(auto) eval_tendency(auto&& t1, auto&& t2) { return t1*t2; }
    static decltype(auto) partial_t1(auto&& t1, auto&& t2, auto&& adjoint) { return t2 * adjoint; }
    static decltype(auto) partial_t2(auto&& t1, auto&& t2, auto&& adjoint) { return t1 * adjoint; }
};

struct Or {
    static decltype(auto) eval_discrete(auto&& x1, auto&& x2) { return x1 || x2; }
    static decltype(auto) eval_tendency(auto&& t1, auto&& t2) { return t1 + t2 - t1 * t2; }
    static decltype(auto) partial_t1(auto&& t1, auto&& t2, auto&& adjoint) { return (1-t2) * adjoint; }
    static decltype(auto) partial_t2(auto&& t1, auto&& t2, auto&& adjoint) { return (1-t1) * adjoint; }
};

// ================= Logical Unary =================

struct Not {
    static decltype(auto) eval_discrete(auto&& x1) { return !x1; }
    static decltype(auto) eval_tendency(auto&& t1) { return 1-t1; }
    static decltype(auto) partial(auto&& t1, auto&& adjoint) { return -1 * adjoint; }
};



// An op that works on values or booleans. TODO: Should it go or is it a good base? Some of the code needlessly relies on it
// An op (abstract here), which comes with the promise of 
// returning a value_t, or boolean_t (upon combination with further inheritance)
template<typename T> class evaluable_op_t : public op {
public:
    using ResultType = T; // doesn't even work, because it has to be explicitly mentioned in base...
};

// Abstract (inherited) predecessor made for ({SType, SBool}) -> {SType, SBool} ops . Potentaially unwraps there types,
// ... inherited classes can just use the accessors: val(), deriv(), ...
template<typename T, typename T1>
class unary_op_template {
protected:
    using UT = unwrapped_type_t<T1>; // removes cvrefs, and unwraps SType
public:
    UT _arg1; //TODO: PROTECTED!!! just test-wise

    unary_op_template(T1&& arg1) 
    : _arg1(unwrap(std::forward<T1>(arg1))) {}

    // if members are not pointer-types, they must have been moved here, or lvalue-references. 
    // NOTE: value_op lvalues are a real problem! STypes can thus only be assigned from rvalues?
    // ... also disallow numeric members from being backpropagated, since they can't
    inline void evaluate_potential_member_args() {
        if constexpr(!is_tapes_pointer_v<UT> && !std::is_pointer_v<UT> && !is_numeric_v<UT>) _arg1.evaluate();
    }    
    inline void backprop_potential_member_args() {
        if constexpr(!is_tapes_pointer_v<UT> && !std::is_pointer_v<UT> && !is_numeric_v<UT>) _arg1.backprop();
    }
};


// Abstract predecessor for ({SType, SBool} x {SType, SBool}) -> {SType, SBool}, ops. Potentially unwraps there types,
// ... inherited classes can just use the accessors: val(), deriv(), ...
template<typename T, typename T1, typename T2>
class binary_op_template {
protected:
    // these two remove the cvrefs from type, and 
    using UT1 = unwrapped_type_t<T1>;
    using UT2 = unwrapped_type_t<T2>;
    // either (or both) of T1, T2 can be pointer types. One may be a number
    // here, pointer describes a tape-position, and unwarp the underlying pointer from the SType (lvalues that receive rvalues)
public:
    UT1 _arg1; //TODO: PROTECTED!!! just test-wise
    UT2 _arg2; //TODO: PROTECTED!!! just test-wise

    // BTW: no problems with const-input, since UT1, UT2 go through remove_cvref!
    binary_op_template(T1&& arg1, T2&& arg2) 
    : _arg1(unwrap(std::forward<T1>(arg1))), _arg2(unwrap(std::forward<T2>(arg2))) {}

    // if members are not pointer-types, they must have been moved here, or lvalue-references. Among those moved types, backprop must be stopped at numeric ones, since they can't 
    // NOTE: value_op lvalues are a real problem! STypes can thus only be assigned from rvalues?
    inline void evaluate_potential_member_args() {
        if constexpr(!is_tapes_pointer_v<UT1> && !std::is_pointer_v<UT1> && !is_numeric_v<UT1>) _arg1.evaluate();
        if constexpr(!is_tapes_pointer_v<UT2> && !std::is_pointer_v<UT2> && !is_numeric_v<UT2>) _arg2.evaluate();
    }    
    inline void backprop_potential_member_args() {
        if constexpr(!is_tapes_pointer_v<UT1> && !std::is_pointer_v<UT1> && !is_numeric_v<UT1>) _arg1.backprop();
        if constexpr(!is_tapes_pointer_v<UT2> && !std::is_pointer_v<UT2> && !is_numeric_v<UT2>) _arg2.backprop();
    }
};

template<typename T>
class evaluable_value_op_t : public evaluable_op_t<T>, public value_t<T> {   
public:
    evaluable_value_op_t() : value_t<T>{0,0} {}
};
template<typename T>
class evaluable_boolean_op_t : public evaluable_op_t<T>, public boolean_t<T> {
public:
    evaluable_boolean_op_t() : evaluable_op_t<T>(), boolean_t<T>{0,0,0} {}
};


// ================================ Now that we arrived at the concrete ones, we can have requirements! ==========================
// is the SType's content castable to boolean_t?
template <class C> concept boolean_t_castable_ = requires(C c) { []<typename X>(boolean_t<X>&){}(c); };
template<typename T> static constexpr bool is_boolean_t_castable_stype_content_vh = false;
template<typename T> static constexpr bool is_boolean_t_castable_stype_content_vh<SType<T>> = boolean_t_castable_<std::remove_cvref_t<T>>;
template<typename ST> static constexpr bool is_boolean_t_castable_stype_content_v = is_boolean_t_castable_stype_content_vh<std::remove_cvref_t<ST>>;

// is the SType's content castable to boolean_t?
template <class C> concept value_t_castable_ = requires(C c) { []<typename X>(value_t<X>&){}(c); };
template<typename T> static constexpr bool is_value_t_castable_stype_content_vh = false;
template<typename T> static constexpr bool is_value_t_castable_stype_content_vh<SType<T>> = value_t_castable_<std::remove_cvref_t<T>>;
template<typename ST> static constexpr bool is_value_t_castable_stype_content_v = is_value_t_castable_stype_content_vh<std::remove_cvref_t<ST>>;

// is SType content numeric?
template<typename T> static constexpr bool is_numeric_stype_content_vh = false;
template<typename T> static constexpr bool is_numeric_stype_content_vh<SType<T>> = is_numeric_v<std::remove_cvref_t<T>>;
template<typename ST> static constexpr bool is_numeric_stype_content_v = is_numeric_stype_content_vh<std::remove_cvref_t<ST>>;

// is SType content bool?
template<typename T> static constexpr bool is_bool_stype_content_vh = false;
template<typename T> static constexpr bool is_bool_stype_content_vh<SType<T>> = is_bool_v<std::remove_cvref_t<T>>;
template<typename ST> static constexpr bool is_bool_stype_content_v = is_bool_stype_content_vh<std::remove_cvref_t<ST>>;





// is something an evaluable op?
template <class C> concept derived_of_evaluable_op_ = requires(C c) { []<typename X>(evaluable_op_t<X>&){}(c); }; // (even though we could also just test op there... TODO: and we can as soon as we have a typed tape)



// to be honest: all the following could also exist in a simpler form, since the template-checks usually contain the value_t (or boolean_t) check separately. evaluable_value_op is already descendant of boolean_t. This one's pointer-check is necessary though 
template <class C> concept derived_of_evaluable_value_op_ = requires(C c) { []<typename X>(evaluable_value_op_t<X>&){}(c); };
template <class C> concept derived_of_evaluable_boolean_op_ = requires(C c) { []<typename X>(evaluable_boolean_op_t<X>&){}(c); };
template<typename T> static constexpr bool derived_of_evaluable_value_op_or_pointer_v = derived_of_evaluable_value_op_<std::remove_pointer_t<std::remove_cvref_t<T>>>;
template<typename T> static constexpr bool derived_of_evaluable_boolean_op_or_pointer_v = derived_of_evaluable_boolean_op_<std::remove_pointer_t<std::remove_cvref_t<T>>>; // && is_boolean_t_derived_v<std::remove_pointer_t<std::remove_cvref_t<T>>>;


// This is a Value op, which emulates "(numeric, numeric) -> numeric"
// Requirement: Inputs {T1, T2} have to be SType, a numeric type, or evaluable_op_t derivative. 
// At least one of {T1, T2} has to be evaluable_op_t derivative or SType
// SType-s must have nested type of value_t derived optype or numeric type
template<typename Op, typename T, typename T1, typename T2> 
requires (
    (is_value_t_castable_stype_content_v<T1> || is_numeric_stype_content_v<T1> || is_numeric_v<T1> || derived_of_evaluable_value_op_or_pointer_v<T1> ) && 
    (is_value_t_castable_stype_content_v<T2> || is_numeric_stype_content_v<T2> || is_numeric_v<T2> || derived_of_evaluable_value_op_or_pointer_v<T2> ) && 
    (is_stype_v<T1> || is_stype_v<T2> || derived_of_evaluable_value_op_or_pointer_v<T1> || derived_of_evaluable_value_op_or_pointer_v<T2>) 
)
class binary_value_op : public binary_op_template<T, T1, T2>, public evaluable_value_op_t<T>  { // TOOD: ValueOp intermediary
public:
    // using base_value_type = value_t<T>;
    using bot = binary_op_template<T, T1, T2>;
    using UT1 = typename bot::UT1;
    using UT2 = typename bot::UT2;

    binary_value_op(T1&& arg1, T2&& arg2) : 
        binary_op_template<T, T1, T2>(std::forward<T1>(arg1), std::forward<T2>(arg2)), evaluable_value_op_t<T>() {}

    void evaluate() {
        this->evaluate_potential_member_args(); // if args are rvalues, they aren't evaluated trough the tape!
        this->value = Op::eval(val(maybe_deref(bot::_arg1)), val(maybe_deref(bot::_arg2))); 
    }
    void backprop() {
        // Note: No need (and possibility) to backpropagate to numeric. Only bases of value_t, evaluable_op_t (or pointers to such)
        if constexpr(!is_numeric_v<UT1>) { maybe_deref(bot::_arg1).partial += Op::partial_x1(val(maybe_deref(bot::_arg1)), val(maybe_deref(bot::_arg2)), this->partial); }
        if constexpr(!is_numeric_v<UT2>) { maybe_deref(bot::_arg2).partial += Op::partial_x2(val(maybe_deref(bot::_arg1)), val(maybe_deref(bot::_arg2)), this->partial); }
        // rvalues don't backpropagate on the tape. force them to do it internally here!
        this->backprop_potential_member_args();
        // if constexpr(!is_numeric_v<UT1>) { std::cout << "bv d " << maybe_deref(bot::_arg1).partial << " " << val(maybe_deref(bot::_arg1)) << std::endl; }
        // if constexpr(!is_numeric_v<UT2>) { std::cout << "bv d " << maybe_deref(bot::_arg2).partial << " " << val(maybe_deref(bot::_arg2)) << std::endl; }
    }
};


template<typename Op, typename T, typename T1> 
requires (
    (is_value_t_castable_stype_content_v<T1> || is_numeric_stype_content_v<T1> || is_numeric_v<T1> || derived_of_evaluable_value_op_or_pointer_v<T1> ) && 
    (is_stype_v<T1> || derived_of_evaluable_value_op_or_pointer_v<T1>) 
)
class unary_value_op : public unary_op_template<T, T1>, public evaluable_value_op_t<T>  { 
public:
    // using base_value_type = value_t<T>;
    using uot = unary_op_template<T, T1>;
    using UT = typename uot::UT;

    unary_value_op(T1&& arg1) : 
        unary_op_template<T, T1>(std::forward<T1>(arg1)), evaluable_value_op_t<T>() {}

    void evaluate() {
        this->evaluate_potential_member_args(); // if args are rvalues, they aren't evaluated trough the tape!
        this->value = Op::eval(val(maybe_deref(uot::_arg1))); 
    }
    void backprop() {
        // Note: No need (and possibility) to backpropagate to numeric. Only bases of value_t, evaluable_op_t (or pointers to such)
        if constexpr(!is_numeric_v<UT>) { maybe_deref(uot::_arg1).partial += Op::partial(val(maybe_deref(uot::_arg1)), this->partial); }
        // if constexpr(!is_numeric_v<UT2>) { maybe_deref(bot::_arg2).partial += Op::partial_x2(val(maybe_deref(bot::_arg1)), val(maybe_deref(bot::_arg2)), this->partial); }
        // rvalues don't backpropagate on the tape. force them to do it internally here!
        this->backprop_potential_member_args();
        // if constexpr(!is_numeric_v<UT1>) { std::cout << "bv d " << maybe_deref(bot::_arg1).partial << " " << val(maybe_deref(bot::_arg1)) << std::endl; }
        // if constexpr(!is_numeric_v<UT2>) { std::cout << "bv d " << maybe_deref(bot::_arg2).partial << " " << val(maybe_deref(bot::_arg2)) << std::endl; }
    }
};

// // value op unary. E.g. sin, cos, exp, ...
// // numeric -> numeric
// template<typename Op, typename T, typename T1>
// requires (
//     (is_value_t_castable_stype_content_v<T1> || is_numeric_stype_content_v<T1> || is_numeric_v<T1> || derived_of_evaluable_value_op_or_pointer_v<T1> ) && 
//     (is_stype_v<T1> || derived_of_evaluable_value_op_or_pointer_v<T1> )
// )
// class unary_value_op : public unary_op_template<T, T1>, evaluable_value_op_t<T> {
// public:
//     using uot = unary_op_template<T, T1>;
//     using UT1 = typename uot::UT1;

//     unary_value_op(T1&& arg) : unary_op_template<T, T1>(std::forward<T1>(uot::_arg1)), evaluable_value_op_t<T>() {}

//     void evaluate() {
//         this->evaluate_potential_member_args(); // if args are rvalues, they aren't evaluated trough the tape!
//         this->value = Op::eval(val(maybe_deref(uot::_arg1))); 
//     }

//     void backprop() {
//         // Note: No need (and possibility) to backpropagate to numeric. Only bases of value_t, evaluable_op_t (or pointers to such)
//         if constexpr(!is_numeric_v<UT1>) { maybe_deref(uot::_arg1).partial += Op::partial(val(maybe_deref(uot::_arg1)), this->partial); }

//         this->backprop_potential_member_args();
//     }

// };

// boolean op unary. Probably only "!"
// bool -> bool (makes a clause out of a comparison_op or clause_op)
template<typename Op, typename T, typename T1>
requires (
    (is_boolean_t_castable_stype_content_v<T1> || is_bool_stype_content_v<T1> || is_bool_v<T1> || derived_of_evaluable_boolean_op_or_pointer_v<T1> ) && 
    (is_stype_v<T1> || derived_of_evaluable_boolean_op_or_pointer_v<T1> )
)
class unary_clause_op : public unary_op_template<T, T1>, public evaluable_boolean_op_t<T> {
public:
    using uot = unary_op_template<T, T1>;
    using UT1 = typename uot::UT;

    unary_clause_op(T1&& arg) : unary_op_template<T, T1>(std::forward<T1>(arg)), evaluable_boolean_op_t<T>() {}

    void evaluate() {
        this->evaluate_potential_member_args(); // if args are rvalues, they aren't evaluated trough the tape!
        this->discrete = Op::eval_discrete(disc(maybe_deref(uot::_arg1))); 
        this->tendency = Op::eval_tendency(tend(maybe_deref(uot::_arg1))); 
    }

    void backprop() {
        // // Note: No need (and possibility) to backpropagate to numeric. Only bases of value_t, evaluable_op_t (or pointers to such)
        if constexpr(!is_bool_v<UT1>) { maybe_deref(uot::_arg1).partial += Op::partial(tend(maybe_deref(uot::_arg1)), this->partial); }
        this->backprop_potential_member_args();
    }

};


// This is a Comparison op, which emulates "(numeric, numeric) -> boolean"
// Requirement: Inputs {T1, T2} have to be SType, or one may be a numeric type. Its input SType_s must have value_t derived optype or numeric type nested
template<typename Op, typename T, typename T1, typename T2> 
requires (
    (is_value_t_castable_stype_content_v<T1> || is_numeric_stype_content_v<T1> || is_numeric_v<T1> || derived_of_evaluable_value_op_or_pointer_v<T1>) && 
    (is_value_t_castable_stype_content_v<T2> || is_numeric_stype_content_v<T2> || is_numeric_v<T2> || derived_of_evaluable_value_op_or_pointer_v<T2>) && 
    (is_stype_v<T1> || is_stype_v<T2> || derived_of_evaluable_value_op_or_pointer_v<T1> || derived_of_evaluable_value_op_or_pointer_v<T2>) 
)
class binary_comparison_op : public binary_op_template<T, T1, T2>, public evaluable_boolean_op_t<T>   { // boolean_t type-op pushed to tape
public:
    // using base_value_type = value_t<T>;
    // using ResultType = T;
    using bot = binary_op_template<T, T1, T2>;
    using UT1 = typename bot::UT1;
    using UT2 = typename bot::UT2;

    binary_comparison_op(T1&& arg1, T2&& arg2) : 
        binary_op_template<T, T1, T2>(std::forward<T1>(arg1), std::forward<T2>(arg2)), evaluable_boolean_op_t<T>() {}

    void evaluate() {
        this->evaluate_potential_member_args();
        auto&& a1 = val(maybe_deref(bot::_arg1));
        auto&& a2 = val(maybe_deref(bot::_arg2));
        this->discrete = Op::eval_discrete(a1, a2);
        // if we're comparing to an inf, who the hell cares what the tendency is! (same as discrete) 
        this->tendency = (std::isinf(a1) || std::isinf(a2)) ? this->discrete : Op::eval_tendency(a1, a2); 
    }
    void backprop() {

        // TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // The question here is whether inf <= inf  should have derivative 0.25, or 0 (as currently done! but tecnhically wrong)
        // TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if constexpr(!is_numeric_v<UT1>) {
            auto&& a1 = val(maybe_deref(bot::_arg1));
            auto&& a2 = val(maybe_deref(bot::_arg2));
            maybe_deref(bot::_arg1).partial += (std::isinf(a1) || std::isinf(a2)) ? 0 : Op::partial_x1(a1, a2, this->tendency, this->partial); 
        }
        if constexpr(!is_numeric_v<UT2>) {
            auto&& a1 = val(maybe_deref(bot::_arg1));
            auto&& a2 = val(maybe_deref(bot::_arg2));
            maybe_deref(bot::_arg2).partial += (std::isinf(a1) || std::isinf(a2)) ? 0 : Op::partial_x2(a1, a2, this->tendency, this->partial); 
        }
        this->backprop_potential_member_args();
        // if constexpr(!is_numeric_v<UT1>) { std::cout << "bcomp d " << maybe_deref(bot::_arg1).partial << std::endl; }
        // if constexpr(!is_numeric_v<UT2>) { std::cout << "bcomp d " << maybe_deref(bot::_arg2).partial << std::endl; }
    }

};

// This is a Clause Op, which emulates "(boolean, boolean) -> boolean"
// Requirement: Inputs {T1, T2} have to be SType, or one may be bool. Its input SType_s must have boolean_t derived optype nested.
template<typename Op, typename T, typename T1, typename T2> 
requires 
    (is_boolean_t_castable_stype_content_v<T1> || is_bool_v<T1> || derived_of_evaluable_boolean_op_or_pointer_v<T1>) && 
    (is_boolean_t_castable_stype_content_v<T2> || is_bool_v<T2> || derived_of_evaluable_boolean_op_or_pointer_v<T2>) && 
    (is_stype_v<T1> || is_stype_v<T2> || derived_of_evaluable_boolean_op_or_pointer_v<T1> || derived_of_evaluable_boolean_op_or_pointer_v<T2>)
class binary_clause_op : public binary_op_template<T, T1, T2>, public evaluable_boolean_op_t<T>   {
public:
    // using ResultType = T;
    using bot = binary_op_template<T, T1, T2>;
    using UT1 = typename bot::UT1;
    using UT2 = typename bot::UT2;

    binary_clause_op(T1&& arg1, T2&& arg2) : 
        binary_op_template<T, T1, T2>(std::forward<T1>(arg1), std::forward<T2>(arg2)), evaluable_boolean_op_t<T>() {}

    void evaluate() {
        this->evaluate_potential_member_args();
        this->discrete = Op::eval_discrete(disc(maybe_deref(bot::_arg1)), disc(maybe_deref(bot::_arg2)));
        this->tendency = Op::eval_tendency(tend(maybe_deref(bot::_arg1)), tend(maybe_deref(bot::_arg2)));
    }
    void backprop() {
        /// TODO: This in evaluated_reverse!
        if constexpr(!is_bool_v<T1>) maybe_deref(bot::_arg1).partial += Op::partial_t1(tend(maybe_deref(bot::_arg1)), tend(maybe_deref(bot::_arg2)), this->partial);
        if constexpr(!is_bool_v<T2>) maybe_deref(bot::_arg2).partial += Op::partial_t2(tend(maybe_deref(bot::_arg1)), tend(maybe_deref(bot::_arg2)), this->partial);

        this->backprop_potential_member_args();
        // if constexpr(!is_bool_v<UT1>) { std::cout << "bcl d " << maybe_deref(bot::_arg1).partial << std::endl; }
        // if constexpr(!is_bool_v<UT2>) { std::cout << "bcl d " << maybe_deref(bot::_arg2).partial << std::endl; }
    } 

};

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// Requirement: Input T1 has to be SType.
template<typename T, typename T1, typename T2> 
requires 
    (is_value_t_castable_stype_content_v<T1> || is_numeric_stype_content_v<T1>) && 
    (is_value_t_castable_stype_content_v<T2> || is_numeric_stype_content_v<T2>|| is_numeric_v<T2> || derived_of_evaluable_value_op_or_pointer_v<T2>)
    // && (is_stype_v<T1> || is_stype_v<T2> || derived_of_evaluable_boolean_op_or_pointer_v<T>)
class assign_op : public binary_op_template<T, T1, T2>, public evaluable_op_t<T>   {
public:

    using bot = binary_op_template<T, T1, T2>;
    // even though those aren't references, they are always pointers in the case of STypes! 
    // so we require that T1 is an SType (=>UT1 is its pointer)  
    using UT1 = typename bot::UT1;
    using UT2 = typename bot::UT2;

    value_t<T> saved_value = {0, 0}; // will be assigned to arg1! We always have default-deriv 1!

    // let's say we have x = y;
    // "target" x, "source" y;
    assign_op(T1&& arg1, T2&& arg2) : 
        binary_op_template<T, T1, T2>(std::forward<T1>(arg1), std::forward<T2>(arg2)), 
        evaluable_op_t<T>() {} 

    void evaluate() {
        this->evaluate_potential_member_args(); // arg1 must be already eval'd, butmaybe arg2 isn't...
        saved_value = maybe_deref(bot::_arg1); // preserve value of target (and deriv, which isn't set yet)
        val(maybe_deref(bot::_arg1)) = val(maybe_deref(bot::_arg2)); // assign value (and derivative if it's even set)
        if constexpr(is_value_t_castable_stype_content_v<T2>) maybe_deref(bot::_arg1).partial = maybe_deref(bot::_arg2).partial; // but if it has a derivative, we might as well reset too for consistency
    }

    void backprop() {
        // 1) backpropagate our current derivative to our "source", as-is
        deriv(maybe_deref(bot::_arg2)) += deriv(maybe_deref(bot::_arg1));
        // 2) make the target  (and the derivative, which is the old non-set one)!
        maybe_deref(bot::_arg1) = saved_value; 
        // val(maybe_deref(bot::_arg1)) = val(saved_value); 
        // deriv(maybe_deref(bot::_arg1)) += deriv(saved_value);  // not that this matters. Should be 0 anyway
        this->backprop_potential_member_args(); // useful? necessary? probably wouldn't even function...
    }

};
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////



// this op is what's evaluated inside the if-statements
template<typename T, typename T1> 
requires (is_boolean_t_castable_stype_content_v<T1> || derived_of_evaluable_boolean_op_or_pointer_v<T1>)
class split_op : public unary_op_template<T, T1>, public evaluable_boolean_op_t<T> {
public:
    // using ResultType = T;
    using uot = unary_op_template<T, T1>; // holds _arg1
    using UT = typename uot::UT;

    bool _eval_reverse;
    bool pure_discrete; // needed for backprop

    split_op(T1&& arg1, bool const eval_reverse) : 
        unary_op_template<T, T1>(std::forward<T1>(arg1)), evaluable_boolean_op_t<T>(), _eval_reverse(eval_reverse) {}

    void evaluate() {
        this->evaluate_potential_member_args();

        // 1) evaluate pure, and flip if we're reversing
        pure_discrete = disc(maybe_deref(uot::_arg1));
        this->discrete = _eval_reverse ? ! pure_discrete : pure_discrete;

        // 2) get the main-contribution (strength of branch, always > 0.5)
        auto&& main_contribution = pure_discrete ? tend(maybe_deref(uot::_arg1)) : 1-tend(maybe_deref(uot::_arg1));

        // 3) main-contribution is inversed if we're inversing
        this->tendency = _eval_reverse ? 1-main_contribution : main_contribution; 
    }

    void backprop() {

        // partial is tendency_a (reverse 3)
        auto&& main_contribution_a = _eval_reverse ? -1 * this->partial :  this->partial;

        // (reverse 2))
        maybe_deref(uot::_arg1).partial += pure_discrete ?  main_contribution_a : - main_contribution_a;

        this->backprop_potential_member_args();
        // if constexpr(!is_bool_v<UT>) { std::cout << "spl d " << maybe_deref(uot::_arg1).partial << std::endl; }

    } 

};


#include "operators.hpp"