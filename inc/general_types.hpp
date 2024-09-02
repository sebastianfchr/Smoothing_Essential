#pragma once 
#include <iostream>
// These are the val and deriv accessors!
 
template<typename T>
struct value_t { 
    T value;
    T partial;
};

template<typename T>
struct boolean_t {
    bool discrete;
    // Note that we cannot scrap discrete!
    T tendency;
    T partial; // this is the partial of the tendency! 
};

// derived of value_t
template <class C> concept is_value_t_derived_ = requires(C c) { []<typename X>(value_t<X>&){}(c); };
template<typename T> static constexpr bool is_value_t_derived_v = is_value_t_derived_<std::remove_cvref<T>>;

// derived of boolean_t
template <class C> concept is_boolean_t_derived_ = requires(C c) { []<typename X>(boolean_t<X>&){}(c); };
template<typename T> static constexpr bool is_boolean_t_derived_v = is_boolean_t_derived_<std::remove_cvref<T>>;



// ============= actual op
class op {
public:
    virtual void evaluate() = 0;
    virtual void backprop() = 0;
};


template<typename T>
std::ostream& operator<<(std::ostream& o, const value_t<T>& v) {
    o << "{" << v.value << ", " << v.partial << "}"; 
    return o;
}

template<typename T>
std::ostream& operator<<(std::ostream& o, const boolean_t<T>& v) {
    o << "{" << v.discrete << ", " << v.tendency << ", " << v.partial << "}"; 
    return o;
}

// ================ VAL(), DERIV() =========================
// * Applied vaculously to any type!
// or we apply it to what comes out of val(...), which is a value-reference on a dereferenced pointer (case 2), 
//      ... in case 2, therefore rvalref is enough!
// A NOTE: in many cases, this will be a base-to-derived pointer (i.e. value_t<T>& a = binary_value_op<Op, T, ...>). And that's all fine (works similar to pointer and doesn't re-construct anything!)
template<typename T> requires is_numeric_v<T>
constexpr T&& val(T&& v) { return std::forward<T>(v); }
template<typename T>  
constexpr auto& val(value_t<T>& v) { return v.value; }
template<typename T>  
constexpr auto& val(value_t<T> const& v) { return v.value; }
// constexpr auto& val(value_t<T>&& v) { return v.value; }

template<typename T> requires is_numeric_v<T>
constexpr T&& deriv(T&& v) { return v; }
template<typename T> // same as (A NOTE)
constexpr auto& deriv(value_t<T>& v) { return v.partial; }
template<typename T> // same as (A NOTE)
constexpr auto& deriv(value_t<T> const& v) { return v.partial; }

// ================ DIS(), TEND() =========================
template<typename T> requires is_numeric_v<T>
constexpr T&& disc(T&& v) { return v; }
template<typename T>  
constexpr auto& disc(boolean_t<T>& v) { return v.discrete; }

template<typename T> requires is_numeric_v<T>
constexpr T&& tend(T&& v) { return v; }
template<typename T>  
constexpr auto& tend(boolean_t<T>& v) { return v.tendency; }

// template<typename T> requires is_numeric_v<T>
// constexpr T&& deriv(T&& v) { return v; }
template<typename T> // same as (A NOTE)
constexpr auto& deriv(boolean_t<T>& v) { return v.partial; }


// ================================== marker

enum marker_t {
    eval_main,          // eval according to discrete path this round, but eval complementary next round
    eval_main_only,     // eval according to discrete path this round, but faces removal (because complementary eval can be omitted) 
    eval_complement     // eval complementary this round, faces removal
};

bool to_bool(const marker_t& m) { 
    // first two options are true (to mean: main_contribution evaluated) 
    return (m == eval_main || m == eval_main_only); 
}