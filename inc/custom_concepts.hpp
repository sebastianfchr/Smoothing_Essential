#pragma once 
#include <type_traits> 
#include <tuple>
#include <concepts>


// first argumnent is same type to either of the following arguments
template<typename T, typename R, typename... Rs>
struct is_either { static constexpr bool value = std::is_same<T,R>::value || is_either<T, Rs...>::value; };
template<typename T, typename R>
struct is_either<T, R> { static constexpr bool value = std::is_same<T,R>::value; };


// (T either of R, Rs...) AND (Q either of R, Rs...)
template<typename T, typename Q, typename R, typename... Rs>
struct are_either { static constexpr bool value = is_either<T,R,Rs...>::value && is_either<Q,R,Rs...>::value; };


// T is base of either R, or any of Rs...
template<typename T, typename R, typename ...Rs>
struct is_base_of_either { static constexpr bool value = std::is_base_of_v<T, R> || is_base_of_either<T, Rs...>::value; };
template<typename T, typename R>
struct is_base_of_either<T, R> { static constexpr bool value = std::is_base_of_v<T, R>; };



// TODO: this can be shortened
// R is derived from (or same as) Q or T 
template<typename T, typename Q, typename R>
struct is_either_or_desc_thereof { static constexpr bool value = std::is_base_of_v<T, R> || std::is_base_of_v<Q, R>; };
// third and following argument are either equal or derived from the first two
template<typename T, typename Q, typename R, typename... Rs>
struct are_either_or_desc_thereof{ 
    static constexpr bool value = is_either_or_desc_thereof<T,Q,R>::value && are_either_or_desc_thereof<T,Q,Rs...>::value;
};
// base case for above
template<typename T, typename Q, typename R>
struct are_either_or_desc_thereof<T,Q,R> { 
    static constexpr bool value = is_either_or_desc_thereof<T,Q,R>::value;
};


// all descendants or same type as first one
template<typename T, typename R, typename ... Rs>
struct base_of_all { static constexpr bool value = std::is_base_of_v<std::decay_t<T>,std::decay_t<R>> && base_of_all<std::decay_t<T>, Rs...>::value; };
template<typename T, typename R>
struct base_of_all<T, R> { static constexpr bool value = std::is_base_of_v<std::decay_t<T>,std::decay_t<R>>; };


// R, Rs... are all of type T
template<typename T, typename R, typename... Rs>
struct are_all { static constexpr bool value = std::is_same<T,R>::value && are_all<T, Rs...>::value; };
template<typename T, typename R>
struct are_all<T, R> { static constexpr bool value = std::is_same<T,R>::value; };


// bool, char, float, double, long double...
template<typename T> static constexpr bool is_numeric_v = std::is_arithmetic_v<std::remove_cvref_t<T>> || std::is_floating_point_v<std::remove_cvref_t<T>>;
template<typename T> static constexpr bool is_bool_v = std::is_same_v<std::remove_cvref_t<T>, bool>;
