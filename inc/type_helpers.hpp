#pragma once
template<typename T> struct tapes_pointer;

// type-check for tapes pointer. Is the last one necessary? There's no way a tapes_pointer occurs with cvref anywhere...!
template<typename T> constexpr bool is_tapes_pointer_vh = false;
template<typename T> constexpr bool is_tapes_pointer_vh<tapes_pointer<T>> = true;
template<typename T> constexpr bool is_tapes_pointer_v = is_tapes_pointer_vh<std::remove_cvref_t<T>>;

template<typename T>
class SType ;
 
// Determine whether it's SType
template<typename T> struct is_stype_h { static constexpr bool value = false; };
template<typename T> struct is_stype_h<SType<T>> { static constexpr bool value = true; };
template<typename T> static constexpr bool is_stype_v = is_stype_h<std::remove_cvref_t<T>>::value;



// unwrapping is meant to:
// * if lvalue: remove const and ref, so that value is purely copied to tape
// * if rvalue: leave it as-is
// * additionally, if SType: unwrap and use the tape-pointer PtrT inside
// default leaves the type intact, except ...
template<typename T> struct unwrapped_type_h {  using type = T; };
// when specialized for SType, we get its Pointer Type! (which is defined to be the )
template<typename TT> struct unwrapped_type_h<SType<TT>> { using type = SType<TT>::PtrT; };
// before doing all that, peel cvref for lvalues and put into above expressions
template<typename T> 
using unwrapped_type_t = unwrapped_type_h<std::remove_cvref_t<T>>::type;


// unwrap-operator is used to extract the value-pointer from the SType (if applicable). TODO: constexpr useful here?
// 1) unwrap SType
template<typename T>
constexpr SType<T>::PtrT unwrap(SType<T> const& potential_is_stype) { return potential_is_stype.valptr; }
template<typename T>
constexpr SType<T>::PtrT unwrap(SType<T>& potential_is_stype) { return potential_is_stype.valptr; }
// 1.1) in my opinion, this happens when STypes are basically intermediate types in an aggregate operation
template<typename T>
constexpr SType<T>::PtrT unwrap(SType<T>&& potential_is_stype) { return potential_is_stype.valptr; }
// 2) just pass on all other types
template<typename T>
constexpr T&& unwrap(T&& potential_isnt_stype) { return std::forward<T>(potential_isnt_stype); }


// Deref normal pointers and tapes_pointers. Everything else is just forwarded.
template<typename TT> requires (!is_tapes_pointer_v<TT> && !std::is_pointer_v<std::remove_cvref_t<TT>>)  // this constraint excludes later
constexpr TT&& maybe_deref(TT&& value) { return std::forward<TT>(value); }
template<typename TT>       
constexpr auto& maybe_deref(tapes_pointer<TT> tape_pointer) {  return *tape_pointer; } 
template<typename TT>       
constexpr auto& maybe_deref(TT* plain_pointer) {  return *plain_pointer; }



#include <typeinfo>
#include <cxxabi.h>
#include <type_traits>
template<typename T>
void print_typename(T&& x){
    int status;
    std::cout << abi::__cxa_demangle(typeid(std::forward<T>(x)).name(), NULL, NULL, &status)  << std::endl;
}


