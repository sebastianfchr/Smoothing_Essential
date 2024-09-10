#pragma once
#include <type_traits>
#include "custom_concepts.hpp"

#include "ValueOps.hpp"
#include "BooleanOps.hpp"
#include "STape.hpp" // new!
#include <iostream>


template<typename T>
class SType {

    // Look at constructor: SType is either initialized with [evaluable_boolean_op // evaluable_value_op] or numeric value!
    // in the prior case, take this op. In the latter, make a value_t from the numeric     
    using Type = typename std::conditional_t<derived_of_evaluable_value_op_<T> || derived_of_evaluable_boolean_op_<T>, 
                                                T,
                                                value_t<T> >;    
public:
    using PtrT = tapes_pointer<Type>; // this PtrT emulates a pointer to our AllTape! 

    PtrT valptr; 

    // this one's for the returns [if we don't do this, they want to use the deleted operator= WHY?]
    // TODO: THIS IS FALSE!!!
    SType(SType const& other) = default;
    SType(SType&& other) = default;

    // we only want to permit our way of assignment to STypes ...
    SType& operator=(const SType& other) = delete;
    SType& operator=(SType&& other) = delete;

    // ... which is an assignment through assign_op (re-assignments managed in line with Adjoint rules)
    template<typename TT> requires (is_value_t_castable_stype_content_v<TT> || is_numeric_stype_content_v<TT>|| is_numeric_v<TT>)
    SType& operator=(TT&& r) {
        auto&& content_ptr = binary_creator<double>::create_assign_op_pointer(*this, r);
        content_ptr->evaluate();
        tape::get_tape_ptr()->push_back_get_index(content_ptr);
        return *this;
    }

    // default-constructed must be a dummy-type, not even a valptr! Because default operator= will use constructors!
    SType() : valptr(tape::get_tape_ptr()->push_back_get_index(value_t<T>{0,0})) {} 

    // this constructor is used inside 
    // SType(T* const& valop_ptr) requires std::is_base_of_v<op, T> 
    SType(T* const& valop_ptr) requires (derived_of_evaluable_value_op_<T> || derived_of_evaluable_boolean_op_<T>) 
    : valptr(tape::get_tape_ptr()->push_back_get_index(valop_ptr)) {
        // std::cout << "\n used ptr-formulation" << std::endl; 
        // print_typename(*valop_ptr);
    }


    // value-constructor of numeric values. 
    // TODO: We could make this rvalue-constructor. But then our T in value_t would have to be remove_cvref...
    // ... solution could just be to make an additional rvalue-constructor
    SType(T v) requires (!std::is_pointer_v<T> && (is_numeric_v<T>))
    : valptr(tape::get_tape_ptr()->push_back_get_index(value_t<T>{std::move(v), 0})) { 
    }

    auto& get_value() {
        return *valptr;        
    }

};


// triggers val() and deriv() for the underlying value_t-s
template<typename T> requires (is_value_t_castable_stype_content_v<T> || is_numeric_stype_content_v<T>)
auto&& value(T&& s) { return val(s.get_value()); }
template<typename T> requires (is_value_t_castable_stype_content_v<T> || is_numeric_stype_content_v<T> || is_boolean_t_castable_stype_content_v<T>)
auto&& derivative(T&& s) { return deriv(s.get_value()); }
template<typename T> requires (is_boolean_t_castable_stype_content_v<T>)
auto&& discrete(T&& s) { return disc(s.get_value()); }



class SOutType {
private:
public:


    value_t<double> aggregated_value;
    std::weak_ptr<AdjointTape> belonging_tape; // TODO: pointer to the tape making sure we cannot assign to anymore if expired!
    std::size_t souttype_register_ind;

    SOutType() : aggregated_value{0,0}, belonging_tape(tape::get_tape_ptr()), souttype_register_ind(belonging_tape.lock()->register_souttype()) {}

    // numeric; direct  construction: push onto valtape, and set the valptr reg!   
    template<typename T> 
    SOutType& operator=(T&& op_or_stype_or_val) {
        // TODO SEB: force T to be an op! If it is a value, it needs to be turned into one!
        // TODO SEB: force T to be an op! If it is a value, it needs to be turned into one!
        // TODO SEB: force T to be an op! If it is a value, it needs to be turned into one!
        // TODO SEB: force T to be an op! If it is a value, it needs to be turned into one!
        // TODO SEB: force T to be an op! If it is a value, it needs to be turned into one!
        
        if(belonging_tape.expired()) throw std::logic_error("This SOutType's tape has been destroyed. Assignment must happen within same scope!");

        // Case 1): this SOutType had an old assignment. New assignment means
        // that old assignment is overwritten. Unused, Delete!
        if (belonging_tape.lock()->souttype_register[souttype_register_ind] != nullptr){
            delete belonging_tape.lock()->souttype_register[souttype_register_ind];
            belonging_tape.lock()->souttype_register[souttype_register_ind] = new out_op<T>(aggregated_value, std::forward<T>(op_or_stype_or_val));
        } else {
        // Case 2) no assignment yet 
            belonging_tape.lock()->souttype_register[souttype_register_ind] = new out_op<T>(aggregated_value, std::forward<T>(op_or_stype_or_val));
        }
        return *this;
    }

    ~SOutType() {
        // Dealloc after scope ends. Btw: Inputs to pointer are dangling since #SMOOTHING_END
        // since we're owning, clean up the content
        if (!belonging_tape.expired()) delete tape::get_tape_ptr()->souttype_register[souttype_register_ind];
    }

    void seed(const double& d) { 
        deriv(aggregated_value) = d; 
    }

    value_t<double> get_value() {
        return aggregated_value;
    }
    // backpropagate: 

};


// these can also be applied directly to SType!

template<typename T> requires std::is_same_v<std::remove_cvref_t<T>, SOutType>
auto value(T&& s) { return s.aggregated_value.value; }
template<typename T> requires std::is_same_v<std::remove_cvref_t<T>, SOutType>
auto derivative(T&& s) { return s.aggregated_value.derivative; }
// template<typename T> requires (is_value_t_castable_stype_content_v<T> || is_numeric_stype_content_v<T> || is_boolean_t_castable_stype_content_v<T>)
// auto&& derivative(T&& s) { return deriv(s.get_value()); }
// template<typename T> requires (is_boolean_t_castable_stype_content_v<T>)
// auto&& discrete(T&& s) { return disc(s.get_value()); }


