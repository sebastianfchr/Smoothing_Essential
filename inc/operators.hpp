
/// ================== TOP ISSUES ======================
// * make sure only early STypes are kept, rest is deleted
// * use contribution inside SOutType additoon
// * make sure backprop() works inside SOutType

// binary_creator<Add, double>::create_op(1, 2)

// template<typename Op, typename T> or just template<typename T>, depending on called op
template<typename ...Op_and_T>
struct binary_creator {
    
    template<typename T1, typename T2>
    static decltype(auto) create_val_op(T1&& x1, T2&& x2) { return binary_value_op<Op_and_T..., T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }
    template<typename T1, typename T2>
    static decltype(auto) create_val_op_pointer(T1&& x1, T2&& x2) { return new binary_value_op<Op_and_T..., T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }

    template<typename T1, typename T2>
    static decltype(auto) create_comparison_op(T1&& x1, T2&& x2) { return binary_comparison_op<Op_and_T..., T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }
    template<typename T1, typename T2>
    static decltype(auto) create_comparison_op_pointer(T1&& x1, T2&& x2) { return new binary_comparison_op<Op_and_T..., T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }

    template<typename T1, typename T2>
    static decltype(auto) create_clause_op(T1&& x1, T2&& x2) { return binary_clause_op<Op_and_T..., T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }
    template<typename T1, typename T2>
    static decltype(auto) create_clause_op_pointer(T1&& x1, T2&& x2) { return new binary_clause_op<Op_and_T..., T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }

    template<typename T1, typename T2>
    static decltype(auto) create_assign_op(T1&& target, T2&& source) { return assign_op<Op_and_T..., T1, T2>(std::forward<T1>(target), std::forward<T2>(source)); }
    template<typename T1, typename T2>
    static decltype(auto) create_assign_op_pointer(T1&& target, T2&& source) { return new assign_op<Op_and_T..., T1, T2>(std::forward<T1>(target), std::forward<T2>(source)); }
};

// template<typename Op, typename T>
// struct binary_creator {
    
//     template<typename T1, typename T2>
//     static decltype(auto) create_val_op(T1&& x1, T2&& x2) { return binary_value_op<Op, T, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }
//     template<typename T1, typename T2>
//     static decltype(auto) create_val_op_pointer(T1&& x1, T2&& x2) { return new binary_value_op<Op, T, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }

//     template<typename T1, typename T2>
//     static decltype(auto) create_comparison_op(T1&& x1, T2&& x2) { return binary_comparison_op<Op, T, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }
//     template<typename T1, typename T2>
//     static decltype(auto) create_comparison_op_pointer(T1&& x1, T2&& x2) { return new binary_comparison_op<Op, T, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }

//     template<typename T1, typename T2>
//     static decltype(auto) create_clause_op(T1&& x1, T2&& x2) { return binary_clause_op<Op, T, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }
//     template<typename T1, typename T2>
//     static decltype(auto) create_clause_op_pointer(T1&& x1, T2&& x2) { return new binary_clause_op<Op, T, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2)); }

// };

template<typename Op, typename T> 
struct unary_creator {

    template<typename T1>
    static decltype(auto) create_val_op(T1&& x1) { return unary_value_op<Op, T, T1>(std::forward<T1>(x1)); }
    template<typename T1>
    static decltype(auto) create_val_op_pointer(T1&& x1) { return new unary_value_op<Op, T, T1>(std::forward<T1>(x1)); }

    template<typename T1>
    static decltype(auto) create_clause_op(T1&& x1) { return unary_clause_op<Op, T, T1>(std::forward<T1>(x1)); }
    template<typename T1>
    static decltype(auto) create_clause_op_pointer(T1&& x1) { return new unary_clause_op<Op, T, T1>(std::forward<T1>(x1)); }
    
};



// ================ Value Ops Unary ==================

// TODO: uncomment unary_value_op

// ================ Value Ops Binary ==================
template<typename T1, typename T2> requires (is_stype_v<T1> || is_stype_v<T2>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) operator+(T1&& x1, T2&& x2) {

    // question is whether it should be added to the tape here, or in the SType.
    // probably in the SType is correct: 
    // if we want to aggregate binary_value_ops into intermediary-types, we can do so and push them 
    // only once they are instantiated by an SType
    // auto*&& bo = new binary_value_op<Add, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    auto*&& bo = binary_creator<Add, double>::create_val_op_pointer(std::forward<T1>(x1), std::forward<T2>(x2));
    bo->evaluate();
    return SType(bo);
}
template<typename T1, typename T2> requires (is_stype_v<T1> || is_stype_v<T2>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) operator*(T1&& x1, T2&& x2) {
    // auto*&& bo = new binary_value_op<Multiply, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    auto*&& bo = binary_creator<Multiply, double>::create_val_op_pointer(std::forward<T1>(x1), std::forward<T2>(x2));
    bo->evaluate();
    return SType(bo);
}
template<typename T1, typename T2> requires (is_stype_v<T1> || is_stype_v<T2>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) operator-(T1&& x1, T2&& x2) {
    // auto*&& bo = new binary_value_op<Subtract, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    auto*&& bo = binary_creator<Subtract, double>::create_val_op_pointer(std::forward<T1>(x1), std::forward<T2>(x2));
    bo->evaluate();
    return SType(bo);
}


template<typename T1, typename T2> requires (is_stype_v<T1> || is_stype_v<T2>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) operator/(T1&& x1, T2&& x2) {
    // auto*&& bo = new binary_value_op<Divide, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    auto*&& bo = binary_creator<Divide, double>::create_val_op_pointer(std::forward<T1>(x1), std::forward<T2>(x2));
    bo->evaluate();
    return SType(bo);
}
template<typename T1, typename T2> requires (is_stype_v<T1> || is_stype_v<T2>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) pow(T1&& x1, T2&& x2) {
    // auto*&& bo = new binary_value_op<Divide, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    auto*&& bo = binary_creator<Pow, double>::create_val_op_pointer(std::forward<T1>(x1), std::forward<T2>(x2));
    bo->evaluate();
    return SType(bo);
}


// =========== Value Ops unary ===============
template<typename T1> requires (is_stype_v<T1>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) operator-(T1&& x1) {
    // auto*&& bo = new binary_value_op<Divide, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    auto*&& bo = unary_creator<Negative, double>::create_val_op_pointer(std::forward<T1>(x1));
    bo->evaluate();
    return SType(bo);
}

template<typename T1> requires (is_stype_v<T1>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) sin(T1&& x1) {
    // auto*&& bo = new binary_value_op<Divide, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    auto*&& bo = unary_creator<Sin, double>::create_val_op_pointer(std::forward<T1>(x1));
    bo->evaluate();
    return SType(bo);
}

template<typename T1> requires (is_stype_v<T1>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) cos(T1&& x1) {
    // auto*&& bo = new binary_value_op<Divide, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    auto*&& bo = unary_creator<Cos, double>::create_val_op_pointer(std::forward<T1>(x1));
    bo->evaluate();
    return SType(bo);
}


// =========== Comparison ===========

template<typename T1, typename T2> requires (is_stype_v<T1> || is_stype_v<T2>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) operator<(T1&& x1, T2&& x2) {
    auto*&& bo = new binary_comparison_op<Lt, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    // auto*&& bo = binary_creator<Lt, double>::create_val_op_pointer<T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    bo->evaluate();  
    return SType(bo);
}

template<typename T1, typename T2> requires (is_stype_v<T1> || is_stype_v<T2>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) operator<=(T1&& x1, T2&& x2) {
    auto*&& bo = new binary_comparison_op<Le, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    // auto*&& bo = binary_creator<Lt, double>::create_val_op_pointer<T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    bo->evaluate();  
    return SType(bo);
}

template<typename T1, typename T2> requires (is_stype_v<T1> || is_stype_v<T2>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) operator>(T1&& x1, T2&& x2) {
    auto*&& bo = new binary_comparison_op<Gt, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    // auto*&& bo = binary_creator<Lt, double>::create_val_op_pointer<T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    bo->evaluate();  
    return SType(bo);
}

template<typename T1, typename T2> requires (is_stype_v<T1> || is_stype_v<T2>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) operator>=(T1&& x1, T2&& x2) {
    auto*&& bo = new binary_comparison_op<Ge, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    // auto*&& bo = binary_creator<Lt, double>::create_val_op_pointer<T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    bo->evaluate();  
    return SType(bo);
}

// =========== Clause Binary ===========

template<typename T1, typename T2> requires (is_stype_v<T1> || is_stype_v<T2>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) operator&&(T1&& x1, T2&& x2) {
    auto*&& bo = new binary_clause_op<And, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    bo->evaluate();
    return SType(bo);
}

template<typename T1, typename T2> requires (is_stype_v<T1> || is_stype_v<T2>) //is_base_of_either<S, T1, T2>::value 
decltype(auto) operator||(T1&& x1, T2&& x2) {
    auto*&& bo = new binary_clause_op<Or, double, T1, T2>(std::forward<T1>(x1), std::forward<T2>(x2));
    bo->evaluate();
    return SType(bo);
}

// =========== Clause Unary ===========

template<typename T1> requires (is_stype_v<T1>) 
decltype(auto) operator!(T1&& x) {
    auto*&& co = unary_creator<Not, double>::create_clause_op_pointer(std::forward<T1>(x));
    co->evaluate();
    return SType(co);
}
