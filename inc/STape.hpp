#pragma once
#include <vector>
#include <stdexcept>
#include "ValueOps.hpp"
#include "BooleanOps.hpp"
#include "general_types.hpp"
#include <functional>
#include <memory>

// This is an attempt to intergrate all value types into one tape!
class ValTape {
private:
public:

    std::function<void()> function_body; // repetitively called function 


    std::vector<op* > ops;
    std::vector<value_t<long double>> vals_ld; //
    std::vector<value_t<double>> vals_d; //
    std::vector<value_t<float>> vals_f; //
    std::vector<value_t<int>> vals_i; //
    std::vector<boolean_t<long double>> bools_ld; //
    std::vector<boolean_t<double>> bools_d; //
    std::vector<boolean_t<float>> bools_f; //
    std::vector<boolean_t<int>> bools_i; //

    // #SMOOTHING describes a function body that's called repetitively. All values therein are 
    // temporary. Thus: clear tape after after execution. Outersize the cut-off point on each tape, 
    // after which values are temporary, and deletable (after execution)   
    std::size_t ops_outersize;
    std::size_t v_ld_c_outersize, v_d_c_outersize, v_f_c_outersize, v_i_c_outersize; 
    // ... same for booleans!
    std::size_t b_ld_c_outersize, b_d_c_outersize, b_f_c_outersize, b_i_c_outersize; 

    // This needs to be called before the first execution
    // to idenfity the boundaries in the above vectors until which can be cleared.
    // (we can clear anything that is created in the function-body, but not before!)
    inline void prepare_valtape() {
        // 1) mark the points after which deletion will happen 
        ops_outersize = ops.size();
        v_ld_c_outersize = vals_ld.size(); 
        
        v_d_c_outersize = vals_d.size(); 
        
        v_f_c_outersize = vals_f.size(); v_i_c_outersize = vals_i.size();  
        b_ld_c_outersize = bools_ld.size(); b_d_c_outersize = bools_d.size(); b_f_c_outersize = bools_f.size(); b_i_c_outersize = bools_i.size();     
    }
    
    // called in tape_next_round()
    void clear_valtape() {
        // erase all vals that were temporarily created in body(), 
        // since they will be re-created in successive executions


        // free pointers from ops_outersize onwards, and then erase vector entries
        for(auto oit = ops.begin()+ops_outersize; oit != ops.end(); oit++) delete *oit;
        ops.erase(ops.begin()+ops_outersize, ops.end()); // hopefully this avoids memory reshuffling
        // for the non-pointer vectors, only need to erase the vector entries
        vals_ld.erase(vals_ld.begin()+v_ld_c_outersize, vals_ld.end()); vals_d.erase(vals_d.begin()+v_d_c_outersize, vals_d.end()); vals_f.erase(vals_f.begin()+v_f_c_outersize, vals_f.end()); vals_i.erase(vals_i.begin()+v_i_c_outersize, vals_i.end());
        bools_ld.erase(bools_ld.begin()+b_ld_c_outersize, bools_ld.end()); bools_d.erase(bools_d.begin()+b_d_c_outersize, bools_d.end()); bools_f.erase(bools_f.begin()+b_f_c_outersize, bools_f.end()); bools_i.erase(bools_i.begin()+b_i_c_outersize, bools_i.end());

    }


    template<typename T>
    auto& get_tape() {
        if constexpr(std::is_same_v<T, value_t<long double>>) return vals_ld; 
        else if constexpr(std::is_same_v<T, value_t<double>>) return vals_d; 
        else if constexpr(std::is_same_v<T, value_t<float>>) return vals_f; 
        else if constexpr(std::is_same_v<T, value_t<int>>) return vals_i;
        else if constexpr (derived_of_evaluable_op_<T>) return ops; 
    }

    // Used for the value_t initializer of SType
    template<typename T> std::size_t push_back_get_index(value_t<T>&& v) {
        get_tape<value_t<T>>().push_back(std::forward<value_t<T>>(v));
        return get_tape<value_t<T>>().size()-1; // pointer to last element OF T-tape!!
    }

    std::size_t push_back_get_index(op* const& vop) {
        ops.push_back(vop);
        return ops.size()-1; // pointer to last element OF T-tape!!
    }


    void forward() { for (auto&& op : ops) op->evaluate(); }

    void backprop() {
        for (auto rit = ops.rbegin(); rit != ops.rend(); rit++) (*rit)->backprop(); // iterator on pointer, thus **
    }

    virtual ~ValTape() {
        for(auto*& op : ops) delete op;
        // std::cout << "~ValTape" << std::endl;
        // all std::vectors are cleared upon destruction
    }
};


class out_op_general {
    // abstract and
public: 
    virtual void eval_forward_backward() = 0;
}; // AdjointTape needs this!

class AdjointTape : public ValTape {
private:
public:
    // split_op_markers remain across executions for tracking. They are
    // falsified or removed in tape_next_round(), and potentially added to
    // by make_split(), if we're branching beyond previous recordings
    std::vector<marker_t> split_op_markers;
    // this one is reset in next_round() for counting the already encountered
    // branchings which are traversed again (potentially falsified)
    std::size_t num_current_split_ops = 0;

    // this vector tracks all split_op-s. It doesn't own any of them!
    evaluable_boolean_op_t<double>* split_op_product_tracker;
    double get_contribution() { return tend(*split_op_product_tracker); } // TODO: Use this or delete this. Could be used in out_op 


    // map of values to restore primal_result <pointer, ...


    std::vector<out_op_general*> souttype_register;

    std::size_t register_souttype() {
        // the SOutType owns the pointer, so all we have to find a stable place 
        // stable place (given as position on this tape) of the pointer
        souttype_register.push_back(nullptr);
        return souttype_register.size()-1;
    }

    // calculating the souttypes has to happen at the end of a tape-recording!
    void forward_backward_souttypes(){
        // note that the whole souttype_register gets set to nullptr at clear_tape()
        // so the check makes sure that we only backprop the ones which have gotten assignments in this iteration
        for(auto*& s : souttype_register) 
            if (s != nullptr){ 
                s->eval_forward_backward(); 
                delete s; s=nullptr; // out_op-s must be removed to start fresh in next round
            }
    }


    // called in tape_next_round()
    void clear_tape() {

        
        // NOTE: souttype_register goes beyond the tape's existence. Quite ok since its
        // pointers are deallocated at ~SOutType, and relevant entry erased
        // for(auto*& ptr : souttype_register) ptr = nullptr;

        ValTape::clear_valtape(); // clear values created inside function-body and ops

        split_op_product_tracker = nullptr; // non-owning, optape owns

        // next run will bring new split_ops: start at 0 to count previously encountered traversals
        num_current_split_ops = 0; 
    }


    bool tape_next_round() {
        
        clear_tape(); // delete ops and values created inside the re-executing function-lambda
        
        // remove top sequence of markers that are 
        // 1) (eval_complement) evaluated composite ("falsified") markers                
        // 2) (eval_main_only) evaluated main contribution, but composite contribution not relevant flipping 
        int ind = split_op_markers.size()-1;
        while(ind >= 0 && (split_op_markers[ind]==eval_complement || split_op_markers[ind]==eval_main_only) ) {
            split_op_markers.pop_back();
            ind--;
        }
        
        if(ind >= 0) { // any more markers? (equivalent to split_op_markers.size() >= 1)
            // as long as there are markers after removal of top sequence, top must be eval_main.
            split_op_markers[ind] = eval_complement;  // flip last (must have been eval_main)
            // for(int i=0; i<split_op_markers.size(); i++) std::cout << split_op_markers[i] << " ";

            return true;                    // there is a next round!
        } else {

            // for(int i=0; i<split_op_markers.size(); i++) std::cout << split_op_markers[i] << " ";
            return false;                   // all have been removed, so no next round!
        }

   


    }

    // turns a boolean op into a splitOp!
    template<typename T> requires (derived_of_evaluable_value_op_or_pointer_v<T> || is_stype_v<T>)
    auto make_split(T&& x) {

        ++num_current_split_ops; // new op, new marker

        // If marker is for new branch, always start with true (main contribution)! 
        // Else keep decision recorded in relevant entry of split_op_markes
        const bool& push_new_marker = num_current_split_ops > split_op_markers.size();  
        bool marker_i = push_new_marker ? true : to_bool(split_op_markers[num_current_split_ops-1]);

        // evaluate current split_op
        auto* sop_ptr = new split_op<double, T>(std::forward<T>(x), !marker_i); // second argument means "inverted pass?" must first be false
        sop_ptr->evaluate();

        // in case the split op is a new one...
        if(push_new_marker) {
            // 1) push a marker that has it removed after evaluating main-contr. (if tend evaluates to 1 or 0)
            // 2) push a marker that will also evaluate complement-contr eventually (if tend evaluates to (0,1))
            marker_t marker = (sop_ptr->tendency == 1.0 || sop_ptr->tendency == 0.0) ? eval_main_only : eval_main;
            split_op_markers.push_back(marker);
        } 

        // TODO: eventually, we want to throw it directly on the optape without SType 
        SType split_stype(sop_ptr); // with this, pointer is on the optape!


        if(split_op_product_tracker == nullptr) { // first tracked element? push back
            split_op_product_tracker = sop_ptr; 

        } else { 
            // if any splits have been made, we must multiply with the product of all splits
            // thus, note that the split_op_product tracker contains the successively multiplied 
            // products of the splits and the outgoing value needs to be multiplied by them!

            auto* accum_sop_ptr = binary_creator<And, double>::create_clause_op_pointer(
                    split_op_product_tracker, 
                    sop_ptr);


            accum_sop_ptr->evaluate();

            // once pushed back to ops, we're sure it's deleted at the end.
            // it also allows the split_op_product_tracker to be non-owning 
            ops.push_back(accum_sop_ptr); 
            // update product to most recent
            split_op_product_tracker = accum_sop_ptr;
                     
        }

        return std::move(split_stype); // we want to return it as an SType, to show it's already pushed!

    }

    virtual ~AdjointTape() {
        // std::cout << "~AdjointTape" << std::endl;
    }

};


// central tape
class tape {
    static std::weak_ptr<AdjointTape> adtape;
public:
    
    // assign the (weak pointer) central tape , and give back a shared_ptr to it! 
    // Tape is destroyed if the last shared pointer dies
    static std::shared_ptr<AdjointTape> make_smart_tape_ptr() {
        if(!adtape.expired()) { throw std::logic_error("This tape still holds a valid pointer. Deallocate first, or use existing one [get_tape_ptr]"); }
        auto stp = std::shared_ptr<AdjointTape>(new AdjointTape());
        tape::adtape = stp;
        return stp;
    }
    
    static std::shared_ptr<AdjointTape> get_tape_ptr() {
        if(adtape.expired()) { throw std::logic_error("This tape holds no valid pointer. Assign one first [new_smart_tape_ptr()]"); }
        return adtape.lock(); // during this cast, we're counting!
    }

};

// assign expired
std::weak_ptr<AdjointTape> tape::adtape = std::weak_ptr<AdjointTape>();



// TODO: DEPRECATE
AdjointTape valtape;


#define SMOOTHING()\
tape::get_tape_ptr()->function_body = [&](){
#define SMOOTHING_END()\
};


// assumes this is used together with SMOOTHING
#define BACKPROP()\
do {\
    tape::get_tape_ptr()->prepare_valtape();\
    tape::get_tape_ptr()->function_body();\
    tape::get_tape_ptr()->forward_backward_souttypes();\
    tape::get_tape_ptr()->backprop();\
} while (tape::get_tape_ptr()->tape_next_round());

// ================ this is interactive backprop. Whyever we need this... ==========
#define BACKPROP_BEGIN()\
do {\
    tape::get_tape_ptr()->prepare_valtape();\
    tape::get_tape_ptr()->function_body();\
    tape::get_tape_ptr()->forward_backward_souttypes();\
    tape::get_tape_ptr()->backprop();

#define BACKPROP_END()\
} while (tape::get_tape_ptr()->tape_next_round());





#define IF(XCOND)\
if(SType(tape::get_tape_ptr()->make_split(XCOND)).valptr->discrete) 

#define ELSE()\
else






// Pointer to the ValueTape (or AdjointTape), which distinguishes which fetching is done: 
// numeric types reside on vectors, ops reside on vector of pointers
template<typename T>
struct tapes_pointer {
    std::size_t tape_position; // specified, btw, through tape-type!
    tapes_pointer(std::size_t tapes_position) : tape_position(tapes_position) {}
    
    T& operator*() {
        if constexpr(std::is_base_of_v<op, T>) {
            // in this case, T is sume derived class of ValueOp. We need to bring this info back, because tape only knows it as op*
            return *static_cast<T*>(tape::get_tape_ptr()->get_tape<T>().at(tape_position));
            // auto &q = *static_cast<T*>(tape::get_tape_ptr()->get_tape<T>().at(tape_position));
            // return q;
        }
        else { return tape::get_tape_ptr()->get_tape<T>().at(tape_position); }
    }

    T* operator->(){
        // if constexpr(std::is_base_of_v<op, T>) return &(*(tape::get_tape_ptr()->get_tape<T>().at(tape_position)));
        if constexpr(std::is_base_of_v<op, T>) {
            auto q = static_cast<T*>(tape::get_tape_ptr()->get_tape<T>().at(tape_position));
            return q; 
        } 
        else return &(tape::get_tape_ptr()->get_tape<T>().at(tape_position));
    }

};


// ===================== MUST BE SOLVED DIFFERENTLY EVENTUALLY ========================
template<typename T> // TODO: NO IDEA WHETHER THE REQUIREMENTS ARE CORRECT
requires (is_value_t_castable_stype_content_v<T> || is_numeric_stype_content_v<T> || is_numeric_v<T> || derived_of_evaluable_value_op_or_pointer_v<T>)
class out_op : public out_op_general, public unary_op_template<double, T> {
private:
public:

    value_t<double>& _dest; // SOutType's value
    using UT = unary_op_template<double, T>::UT;
    
    // this op associates _arg1 with an SOutType's value_t (defined in operator=())
    out_op(value_t<double>& dest, T&& arg) : unary_op_template<double, T>(std::forward<T>(arg)), _dest(dest) {}

    // this function is called in valtape's forward_backward_souttypes
    void eval_forward_backward() { 

        // 1) forward: increment SOutType's value by factor of accumulated SplitOps
        this->evaluate_potential_member_args();
        
        // if there are no split_ops (because split_op_product_tracker is nullptr), factor is 1
        bool&& splits_present = tape::get_tape_ptr()->split_op_product_tracker != nullptr;
        auto&& cont = splits_present ? tend(*(tape::get_tape_ptr()->split_op_product_tracker)) : 1;

        _dest.value += val(maybe_deref(unary_op_template<double, T>::_arg1)) * cont;

        // 2) backward 
        // 2.1 backprop multiplication  
        if constexpr(!is_numeric_v<UT>) {
            deriv(maybe_deref(unary_op_template<double, T>::_arg1)) = cont * _dest.partial;
        }
        // 2.2 backprop multiplication
        if(splits_present)  
            deriv(*(tape::get_tape_ptr()->split_op_product_tracker)) = val(maybe_deref(unary_op_template<double, T>::_arg1)) * _dest.partial;
        this->backprop_potential_member_args();

    }

};




