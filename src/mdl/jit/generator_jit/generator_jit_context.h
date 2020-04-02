/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#ifndef MDL_GENERATOR_JIT_CONTEXT_H
#define MDL_GENERATOR_JIT_CONTEXT_H 1

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/MDBuilder.h>

#include <mi/mdl/mdl_statements.h>
#include <mi/mdl/mdl_types.h>

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>

#include "generator_jit_type_map.h"

namespace llvm {
class BasicBlock;
class Constant;
class ConstantInt;
class ConstantFP;
class DIBuilder;
class DIFile;
class Function;
class Type;
class Value;
}  // llvm

namespace mi {
namespace mdl {

class IDefinition;
class IParameter;
class IResource_manager;
class ISymbol;
class LLVM_context_data;
class LLVM_code_generator;
class Position;
class Exc_location;
class Function_instance;

///
/// Carries the context of the compilation which stores information needed
/// for compilation like the current basic block, pointers to special parameters etc.
///
class Function_context {
public:
    /// The call mode for tex_lookup functions.
    enum Tex_lookup_call_mode {
        TLCM_VTABLE,   ///< get the tex_lookup() functions from a vtable access
        TLCM_DIRECT,   ///< call tex_lookup() functions directly
        TLCM_OPTIX_CP, ///< call tex_lookup() functions through an OptiX bindless callable program
    };

    /// RAII-like block scope handler.
    class Block_scope {
    public:
        /// Constructor.
        Block_scope(Function_context &ctx, mi::mdl::Position const *pos)
        : m_ctx(ctx)
        {
            ctx.push_block_scope(pos);
        }

        /// Constructor.
        Block_scope(Function_context &ctx, mi::mdl::IStatement const *stmt)
            : m_ctx(ctx)
        {
            ctx.push_block_scope(&stmt->access_position());
        }

        /// Destructor.
        ~Block_scope()
        {
            m_ctx.pop_block_scope();
        }

    private:
        /// The function context.
        Function_context &m_ctx;
    };

    /// RAII-like break destination scope handler.
    class Break_destination {
    public:
        /// Constructor.
        Break_destination(Function_context &ctx, llvm::BasicBlock *dest)
        : m_ctx(ctx)
        {
            ctx.push_break(dest);
        }

        /// Destructor.
        ~Break_destination() {
            m_ctx.pop_break();
        }

    private:
        /// The function context.
        Function_context &m_ctx;
    };

    /// RAII-like continue destination scope handler.
    class Continue_destination {
    public:
        /// Constructor.
        Continue_destination(Function_context &ctx, llvm::BasicBlock *dest)
            : m_ctx(ctx)
        {
            ctx.push_continue(dest);
        }

        /// Destructor.
        ~Continue_destination() {
            m_ctx.pop_continue();
        }

    private:
        /// The function context.
        Function_context &m_ctx;
    };

public:
    /// Constructor, creates a context for a function including its first scope.
    ///
    /// \param alloc           the allocator
    /// \param code_gen        the code generator
    /// \param func_inst       the MDL function instance
    /// \param function        the function which will be build using this context
    /// \param flags           the function flags
    Function_context(
        mi::mdl::IAllocator        *alloc,
        LLVM_code_generator        &code_gen,
        Function_instance const    &func_inst,
        llvm::Function             *function,
        unsigned                   flags);

    /// Constructor, creates a context for modification of an already existing function.
    /// Array instances and the create_return functions are not available in this mode.
    ///
    /// \param alloc                 the allocator
    /// \param code_gen              the code generator
    /// \param function              the function which will be modified using this context
    /// \param flags                 the function flags
    /// \param optimize_on_finalize  specifies, whether the function should get optimized, when
    ///                              the destructor of the context is executed
    Function_context(
        mi::mdl::IAllocator        *alloc,
        LLVM_code_generator        &code_gen,
        llvm::Function             *function,
        unsigned                   flags,
        bool                       optimize_on_finalize);

    /// Destructor, closes the last scope and fills the end block, if the context was not
    /// created for an already existing function.
    ~Function_context();

    /// Get the allocator.
    mi::mdl::IAllocator *get_allocator() const { return m_arena.get_allocator(); }

    /// Get the first (real) parameter of the current function.
    llvm::Function::arg_iterator get_first_parameter();

    /// Returns true if the current function uses an exec_ctx parameter.
    bool has_exec_ctx_parameter() const;

    /// Get the exec_ctx parameter of the current function.
    llvm::Value *get_exec_ctx_parameter();

    /// Get the state parameter of the current function.
    ///
    /// \param exec_ctx  if non-null, the execution context to retrieve the state from
    llvm::Value *get_state_parameter(llvm::Value *exec_ctx=NULL);

    /// Get the resource_data parameter of the current function.
    ///
    /// \param exec_ctx  if non-null, the execution context to retrieve the state from
    llvm::Value *get_resource_data_parameter(llvm::Value *exec_ctx=NULL);

    /// Get the exc_state parameter of the current function.
    ///
    /// \param exec_ctx  if non-null, the execution context to retrieve the state from
    llvm::Value *get_exc_state_parameter(llvm::Value *exec_ctx=NULL);

    /// Get the cap_args parameter of the current function.
    ///
    /// \param exec_ctx  if non-null, the execution context to retrieve the state from
    llvm::Value *get_cap_args_parameter(llvm::Value *exec_ctx=NULL);

    /// Get the lambda_results parameter of the current function.
    ///
    /// \param exec_ctx  if non-null, the execution context to retrieve the state from
    llvm::Value *get_lambda_results_parameter(llvm::Value *exec_ctx=NULL);


    /// Get the object_id value of the current function.
    llvm::Value *get_object_id_value();

    /// Get the wavelength_min() value of the current function.
    llvm::Value *get_wavelength_min_value();

    /// Get the wavelength_max() value of the current function.
    llvm::Value *get_wavelength_max_value();

    /// Get the world-to-object transform (matrix) value of the current function.
    llvm::Value *get_w2o_transform_value();

    /// Get the object-to-world transform (matrix) value of the current function.
    llvm::Value *get_o2w_transform_value();

    /// Get the current break destination.
    llvm::BasicBlock *tos_break();

    /// Get the current continue destination.
    llvm::BasicBlock *tos_continue();

    /// Retrieve the LLVM context data for a MDL variable definition.
    ///
    /// \param var_def    the definition
    LLVM_context_data *get_context_data(
        mi::mdl::IDefinition const *var_def);

    /// Retrieve the LLVM context data for a lambda parameter.
    ///
    /// \param idx  the parameter index
    LLVM_context_data *get_context_data(
        size_t idx);

    /// Create the LLVM context data for a MDL variable/parameter definition.
    ///
    /// \param def           the definition
    /// \param value         the initial value of the variable/parameter
    /// \param by_reference  true if value is the address, false if it is the real value
    LLVM_context_data *create_context_data(
        mi::mdl::IDefinition const *def,
        llvm::Value                *value,
        bool                       by_reference);

    /// Create the LLVM context data for a lambda function parameter.
    ///
    /// \param idx           the parameter index
    /// \param value         the initial value of the variable/parameter
    /// \param by_reference  true if value is the address, false if it is the real value
    LLVM_context_data *create_context_data(
        size_t      idx,
        llvm::Value *value,
        bool        by_reference);

    /// Forward construction from the function context to the builder.
    llvm::IRBuilder<>* operator->() { return &m_ir_builder; }

    /// Creates a new Basic Block with the given name.
    ///
    /// \param name  the name of the block
    llvm::BasicBlock *create_bb(char const *name) {
        return create_bb(m_llvm_context, name, m_function);
    }

    /// Create a new local variable and return its address.
    ///
    /// \param type   the LLVM type of the local
    /// \param name   its name
    llvm::Value *create_local(llvm::Type *type, char const *name) {
        const llvm::DataLayout &DL = m_function->getParent()->getDataLayout();
        return new llvm::AllocaInst(
            type, DL.getAllocaAddrSpace(), nullptr, name, &*m_function->front().begin());
    }

    /// Create a new local array variable and return its address.
    ///
    /// \param type        the LLVM element type of the local
    /// \param array_size  the size of the array
    /// \param name        its name
    llvm::Value *create_local(llvm::Type *type, unsigned array_size, char const *name) {
        const llvm::DataLayout &DL = m_function->getParent()->getDataLayout();
        return new llvm::AllocaInst(
            type,
            DL.getAllocaAddrSpace(),
            get_constant(int(array_size)),
            name,
            &*m_function->front().begin());
    }

    /// Creates a void return at the current block.
    void create_void_return();

    /// Creates a return at the current block.
    ///
    /// \param expr  the return value
    void create_return(llvm::Value *expr);

    /// Creates a Br instruction and set the current block to unreachable.
    ///
    /// \param dest  the destination BB
    void create_jmp(llvm::BasicBlock *dest);

    /// Creates a &ptr[index] operation, index is assured to be in bounds.
    ///
    /// \param ptr    a pointer to an aggregate value
    /// \param index  the index of an aggregate value member
    llvm::Value *create_simple_gep_in_bounds(llvm::Value *ptr, llvm::Value *index);

    /// Creates a &ptr[index] operation, index is assured to be in bounds.
    ///
    /// \param ptr    a pointer to an aggregate value
    /// \param index  the index of an aggregate value member
    llvm::Value *create_simple_gep_in_bounds(llvm::Value *ptr, unsigned index);

    /// Convert a boolean value.
    ///
    /// \param b  the value
    llvm::ConstantInt *get_constant(mi::mdl::IValue_bool const *b);

    /// Convert a boolean value.
    ///
    /// \param b  the value
    llvm::ConstantInt *get_constant(bool b);

    /// Convert an integer value.
    ///
    /// \param i  the value
    llvm::ConstantInt *get_constant(mi::mdl::IValue_int_valued const *i);

    /// Convert an integer value.
    ///
    /// \param i  the value
    llvm::ConstantInt *get_constant(int i);

    /// Convert a size_t value.
    ///
    /// \param i  the value
    llvm::ConstantInt *get_constant(size_t u);

    /// Convert a float value.
    ///
    /// \param f  the value
    llvm::ConstantFP *get_constant(mi::mdl::IValue_float const *f);

    /// Convert a float value.
    ///
    /// \param f  the value
    llvm::ConstantFP *get_constant(float f);

    /// Convert a double value.
    ///
    /// \param d  the value
    llvm::ConstantFP *get_constant(mi::mdl::IValue_double const *d);

    /// Convert a double value.
    ///
    /// \param d  the value
    llvm::ConstantFP *get_constant(double d);

    /// Convert a string value.
    ///
    /// \param s  the value
    llvm::Value *get_constant(mi::mdl::IValue_string const *s);

    /// Convert a string value.
    ///
    /// \param s  the value
    llvm::Value *get_constant(char const *s);

    /// Get a int/FP/vector splat constant.
    ///
    /// \param type  the destination type
    /// \param v     the value
    llvm::Constant *get_constant(llvm::Type *type, int v);

    /// Get a FP/vector splat constant.
    ///
    /// \param type  the destination type
    /// \param v     the value
    llvm::Constant *get_constant(llvm::Type *type, double v);

    /// Convert a resource value.
    ///
    /// \param r  the value
    llvm::ConstantInt *get_constant(mi::mdl::IValue_resource const *r);

    /// Get a constant from an MDL IValue.
    ///
    /// \param v  the value
    llvm::Value *get_constant(mi::mdl::IValue const *v);

    /// Get a vector3 constant.
    ///
    /// \param vtype  a LLVM vector type, must be of at least length 3, longer vectors
    ///               will be filled up with zero
    /// \param x      first component
    /// \param y      second component
    /// \param z      third component
    llvm::Constant *get_constant(llvm::VectorType *vtype, float x, float y, float z);

    /// Get a vector4 constant.
    ///
    /// \param vtype  a LLVM vector type, must be of at least length 3, longer vectors
    ///               will be filled up with zero
    /// \param x      first component
    /// \param y      second component
    /// \param z      third component
    /// \param w      fourth component
    llvm::Constant *get_constant(llvm::VectorType *vtype, float x, float y, float z, float w);

    /// Get the file name of a module.
    ///
    /// \param mod  the MDL module
    llvm::Value *get_module_name(mi::mdl::IModule const *mod);

    /// Get a shuffle mask.
    ///
    /// \param mask  the mask values
    llvm::Constant *get_shuffle(llvm::ArrayRef<int> values);

    /// Creates a single LLVM addition instruction (integer OR FP).
    ///
    /// \param type  the LLVM result type
    /// \param lhs   the left operand
    /// \param rhs   the right operand
    llvm::Value *create_add(llvm::Type *res_type, llvm::Value *lhs, llvm::Value *rhs);

    /// Creates a single LLVM subtraction instruction (integer OR FP).
    ///
    /// \param type  the LLVM result type
    /// \param lhs   the left operand
    /// \param rhs   the right operand
    llvm::Value *create_sub(llvm::Type *res_type, llvm::Value *lhs, llvm::Value *rhs);

    /// Creates a single LLVM multiplication instruction (integer OR FP).
    ///
    /// \param type  the LLVM result type
    /// \param lhs   the left operand
    /// \param rhs   the right operand
    llvm::Value *create_mul(llvm::Type *res_type, llvm::Value *lhs, llvm::Value *rhs);

    /// Creates LLVM code for a 3-element vector by state matrix multiplication.
    /// Note: the state matrices have a different LLVM type than normal matrices.
    ///
    /// \param type                the LLVM result type
    /// \param lhs_V3              the left operand, the 3-element vector
    /// \param rhs_M               the right operand, the state matrix
    /// \param ignore_translation  if true, the translation component of the matrix is ignored
    /// \param transposed          use the transposed matrix for the multiplication.
    ///                            not supported in combination with ignore_translation = false
    llvm::Value *create_mul_state_V3xM(
        llvm::Type *res_type,
        llvm::Value *lhs_V,
        llvm::Value *rhs_M,
        bool ignore_translation,
        bool transposed);

    /// Creates LLVM code for a state matrix by dual 3-element vector multiplication.
    /// Note: the state matrices have a different LLVM type than normal matrices.
    ///
    /// \param type                the LLVM result type
    /// \param lhs_V               the left operand, the dual vector
    /// \param rhs_M               the right operand, the matrix
    /// \param ignore_translation  if true, the translation component of the matrix is ignored
    /// \param transposed          use the transposed matrix for the multiplication.
    ///                            not supported in combination with ignore_translation = false
    llvm::Value *create_deriv_mul_state_V3xM(
        llvm::Type *res_type,
        llvm::Value *lhs_V,
        llvm::Value *rhs_M,
        bool ignore_translation,
        bool transposed);

    /// Creates a single LLVM FP division instruction.
    ///
    /// \param type  the LLVM result type
    /// \param lhs   the left operand
    /// \param rhs   the right operand
    llvm::Value *create_fdiv(llvm::Type *res_type, llvm::Value *lhs, llvm::Value *rhs);

    /// Creates addition instructions for two dual values.
    ///
    /// \param type  the LLVM result type of the value component
    /// \param lhs   the left operand
    /// \param rhs   the right operand
    llvm::Value *create_deriv_add(llvm::Type *res_type, llvm::Value *lhs, llvm::Value *rhs);

    /// Creates subtraction instructions for two dual values.
    ///
    /// \param type  the LLVM result type of the value component
    /// \param lhs   the left operand
    /// \param rhs   the right operand
    llvm::Value *create_deriv_sub(llvm::Type *res_type, llvm::Value *lhs, llvm::Value *rhs);

    /// Creates multiplication instructions for two dual values.
    ///
    /// \param type  the LLVM result type of the value component
    /// \param lhs   the left operand
    /// \param rhs   the right operand
    llvm::Value *create_deriv_mul(llvm::Type *res_type, llvm::Value *lhs, llvm::Value *rhs);

    /// Creates division instructions for two dual values.
    ///
    /// \param type  the LLVM result type of the value component
    /// \param lhs   the left operand
    /// \param rhs   the right operand
    llvm::Value *create_deriv_fdiv(llvm::Type *res_type, llvm::Value *lhs, llvm::Value *rhs);

    /// Creates a cross product on non-dual vectors.
    ///
    /// \param lhs   the left operand
    /// \param rhs   the right operand
    llvm::Value *create_cross(llvm::Value *lhs, llvm::Value *rhs);

    /// Creates a splat vector from a scalar value.
    ///
    /// \param type  the LLVM result type
    /// \param v     the scalar value
    llvm::Value *create_vector_splat(llvm::VectorType *res_type, llvm::Value *v);

    /// Creates a splat value from a scalar value.
    ///
    /// \param type  the LLVM result type
    /// \param v     the scalar value
    llvm::Value *create_splat(llvm::Type *res_type, llvm::Value *v);

    /// Get the number of elements for a vector or an array.
    ///
    /// \param val   the value
    unsigned get_num_elements(llvm::Value *val);

    /// Creates a ExtractValue or ExtractElement instruction.
    ///
    /// \param comp_val  the compound value to extract from
    /// \param index     the index at which a subcomponent should be extracted
    llvm::Value *create_extract(llvm::Value *val, unsigned index);

    /// Extracts a value from a compound value, also supporting derivative values.
    ///
    /// \param comp_val  the compound value to extract from
    /// \param index     the index at which a subcomponent should be extracted
    llvm::Value *create_extract_allow_deriv(llvm::Value *val, unsigned index);

    /// Creates an InsertValue or InsertElement instruction.
    ///
    /// \param agg_val   the compound aggregate value
    /// \param val       the value to insert
    /// \param index     the index at which the should be inserted
    llvm::Value *create_insert(llvm::Value *agg_value, llvm::Value *val, unsigned index);

    /// Creates a bounds check compare.
    ///
    /// \param index        the index value (int in MDL)
    /// \param bound        the bound value (size_t)
    /// \param ok_bb        the basic block to branch to if the index is in bounds
    /// \param fail_bb      the basic block to branch to if the index is out of bounds
    void create_bounds_check_cmp(
        llvm::Value        *index,
        llvm::Value        *bound,
        llvm::BasicBlock   *ok_bb,
        llvm::BasicBlock   *fail_bb);

    /// Creates a bounds check using an exception.
    ///
    /// \param index        the index value (int in MDL)
    /// \param bound        the bound value (size_t)
    /// \param loc          the location of the index expression
    void create_bounds_check_with_exception(
        llvm::Value        *index,
        llvm::Value        *bound,
        Exc_location const &loc);

    /// Selects the given value if the index is smaller than the bound value.
    /// Otherwise selects the out-of-bounds value.
    ///
    /// \param index              the index value (int in MDL)
    /// \param bound              the bound value (size_t)
    /// \param val                the value to select, if the index is within the bounds
    /// \param out_of_bounds_val  the value to select, if the index is out-of-bounds
    ///
    /// \returns the selected value
    llvm::Value *create_select_if_in_bounds(
        llvm::Value        *index,
        llvm::Value        *bound,
        llvm::Value        *val,
        llvm::Value        *out_of_bounds_val);

    /// Creates a non-zero check compare.
    ///
    /// \param v            the value to check (int in MDL)
    /// \param non_zero_bb  the basic block to branch to if the value is non-zero
    /// \param zero_bb      the basic block to branch to if the value is zero
    void create_non_zero_check_cmp(
        llvm::Value        *v,
        llvm::BasicBlock   *non_zero_bb,
        llvm::BasicBlock   *zero_bb);

    /// Creates a div-by-zero check using an exception.
    ///
    /// \param v            the right hand side division value
    /// \param loc          the location of the index expression
    void create_div_check_with_exception(
        llvm::Value        *v,
        Exc_location const &loc);

    /// Create a weighted conditional branch.
    ///
    /// \param cond          the condition
    /// \param true_bb       the branch target if cond is true
    /// \param false_bb      the branch target if cond is false
    /// \param true_weight   the weight for the true branch
    /// \param false_weight  the weight for the false branch
    llvm::BranchInst *CreateWeightedCondBr(
        llvm::Value      *cond,
        llvm::BasicBlock *true_bb,
        llvm::BasicBlock *false_bb,
        uint32_t         true_weight,
        uint32_t         false_weight);

    /// Set the current source position.
    ///
    /// \param pos  the position
    void set_curr_pos(mi::mdl::Position const *pos) {
        if (pos != NULL)
            set_curr_pos(*pos);
    }

    /// Set the current source position.
    ///
    /// \param pos  the position
    void set_curr_pos(mi::mdl::Position const &pos);

    /// Get the current source position.
    mi::mdl::Position const *get_curr_pos() const { return m_curr_pos; }

    /// Add debug info for a variable declaration.
    ///
    /// \param var_def  the definition of the variable
    void add_debug_info(mi::mdl::IDefinition const *var_def);

    /// Make the given function parameter accessible.
    ///
    /// \param param  the function parameter
    void make_accessible(mi::mdl::IParameter const *param);

    /// Find a parameter for a given array size.
    ///
    /// \param sym  the array size
    mi::mdl::IDefinition const *find_parameter_for_size(
        mi::mdl::ISymbol const *sym) const;

    /// Get the "unreachable" block of the current function, create one if necessary.
    llvm::BasicBlock *get_unreachable_bb();

    /// Get the current function.
    llvm::Function *get_function() { return m_function; }

    void move_to_body_start() {
        m_ir_builder.SetInsertPoint(m_body_start_point);
    }

    /// Register a resource value and return its index.
    ///
    /// \param resource  the resource value
    size_t get_resource_index(mi::mdl::IValue_resource const *resource);

    /// Get the array base address of an deferred-sized array.
    ///
    /// \param arr_desc  the array descriptor
    llvm::Value *get_deferred_base(llvm::Value *arr_desc);

    /// Get the array size of an deferred-sized array (type size_t).
    ///
    /// \param arr_desc  the array descriptor
    llvm::Value *get_deferred_size(llvm::Value *arr_desc);

    /// Get the array base address of an deferred-sized array.
    ///
    /// \param arr_desc_ptr  a pointer to the array descriptor
    llvm::Value *get_deferred_base_from_ptr(llvm::Value *arr_desc_ptr);

    /// Get the array size of an deferred-sized array (type size_t).
    ///
    /// \param arr_desc_ptr  a pointer to the array descriptor
    llvm::Value *get_deferred_size_from_ptr(llvm::Value *arr_desc_ptr);

    /// Set the array base address of an deferred-sized array.
    ///
    /// \param arr_desc_ptr  a pointer to the array descriptor
    /// \param arr           the array pointer
    void set_deferred_base(llvm::Value *arr_desc_ptr, llvm::Value *arr);

    /// Set the array size of an deferred-sized array (type size_t).
    ///
    /// \param arr_desc_ptr  a pointer to the array descriptor
    /// \param size          the array size
    void set_deferred_size(llvm::Value *arr_desc_ptr, llvm::Value *size);

    /// Returns true, if the given LLVM type is a derivative type.
    ///
    /// \param type  the type to check
    bool is_deriv_type(llvm::Type *type);

    /// Get the base value LLVM type of a derivative LLVM type.
    ///
    /// \param type  the LLVM type
    ///
    /// \returns the base value type of the derivative type or NULL if it is not a derivative type
    llvm::Type *get_deriv_base_type(llvm::Type *type);

    /// Get a dual value.
    ///
    /// \param val  the value component for the dual value
    /// \param dx   the dx component for the dual value
    /// \param dy   the dy component for the dual value
    llvm::Value *get_dual(llvm::Value *val, llvm::Value *dx, llvm::Value *dy);

    /// Get a dual value with dx and dy set to zero.
    ///
    /// \param val  the value component for the dual value
    llvm::Value *get_dual(llvm::Value *val);

    /// Get a component of the dual value.
    ///
    /// \param dual        the dual value
    /// \param comp_index  the index of the component: 0 = value, 1 = dx, 2 = dy
    llvm::Value *get_dual_comp(llvm::Value *dual, unsigned int comp_index);

    /// Get the value of the dual value, or the value itself for non-dual values.
    ///
    /// \param dual  the dual value
    llvm::Value *get_dual_val(llvm::Value *dual);

    /// Get the pointer to the value component of a dual value.
    ///
    /// \param dual_ptr  the pointer to the dual value
    llvm::Value *get_dual_val_ptr(llvm::Value *dual_ptr);

    /// Get the dx component of the dual value, or zero for non-dual values.
    ///
    /// \param dual  the dual value
    llvm::Value *get_dual_dx(llvm::Value *dual);

    /// Get the dy component of the dual value, or zero for non-dual values.
    ///
    /// \param dual  the dual value
    llvm::Value *get_dual_dy(llvm::Value *dual);

    /// Extract a dual component from a dual compound value.
    ///
    /// \param compound_val  the compound value
    /// \param index         the index of the component to extract
    llvm::Value *extract_dual(llvm::Value *compound_val, unsigned int index);

    /// Get a pointer type from a base type.
    ///
    /// \param type  the base type
    llvm::PointerType *get_ptr(llvm::Type *type);

    /// Load a value and convert the representation.
    ///
    /// \param dst_type  the LLVM destination type
    /// \param ptr       a pointer of a desired value
    ///
    /// Loads a value from a pointer and convert its representation
    /// to a "compatible one", i.e. the MDL type of both values must be
    /// the same. Allows to convert from array representation to vector
    /// representation.
    llvm::Value *load_and_convert(llvm::Type *dst_type, llvm::Value *ptr);

    /// Convert a value and store it into a pointer location.
    ///
    /// \param value   the LLVM value to store
    /// \param ptr     a pointer to the desired location
    ///
    /// Convert the representation of a value to a "compatible one",
    /// i.e. the MDL type of the value and the location must be
    /// the same. Allows to convert from array representation to vector
    /// representation.
    llvm::StoreInst *convert_and_store(llvm::Value *value, llvm::Value *ptr);

    /// Store a int2(0,0) into a pointer location.
    ///
    /// \param ptr     a pointer to the desired location
    llvm::StoreInst *store_int2_zero(llvm::Value *ptr);

    /// Get the real return type of the function (i.e. does NOT return void for sret functions).
    llvm::Type *get_return_type() const;

    /// Get the real non-derivative return type of the function (i.e. does NOT return void for sret
    /// functions, for derivative return types returns the type of the value component).
    llvm::Type *get_non_deriv_return_type() const;

    /// Override a (possibly not existing) lambda_results parameter.
    void override_lambda_results(llvm::Value *lambda_results)
    {
        m_lambda_results_override = lambda_results;
    }

    /// Map type sizes due to function instancing.
    ///
    /// \param type  an MDL type
    ///
    /// \return if type is a deferred size array type and instancing is enabled, the immediate
    ///         size instance of this type, else -1
    int instantiate_type_size(
        mi::mdl::IType const *type) const MDL_WARN_UNUSED_RESULT;

    /// Get the tex_lookup function address for a given vtable index.
    ///
    /// \param self   the value representing the this pointer for retrieving the vtable
    /// \param index  the vtable index.
    llvm::Value *get_tex_lookup_func(
        llvm::Value                                    *self,
        mi::mdl::Type_mapper::Tex_handler_vtable_index index);

    /// Get the current line (computed from current position debug info)
    int get_dbg_curr_line();

    /// Get the current file name (computed from current position debug info)
    char const *get_dbg_curr_filename();

    /// Get the current function name.
    char const *get_dbg_curr_function();

    /// Returns true, if the value \p val is a constant with a value equal to \p int_val.
    ///
    /// \param val      the LLVM value to check
    /// \param int_val  the integer value to compare to
    bool is_constant_value(llvm::Value *val, int int_val);

    /// Get the code generator.
    LLVM_code_generator &get_code_gen() { return m_code_gen; }

private:
    /// Pushes a break destination on the break stack.
    void push_break(llvm::BasicBlock *dest);

    /// Pop a break destination from the break stack.
    void pop_break();

    /// Pushes a continue destination on the continue stack.
    void push_continue(llvm::BasicBlock *dest);

    /// Pop a continue destination from the continue stack.
    void pop_continue();

    /// Creates a new block scope and push it.
    ///
    /// \param pos  the position of the block if any
    void push_block_scope(mi::mdl::Position const *pos);

    /// Pop a variable scope, creating running if necessary.
    void pop_block_scope();

    /// Get the current LLVM debug info scope.
    llvm::DIScope *get_debug_info_scope();

    /// Creates a new Basic Block with the given name inside the given function.
    ///
    /// \param llvm_context  the LLVM context to use
    /// \param name          the name of the block
    /// \param func          the owner function of the block
    static llvm::BasicBlock *create_bb(
        llvm::LLVMContext &llvm_context,
        const char        *name,
        llvm::Function    *func)
    {
        return llvm::BasicBlock::Create(llvm_context, name, func);
    }

    /// Creates an out-of-bounds exception call
    ///
    /// \param index        the index value (int in MDL)
    /// \param bound        the bound value (size_t)
    /// \param loc          the location of the index expression
    void create_oob_exception_call(
        llvm::Value        *index,
        llvm::Value        *bound,
        Exc_location const &loc);

    /// Creates a div-by-zero check using an exception.
    ///
    /// \param loc          the location of the index expression
    void create_div_exception_call(
        Exc_location const &loc);

private:
    // Not to be implemented
    Function_context(Function_context const &) MDL_DELETED_FUNCTION;
    Function_context &operator=(Function_context const &) MDL_DELETED_FUNCTION;

private:
    typedef mi::mdl::ptr_hash_map<
        void const,
        LLVM_context_data *>::Type Var_context_data_map;

    typedef ptr_hash_map<IType_array_size const, int>::Type Array_size_map;

    /// The memory arena.
    mi::mdl::Memory_arena m_arena;

    /// The builder for memory arena objects.
    mi::mdl::Arena_builder m_arena_builder;

    /// The map storing context data for variable definitions.
    Var_context_data_map m_var_context_data;

    /// The map for instantiated array sizes.
    Array_size_map m_array_size_map;

    /// The type mapper.
    Type_mapper &m_type_mapper;

    /// The used LLVM thread context.
    llvm::LLVMContext &m_llvm_context;

    /// Used to add instructions, stores current insertion point (basic block).
    llvm::IRBuilder<> m_ir_builder;

    /// Used to create MetaData nodes.
    llvm::MDBuilder m_md_builder;

    /// The debug info builder if any.
    llvm::DIBuilder *m_di_builder;

    /// The debug info for the current file if any.
    llvm::DIFile *m_di_file;

    /// The current function.
    llvm::Function *m_function;

    /// The Start block of the function.
    llvm::BasicBlock *m_start_bb;

    /// The End block of the function. Only available when context is not in modification mode.
    llvm::BasicBlock *m_end_bb;

    /// The unreachable block of the function.
    llvm::BasicBlock *m_unreachable_bb;

    /// First instruction after all Alloca instructions in the start block.
    /// Any non-Alloca instructions may be inserted here.
    llvm::Instruction *m_body_start_point;

    /// The address of a store taking the return value.
    llvm::Value *m_retval_adr;

    /// A value to override a (possibly not existing) lambda_results parameter.
    llvm::Value *m_lambda_results_override;

    /// The current source position.
    mi::mdl::Position const *m_curr_pos;

    /// If non-NULL, the resource manager to handle resource values.
    IResource_manager *m_res_manager;

    /// The LLVM code generator that owns this context.
    LLVM_code_generator &m_code_gen;

    /// The function flags, a bitset of LLVM_context_data::Flags
    unsigned m_flags;

    /// If true, the function gets optimized when the destructor is called.
    bool m_optimize_on_finalize;

    /// If true, generate full debug info.
    bool m_full_debug_info;

    typedef stack<llvm::BasicBlock *>::Type BB_stack;

    /// The break stack.
    BB_stack m_break_stack;

    /// The continue stack.
    BB_stack m_continue_stack;

    typedef stack<llvm::DILexicalBlock *>::Type DILB_stack;

    /// The lexical block for the current function.
    llvm::DISubprogram *m_di_function;

    /// The stack for debug info lexical blocks.
    DILB_stack m_dilb_stack;

    typedef mi::mdl::vector<mi::mdl::IDefinition const *>::Type Definition_vector;

    /// Helper: Accessible function parameters when translating an expression.
    Definition_vector m_accesible_parameters;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_CONTEXT_H

