/******************************************************************************
 * Copyright (c) 2013-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_LLVM_H
#define MDL_GENERATOR_JIT_LLVM_H 1

#include <csetjmp>

#include <mi/base/atom.h>
#include <mi/base/iinterface.h>
#include <mi/base/lock.h>

#include <llvm/ADT/OwningPtr.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/Compiler.h>
#include <llvm/DebugInfo.h>

#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mdl/compiler/compilercore/compilercore_bitset.h>
#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include <mdl/compiler/compilercore/compilercore_function_instance.h>

#include "generator_jit_type_map.h"
#include "generator_jit_context.h"

namespace llvm {
    class Argument;
    class DIBuilder;
    class DIFile;
    class ExecutionEngine;
    class Function;
    class Module;
}  // llvm

namespace mi {

namespace mdl {

class Df_component_info;
class DAG_call;
class DAG_node;
class Derivative_infos;
class Func_deriv_info;
class Function_context;
class Function_instance;
class ICall_name_resolver;
class IDag_builder;
class ILambda_call_transformer;
class ILambda_resource_attribute;
class IMDL_exception_handler;
class IModule;
class IResource_manager;
class IStatement;
class IStatement_compound;
class Lambda_function;
class Distribution_function;
class MDL;
class MDL_runtime_creator;

/// The Jitted code interface holds jitted code.
class IJitted_code : public
    mi::base::Interface_declare<0x933809eb,0x0449,0x4c29,0xaf,0x04,0xc4,0x90,0x1e,0xaa,0xd3,0x3e>
{
};

///
/// The Jitted code object holds jitted code.
///
class Jitted_code : public Allocator_interface_implement<IJitted_code>
{
    typedef Allocator_interface_implement<IJitted_code> Base;
    friend class Allocator_builder;
public:
    /// Retrieve the global LLVM context.
    llvm::LLVMContext *get_llvm_context() { return m_llvm_context; }

    /// Add this LLVM module to the execution engine.
    ///
    /// \param llvm_module  the LLVM module, takes ownership
    void add_llvm_module(llvm::Module *llvm_module);

    /// Remove this module from the execution engine and delete it.
    ///
    /// \param llvm_module  the LLVM module
    void delete_llvm_module(llvm::Module *llvm_module);

    /// JIT compile the given LLVM function.
    ///
    /// \param func  the LLVM function to compile
    ///
    /// \return The address of the jitted code of the function.
    ///
    /// \note: the module of this function must be added in advance
    void *jit_compile(llvm::Function *func);

    /// Get the only instance.
    ///
    /// \param alloc    the allocator
    static Jitted_code *get_instance(mi::mdl::IAllocator *alloc);

    /// Registers a native function for symbol resolving by the JIT.
    ///
    /// \param func_name  The mangled name as used for symbol resolving.
    /// \param address    The address of the native implementation of the function.
    void register_function(llvm::StringRef const &func_name, void *address);

    /// Get the layout data for the current JITer target.
    llvm::DataLayout const *get_layout_data() const;

private:
    /// Constructor.
    ///
    /// \param alloc    the allocator
    explicit Jitted_code(mi::mdl::IAllocator *alloc);

    /// Destructor.
    ///
    /// Terminates all jitted code modules.
    ~Jitted_code();

    /// One time LLVM initialization.
    static void init_llvm();

private:
    // NOT implemented
    Jitted_code(Jitted_code const &) LLVM_DELETED_FUNCTION;
    Jitted_code &operator=(Jitted_code const &) LLVM_DELETED_FUNCTION;

private:
    /// The singleton.
    static Jitted_code *m_instance;

    /// The singleton lock.
    static mi::base::Lock m_singleton_lock;

    /// Set for the very first time of the singleton creation.
    static bool m_first_time_init;

    /// The LLVMContext to use for the code generator.
    llvm::LLVMContext *m_llvm_context;

    /// The global ExecutionEngine.
    llvm::ExecutionEngine *m_execution_engine;
};

///
/// Class providing information about an internal function.
///
class Internal_function
{
public:
    // Same as LLVM_context_data
    enum Flag {
        FL_NONE           = 0,        ///< No flags.
        FL_SRET           = 1 << 0,   ///< Uses struct return.
        FL_HAS_STATE      = 1 << 1,   ///< Has state parameter.
        FL_HAS_RES        = 1 << 2,   ///< Has resource_data parameter.
        FL_HAS_EXC        = 1 << 3,   ///< Has exc_state parameter.
        FL_HAS_CAP_ARGS   = 1 << 4,   ///< Has captured arguments parameter.
        FL_HAS_LMBD_RES   = 1 << 5,   ///< Has lambda_results parameter (libbsdf).
        FL_HAS_OBJ_ID     = 1 << 6,   ///< Has object_id parameter.
        FL_HAS_TRANSFORMS = 1 << 7,   ///< Has transform (matrix) parameters.
        FL_UNALIGNED_RET  = 1 << 8,   ///< The address for the struct return is might be unaligned.
        FL_HAS_EXEC_CTX   = 1 << 9,   ///< The state, resource_data, exc_state, captured_arguments
                                      ///< and lambda_results parameter is provided in an execution
                                      ///< context parameter. The other flags should also be set
                                      ///< to indicate, that the information is available.
    };  // can be or'ed

    typedef unsigned Flags;

    enum Kind {
        KI_STATE_SET_NORMAL,                    ///< Kind of state::set_normal(float3)
        KI_STATE_GET_TEXTURE_RESULTS,           ///< Kind of state::get_texture_results()
        KI_STATE_GET_ARG_BLOCK,                 ///< Kind of state::get_arg_block()
        KI_STATE_GET_RO_DATA_SEGMENT,           ///< Kind of state::get_ro_data_segment()
        KI_STATE_OBJECT_ID,                     ///< Kind of state::object_id()
        KI_STATE_CALL_LAMBDA_FLOAT,             ///< Kind of state::call_lambda_float(int)
        KI_STATE_CALL_LAMBDA_FLOAT3,            ///< Kind of state::call_lambda_float3(int)
        
        /// Kind of df::bsdf_measurement_resolution(int,int)
        KI_DF_BSDF_MEASUREMENT_RESOLUTION,      

        /// Kind of df::bsdf_measurement_evaluate(int,float2,float2,int)
        KI_DF_BSDF_MEASUREMENT_EVALUATE,

        /// Kind of df::bsdf_measurement_sample(int,float2,float3,int)
        KI_DF_BSDF_MEASUREMENT_SAMPLE,

        /// Kind of df::bsdf_measurement_pdf(int,float2,float2,int)
        KI_DF_BSDF_MEASUREMENT_PDF,

        /// Kind of df::bsdf_measurement_albedos(int,float2)
        KI_DF_BSDF_MEASUREMENT_ALBEDOS,

        /// Kind of df::_light_profile_evaluate(int,float2)
        KI_DF_LIGHT_PROFILE_EVALUATE,

        /// Kind of df::_light_profile_sample(int,float3)
        KI_DF_LIGHT_PROFILE_SAMPLE,

        /// Kind of df::_light_profile_pdf(int,float2)
        KI_DF_LIGHT_PROFILE_PDF,

        KI_NUM_INTERNAL_FUNCTIONS
    };

    /// Construct an internal function.
    ///
    /// \param arena          the arena
    /// \param name           the name of the internal function
    /// \param mangled_name   the mangled name of the internal function (because the mangler would
    ///                       need an IDefinition)
    /// \param kind           the kind of the internal function
    /// \param flags          the LLVM_context_data flags
    /// \param ret_type       the return type of the internal function
    /// \param param_types    the parameter types of the internal function
    /// \param param_names    the parameter names of the internal function
    Internal_function(
        Memory_arena *arena,
        char const *name,
        char const *mangled_name,
        Kind kind,
        Flags flags,
        llvm::Type *ret_type,
        Array_ref<IType const *> const &param_types,
        Array_ref<char const *> const &param_names);

    /// Get the name of the internal function.
    char const *get_name() const { return m_name; }

    /// Get the mangled name of the internal function.
    char const *get_mangled_name() const { return m_mangled_name; }

    /// Get the kind of the internal function.
    Kind get_kind() const { return m_kind; }

    /// Get the LLVM_context_data flags of the internal function.
    Flags get_flags() const { return m_flags; }

    /// Get the return type of the internal function.
    llvm::Type *get_return_type() const { return m_ret_type; }

    /// Get the number of parameters.
    size_t get_parameter_number() const;

    /// Get the parameter type of the internal function at the given index.
    IType const *get_parameter_type(size_t index) const;

    /// Get the parameter names of the internal function.
    char const *get_parameter_name(size_t index) const;


private:
    /// The name of the internal function.
    char const *m_name;

    /// The mangled name of the internal function.
    char const *m_mangled_name;

    /// The kind of the internal function.
    Kind m_kind;

    /// The LLVM_context_data flags of the internal function.
    Flags m_flags;

    /// The return type of the internal function.
    llvm::Type *m_ret_type;

    /// The parameter types of the internal function.
    Arena_VLA<IType const *> m_param_types;

    /// The parameter names of the internal function.
    Arena_VLA<char const *> m_param_names;
};

/// Helper class to handle exception locations.
class Exc_location
{
public:
    /// Constructor.
    ///
    /// \param code_gen  the LLVM code generator
    /// \param pos       the MDL position of the exceptional expression if any
    Exc_location(
        LLVM_code_generator const &code_gen,
        mi::mdl::Position const   *pos);

    /// Get the module.
    mi::mdl::IModule const *get_module() const { return m_mod; }

    /// Get the line.
    unsigned get_line() const { return m_line; }

private:
    /// The module of the exception occurrence.
    mi::mdl::IModule const *m_mod;

    /// The line of the exception occurrence.
    unsigned m_line;
};

///
/// LLVM backend context data.
///
/// Note: this is a thin wrapper around a union. Object of this type are allocated
/// on a memory arena, hence to destructor.
///
class LLVM_context_data
{
public:
    enum Flag {
        FL_NONE           = 0,        ///< No flags.
        FL_SRET           = 1 << 0,   ///< Uses struct return.
        FL_HAS_STATE      = 1 << 1,   ///< Has state parameter.
        FL_HAS_RES        = 1 << 2,   ///< Has resource_data parameter.
        FL_HAS_EXC        = 1 << 3,   ///< Has exc_state parameter.
        FL_HAS_CAP_ARGS   = 1 << 4,   ///< Has captured arguments parameter.
        FL_HAS_LMBD_RES   = 1 << 5,   ///< Has lambda_results parameter (libbsdf).
        FL_HAS_OBJ_ID     = 1 << 6,   ///< Has object_id parameter.
        FL_HAS_TRANSFORMS = 1 << 7,   ///< Has transform (matrix) parameters
        FL_UNALIGNED_RET  = 1 << 8,   ///< The address for the struct return is might be unaligned.
        FL_HAS_EXEC_CTX   = 1 << 9,   ///< The state, resource_data, exc_state, captured_arguments
                                      ///< and lambda_results parameter is provided in an execution
                                      ///< context parameter. The other flags should also be set
                                      ///< to indicate, that the information is available.
    };  // can be or'ed

    typedef unsigned Flags;

public:
    /// Constructor for a function.
    ///
    /// \param func      the LLVM function object
    /// \param ret_type  the real return type of this function
    /// \param flags     flags bitset
    LLVM_context_data(
        llvm::Function *func,
        llvm::Type     *ret_type,
        Flags          flags)
    {
        u.f.m_function = func;
        u.f.m_ret_type = ret_type;
        u.f.m_flags    = flags;
    }

    /// Constructor for a variable that needs an address (lvalue).
    ///
    /// \param ctx           the function context of this variable life time
    /// \param var_type      the type of the variable
    /// \param var_name      the name of the variable
    LLVM_context_data(
        Function_context *ctx,
        llvm::Type       *var_type,
        char const       *var_name)
    {
        u.v.m_ctx       = ctx;
        u.v.m_var_type  = var_type;
        u.v.m_var_adr   = ctx->create_local(var_type, var_name);
        u.v.m_var_value = NULL;
    }

    /// Constructor for a variable/parameter.
    ///
    /// \param ctx           the function context of this variable life time
    /// \param value         the initial value of the variable
    /// \param name          the name of the variable/parameter
    /// \param by_reference  if true, value is the address of the variable, else its value
    LLVM_context_data(
        Function_context *ctx,
        llvm::Value      *value,
        char const       *name,
        bool             by_reference)
    {
        u.v.m_ctx      = ctx;
        u.v.m_var_type = value->getType();
        if (by_reference) {
            u.v.m_var_adr   = value;
            u.v.m_var_value = NULL;
        } else {
            u.v.m_var_adr   = NULL;
            u.v.m_var_value = value;
        }
    }

    /// Get the LLVM function object.
    llvm::Function *get_function() const { return u.f.m_function; }

    /// Get the return type of the function.
    llvm::Type *get_return_type() const { return u.f.m_ret_type; }

    /// Returns true if the current function uses struct return.
    bool is_sret_return() const { return (u.f.m_flags & FL_SRET) != 0; }

    /// Returns true if the current function has an execution context parameter
    /// containing the state, resource_data, exc_state, lambda_results and captured arguments.
    bool has_exec_ctx_param() const { return (u.f.m_flags & FL_HAS_EXEC_CTX) != 0; }

    /// Returns true if the current function has a state parameter.
    bool has_state_param() const { return (u.f.m_flags & FL_HAS_STATE) != 0; }

    /// Returns true if the current function has a resource_data parameter.
    bool has_resource_data_param() const { return (u.f.m_flags & FL_HAS_RES) != 0; }

    /// Returns true if the current function has a exc_state parameter.
    bool has_exc_state_param() const { return (u.f.m_flags & FL_HAS_EXC) != 0; }

    /// Returns true if the current function has a captured_arguments parameter.
    bool has_captured_args_param() const { return (u.f.m_flags & FL_HAS_CAP_ARGS) != 0; }

    /// Returns true if the current function has a lambda_results parameter.
    bool has_lambda_results_param() const { return (u.f.m_flags & FL_HAS_LMBD_RES) != 0; }

    /// Returns true if the current function has an object_id parameter.
    bool has_object_id_param() const { return (u.f.m_flags & FL_HAS_OBJ_ID) != 0; }

    /// Returns true if the current function has two transform (matrix) parameters.
    bool has_transform_params() const { return (u.f.m_flags & FL_HAS_TRANSFORMS) != 0; }

    /// Get the function flags.
    Flags get_function_flags() const { return u.f.m_flags; }

    /// Get the address of a variable.
    llvm::Value *get_var_address() const { return u.v.m_var_adr; }

    /// Get the type of a variable.
    llvm::Type *get_var_type() const { return u.v.m_var_type; }

    /// Get the value of a variable.
    llvm::Value *get_var_value() const {
        if (u.v.m_var_value != NULL)
            return u.v.m_var_value;

        Function_context &ctx = *u.v.m_ctx;
        llvm::Value *l = ctx->CreateLoad(u.v.m_var_adr);
        return l;
    }

    /// Set the value of a variable.
    ///
    /// \param data  the new value of the variable
    void set_var_value(llvm::Value *data) const {
        (*u.v.m_ctx)->CreateStore(data, u.v.m_var_adr);
    }

    /// Get the offset to the first true function parameter.
    size_t get_func_param_offset() const {
        size_t ofs = 0;
        if (u.f.m_flags & FL_SRET)           ++ofs;
        if (u.f.m_flags & FL_HAS_EXEC_CTX) ++ofs;
        else {
            if (u.f.m_flags & FL_HAS_STATE)      ++ofs;
            if (u.f.m_flags & FL_HAS_RES)        ++ofs;
            if (u.f.m_flags & FL_HAS_EXC)        ++ofs;
            if (u.f.m_flags & FL_HAS_CAP_ARGS)   ++ofs;
            if (u.f.m_flags & FL_HAS_LMBD_RES)   ++ofs;
        }
        if (u.f.m_flags & FL_HAS_OBJ_ID)     ++ofs;
        if (u.f.m_flags & FL_HAS_TRANSFORMS) ofs += 2;
        return ofs;
    }

private:
    union {
        struct {
            /// The LLVM function object.
            llvm::Function *m_function;

            /// The real LLVM return type of the function.
            llvm::Type *m_ret_type;

            /// Flags bitset.
            Flags m_flags;
        } f;
        struct {
            /// The function context of this variable life time.
            Function_context *m_ctx;

            /// The LLVM type of the variable.
            llvm::Type *m_var_type;

            /// The address of a variable if exist.
            llvm::Value *m_var_adr;

            /// The value of a variable if no address exists so far.
            llvm::Value *m_var_value;
        } v;
    } u;
};

/// Helper class to represent an expression result that can be handled like a value
/// OR like a memory address containing the value.
class Expression_result {
public:
    /// Construct an Expression result from a pointer (reference).
    static Expression_result ptr(llvm::Value *ptr) {
        return Expression_result(ptr, /*is_value=*/false);
    }

    /// Construct an Expression result from a value.
    static Expression_result value(llvm::Value *value) {
        return Expression_result(value, /*is_value=*/true);
    }

    /// Create an undefined result of the given type.
    static Expression_result undef(llvm::Type *type) {
        return Expression_result::value(llvm::UndefValue::get(type));
    }

    /// An unset result, do NOT use it.
    static Expression_result unset() {
        return Expression_result();
    }

    /// Returns a pointer to the value. Puts it in a newly allocated temporary if necessary.
    llvm::Value *as_ptr(Function_context &context) {
        if (m_is_value) {
            // do not add debug info here, it is not clear, when this is executed
            llvm::Value *ptr = context.create_local(m_content->getType(), "tmp");
            context->CreateStore(m_content, ptr);
            return ptr;
        } else {
            return m_content;
        }
    }

    /// Return the value.
    llvm::Value *as_value(Function_context &context) {
        if (m_is_value) {
            return m_content;
        } else {
            // do not add debug info here, it is not clear, when this is executed
            return context->CreateLoad(m_content);
        }
    }

    /// Ensures that the expression result is a derivative value or not, by stripping or zero
    /// extending.
    ///
    /// \param context                the function context
    /// \param should_be_deriv_value  if true, the expression result value will be zero extended
    ///                               to a derivative value, if necessary,
    ///                               if false, any derivative information will be stripped
    void ensure_deriv_result(Function_context &context, bool should_be_deriv_value)
    {
        if (should_be_deriv_value) {
            if (!is_deriv_value(context)) {
                llvm::Value *val = as_value(context);
                m_is_value = true;
                m_content = context.get_dual(val);
            }
        } else {
            if (m_is_value) {
                m_content = context.get_dual_val(m_content);
            } else {
                if (context.is_deriv_type(m_content->getType()->getPointerElementType())) {
                    // change pointer to dual into pointer to value component of dual
                    m_content = context.get_dual_val_ptr(m_content);
                }
            }
        }
    }

    /// Returns true, if the expression result is a derivative value or a pointer to such value.
    bool is_deriv_value(Function_context &context) {
        if (m_is_value)
            return context.is_deriv_type(m_content->getType());
        else
            return context.is_deriv_type(m_content->getType()->getPointerElementType());
    }

    /// Returns true if this expression result is a constant.
    bool is_constant() const {
        if (m_is_value) {
            // a constant
            return llvm::isa<llvm::Constant>(m_content);
        } else if (llvm::GlobalVariable *gv = llvm::dyn_cast<llvm::GlobalVariable>(m_content)) {
            // The address of a constant
            return gv->isConstant();
        }
        return false;
    }

    /// Return true if it is unset.
    bool is_unset() const { return m_content == NULL; }

    /// Return true if this expression result represents a value.
    bool is_value() const { return m_is_value; }

    /// Return the type of the value.
    llvm::Type *get_value_type() const {
        if (m_is_value)
            return m_content->getType();
        else
            return m_content->getType()->getPointerElementType();
    }

private:
    /// Constructor.
    Expression_result(llvm::Value *content, bool is_value)
    : m_content(content), m_is_value(is_value)
    {}

public:
    /// Default constructor, creates an unset result.
    Expression_result()
    : m_content(NULL), m_is_value(true)
    {}

private:
    /// The value/ptr itself.
    llvm::Value *m_content;

    /// True, if m_content represent a value, false if its a pointer.
    bool m_is_value;
};

/// A Helper class to unify access to call AST expressions and call DAG expressions.
class ICall_expr {
public:
    /// Check if the called function is an array constructor.
    virtual bool is_array_constructor() const = 0;

    /// Return the semantics of a the called function.
    virtual mi::mdl::IDefinition::Semantics get_semantics() const = 0;

    /// Get the callee definition of one exists.
    ///
    /// \param code_gen  the LLVM code generator
    virtual mi::mdl::IDefinition const *get_callee_definition(
        LLVM_code_generator &code_gen) const = 0;

    /// Get the number of arguments.
    virtual size_t get_argument_count() const = 0;

    /// Translate the i'th argument.
    ///
    /// \param code_gen       the LLVM code generator
    /// \param ctx            the current function context
    /// \param i              the argument index
    /// \param return_derivs  true, iff the user of the argument expects a derivative value
    virtual Expression_result translate_argument(
        LLVM_code_generator &code_gen,
        Function_context    &ctx,
        size_t              i,
        bool                return_derivs) const = 0;

    /// Translate the i'th argument.
    ///
    /// \param code_gen       the LLVM code generator
    /// \param ctx            the current function context
    /// \param i              the argument index
    /// \param return_derivs  true, iff the user of the argument expects a derivative value
    llvm::Value *translate_argument_value(
        LLVM_code_generator &code_gen,
        Function_context    &ctx,
        size_t              i,
        bool                return_derivs) const
    {
        return translate_argument(code_gen, ctx, i, return_derivs).as_value(ctx);
    }

    /// Get the LLVM context data of the callee.
    ///
    /// \param code_gen       the LLVM code generator
    /// \param args           the argument mappings for this function call
    /// \param return_derivs  if true, the function should return derivatives
    virtual LLVM_context_data *get_callee_context(
        LLVM_code_generator                      &code_gen,
        Function_instance::Array_instances const &arg,
        bool                                     return_derivs) const = 0;

    /// Get the result type of the call.
    virtual mi::mdl::IType const *get_type() const = 0;

    /// Get the type of the i'th call argument.
    ///
    /// \param i    the argument index
    virtual mi::mdl::IType const *get_argument_type(
        size_t i) const = 0;

    /// Get the source position of the i'th call argument.
    ///
    /// \param i  the argument index
    virtual mi::mdl::Position const *get_argument_position(size_t i) const = 0;

    /// Get the i'th call argument if it is a constant.
    ///
    /// \param i  the argument index
    ///
    /// \returns the value of the i'th argument if it is a constant, NULL otherwise
    virtual mi::mdl::IValue const *get_const_argument(size_t i) const = 0;

    /// If this is a DS_INTRINSIC_DAG_FIELD_ACCESS, the accessed field index, else -1.
    virtual int get_field_index(
        LLVM_code_generator &code_gen,
        mi::mdl::IAllocator *alloc) const = 0;

    /// Assume the first argument is a boolean branch condition and translate it.
    ///
    /// \param code_gen  the LLVM code generator
    /// \param ctx       the current function context
    /// \param true_bb   branch target for the true case
    /// \param false_bb  branch target for the false case
    virtual void translate_boolean_branch(
        LLVM_code_generator &code_gen,
        Function_context    &ctx,
        llvm::BasicBlock    *true_bb,
        llvm::BasicBlock    *false_bb) const = 0;

    /// If possible, convert into a DAG_call node.
    virtual mi::mdl::DAG_call const *as_dag_call() const = 0;

    /// If possible, convert into an AST expression.
    virtual mi::mdl::IExpression_call const *as_expr_call() const = 0;

    /// Get the source position of this call itself.
    virtual mi::mdl::Position const *get_position() const = 0;

    /// Returns true, if the call should return derivatives.
    ///
    /// \param code_gen  the LLVM code generator
    virtual bool returns_derivatives(LLVM_code_generator &code_gen) const = 0;
};

/// Helper struct for a texture attribute entry.
struct Texture_attribute_entry {
    /// Constructor.
    ///
    /// \param valid   True, if this is a valid texture
    /// \param width   the width of the texture
    /// \param height  the height of the texture (or 0 if not applicable)
    /// \param depth   the depth of the texture (or 0 if not applicable)
    Texture_attribute_entry(bool valid = false, int width = 0, int height = 0, int depth = 0)
    : valid(valid), width(width), height(height), depth(depth)
    {
    }

    bool valid;
    int width, height, depth;
};

/// Helper struct for a light profile attribute entry.
struct Light_profile_attribute_entry {
    /// Constructor.
    ///
    /// \param valid    True, if this is a valid light profile
    /// \param power    the result of light_profile_power()
    /// \param maximum  the result of light_profile_maximum()
    Light_profile_attribute_entry(bool valid = false, float power = 0.0f, float maximum = 0.0f)
    : valid(valid), power(power), maximum(maximum)
    {
    }

    bool valid;
    float power, maximum;
};

/// Helper struct for a bsdf measurement attribute entry.
struct Bsdf_measurement_attribute_entry {
    /// Constructor.
    ///
    /// \param valid    True, if this is a valid bsdf measurement
    Bsdf_measurement_attribute_entry(bool valid = false)
        : valid(valid)
    {
    }

    bool valid;
};

typedef vector<Texture_attribute_entry>::Type          Texture_table;
typedef vector<Light_profile_attribute_entry>::Type    Light_profile_table;
typedef vector<Bsdf_measurement_attribute_entry>::Type Bsdf_measurement_table;
typedef vector<string>::Type                           String_table;

///
/// Implementation of the LLVM jit code generator.
///
class LLVM_code_generator
{
    friend class Allocator_builder;
    friend class MDL_runtime_creator;
    friend class Code_generator_jit;
    friend class Function_context;
    friend class Df_component_info;
    friend class Derivative_infos;
public:
    static char const MESSAGE_CLASS = 'J';

    typedef Type_mapper::State_subset_mode State_subset_mode;
    typedef Type_mapper::Type_mapping_mode Type_mapping_mode;

    typedef IGenerated_code_lambda_function::State_usage  State_usage;

    typedef vector<size_t>::Type Offset_vector;
    typedef vector<mi::mdl::IType const *>::Type Type_vector;
    typedef vector<llvm::Function *>::Type Function_vector;

    ///
    /// Debug modes for the generated JIT code.
    ///
    enum Jit_debug_mode {
        JDBG_NONE     = 0,   /**< Default. Do not generate enter()/leave calls at all. */
        JDBG_STACK    = 1,   /**< Generate enter()/leave() calls in jitted code,
                                  maintain an internal call stack. */
        JDBG_PRINT    = 2,   /**< Generate enter()/leave() calls in jitted code,
                                  print enter/leave messages to stdout. */
        JDBG_BP_ENTER = 4,   /**< Generate enter()/leave() calls in jitted code,
                                  break point on function enter. */
        JDBG_PAUSE    = 8,   /**< Generate enter()/leave() calls in jitted code,
                                  pause on function enter (to connect the debugger). */
    }; // can be or'ed

    /// The coordinate space encoding, must match the definitions in state.mdl.
    enum coordinate_space {
        coordinate_internal,
        coordinate_object,
        coordinate_world
    };

    /// States in code generation of a distribution function, responsible for choosing different
    /// data types and selecting different library functions during code generation.
    enum Distribution_function_state {
        DFSTATE_NONE,       ///< Not generating a distribution function
        DFSTATE_INIT,       ///< Generating BSDF init functions
        DFSTATE_SAMPLE,     ///< Generating BSDF sample functions
        DFSTATE_EVALUATE,   ///< Generating BSDF evaluate functions
        DFSTATE_PDF,        ///< Generating BSDF PDF functions
        DFSTATE_END_STATE
    };

    /// The kinds of resource lookup tables we have.
    enum Resource_table_kind {
        RTK_TEXTURE,            ///< Texture attributes.
        RTK_LIGHT_PROFILE,      ///< Light profile attributes.
        RTK_BSDF_MEASUREMENT,   ///< Bsdf measurement attributes.
        RTK_STRINGS,            ///< String table.
        RTK_LAST = RTK_STRINGS
    };

    /// RAII helper class.
    class MDL_module_scope {
    public:
        /// Constructor.
        MDL_module_scope(LLVM_code_generator &generator, mi::mdl::IModule const *mod)
        : m_generator(generator)
        {
            generator.push_module(mod);
        }

        /// Destructor.
        ~MDL_module_scope() { m_generator.pop_module(); }

    private:
        /// The generator containing the stack.
        LLVM_code_generator &m_generator;
    };

    /// The exception state.
    struct Exc_state {
        IMDL_exception_handler *handler;  ///< The exception handler if any.
        mi::base::Atom32       *abort;    ///< Points to the abort flag.
                                          ///  The long_jump buffer for abort on exception.
        jmp_buf                env;       // PVS: -V730_NOINIT

        /// Constructor.
        Exc_state(IMDL_exception_handler *handler, mi::base::Atom32 &abort)
        : handler(handler), abort(&abort)
        {
        }
    };

    /// Helper value type for the global value map.
    struct Value_offset_pair {
        /// Constructor.
        Value_offset_pair(
            llvm::Value *value,
            bool        is_offset)
        : value(value), is_offset(is_offset)
        {
        }

        /// Default Constructor (needed by the hash map).
        Value_offset_pair()
        : value(NULL), is_offset(false)
        {
        }

        llvm::Value *value;
        bool        is_offset;
    };


public:
    /// Constructor.
    ///
    /// \param jitted_code          the jitted code object
    /// \param compiler             the MDL compiler
    /// \param messages             messages object
    /// \param context              the LLVM context to be used for this generator
    /// \param ptx_mode             if true, ptx will be targeted
    /// \param tm_mode              if ptx_mode is false, the type mapping mode
    /// \param sm_version           if ptx_mode is true, the SM_version we compile for
    /// \param has_tex_handler      True if a texture handler interface is available
    /// \param state_mode           the supported state subset
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    /// \param options              the backend options
    /// \param incremental          if true, this code generator will allow incremental compilation
    /// \param state_mapping        how to map the MDL state
    /// \param res_manag            if non-NULL, the resource manager to use
    /// \param enable_debug         if true, generate debug info
    ///
    /// \note Exceptions are always disabled if \c ptx_mode is true.
    LLVM_code_generator(
        Jitted_code        *jitted_code,
        MDL                *compiler,
        Messages_impl      &messages,
        llvm::LLVMContext  &context,
        bool               ptx_mode,
        Type_mapping_mode  tm_mode,
        unsigned           sm_version,
        bool               has_tex_handler,
        State_subset_mode  state_mode,
        unsigned           num_texture_spaces,
        unsigned           num_texture_results,
        Options_impl const &options,
        bool               incremental,
        unsigned           state_mapping,
        IResource_manager  *res_manag,
        bool               enable_debug);

    /// Destructor.
    ~LLVM_code_generator();

    /// Returns true if strings are mapped to IDs.
    bool strings_mapped_to_ids() const { return m_type_mapper.strings_mapped_to_ids(); }

    /// Enable name mangling.
    void enable_name_mangling() { m_mangle_name = true; }

    /// Enable the generation of the RO segment if allowed.
    void enable_ro_data_segment() { m_use_ro_data_segment = m_enable_ro_segment; }

    /// Mark all exported functions as entry points.
    void mark_exported_funcs_as_entries() { m_exported_funcs_are_entries = true; }

    /// Get the layout data of the target machine.
    llvm::DataLayout const *get_target_layout_data() const;

    /// Get the read-only segment if one was generated.
    ///
    /// \param[out] size  get the size of the segment
    unsigned char const *get_ro_segment(size_t &size) const;

    /// Check if the given value can be stored in the RO data segment.
    ///
    /// \param t  an MDL type to check
    bool can_be_stored_is_ro_segment(IType const *t);

    /// Compile all functions of a module.
    ///
    /// \param module   the module to compile
    ///
    /// \return the created LLVM module or NULL on compilation errors
    llvm::Module *compile_module(
        mi::mdl::IModule const *module);

    /// Compile a distribution function into an LLVM Module and return the LLVM module.
    ///
    /// \param incremental  if true, the module will not be finished
    /// \param dist_func    the distribution function
    /// \param resolver     the call resolver interface to be used
    /// \param llvm_funcs   the generated LLVM functions
    ///
    /// \returns The LLVM module containing the generated functions for this material
    ///          or NULL on compilation errors.
    llvm::Module *compile_distribution_function(
        bool                        incremental,
        Distribution_function const &dist_func,
        ICall_name_resolver const   *resolver,
        Function_vector             &llvm_funcs);

    /// Compile an environment lambda function into an LLVM Module and return the LLVM function.
    ///
    /// \param incremental  if true, the module will not be finished
    /// \param lambda       the lambda function
    /// \param resolver     the call resolver interface to be used
    ///
    /// \returns The LLVM function for this lambda function or NULL on compilation errors.
    llvm::Function *compile_environment_lambda(
        bool                      incremental,
        Lambda_function const     &lambda,
        ICall_name_resolver const *resolver);

    /// Compile an constant lambda function into an LLVM Module and return the LLVM function.
    ///
    /// \param lambda           the lambda function
    /// \param resolver         the call resolver interface to be used
    /// \param attr             an interface to retrieve resource attributes
    /// \param world_to_object  the world-to-object transformation matrix for this function
    /// \param object_to_world  the object-to-world transformation matrix for this function
    /// \param object_id        the result of state::object_id() for this function
    ///
    /// \return the compiled function or NULL on compilation errors
    llvm::Function  *compile_const_lambda(
        Lambda_function const      &lambda,
        ICall_name_resolver const  *resolver,
        ILambda_resource_attribute *attr,
        Float4_struct const        world_to_object[4],
        Float4_struct const        object_to_world[4],
        int                        object_id);

    /// Compile a switch lambda function into an LLVM Module and return the LLVM function.
    ///
    /// \param incremental  if true, the module will not be finished
    /// \param lambda       the lambda function
    /// \param resolver     the call resolver interface to be used
    ///
    /// \returns The LLVM function for this lambda function or NULL on compilation errors.
    llvm::Function *compile_switch_lambda(
        bool                      incremental,
        Lambda_function const     &lambda,
        ICall_name_resolver const *resolver);

    /// Compile a generic lambda function into an LLVM Module and return the LLVM function.
    ///
    /// \param incremental  if true, the module will not be finished
    /// \param lambda       the lambda function
    /// \param resolver     the call resolver interface to be used
    /// \param transformer  if non-NULL, a DAG call transformer
    ///
    /// \return the compiled function or NULL on compilation errors
    ///
    /// \note the lambda function must have only one root expression.
    llvm::Function *compile_generic_lambda(
        bool                      incremental,
        Lambda_function const     &lambda,
        ICall_name_resolver const *resolver,
        ILambda_call_transformer  *transformer);

    /// Retrieve the LLVM module.
    llvm::Module const *get_llvm_module() const { return m_module; }

    /// Retrieve the LLVM module.
    llvm::Module *get_llvm_module() { return m_module; }

    /// Retrieve a reference to the LLVM context that will own the LLVM module.
    llvm::LLVMContext &get_llvm_context() { return m_llvm_context; }

    /// Retrieve the resource manager if any.
    IResource_manager *get_resource_manager() const { return m_res_manager; }

    /// Get a reference to the type mapper.
    Type_mapper &get_type_mapper() { return m_type_mapper; }

    /// Get the debug info builder if any.
    llvm::DIBuilder *get_debug_info_builder() const { return m_di_builder; }

    /// Get the debug info file entry.
    llvm::DIFile &get_debug_info_file_entry() { return m_di_file; }

    /// Returns true if the LLVM code is optimized.
    bool is_optimized() const { return m_opt_level > 0; }

    /// Returns true if fast-math is enabled.
    bool is_fast_math_enabled() const { return m_fast_math; }

    /// Return true if finite-math is enabled.
    bool is_finite_math_enabled() const { return m_finite_math; }

    /// Return true if reciprocal math is enabled (i.e. a/b = a * 1/b).
    bool is_reciprocal_math_enabled() const { return m_reciprocal_math; }

    /// Return true, if the state parameter was used inside the generated code.
    ///
    /// \note Currently this property is calculated statically at code generation
    /// time. It could be improved if the generated code AFTER the optimization is
    /// analyzed ...
    bool get_state_param_usage() const { return m_uses_state_param; }

    /// Get the potential render state usage of the currently compiled entity.
    IGenerated_code_lambda_function::State_usage get_render_state_usage() const
    {
        return m_render_state_usage;
    }

    /// Get the MDL types of the captured arguments if any.
    Type_vector const &get_captured_argument_mdl_types() const {
        return m_captured_args_mdl_types;
    }

    /// Get the LLVM type of all captured arguments if any.
    llvm::StructType *get_captured_arguments_llvm_type() const {
        return m_captured_args_type;
    }

    /// Returns true if the render state includes the uniform state.
    bool state_include_uniform_state() const {
        return m_type_mapper.state_include_uniform_state();
    }

    /// Create resource attribute lookup tables for a lambda function if necessary.
    ///
    /// \param lambda  the lambda function
    void create_resource_tables(Lambda_function const &lambda);

    /// Create the argument struct for captured material parameters.
    ///
    /// \param context  the LLVM context
    /// \param lambda   the lambda function
    void create_captured_argument_struct(
        llvm::LLVMContext     &context,
        Lambda_function const &lambda);

    /// Translate an expression to LLVM IR, returning its value.
    ///
    /// \param ctx            the function context
    /// \param expr           the expression to translate
    /// \param return_derivs  true, iff the user of the expression expects a derivative value
    llvm::Value *translate_expression_value(
        Function_context           &ctx,
        mi::mdl::IExpression const *expr,
        bool                       return_derivs);

    /// Translate an (r-value) expression to LLVM IR.
    ///
    /// \param ctx            the function context
    /// \param expr           the expression to translate
    /// \param return_derivs  true, iff the user of the expression expects a derivative value
    Expression_result translate_expression(
        Function_context           &ctx,
        mi::mdl::IExpression const *expr,
        bool                       return_derivs);

    /// Translate a DAG node into LLVM IR.
    ///
    /// \param ctx       the current function context
    /// \param node      the DAG node to translate
    /// \param resolver  the entity name resolver
    Expression_result translate_node(
        Function_context                   &ctx,
        mi::mdl::DAG_node const            *node,
        mi::mdl::ICall_name_resolver const *resolver);

    /// Create a branch from a boolean expression (with short cut evaluation).
    ///
    /// \param ctx        the function context
    /// \param cond       the boolean conditional expression
    /// \param true_bb    the branch target in the true case
    /// \param false_bb   the branch target in the false case
    void translate_boolean_branch(
        Function_context           &ctx,
        mi::mdl::IExpression const *cond,
        llvm::BasicBlock           *true_bb,
        llvm::BasicBlock           *false_bb);

    /// Create a branch from a boolean expression (with short cut evaluation).
    ///
    /// \param ctx        the function context
    /// \param cond       the boolean conditional expression
    /// \param resolver   the entity name resolver
    /// \param true_bb    the branch target in the true case
    /// \param false_bb   the branch target in the false case
    void translate_boolean_branch(
        Function_context                   &ctx,
        mi::mdl::ICall_name_resolver const *resolver,
        mi::mdl::DAG_node const            *cond,
        llvm::BasicBlock                   *true_bb,
        llvm::BasicBlock                   *false_bb);

    /// Retrieve the LLVM context data for a MDL function instance, create it if not available.
    ///
    /// \param owner        the owner of the definition
    /// \param inst         the function instance
    /// \param module_name  the name of the owner module if owner is NULL
    LLVM_context_data *get_or_create_context_data(
        mi::mdl::IModule const  *owner,
        Function_instance const &inst,
        char const              *module_name = NULL);

    /// Retrieve the LLVM context data for a MDL function instance, return NULL if not available.
    ///
    /// \param func_instance  the function instance
    LLVM_context_data *get_context_data(
        Function_instance const &func_instance);

    /// Retrieve the LLVM context data for a lambda function, create it if not available.
    ///
    /// \param lambda        the lambda function
    LLVM_context_data *get_or_create_context_data(
        mi::mdl::Lambda_function const *lambda);

    /// Create an LLVM context data for an existing LLVM function.
    ///
    /// \param def            the function definition
    /// \param return_derivs  if true, the function returns derivatives
    /// \param func           the LLVM function
    LLVM_context_data *create_context_data(
        IDefinition const *def,
        bool               return_derivs,
        llvm::Function    *func);

    /// Create an LLVM context data for an existing LLVM function.
    ///
    /// \param def        the function definition
    /// \param func       the LLVM function
    LLVM_context_data *create_context_data(
        Internal_function const *int_func,
        llvm::Function *func);

    /// Get the top level module on the stack.
    ///
    /// \note Does NOT increase the reference count of the returned
    ///       module, do NOT decrease it just because of this call.
    mi::mdl::IModule const *tos_module() const;

    /// Retrieve the out-of-bounds reporting routine.
    llvm::Function *get_out_of_bounds() const;

    /// Retrieve the div-by-zero reporting routine.
    llvm::Function *get_div_by_zero() const;

    /// Compile the given module into PTX code.
    ///
    /// \param module       the LLVM module to JIT compile
    /// \param code         will be filled with the PTX code
    void ptx_compile(llvm::Module *module, string &code);

    /// Compile the given module into LLVM-IR code.
    ///
    /// \param module       the LLVM module to JIT compile
    /// \param code         will be filled with the LLVM-IR code
    void llvm_ir_compile(llvm::Module *module, string &code);

    /// Compile the given module into LLVM-BC code.
    ///
    /// \param module       the LLVM module to JIT compile
    /// \param code         will be filled with the LLVM-BC code
    void llvm_bc_compile(llvm::Module *module, string &code);

    /// Set a call transformer for DAG calls.
    ///
    /// \param transformer  the call transformer
    /// \param builder      the DAG builder to be used by the call transformer
    void set_call_transformer(
        mi::mdl::ILambda_call_transformer *transformer,
        mi::mdl::IDag_builder             *builder)
    {
        m_transformer   = transformer;
        m_trans_builder = builder;
    }

    /// Prepare a dummy resource attribute table only containing an invalid resource and declare
    /// the access functions.
    /// To be used, when no corresponding resources were registered but the attribute table
    /// is accessed.
    ///
    /// \param kind  the resource table kind to prepare
    void init_dummy_attribute_table(Resource_table_kind kind);


    /// Initialize the string attribute table, adding the unknown string and declaring
    /// the access functions.
    void init_string_attribute_table();

    /// Get a attribute lookup table.
    ///
    /// \param ctx   the context data of the current function
    /// \param kind  the requested resource table kind
    ///
    /// \return the table
    llvm::Value *get_attribute_table(
        Function_context    &ctx,
        Resource_table_kind kind);

    /// Get a attribute lookup table size.
    ///
    /// \param ctx   the context data of the current function
    /// \param kind  the requested resource table kind
    ///
    /// \return the table size
    llvm::Value *get_attribute_table_size(
        Function_context    &ctx,
        Resource_table_kind kind);

    /// Set the uniform state for evaluating uniform state functions.
    ///
    /// \param world_to_object  if non-NULL, use this matrix to implement world-to-object
    ///                         transform calls, otherwise get the matrix from the state parameter
    /// \param object_to_world  if non-NULL, use this matrix to implement object-to-world
    ///                         transform calls, otherwise get the matrix from the state parameter
    /// \param object_id        the result of state::object_id()
    void set_uniform_state(
        Float4_struct const world_to_object[4],
        Float4_struct const object_to_world[4],
        int                 object_id)
    {
        m_world_to_object = world_to_object;
        m_object_to_world = object_to_world;
        m_object_id       = object_id;
    }

    /// Set the uniform object id.
    ///
    /// \param object_id        the result of state::object_id()
    void set_object_id(int object_id)
    {
        m_object_id = object_id;
    }

    /// Set the world-to-object transformation matrix.
    ///
    /// \param w2o        the world-to-object matrix used to implement state::transform*()
    /// \param o2w        the object-to-world matrix used to implement state::transform*()
    void set_transforms(
        mi::mdl::IValue_matrix const *w2o,
        mi::mdl::IValue_matrix const *o2w);

    /// Set the uniform world-to-object transform matrix.
    ///
    /// \param world_to_object  if non-NULL, use this matrix to implement world-to-object
    ///                         transform calls, otherwise get the matrix from the state parameter
    /// \param object_to_world  if non-NULL, use this matrix to implement object-to_world
    ///                         transform calls, otherwise get the matrix from the state parameter
    void set_transforms(
        Float4_struct const world_to_object[4],
        Float4_struct const object_to_world[4])
    {
        m_world_to_object = world_to_object;
        m_object_to_world = object_to_world;
    }

    /// Get the texture results pointer from the state.
    ///
    /// \param ctx   the context data of the current function
    llvm::Value *get_texture_results(Function_context &ctx);

    /// Get the read-only data segment pointer from the state.
    ///
    /// \param ctx   the context data of the current function
    llvm::Value *get_ro_data_segment(Function_context &ctx);

    /// Get the current object_id value from the uniform state.
    ///
    /// \param ctx   the context data of the current function
    llvm::Value *get_current_object_id(Function_context &ctx);

    /// Get the LLVM value of the current world-to-object matrix.
    ///
    /// \param ctx   the context data of the current function
    llvm::Value *get_w2o_transform_value(Function_context &ctx);

    /// Get the LLVM value of the current object-to-world matrix.
    ///
    /// \param ctx   the context data of the current function
    llvm::Value *get_o2w_transform_value(Function_context &ctx);

    /// Disable array instancing support.
    ///
    /// \note This is currently only possible for CPU executed code.
    void disable_function_instancing();

    /// Get an LLVM type for an MDL type.
    ///
    /// \param type      the MDL type
    /// \param arr_size  if >= 0, the instantiated array size of type
    llvm::Type *lookup_type(
        mdl::IType const *type,
        int              arr_size = -1);

    /// Get an LLVM type for the result of a call expression.
    /// If necessary, a derivative type will be used.
    ///
    /// \param ctx        the context data of current function
    /// \param call_expr  the call expression
    llvm::Type *lookup_type_or_deriv_type(
        Function_context &ctx,
        mi::mdl::ICall_expr const *call_expr);

    /// Get an LLVM type for the result of a expression.
    /// If necessary, a derivative type will be used.
    ///
    /// \param ctx        the context data of current function
    /// \param expr       the expression
    llvm::Type *lookup_type_or_deriv_type(
        Function_context &ctx,
        mi::mdl::IExpression const *expr);

    /// Returns true if for the given expression derivatives should be calculated.
    bool is_deriv_expr(mi::mdl::IExpression const *expr) const;

    /// Returns true if for the given variable derivatives should be calculated.
    bool is_deriv_var(mi::mdl::IDefinition const *def) const;

    /// Returns whether the texture runtime uses derivatives.
    bool is_texruntime_with_derivs() const {
        return m_texruntime_with_derivs;
    }

    /// Drop an LLVM module and clear the layout cache.
    void drop_llvm_module(llvm::Module *module);

    /// Get the call mode for tex_lookup() functions.
    Function_context::Tex_lookup_call_mode get_tex_lookup_call_mode() const {
        return m_tex_calls_mode;
    }

    /// Get the option rt_callable_program_from_id(_64) function.
    llvm::Function *get_optix_cp_from_id();

    /// Set a resource manager.
    ///
    /// \param manag  the manager
    void set_resource_manag(IResource_manager *manag) { m_res_manager = manag; }

    /// Register all native runtime functions with the Jitted_code object.
    /// Should only be called by the Jitted_code constructor.
    ///
    /// \param jitted_code  the Jitted_code object containing the JIT
    static void register_native_runtime_functions(Jitted_code *jitted_code);

private:
    /// Helper to retrieve the allocator.
    mi::mdl::IAllocator *get_allocator() { return m_arena.get_allocator(); }

    /// Mangle a name for PTX output.
    ///
    /// \param def    the function to mangle
    /// \name_prefix  the MDL package prefix
    string mangle(mi::mdl::Function_instance const &inst, char const *name_prefix);

    /// Get the first (real) parameter of the given function.
    ///
    /// \param func  a LLVM function
    /// \param ctx   the context data of this function
    llvm::Function::arg_iterator get_first_parameter(
        llvm::Function          *func,
        LLVM_context_data const *ctx);

    /// Optimize an LLVM function.
    ///
    /// \param func  The LLVM function to optimize.
    ///
    /// \return true if function was modified, false otherwise
    bool optimize(llvm::Function *func);

    /// Optimize LLVM code.
    ///
    /// \param module  The LLVM module to optimize.
    ///
    /// \return true if module was modified, false otherwise
    bool optimize(llvm::Module *module);

    /// Check if a given type needs reference return calling convention.
    ///
    /// \param type  the type to check
    bool need_reference_return(mi::mdl::IType const *type) const;

    /// Check if the given parameter type must be passed by reference.
    ///
    /// \param type   the type of the parameter
    bool is_passed_by_reference(mi::mdl::IType const *type) const;

    /// Determine the function context flags for a function definition.
    ///
    /// \param def    the function definition
    LLVM_context_data::Flags get_function_flags(IDefinition const *def);


    /// Declares an LLVM function from a MDL function instance.
    ///
    /// \param owner        the MDL owner module of the function
    /// \param inst         the function instance
    /// \param name_prefix  the name prefix for this function
    LLVM_context_data *declare_function(
        mi::mdl::IModule const *owner,
        Function_instance const &inst,
        char const             *name_prefix);

    /// Declares an LLVM function from a internal function instance.
    ///
    /// \param inst         the function instance
    LLVM_context_data *declare_internal_function(
        Function_instance const &inst);

    /// Declares an LLVM function from a lambda function.
    ///
    /// \param lambda       the lambda function
    LLVM_context_data *declare_lambda(
        mi::mdl::Lambda_function const *lambda);

    /// Returns true if the given variable needs storage to be allocated.
    ///
    /// \param var_def  the definition of the variable
    bool need_storage_for_var(mi::mdl::IDefinition const *var_def) const;

    /// Compile a function instance.
    ///
    /// \param inst       the function instance
    /// \param func_decl  the function declaration
    void compile_function_instance(
        Function_instance const              &inst,
        mi::mdl::IDeclaration_function const *func_decl);

    /// Push a module on the stack.
    void push_module(mi::mdl::IModule const *module);

    /// Pop a module from the stack.
    void pop_module();

    /// Create extra instructions at function start for debugging.
    ///
    /// \param func  the LLVM function that is entered
    void enter_function(llvm::Function *func);

    /// Translate a statement to LLVM IR.
    ///
    /// \param ctx   the function context
    /// \param stmt  the statement to translate
    void translate_statement(
        Function_context          &ctx,
        mi::mdl::IStatement const *stmt);

    /// Translate a block statement to LLVM IR.
    ///
    /// \param ctx    the function context
    /// \param block  the block statement to translate
    void translate_block(
        Function_context                   &ctx,
        mi::mdl::IStatement_compound const *block);

    /// Translate a declaration statement to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param decl_stmt  the declaration statement to translate
    void translate_decl_stmt(
        Function_context                      &ctx,
        mi::mdl::IStatement_declaration const *decl_stmt);

    /// Translate a declaration to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param decl_stmt  the declaration statement to translate
    void translate_declaration(
        Function_context            &ctx,
        mi::mdl::IDeclaration const *decl);

    /// Translate a variable declaration to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param var_decl   the declaration to translate
    void translate_var_declaration(
        Function_context                      &ctx,
        mi::mdl::IDeclaration_variable const *var_decl);

    /// Translate an if statement to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param if_stmt    the if statement to translate
    void translate_if(
        Function_context              &ctx,
        mi::mdl::IStatement_if const *if_stmt);

    /// Translate a switch statement to LLVM IR.
    ///
    /// \param ctx          the function context
    /// \param switch_stmt  the switch statement to translate
    void translate_switch(
        Function_context                 &ctx,
        mi::mdl::IStatement_switch const *switch_stmt);

    /// Translate a while statement to LLVM IR.
    ///
    /// \param ctx          the function context
    /// \param while_stmt   the while statement to translate
    void translate_while(
        Function_context                &ctx,
        mi::mdl::IStatement_while const *while_stmt);

    /// Translate a do-while statement to LLVM IR.
    ///
    /// \param ctx          the function context
    /// \param do_stmt      the do-while statement to translate
    void translate_do_while(
        Function_context                   &ctx,
        mi::mdl::IStatement_do_while const *do_stmt);

    /// Translate a for statement to LLVM IR.
    ///
    /// \param ctx          the function context
    /// \param for_stmt     the for statement to translate
    void translate_for(
        Function_context              &ctx,
        mi::mdl::IStatement_for const *for_stmt);

    /// Translate a break statement to LLVM IR.
    ///
    /// \param ctx          the function context
    /// \param break_stmt   the break statement to translate
    void translate_break(
        Function_context                &ctx,
        mi::mdl::IStatement_break const *break_stmt);

    /// Translate a continue statement to LLVM IR.
    ///
    /// \param ctx          the function context
    /// \param cont_stmt    the continue statement to translate
    void translate_continue(
        Function_context                   &ctx,
        mi::mdl::IStatement_continue const *cont_stmt);

    /// Translate a return statement to LLVM IR.
    ///
    /// \param ctx          the function context
    /// \param ret_stmt     the return statement to translate
    void translate_return(
        Function_context                 &ctx,
        mi::mdl::IStatement_return const *ret_stmt);

    /// Calculate &matrix[index], index is assured to be in bounds.
    ///
    /// \param ctx         the function context
    /// \param m_type      the type of the matrix object that is indexed
    /// \param matrix_ptr  the pointer to the matrix object
    /// \param index       the index value (in MDL an signed integer)
    llvm::Value *calc_matrix_index_in_bounds(
        Function_context            &ctx,
        mi::mdl::IType_matrix const *m_type,
        llvm::Value                 *matrix_ptr,
        llvm::Value                 *index);

    /// Translate an l-value index expression to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param comp_type  the type of the compound object that is indexed
    /// \param comp_ptr   the pointer to the compound object
    /// \param index      the index value (in MDL an signed integer)
    /// \param index_pos  the source position of the index expression for bound checks
    llvm::Value *translate_lval_index_expression(
        Function_context        &ctx,
        mi::mdl::IType const    *comp_type,
        llvm::Value             *comp_ptr,
        llvm::Value             *index,
        mi::mdl::Position const *index_pos);

    /// Translate a dual l-value index expression to LLVM IR.
    ///
    /// \param ctx           the function context
    /// \param comp_type     the type of the compound object that is indexed
    /// \param comp_val_ptr  the pointer to the value component of the compound object
    /// \param comp_dx_ptr   the pointer to the dx component of the compound object
    /// \param comp_dy_ptr   the pointer to the dy component of the compound object
    /// \param index         the index value (in MDL an signed integer)
    /// \param index_pos     the source position of the index expression for bound checks
    /// \param[out] adr_val  a reference for the resulting address of the value component
    /// \param[out] adr_dx   a reference for the resulting address of the dx component
    /// \param[out] adr_dy   a reference for the resulting address of the dy component
    void translate_lval_index_expression_dual(
        Function_context        &ctx,
        mi::mdl::IType const    *comp_type,
        llvm::Value             *comp_val_ptr,
        llvm::Value             *comp_dx_ptr,
        llvm::Value             *comp_dy_ptr,
        llvm::Value             *index,
        mi::mdl::Position const *index_pos,
        llvm::Value             *&adr_val,
        llvm::Value             *&adr_dx,
        llvm::Value             *&adr_dy);

    /// Translate an r-value index expression to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param comp_type  the type of the compound object that is indexed
    /// \param compound   the compound object
    /// \param index      the index value (in MDL an signed integer)
    /// \param index_pos  the source position of the index expression for bound checks
    Expression_result translate_index_expression(
        Function_context        &ctx,
        mi::mdl::IType const    *comp_type,
        Expression_result       compound,
        llvm::Value             *index,
        mi::mdl::Position const *index_pos);

    /// Translate an l-value expression to LLVM IR.
    ///
    /// \param ctx    the function context
    /// \param expr   the expression to translate
    llvm::Value *translate_lval_expression(
        Function_context           &ctx,
        mi::mdl::IExpression const *expr);

    /// Translate a dual l-value expression to LLVM IR.
    ///
    /// \param      ctx      the function context
    /// \param      expr     the expression to translate
    /// \param[out] adr_val  a reference for the resulting address of the value component
    /// \param[out] adr_dx   a reference for the resulting address of the dx component
    /// \param[out] adr_dy   a reference for the resulting address of the dy component
    void translate_lval_expression_dual(
        Function_context           &ctx,
        mi::mdl::IExpression const *expr,
        llvm::Value                *&adr_val,
        llvm::Value                *&adr_dx,
        llvm::Value                *&adr_dy);

    /// Append the given value to the RO data segment.
    ///
    /// \param v           the value to append
    /// \param alloc_size  the allocation size for this value
    ///
    /// \return the offset of the value in the RO segment
    size_t add_to_ro_data_segment(
        mi::mdl::IValue const *v,
        size_t                alloc_size);

    /// Creates a global constant for a value in the LLVM IR.
    ///
    /// \param      ctx                the function context
    /// \param[in]  v                  the value to translate
    /// \param[out] is_ro_segment_ofs  if true, the constant is accessed through an offset in the
    ///                                RO segment
    ///
    /// \return the global value (the address)
    llvm::Value *create_global_const(
        Function_context               &ctx,
        mi::mdl::IValue_compound const *v,
        bool                           &is_ro_segment_ofs);

    /// Translate a value to LLVM IR.
    ///
    /// \param ctx    the function context
    /// \param v      the value to translate
    Expression_result translate_value(
        Function_context      &ctx,
        mi::mdl::IValue const *v);

    /// Translate a literal expression to LLVM IR.
    ///
    /// \param ctx    the function context
    /// \param lit    the literal expression to translate
    Expression_result translate_literal(
        Function_context                   &ctx,
        mi::mdl::IExpression_literal const *lit);

    // Translate an unary expression without side-effect to LLVM IR.
    llvm::Value *translate_unary(
        Function_context                     &ctx,
        mi::mdl::IExpression_unary::Operator op,
        llvm::Value                          *arg);

    /// Translate an unary expression to LLVM IR.
    ///
    /// \param ctx            the function context
    /// \param un_expr        the unary expression to translate
    /// \param return_derivs  true, iff the user of the expression expects a derivative value
    Expression_result translate_unary(
        Function_context                 &ctx,
        mi::mdl::IExpression_unary const *un_expr,
        bool                             return_derivs);

    /// Helper for pre/post inc/decrement.
    ///
    /// \param ctx       the function context
    /// \param op        the operator to be executed
    /// \param tp        the result type of the operation
    /// \param old_v     the initial value
    /// \param r         out: the result value
    /// \param v         out: the new value
    void do_inner_inc_dec(
        Function_context                     &ctx,
        mi::mdl::IExpression_unary::Operator op,
        llvm::Type                           *tp,
        llvm::Value                          *old_v,
        llvm::Value                          *&r,
        llvm::Value                          *&v);

    /// Translate an inplace change expression.
    ///
    /// \param ctx       the function context
    /// \param un_expr   the expression to translate
    Expression_result translate_inplace_change_expression(
        Function_context                 &ctx,
        mi::mdl::IExpression_unary const *un_expr);

    /// Translate a binary expression to LLVM IR.
    ///
    /// \param ctx       the function context
    /// \param bin_expr  the binary expression to translate
    /// \param return_derivs  true, iff the user of the expression expects a derivative value
    Expression_result translate_binary(
        Function_context                  &ctx,
        mi::mdl::IExpression_binary const *bin_expr,
        bool                              return_derivs);

    /// Translate a side effect free binary expression to LLVM IR.
    ///
    /// \param ctx       the function context
    /// \param bin_expr  the binary expression to translate
    /// \param return_derivs  true, iff the user of the expression expects a derivative value
    llvm::Value *translate_binary_no_side_effect(
        Function_context                  &ctx,
        mi::mdl::IExpression_binary const *bin_expr,
        bool                              return_derivs);

    /// Translate a side effect free binary expression to LLVM IR.
    ///
    /// \param ctx       the function context
    /// \param bin_expr  the binary expression to translate
    llvm::Value *translate_binary_no_side_effect(
        Function_context          &ctx,
        mi::mdl::ICall_expr const *bin_expr);

    /// Translate a side effect free binary simple expressions that require only one
    /// instruction to LLVM IR.
    ///
    /// \param ctx       the function context
    /// \param op        the operator
    /// \param l         the left hand side expression
    /// \param r         the right hand side expression
    /// \param expr_pos  the source position of the expression
    llvm::Value *translate_binary_basic(
        Function_context                      &ctx,
        mi::mdl::IExpression_binary::Operator op,
        llvm::Value                           *l,
        llvm::Value                           *r,
        mi::mdl::Position const               *expr_pos);

    /// Translate a multiplication expression to LLVM IR.
    ///
    /// \param ctx            the function context
    /// \param res_llvm_type  the LLVM result type of the expression
    /// \param l_type         the MDL type of the left hand side expression
    /// \param lhs            the left hand side expression
    /// \param rhs            the MDL type of the right hand side expression
    /// \param rhs            the right hand side expression
    llvm::Value *translate_multiply(
        Function_context           &ctx,
        llvm::Type                 *res_llvm_type,
        mi::mdl::IType const       *l_type,
        llvm::Value                *l,
        mi::mdl::IType const       *r_type,
        llvm::Value                *r);

    /// Translate a multiplication expression to LLVM IR.
    ///
    /// \param ctx            the function context
    /// \param res_llvm_type  the LLVM result type of the expression
    /// \param lhs            the left hand side expression
    /// \param rhs            the right hand side expression
    /// \param return_derivs  true, iff the user of the expression expects a derivative value
    llvm::Value *translate_multiply(
        Function_context           &ctx,
        llvm::Type                 *res_llvm_type,
        mi::mdl::IExpression const *lhs,
        mi::mdl::IExpression const *rhs,
        bool                       return_derivs);

    /// Translate an assign expression to LLVM IR.
    ///
    /// \param ctx            the function context
    /// \param bin_expr       the assign expression to translate
    /// \param return_derivs  true, iff the user of the expression expects a derivative value
    Expression_result translate_assign(
        Function_context                  &ctx,
        mi::mdl::IExpression_binary const *bin_expr,
        bool                              return_derivs);

    /// Translate a binary compare expression to LLVM IR.
    ///
    /// \param ctx       the function context
    /// \param op        the operator kind to translate
    /// \param l_type    the MDL type of the left argument
    /// \param lv        the left argument value
    /// \param r_type    the MDL type of the right argument
    /// \param rv        the right argument value
    Expression_result translate_compare(
        Function_context                      &ctx,
        mi::mdl::IExpression_binary::Operator op,
        mi::mdl::IType const                  *l_type,
        llvm::Value                           *lv,
        mi::mdl::IType const                  *r_type,
        llvm::Value                           *rv);

    /// Translate a conditional expression to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param cond_expr  the conditional expression to translate
    /// \param return_derivs  true, iff the user of the expression expects a derivative value
    Expression_result translate_conditional(
        Function_context                       &ctx,
        mi::mdl::IExpression_conditional const *cond_expr,
        bool                                   return_derivs);

    /// Create the float4x4 identity matrix.
    ///
    /// \param ctx        the function context
    llvm::Value *create_identity_matrix(
        Function_context &ctx);

    /// Translate a call expression to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    Expression_result translate_call(
        Function_context &ctx,
        ICall_expr const *call_expr);

    /// Returns the BSDF function name suffix for the current distribution function state.
    static char const *get_dist_func_state_suffix(Distribution_function_state state);

    /// Returns the BSDF function name suffix for the current distribution function state.
    char const *get_dist_func_state_suffix() const
    {
        return get_dist_func_state_suffix(m_dist_func_state);
    }

    /// Returns the distribution function state requested by the given call.
    ///
    /// \param call  the LLVM call instruction calling a BSDF member function
    Distribution_function_state get_dist_func_state_from_call(llvm::CallInst *call);

    /// Get the BSDF function for the given semantics and the current distribution function state
    /// from the BSDF library.
    llvm::Function *get_libbsdf_function(
        IDefinition::Semantics sema, 
        IType::Kind kind);

    /// Generate a call to an expression lambda function.
    ///
    /// \param ctx           the function context
    /// \param lambda_index  the index of the precalculated lambda function
    /// \param opt_dest_ptr  an optional destination pointer. The result will be converted
    ///                      before writing it there, if necessary
    /// \returns the result pointer
    Expression_result generate_expr_lambda_call(
        Function_context &ctx,
        size_t           lambda_index,
        llvm::Value      *opt_dest_ptr = NULL);

    /// Translate a precalculated lambda function to LLVM IR.
    ///
    /// \param ctx             the function context
    /// \param lambda_index    the index of the precalculated lambda function
    /// \param expected_type   the result will be converted to this type if necessary
    Expression_result translate_precalculated_lambda(
        Function_context &ctx,
        size_t           lambda_index,
        llvm::Type       *expected_type);

    /// Get the BSDF parameter ID metadata for an instruction.
    ///
    /// \param inst            the instruction for which the metadata should be retrieved
    /// \param kind            kind of distribution function TK_BSDF, TK_EDF, ...
    ///
    /// \returns the ID or a negative value (-1) if instruction does not have this metadata
    int get_metadata_df_param_id(
        llvm::Instruction   *inst, 
        IType::Kind         kind);

    /// Rewrite all usages of a DF component variable using the given weight array and the
    /// BSDF component information.
    ///
    /// \param ctx             the function context
    /// \param inst            the alloca instruction representing the array parameter
    /// \param weight_array    the array containing the component weights, can be local or global
    /// \param comp_info       the bsdf component information to use for the replacements
    /// \param delete_list     list of instructions to be deleted when function is fully processed
    void rewrite_df_component_usages(
        Function_context                           &ctx,
        llvm::AllocaInst                           *inst,
        llvm::Value                                *weight_array,
        Df_component_info                        &comp_info,
        llvm::SmallVector<llvm::Instruction *, 16> &delete_list);

    /// Rewrite the address of a memcpy from a color_bsdf_component to the given weight array.
    ///
    /// \param ctx             the function context
    /// \param weight_array    the array containing the component weights, can be local or global
    /// \param addr_bitcast    the address supposedly used by memcpy calls
    /// \param index           the index into a color_bsdf_component array
    /// \param delete_list     list of instructions to be deleted when function is fully processed
    ///
    /// \returns true if the rewrite was successful
    bool rewrite_weight_memcpy_addr(
        Function_context &ctx,
        llvm::Value *weight_array,
        llvm::BitCastInst *addr_bitcast,
        llvm::Value *index,
        llvm::SmallVector<llvm::Instruction *, 16> &delete_list);

    /// Handle BSDF array parameter during BSDF instantiation.
    ///
    /// \param ctx          the function context
    /// \param inst         the alloca instruction representing the array parameter
    /// \param arg          the DAG node of the BSDF array parameter
    /// \param delete_list  list of instructions to be deleted when function is fully processed
    void handle_df_array_parameter(
        Function_context                           &ctx,
        llvm::AllocaInst                           *inst,
        DAG_node const                             *arg,
        llvm::SmallVector<llvm::Instruction *, 16> &delete_list);

    /// Recursively instantiate a DF from the given DAG node from code in the DF library
    /// according to current distribution function state.
    ///
    /// \param node  the DAG call with DF semantics or a DF constant node.
    ///              For a DAG call, the arguments will be used to instantiate the DF.
    llvm::Function *instantiate_df(
        DAG_node const *node);

    /// Recursively instantiate a ternary operator of type DF.
    ///
    /// \param node  the DAG call of the ternary operator
    ///
    /// \returns a function implementing the ternary operator.
    llvm::Function *instantiate_ternary_df(
        DAG_call const *dag_call);


    /// Translate the current distribution function to LLVM IR.
    ///
    /// \param ctx                  the function context
    /// \param lambda_result_exprs  the list of expression lambda indices for the lambda results
    Expression_result translate_distribution_function(
        Function_context                     &ctx,
        llvm::SmallVector<unsigned, 8> const &lambda_result_exprs);

    /// Translate the init function of the current distribution function to LLVM IR.
    ///
    /// \param ctx                   the function context
    /// \param texture_result_exprs  the list of expression lambda indices for the texture results
    /// \param lambda_result_exprs   the list of expression lambda indices for the lambda results
    void translate_distribution_function_init(
        Function_context                     &ctx,
        llvm::SmallVector<unsigned, 8> const &texture_result_exprs,
        llvm::SmallVector<unsigned, 8> const &lambda_result_exprs);

    /// Translate a DAG intrinsic call expression to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    Expression_result translate_dag_intrinsic(
        Function_context &ctx,
        ICall_expr const *call_expr);

    /// Translate a JIT intrinsic call expression to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    Expression_result translate_jit_intrinsic(
        Function_context &ctx,
        ICall_expr const *call_expr);

    /// Translate a DAG expression lambda call to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    Expression_result translate_dag_call_lambda(
        Function_context &ctx,
        ICall_expr const *call_expr);

    /// Get the argument type instances for a given call.
    ///
    /// \param ctx        the function context
    /// \param call   the call.
    Function_instance::Array_instances get_call_instance(
        Function_context &ctx,
        ICall_expr const *call);

    /// Instantiate a type size using array type instances.
    ///
    /// \param type       the type
    /// \param arr_inst   the array instantiate data
    int instantiate_call_param_type_size(
        IType const                              *type,
        Function_instance::Array_instances const &arr_inst);

    /// Translate a call to an user defined function to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    ///
    /// \return NULL if the call could not be translated
    llvm::Value *translate_call_user_defined_function(
        Function_context &ctx,
        ICall_expr const *call_expr);

    /// Translate a state::transform_*() call expression to LLVM IR.
    ///
    /// \param sema       the semantic of the called function
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    Expression_result translate_transform_call(
        IDefinition::Semantics sema,
        Function_context       &ctx,
        ICall_expr const       *call_expr);

    /// Translate a state::object_id() call expression to LLVM IR.
    ///
    /// \param ctx        the function context
    Expression_result translate_object_id_call(
        Function_context       &ctx);

    /// Check if the given argument is an index and return its bound.
    ///
    /// \param sema  the semantic of the called function
    /// \param idx   the parameter index to check
    ///
    /// \return -1 if the parameter is not an index, its bound (>=0) otherwise
    int is_index_argument(mi::mdl::IDefinition::Semantics sema, int i);

    /// Translate a call to a compiler known function to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    ///
    /// \return NULL if the call could not be translated
    llvm::Value *translate_call_intrinsic_function(
        Function_context          &ctx,
        mi::mdl::ICall_expr const *call_expr);

    /// Get the intrinsic LLVM function for a MDL function.
    ///
    /// \param def            the definition of the MDL function
    /// \param return_derivs  if true, derivatives will be generated for the return value
    llvm::Function *get_intrinsic_function(
        IDefinition const *def,
        bool               return_derivs);

    /// Get the LLVM function for an internal function.
    ///
    /// \param int_func  the internal function description
    llvm::Function *get_internal_function(Internal_function const *int_func);

    /// Call a void runtime function.
    ///
    /// \param ctx     the function context
    /// \param callee  the LLVM function to call
    /// \param args    the call arguments
    void call_rt_func_void(
        Function_context              &ctx,
        llvm::Function                *callee,
        llvm::ArrayRef<llvm::Value *> args);

    /// Call a runtime function.
    ///
    /// \param ctx     the function context
    /// \param callee  the LLVM function to call
    /// \param args    the call arguments
    ///
    /// \return the return value of the function
    llvm::Value *call_rt_func(
        Function_context              &ctx,
        llvm::Function                *callee,
        llvm::ArrayRef<llvm::Value *> args);

    /// Translate a conversion call to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    Expression_result translate_conversion(
        Function_context &ctx,
        ICall_expr const *call_expr);

    /// Translate a conversion call to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    llvm::Value *translate_conversion(
        Function_context     &ctx,
        mi::mdl::IType const *tgt,
        mi::mdl::IType const *src,
        llvm::Value          *v);

    /// Translate a conversion call to a vector type to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param tgt        the target type
    /// \param src        the source type
    /// \param v          the LLVM value to convert
    llvm::Value *translate_vector_conversion(
        Function_context            &ctx,
        mi::mdl::IType_vector const *tgt,
        mi::mdl::IType const        *src,
        llvm::Value                 *v);

    /// Translate a conversion call to a matrix type to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param tgt        the target type
    /// \param src        the source type
    /// \param v          the LLVM value to convert
    llvm::Value *translate_matrix_conversion(
        Function_context            &ctx,
        mi::mdl::IType_matrix const *tgt,
        mi::mdl::IType const        *src,
        llvm::Value                 *v);

    /// Translate a conversion call to the color type to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param src        the source type
    /// \param v          the LLVM value to convert
    llvm::Value *translate_color_conversion(
        Function_context            &ctx,
        mi::mdl::IType const        *src,
        llvm::Value                 *v);

    /// Translate an elemental constructor call to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    Expression_result translate_elemental_constructor(
        Function_context          &ctx,
        mi::mdl::ICall_expr const *call_expr);

    /// Translate a matrix elemental constructor call to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    Expression_result translate_matrix_elemental_constructor(
        Function_context          &ctx,
        mi::mdl::ICall_expr const *call_expr);

    /// Translate a matrix diagonal constructor call to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    Expression_result translate_matrix_diagonal_constructor(
        Function_context          &ctx,
        mi::mdl::ICall_expr const *call_expr);

    /// Translate a color from spectrum constructor call to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    Expression_result translate_color_from_spectrum(
        Function_context          &ctx,
        mi::mdl::ICall_expr const *call_expr);

    /// Translate an array constructor call to LLVM IR.
    ///
    /// \param ctx        the function context
    /// \param call_expr  the call expression to translate
    Expression_result translate_array_constructor_call(
        Function_context &ctx,
        ICall_expr const *call_expr);

    /// Translate a let expression to LLVM IR.
    ///
    /// \param ctx            the function context
    /// \param let_expr       the let expression to translate
    /// \param return_derivs  true, iff the user of the expression expects a derivative value
    Expression_result translate_let(
        Function_context               &ctx,
        mi::mdl::IExpression_let const *let_expr,
        bool                           return_derivs);

    /// Create a matrix by matrix multiplication.
    ///
    /// \param ctx       the function context
    /// \param res_type  the LLVM result type of the multiplication
    /// \param l         the left NxM matrix
    /// \param r         the right MxK matrix
    /// \param N         number of rows of the left matrix
    /// \param M         number of columns of the left and number of rows of the right matrix
    /// \param K         number of columns of the right matrix
    llvm::Value *do_matrix_multiplication(
        Function_context &ctx,
        llvm::Type       *res_type,
        llvm::Value      *l,
        llvm::Value      *r,
        int              N,
        int              M,
        int              K);

    /// Create a vector by matrix multiplication.
    ///
    /// \param ctx       the function context
    /// \param res_type  the LLVM result type of the multiplication
    /// \param l         the left vector
    /// \param r         the right MxK matrix
    /// \param M         size of the left vector and number of rows of the right matrix
    /// \param K         number of columns of the right matrix
    llvm::Value *do_matrix_multiplication_VxM(
        Function_context &ctx,
        llvm::Type       *res_type,
        llvm::Value      *l,
        llvm::Value      *r,
        int              M,
        int              K);

    /// Create a matrix by vector multiplication.
    ///
    /// \param ctx       the function context
    /// \param res_type  the LLVM result type of the multiplication
    /// \param l         the left NxM matrix
    /// \param r         the right vector
    /// \param N         number of rows of the left matrix
    /// \param M         number of columns of the left matrix and size of the right vector
    llvm::Value *do_matrix_multiplication_MxV(
        Function_context &ctx,
        llvm::Type       *res_type,
        llvm::Value      *l,
        llvm::Value      *r,
        int              N,
        int              M);

    /// Compile all functions waiting in the wait queue into the current module.
    void compile_waiting_functions();

    /// Create the RO data segment.
    void create_ro_segment();

    /// Create a new LLVM module.
    ///
    /// \param mod_name   the name of the module
    /// \param mod_fname  the file name of the module's origin
    void create_module(char const *mod_name, char const *mod_fname);

    /// Initialize the current LLVM module with user-specified LLVM implementations.
    bool init_user_modules();

    /// Finalize compilation of the current module that was created by create_module().
    ///
    /// \returns the LLVM module (that was create using create_module()) or NULL on error;
    ///          in that case the module is destroyed
    llvm::Module *finalize_module();

    /// JIT compile all functions of the given module.
    ///
    /// \param module  the LLVM module to JIT compile
    void jit_compile(llvm::Module *module);

    /// Get the address of a JIT compiled LLVM function.
    void *get_entry_point(llvm::Function *func);

    /// Get the number of error messages.
    int get_error_message_count();

    /// Add a JIT backend error message to the messages.
    ///
    /// \param code    the code of the error message
    /// \param params  the message parameters
    void error(int code, Error_params const &params);

    /// Add a JIT backend error message to the messages.
    ///
    /// \param code    the code of the error message
    /// \param param   a string parameter for the error message
    void error(int code, char const *str_param);

    /// Add a JIT backend error message to the messages.
    ///
    /// \param code    the code of the error message
    /// \param param   a string parameter for the error message
    void error(int code, llvm::StringRef const &str_param)
    {
        string str(str_param.data(), str_param.size(), get_allocator());
        error(code, str.c_str());
    }

    /// Add a JIT backend error message to the messages.
    ///
    /// \param code    the code of the error message
    /// \param param   a string parameter for the error message
    void error(int code, std::string const &str_param)
    {
        error(code, str_param.c_str());
    }

    /// Add a JIT backend error message to the messages.
    ///
    /// \param code    the code of the error message
    /// \param param   a string parameter for the error message
    void error(int code, string const &str_param)
    {
        error(code, str_param.c_str());
    }

    /// Create a runtime.
    ///
    /// \param arena_builder        an arena builder
    /// \param code_gen             the code generator to be used
    /// \param ptx_mode             True if PTX is targeted
    /// \param fast_math            True, if fast-math is enabled
    /// \param has_texture_handler  True, if a texture handler interface is available
    /// \param internal_space       the internal space
    static MDL_runtime_creator *create_mdl_runtime(
        mi::mdl::Arena_builder &arena_builder,
        LLVM_code_generator    *code_gen,
        bool                   ptx_mode,
        bool                   fast_math,
        bool                   has_texture_handler,
        char const             *internal_space);

    /// Terminate the runtime.
    ///
    /// \param creator  the runtime creator
    static void terminate_mdl_runtime(MDL_runtime_creator *creator);

    /// Handle out of bounds.
    ///
    /// \param exc_state  the exception state
    /// \param index      the faulty index
    /// \param bound      the object bound ([0 .. bound-1])
    /// \param fname      if non-NULL, the file name of the occurrence
    /// \param line       if non-zero, the line of the occurrence
    static void mdl_out_of_bounds(
        Exc_state  &exc_state,
        int        index,
        size_t     bound,
        char const *fname,
        int        line);

    /// Handle (integer) division by zero.
    ///
    /// \param exc_state  the exception state
    /// \param fname      if non-NULL, the file name of the occurrence
    /// \param line       if non-zero, the line of the occurrence
    static void mdl_div_by_zero(
        Exc_state  &exc_state,
        char const *fname,
        int        line);

    /// Find the definition of a signature of a standard library function.
    ///
    /// \param module_name  the absolute name of a standard library module
    /// \param signature    a (function) signature
    mi::mdl::IDefinition const *find_stdlib_signature(
        char const *module_name,
        char const *signature) const;

    /// Add a texture attribute table.
    ///
    /// \param table  the table data
    void add_texture_attribute_table(
        Texture_table const &table);

    /// Creates the light profile attribute table.
    ///
    /// \param table  the table data
    void add_light_profile_attribute_table(
        Light_profile_table const &table);

    /// Creates the bsdf measurement attribute table.
    ///
    /// \param table  the table data
    void add_bsdf_measurement_attribute_table(
        Bsdf_measurement_table const &table);

    /// Add a string constant mapping to the string table.
    ///
    /// \param s   the string constant to add
    /// \param id  the assigned ID
    void add_string_constant(char const *s, Type_mapper::Tag id);

    /// Get the number of string constants.
    size_t get_string_constant_count() const { return m_string_table.size(); }

    /// Get string constant for a tag.
    ///
    /// \param id  the ID for the string constant
    ///
    /// \return the string constant or NULL if id is out of bounds
    char const *get_string_constant(size_t id) const;

    /// Creates the texture attribute table finally.
    void create_texture_attribute_table();

    /// Creates the light profile attribute table finally.
    void create_light_profile_attribute_table();

    /// Creates the bsdf measurement attribute table finally.
    void create_bsdf_measurement_attribute_table();

    /// Creates the string table finally.
    void create_string_table();


    /// Get libdevice as LLVM bitcode.
    ///
    /// \param[in]  arch             the architecture
    /// \param[out] size             the size of the library in bytes
    /// \param[out] min_ptx_version  if non-zero, the minimum PTX version required
    static unsigned char const *get_libdevice(
        size_t   &size,
        unsigned &min_ptx_version);

    /// Load libdevice.
    ///
    /// \param[in]  llvm_context     the context for the loader
    /// \param[out] min_ptx_version  if non-zero, the minimum PTX version required for the library
    static llvm::Module *load_libdevice(
        llvm::LLVMContext &llvm_context,
        unsigned          &min_ptx_version);

    /// Prepare the internal functions.
    void prepare_internal_functions();

    /// Prepare types and prototypes in the current LLVM module used by libbsdf.
    ///
    /// \param compiler  the MDL compiler
    /// \param libbsdf   the libbsdf LLVM module
    void prepare_libbsdf_prototypes(llvm::Module *libbsdf);

    /// Create the BSDF function types using the BSDF data types from the already linked libbsdf
    /// module.
    ///
    /// \param compiler  the MDL compiler
    void create_bsdf_function_types();

    /// Create the EDF function types using the EDF data types from the already linked libbsdf
    /// module.
    ///
    /// \param compiler  the MDL compiler
    void create_edf_function_types();

    /// Load the libbsdf LLVM module.
    ///
    /// \param llvm_context  the context for the loader
    llvm::Module *load_libbsdf(llvm::LLVMContext &llvm_context);

    /// Determines the semantics for a libbsdf df function name.
    ///
    /// \param name      the name of the function
    ///
    /// \returns the semantics of the function or IDefinition::DS_UNKNOWN if the name
    ///          was not recognized
    IDefinition::Semantics get_libbsdf_function_semantics(llvm::StringRef name);

    /// Check whether the given parameter of the given df function is an array parameter.
    ///
    /// \param sema          the semantics of the df function
    /// \param df_param_idx  the parameter index of the df function
    bool is_libbsdf_array_parameter(IDefinition::Semantics sema, int df_param_idx);

    /// Translate a potential runtime call in a libbsdf function to a call to the according
    /// intrinsic, converting the arguments as necessary.
    ///
    /// \param call      the call instruction to translate
    /// \param ii        the instruction iterator, which will be updated if the call is translated
    /// \param ctx       the context for the translation
    ///
    /// \returns false if there was any error
    bool translate_libbsdf_runtime_call(
        llvm::CallInst *call,
        llvm::BasicBlock::iterator &ii,
        Function_context &ctx);

    /// Transitively walk over the uses of the given argument and mark any calls as DF calls,
    /// storing the provided parameter index as "libbsdf.bsdf_param" or "libbsdf.edf_param" 
    /// or ... metadata.
    ///
    /// \param arg             the argument to be followed
    /// \param df_param_idx    the DF parameter index of the argument
    /// \param kind            kind of distribution function TK_BSDF, TK_EDF, ...
    void mark_df_calls(
        llvm::Argument *arg,        
        int df_param_idx,
        IType::Kind kind);

    /// Load and link libbsdf into the current LLVM module.
    /// It maps the types from libbsdf to our types and resolves referenced API functions
    /// to our intrinsics.
    ///
    /// \returns false if there was any error.
    bool load_and_link_libbsdf();

    /// Clear the DAG-to-LLVM-IR node map.
    void clear_dag_node_map() { m_node_value_map.clear(); }

    /// Reset the lambda function compilation state.
    void reset_lambda_state();

    /// Get the next basic block chain.
    size_t get_next_bb() { return ++m_last_bb; }

    /// Parse a call mode option.
    ///
    /// \param name  a valid call mode name
    static Function_context::Tex_lookup_call_mode parse_call_mode(char const *name);

private:
    /// The memory arena used to allocate context data on.
    mi::mdl::Memory_arena m_arena;

    /// The builder for objects on the memory arena.
    mi::mdl::Arena_builder m_arena_builder;

    /// If non-NULL, a DAG call transformer.
    mi::mdl::ILambda_call_transformer *m_transformer;

    /// The DAG builder to be used by the transformer.
    mi::mdl::IDag_builder             *m_trans_builder;

    /// The context for this code generator.
    llvm::LLVMContext &m_llvm_context;

    /// The supported state subset.
    State_subset_mode m_state_mode;

    /// The internal space for which to compile.
    char const *m_internal_space;

    /// The resource manager if any.
    IResource_manager *m_res_manager;

    /// AN entry in the resource lookup table info.
    struct Resource_lut_info {
        /// The LLVM function to retrieve the lookup table.
        llvm::Function *m_get_lut;

        /// The LLVM function to retrieve the lookup table size.
        llvm::Function *m_get_lut_size;
    };

    /// Resource lookup table handling.
    Resource_lut_info m_lut_info[RTK_LAST + 1];

    /// The current texture table.
    Texture_table m_texture_table;

    /// The current light profile table.
    Light_profile_table m_light_profile_table;

    /// The current bsdf_measurement table.
    Bsdf_measurement_table m_bsdf_measurement_table;

    /// The current string (constant) table.
    String_table m_string_table;

    /// The jitted code singleton.
    mi::base::Handle<Jitted_code> m_jitted_code;

    /// The MDL compiler.
    mi::base::Handle<MDL> m_compiler;

    /// The messages object.
    Messages_impl &m_messages;

    /// The current module.
    llvm::Module *m_module;

    /// A user-specified LLVM implementation of the state module.
    BinaryOptionData m_user_state_module;

    /// The LLVM function pass manager for the current module.
    llvm::OwningPtr<llvm::legacy::FunctionPassManager> m_func_pass_manager;

    /// If true, fast-math transformations are enabled.
    bool m_fast_math;

    /// If true, the read-only segment generation is enabled.
    bool m_enable_ro_segment;

    /// If true, finite-math-only transformations are enabled.
    bool m_finite_math;

    /// If true, reciprocal math transformations are enabled (i.e. a/b = a * 1/b).
    bool m_reciprocal_math;


    /// The runtime creator.
    MDL_runtime_creator *m_runtime;

    /// Cache for the tex_lookup functions once created.
    mutable llvm::Function *m_tex_lookup_functions[mi::mdl::Type_mapper::THV_LAST];

    /// Cache for generated Optix callable programs.
    mutable llvm::Value *m_optix_cps[mi::mdl::Type_mapper::THV_LAST];

    /// The debug info builder if any.
    llvm::DIBuilder *m_di_builder;

    /// DIFile object corresponding to the source file where the current function was defined.
    llvm::DIFile m_di_file;

    /// If non-NULL, this matrix will be used to implement world-to-object
    /// state::transform_*() calls.
    Float4_struct const *m_world_to_object;

    /// If non-NULL, this matrix will be used to implement object-to-world
    /// state::transform_*() calls.
    Float4_struct const *m_object_to_world;

    /// If the world-to-object matrix was created locally, its values are stored here.
    Float4_struct m_world_to_object_store[4];

    /// If the object-to-world matrix was created locally, its values are stored here.
    Float4_struct m_object_to_world_store[4];

    /// The result of state::object_id().
    int m_object_id;

    typedef mi::mdl::hash_map<
        Function_instance,
        LLVM_context_data *,
        Function_instance::Hash<>,
        Function_instance::Equal<> >::Type Context_data_map;

    /// The map storing context data for definitions.
    Context_data_map m_context_data;

    /// The data layout to be used with this code generator.
    llvm::DataLayout m_data_layout;

    /// The type mapper.
    Type_mapper m_type_mapper;

    /// The MDL module stack.
    mi::mdl::vector<mi::mdl::IModule const *>::Type m_module_stack;

    /// Value class to handle waiting functions.
    class Wait_entry {
    public:
        /// Constructor.
        Wait_entry(mi::mdl::IModule const *owner, Function_instance const &inst)
        : m_owner(owner)
        , m_inst(inst)
        {
        }

        /// Get the owner module.
        mi::mdl::IModule const *get_owner() const { return m_owner; }

        /// The function instance.
        Function_instance const &get_instance() const { return m_inst; }

    private:
        /// The owner module.
        mi::mdl::IModule const  *m_owner;
        /// The function instance.
        Function_instance       m_inst;
    };

    typedef mi::mdl::queue<Wait_entry>::Type Function_wait_queue;

    /// The wait queue for functions.
    Function_wait_queue m_functions_q;

    class Value_entry {
    public:
        /// Constructor.
        Value_entry(DAG_node const *node, size_t bb)
        : m_node(node), m_bb(bb)
        {
        }

        /// Get the node.
        DAG_node const *get_node() const { return m_node; }

        /// Get the basic block chain.
        size_t get_bb() const { return m_bb; }
    private:
        /// The node computing the value.
        DAG_node const *m_node;

        /// The basic block-chain that owns the node.
        size_t m_bb;
    };

    struct Value_entry_hash {
        size_t operator()(Value_entry const &ve) const {
            Hash_ptr<DAG_node const> hasher;
            return hasher(ve.get_node()) ^ ve.get_bb();
        }
    };

    struct Value_entry_equal {
        inline unsigned operator()(Value_entry const &a, Value_entry const &b) const {
            return a.get_bb() == b.get_bb() && a.get_node() == b.get_node();
        }
    };

    typedef mi::mdl::hash_map<
        Value_entry const,
        Expression_result,
        Value_entry_hash,
        Value_entry_equal>::Type Node_value_map;

    /// Map to translate DAG expressions to LLVM IR.
    Node_value_map m_node_value_map;

    /// Number of the last created basic block chain.
    size_t m_last_bb;

    /// The current basic block chain.
    size_t m_curr_bb;

    typedef mi::mdl::ptr_hash_map<void const, Value_offset_pair>::Type Global_const_map;

    /// Map large constants to llvm global constants.
    Global_const_map m_global_const_map;

    /// The RO segment once created.
    unsigned char *m_ro_segment;

    /// The next offset in the RO data segment.
    size_t m_next_ro_data_offset;

    typedef list<mi::mdl::IValue const *>::Type Value_list;

    /// The list of all values that goes into the RO data segment.
    Value_list m_ro_data_values;

    /// The option rt_callable_program_from_id(_64) function once created.
    llvm::Function *m_optix_cp_from_id;

    /// The MDL types of all captured arguments if any.
    Type_vector m_captured_args_mdl_types;

    /// The type of all captured arguments if any.
    llvm::StructType *m_captured_args_type;

    /// Optimization level.
    unsigned m_opt_level;

    /// The debug mode.
    Jit_debug_mode m_jit_dbg_mode;

    /// The number of supported texture spaces.
    unsigned m_num_texture_spaces;

    /// The number of texture result entries.
    unsigned m_num_texture_results;

    /// If PTX mode is enabled, the SM_version we compile for.
    unsigned m_sm_version;

    /// If non-zero, the minimum PTX version required.
    unsigned m_min_ptx_version;

    /// The render state usage for the currently compiled entity.
    State_usage m_render_state_usage;

    /// If true, generate debug info.
    bool m_enable_debug;

    /// If true, all exported functions are entry points.
    bool m_exported_funcs_are_entries;

    /// If true, bounds checks exceptions for all index expressions are disabled.
    bool m_bounds_check_exception_disabled;

    /// If true, checks for division by zero exceptions are disabled.
    bool m_divzero_check_exception_disabled;

    /// If true, the state is used in the generated code.
    bool m_uses_state_param;

    /// If true, we are compiling for PTX.
    bool m_ptx_mode;

    /// If true, MDL names are mangled.
    bool m_mangle_name;

    /// If true, function instancing is enabled (default).
    bool m_enable_instancing;

    /// If true, lambda return value is enforced to be sret.
    bool m_lambda_force_sret;

    /// If true, the first lambda function parameter will be passed by reference.
    bool m_lambda_first_param_by_ref;

    /// If true, the (render) state usage of lambda functions is enforced.
    bool m_lambda_force_render_state;

    /// If true, no lambda results parameter will be generated.
    bool m_lambda_force_no_lambda_results;

    /// If true, big constants are places into the RO data segment.
    bool m_use_ro_data_segment;

    /// If true, the libdevice is linked into PTX output.
    bool m_link_libdevice;

    /// If true, this code generator will allow incremental compilation.
    bool m_incremental;

    /// If true, the texture lookup functions with derivatives will be used.
    bool m_texruntime_with_derivs;

    /// If non-null, the derivative analysis information.
    Derivative_infos const *m_deriv_infos;

    /// If non-null, the derivative analysis information of the current function.
    /// Used during compilation of waiting functions.
    Func_deriv_info const *m_cur_func_deriv_info;

    /// The call mode for texture lookup functions.
    Function_context::Tex_lookup_call_mode m_tex_calls_mode;

    /// Current distribution function.
    Distribution_function const *m_dist_func;

    /// Current state of generating a distribution function.
    Distribution_function_state m_dist_func_state;

    typedef mi::mdl::ptr_hash_map<ILambda_function const, llvm::Function *>::Type
        Dist_func_lambda_map;

    /// Map from ILambda_function objects to LLVM functions used for distribution functions.
    Dist_func_lambda_map m_dist_func_lambda_map;

    /// A structure type for storing the results of all lambda functions.
    llvm::StructType *m_lambda_results_struct_type;

    /// Array which maps expression lambda indices to lambda result indices.
    /// For expression lambdas without a lambda result entry the array contains -1.
    mi::mdl::vector<int>::Type m_lambda_result_indices;

    /// A structure type for storing the results of all lambda functions.
    llvm::StructType *m_texture_results_struct_type;

    /// Array which maps expression lambda indices to texture result indices.
    /// For expression lambdas without a texture result entry the array contains -1.
    mi::mdl::vector<int>::Type m_texture_result_indices;

    /// A float3 struct type used in libbsdf.
    llvm::StructType *m_float3_struct_type;


    /// Function type of the BSDF sample function.
    llvm::FunctionType *m_type_bsdf_sample_func;

    /// Return type of the BSDF sample function.
    llvm::Type *m_type_bsdf_sample_data;

    /// Function type of the BSDF evaluate function.
    llvm::FunctionType *m_type_bsdf_evaluate_func;

    /// Return type of the BSDF evaluate function.
    llvm::Type *m_type_bsdf_evaluate_data;

    /// Function type of the BSDF pdf function.
    llvm::FunctionType *m_type_bsdf_pdf_func;

    /// Return type of the BSDF PDF function.
    llvm::Type *m_type_bsdf_pdf_data;

    /// Function type of the EDF sample function.
    llvm::FunctionType *m_type_edf_sample_func;

    /// Return type of the EDF sample function.
    llvm::Type *m_type_edf_sample_data;

    /// Function type of the EDF evaluate function.
    llvm::FunctionType *m_type_edf_evaluate_func;

    /// Return type of the EDF evaluate function.
    llvm::Type *m_type_edf_evaluate_data;

    /// Function type of the EDF pdf function.
    llvm::FunctionType *m_type_edf_pdf_func;

    /// Return type of the EDF PDF function.
    llvm::Type *m_type_edf_pdf_data;


    /// The LLVM metadata kind ID for the BSDF parameter information attached to allocas.
    unsigned m_bsdf_param_metadata_id;

    /// The LLVM metadata kind ID for the EDF parameter information attached to allocas.
    unsigned m_edf_param_metadata_id;

    /// The internal state::set_normal(float3) function, only available for libbsdf.
    Internal_function *m_int_func_state_set_normal;

    /// The internal state::get_texture_results() function, only available for libbsdf.
    Internal_function *m_int_func_state_get_texture_results;

    /// The internal state::get_texture_results() function, only available for libbsdf.
    Internal_function *m_int_func_state_get_arg_block;

    /// The internal state::get_ro_data_segment() function, only available for libbsdf.
    Internal_function *m_int_func_state_get_ro_data_segment;

    /// The internal implementation of the state::object_id() function.
    Internal_function *m_int_func_state_object_id;

    /// The internal state::call_lambda_float(int) function, only available for libbsdf.
    Internal_function *m_int_func_state_call_lambda_float;

    /// The internal state::call_lambda_float3(int) function, only available for libbsdf.
    Internal_function *m_int_func_state_call_lambda_float3;

    /// The internal df::bsdf_measurement_resolution(int,int) function, only available for libbsdf.
    Internal_function *m_int_func_df_bsdf_measurement_resolution;

    /// The internal df::bsdf_measurement_evaluate(int,float2,float2,int) function,
    /// only available for libbsdf.
    Internal_function *m_int_func_df_bsdf_measurement_evaluate;

    /// The internal df::bsdf_measurement_sample(int,float2,float3,int) function,
    /// only available for libbsdf.
    Internal_function *m_int_func_df_bsdf_measurement_sample;

    /// The internal df::bsdf_measurement_pdf(int,float2,float2,int) function,
    /// only available for libbsdf.
    Internal_function *m_int_func_df_bsdf_measurement_pdf;

    /// The internal df::bsdf_measurement_albedos(int,float2) function,
    /// only available for libbsdf.
    Internal_function *m_int_func_df_bsdf_measurement_albedos;

    /// The internal df::light_profile_evaluate(int,float2) function,
    /// only available for libbsdf.
    Internal_function *m_int_func_df_light_profile_evaluate;

    /// The internal df::light_profile_sample(int,float3) function,
    /// only available for libbsdf.
    Internal_function *m_int_func_df_light_profile_sample;

    /// The internal df::light_profile_pdf(int,float2) function,
    /// only available for libbsdf.
    Internal_function *m_int_func_df_light_profile_pdf;
};

/// copysignf implementation for windows runtime.
float copysignf(const float x, const float y);

}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_LLVM_H

