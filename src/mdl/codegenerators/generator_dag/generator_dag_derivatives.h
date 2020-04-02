/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_DAG_DERIVATIVES_H
#define MDL_GENERATOR_DAG_DERIVATIVES_H 1

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include <mdl/compiler/compilercore/compilercore_bitset.h>
#include <mdl/compiler/compilercore/compilercore_function_instance.h>

namespace mi {
namespace mdl {

class DAG_node;
class DAG_call;
class ICall_name_resolver;
class IDefinition;
class IExpression;
class Lambda_function;

/// Class providing derivative analysis information for a function instance.
class Func_deriv_info
{
public:
    /// Constructor.
    Func_deriv_info(Memory_arena *arena, size_t num_params)
    : args_want_derivatives(*arena, num_params + 1)
    , vars_want_derivatives(arena)
    , exprs_want_derivatives(arena)
    , returns_derivatives(false)
    {
    }

    /// Returns true, if the given variable should be a derivative variable.
    ///
    /// \param var_def  the definition of the variable to check
    bool is_derivative_variable(IDefinition const *var_def) const
    {
        return vars_want_derivatives.find(var_def) != vars_want_derivatives.end();
    }

    /// Returns true, if for the given expression derivatives should be generated.
    ///
    /// \param expr  the expression to check
    bool is_derivative_expression(IExpression const *expr) const
    {
        if (IExpression_reference const *ref = as<IExpression_reference>(expr))
            return is_derivative_variable(ref->get_definition());
        return exprs_want_derivatives.find(expr) != exprs_want_derivatives.end();
    }


    /// Specifies whether the i-1-th argument wants to have derivatives.
    /// Index 0 is used to determine, whether any derivatives are wanted at all.
    Arena_Bitset args_want_derivatives;

    typedef Arena_ptr_hash_set<IDefinition const>::Type Definition_set;

    /// Specifies which variables should have a derivative type.
    Definition_set vars_want_derivatives;

    typedef Arena_ptr_hash_set<IExpression const>::Type Expression_set;

    /// Specifies for which expressions derivatives should be generated.
    Expression_set exprs_want_derivatives;

    /// Specifies whether the function instances returns derivatives.
    bool returns_derivatives;
};

/// Class providing derivative analysis information for DAG nodes and function instances.
class Derivative_infos
{
public:
    Derivative_infos(
        IAllocator *alloc)
    : m_alloc(alloc)
    , m_arena(alloc)
    , m_arena_builder(m_arena)
    , m_resolver(NULL)
    , m_deriv_info_dag_map(alloc)
    , m_deriv_func_inst_map(alloc)
    {
    }

    /// Set the call name resolver.
    void set_call_name_resolver(ICall_name_resolver const *resolver)
    {
        m_resolver = resolver;
    }

    /// Retrieve derivative infos for a function instance (ignoring array instantiation).
    Func_deriv_info const *get_function_derivative_infos(
        Function_instance const &func_inst) const;

    /// Retrieve derivative infos for a function instance (ignoring array instantiation)
    /// or calculate it for unknown semantic functions, if it is not available, yet.
    Func_deriv_info *get_or_calc_function_derivative_infos(
        Module const *module,
        Function_instance const &func_inst);

    /// Return true if the given DAG node should calculate derivatives.
    bool should_calc_derivatives(DAG_node const *node) const;

    /// Mark the given DAG node as a node for which derivatives should be calculated.
    void mark_calc_derivatives(DAG_node const *node);

    /// Allocate a Func_deriv_info object owned by this Derivative_infos object.
    ///
    /// \param num_params  the number of parameters of the function
    Func_deriv_info *alloc_function_derivative_infos(size_t num_params) const;

    /// Determine for which arguments of a call derivatives are needed.
    ///
    /// \param call              the DAG call to check
    /// \param want_derivatives  if true, the call should return derivatives
    ///
    /// \returns a bitset with the first bit stating whether any derivatives are needed for
    ///     the arguments and one bot per argument specifying whether this argument needs
    ///     derivatives.
    Bitset call_wants_arg_derivatives(DAG_call const *call, bool want_derivatives);

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The memory arena.
    Memory_arena m_arena;

    /// The arena builder.
    mutable Arena_builder m_arena_builder;

    /// The module resolver.
    ICall_name_resolver const *m_resolver;

    typedef ptr_hash_map<DAG_node const, bool>::Type Deriv_info_dag_map;

    /// Specifies, for which DAG nodes derivatives should be generated.
    Deriv_info_dag_map m_deriv_info_dag_map;

    typedef hash_map<
        Function_instance,
        Func_deriv_info *,
        Function_instance::Hash</*ignore_array_instance=*/ true>,
        Function_instance::Equal</*ignore_array_instance=*/ true> >::Type Deriv_func_inst_map;

    /// A hash map for retrieving derivative information for function instances, ignoring array
    /// instantiation.
    /// get_function_derivative_infos() may insert new entries for definitions with known
    /// semantics, because for example sin(float3) internally uses sin(float) which is not
    /// visible on DAG/AST level.
    mutable Deriv_func_inst_map m_deriv_func_inst_map;
};


/// Helper class to build a new DAG with derivative types where needed.
class Deriv_DAG_builder
{
public:
    /// Constructor.
    ///
    /// \param alloc        the allocator
    /// \param lambda       the lambda which will own the new DAG nodes
    /// \param deriv_infos  the derivative information
    Deriv_DAG_builder(
        IAllocator *alloc,
        Lambda_function &lambda,
        Derivative_infos &deriv_infos);

    /// Gets or creates an MDL derivative type for a given type.
    ///
    /// \param type  the MDL type to get the MDL derivative type for
    IType_struct const *get_deriv_type(IType const *type);

    /// Rebuild an expression with derivative information applied.
    /// This results in a DAG with derivative types and calls returning derivatives marked
    /// with a '#' prefix in the call name, if derivatives are required for this node.
    ///
    /// \param expr              the expression to rebuild
    /// \param want_derivatives  if true, the rebuilt node should return derivatives
    DAG_node const *rebuild(DAG_node const *expr, bool want_derivatives);

    /// Creates a zero value for use a dx or dy component of a dual.
    ///
    /// \param type  the type of the dual component
    IValue const *create_dual_comp_zero(IType const *type);

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The lambda function used to create the DAG nodes and types.
    Lambda_function &m_lambda;

    /// The value factory of the lambda function.
    Value_factory &m_vf;

    /// The type factory of the lambda function.
    Type_factory &m_tf;

    /// The derivative infos.
    Derivative_infos &m_deriv_infos;

    typedef ptr_hash_map<IType const, IType_struct *>::Type Deriv_type_map;

    /// Map from MDL types to their derivative type versions.
    Deriv_type_map m_deriv_type_map;

    typedef ptr_hash_map<DAG_node const, DAG_node const *>::Type Result_cache;

    /// Map from DAG node pointers with want_derivatives encoded in lowest bit to result a DAG node.
    Result_cache m_result_cache;

    /// ID to make type names unique for this builder.
    unsigned m_name_id;

    /// Symbols for the derivative types.
    ISymbol const *sym_val, *sym_dx, *sym_dy;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_DAG_DERIVATIVES_H
