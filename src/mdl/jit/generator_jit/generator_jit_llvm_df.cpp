/***************************************************************************************************
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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
 **************************************************************************************************/
/// \file

#include "pch.h"

#include <algorithm>

#include <mi/mdl/mdl_generated_dag.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Linker/Linker.h>

#include "mdl/compiler/compilercore/compilercore_errors.h"
#include "mdl/compiler/compilercore/compilercore_visitor.h"
#include "mdl/codegenerators/generator_dag/generator_dag_lambda_function.h"
#include "mdl/codegenerators/generator_dag/generator_dag_tools.h"
#include "mdl/codegenerators/generator_dag/generator_dag_walker.h"

#include "generator_jit_llvm.h"


#define DEBUG_TYPE "df_instantiation"

//#define DEBUG_NEW_SCHEDULER   // dumps the material instance graphs and prints scheduler infos

#define DEBUG_INIT_LOOP_SCHEDULER 0
#if DEBUG_INIT_LOOP_SCHEDULER
#include <fstream>
#include <iostream>
#include <iomanip>
#include <llvm/Support/FileSystem.h>
#endif

#ifdef DEBUG_NEW_SCHEDULER
#include <chrono>
#endif

// as we access the parameters by index, add some extra checks to ensure the parameter names
// are correct
#define CHECK_PARAM_NAME(node, idx, name) \
    MDL_ASSERT(strcmp(node->get_parameter_name(idx), name) == 0 && "parameter name mismatch")

namespace mi {
namespace mdl {

template<IDefinition::Semantics sema>
inline bool is(DAG_call const *call) { return call->get_semantic() == sema; }

template<IExpression::Operator op>
inline bool is(DAG_call const *call) { return call->get_semantic() == operator_to_semantic(op); }

/// Interface to retrieve target depended type properties.
class ITarget_type_properties
{
public:
    /// Return the storage size of a type in bytes.
    ///
    /// Returns the maximum number of bytes that may be overwritten by storing the specified type.
    /// If type is a scalable vector type, the scalable property will be set and the runtime size
    /// will be a positive integer multiple of the base size.
    /// For example, returns 12 for float3.
    ///
    /// \param type  the type
    /// \return      the storage size in bytes
    virtual size_t get_store_size(IType const *type) const = 0;

    /// Return the allocation size of a type in bytes.
    ///
    /// Returns the offset in bytes between successive objects of the specified type, including
    /// alignment padding.
    /// If type is a scalable vector type, the scalable property will be set and the runtime size
    /// will be a positive integer multiple of the base size.
    /// This is the amount that is allocatedfor this type. For example,
    /// returns 12 or 16 for float3, depending on alignment.
    ///
    /// \param type  the type
    /// \return      the allocation size in bytes
    virtual size_t get_alloc_size(IType const *type) const = 0;

    /// Returns the minimum ABI-required alignment for the specified type.
    ///
    /// \param type  the type
    /// \return      the ABI alignment in bytes
    virtual size_t get_ABI_alignment(IType const *type) const = 0;

    /// Return the target type of a type.
    ///
    /// \param type  the type
    /// \return      the target type
    virtual void *get_target_type(IType const *type) const = 0;

    /// Return the target type of a type.
    ///
    /// \param type  the type
    /// \return      the target type
    template<typename T>
    T *get_target_type(IType const *type) const
    {
        return static_cast<T *>(get_target_type(type));
    }
};

/// Using the schedule entries as input, update all write_texture evaluations with the correct
/// texture result type indices. This is required because the texture result type is created
/// after the evaluations are initialized during loop schedule creation.
void Loop_schedule::update_texres_indices(mi::mdl::vector<Schedule_entry>::Type &entries) {
    mi::mdl::map<size_t, int>::Type index_map(m_alloc);
    for (auto &e : entries) {
        if (e.texture_result_offset != ~0) {
            index_map.insert({ e.texture_result_offset, e.texture_result_index });
        }
    }
    for (auto &ee : evaluations) {
        switch (ee.kind) {
        case Evaluation::Kind::EK_WRITE_TEXTURE:
            ee.texture_result_index = index_map[ee.texture_result_offset];
            break;
        default:
            break;
        }
    }
}

void Loop_schedule::print_debug_node_set(std::ostream &outs, Node_set &s) {
#if DEBUG_INIT_LOOP_SCHEDULER
    Loop_schedule::Node_ptr_vector vec = s.sorted();
    print_debug_node_ptr_vector(outs, vec);
#endif
}

void Loop_schedule::print_debug_node_ptr_vector(std::ostream &outs, Node_ptr_vector &s) {
#if DEBUG_INIT_LOOP_SCHEDULER
    outs << s.size() << " {";
    int c = 0;
    for (auto &n : s) {
        if (c++ > 0) {
            outs << ", ";
        }
        outs << n->get_id();
    }
    outs << "}";
#endif
}

void Loop_schedule::Node_set::insert(DAG_node const *node) {
    m_nodes.insert(node);
}

void Loop_schedule::Node_set::intersect_with(Node_set const &other) {
    Node_ptr_set new_nodes(m_alloc);

    for (auto n : other.m_nodes) {
        if (m_nodes.find(n) != m_nodes.end()) {
            new_nodes.insert(n);
        }
    }
    std::swap(m_nodes, new_nodes);
}

void Loop_schedule::Node_set::union_with(Node_set const &other) {
    for (auto n : other.m_nodes) {
        m_nodes.insert(n);
    }
}

void Loop_schedule::Node_set::erase(DAG_node const *node) {
    m_nodes.erase(node);
}

void Loop_schedule::Node_set::erase(Node_set const &other) {
    for (auto n : other.m_nodes) {
        m_nodes.erase(n);
    }
}

bool Loop_schedule::Node_set::contains(DAG_node const *node) const {
    return m_nodes.find(node) != m_nodes.end();
}

Loop_schedule::Node_ptr_vector Loop_schedule::Node_set::sorted() const {
    Node_ptr_vector ret(m_alloc);
    ret.insert(ret.begin(), m_nodes.begin(), m_nodes.end());
    std::sort(ret.begin(), ret.end(), [](DAG_node const *a, DAG_node const *b) {
        return a->get_id() < b->get_id();
        });
    return ret;
}

namespace {

/// Helper structure collecting information about DAG nodes.
struct Node_info {
    /// True if the value of this node depends on the evaluation state.
    bool is_eval_state_dependent;

    /// Visit counts depending on the evaluation state.
    unsigned count[Distribution_function::ES_LAST + 1];

    /// Local cost of the node without costs of dependencies.
    unsigned local_cost;

    /// Size of the node result.
    unsigned result_size;

    /// ID to differentiate between walks
    unsigned last_walk_id;

    /// Second ID to differentiate between walks, used when walks are eval state aware
    unsigned last_walk_id_2;

    /// Constructor.
    Node_info()
        : is_eval_state_dependent(false)
        , local_cost(0)
        , result_size(~0)
        , last_walk_id(0)
        , last_walk_id_2(0)
    {
        for (unsigned i = 0; i <= Distribution_function::ES_LAST; ++i) {
            count[i] = 0;
        }
    }

    /// Return true, if this node has already been visited during the given walk.
    bool already_visited(unsigned walk_id) const {
        return last_walk_id == walk_id;
    }

    /// Return true, if this node has already been visited during the given walk for the given
    /// evaluation state.
    bool already_visited(unsigned walk_id, Distribution_function::Eval_state eval_state) const {
        if (!is_eval_state_dependent || eval_state == Distribution_function::ES_BEGIN_STATE) {
            return last_walk_id == walk_id;
        } else {
            return last_walk_id_2 == walk_id;
        }
    }

    /// Mark the node as visited in this walk.
    void mark_visited(unsigned walk_id) {
        last_walk_id = walk_id;
    }

    /// Mark the node as visited in this walk for the given evaluation state.
    void mark_visited(unsigned walk_id, Distribution_function::Eval_state eval_state) {
        if (!is_eval_state_dependent || eval_state == Distribution_function::ES_BEGIN_STATE) {
            last_walk_id = walk_id;
        } else {
            last_walk_id_2 = walk_id;
        }
    }

    /// Increment the counter for the according evaluation state, if the node has not been
    /// visited in this walk, yet, mark as visited in this walk and return the new counter.
    unsigned inc_count(
        Distribution_function::Eval_state eval_state,
        unsigned   walk_id)
    {
        // only increment, if this node has not been seen in this walk, yet
        unsigned inc_val = already_visited(walk_id) ? 0 : 1;
        mark_visited(walk_id);

        // If not state dependent, BEGIN_STATE count is always used.
        if (!is_eval_state_dependent) {
            eval_state = Distribution_function::ES_BEGIN_STATE;
        }

        return count[eval_state] += inc_val;
    }

    /// Get the counter for an evaluation state.
    unsigned get_count(Distribution_function::Eval_state eval_state) const
    {
        // If not state dependent, BEGIN_STATE count is always used.
        if (!is_eval_state_dependent) {
            eval_state = Distribution_function::ES_BEGIN_STATE;
        }
        return count[eval_state];
    }

    /// Get or calculate the result size of this node (store size).
    ///
    /// \param node                    the node of this node info
    /// \param target_type_properties  the target type properties
    unsigned get_result_size(
        DAG_node const                 *node,
        ITarget_type_properties const  &target_type_properties)
    {
        if (result_size != ~0) {
            return result_size;
        }

        return unsigned(target_type_properties.get_store_size(node->get_type()));
    }
};

typedef ptr_hash_map<DAG_node const, Node_info>::Type Node_info_map;

typedef ptr_hash_map<mi::mdl::IDefinition const, unsigned int>::Type Function_cost_map;

/// Helper class managing configurable cost of MDL code.
class Cost_provider
{
public:
    enum Cost_kind
    {
        CK_INVALID = 0,                   ///< Invalid cost kind
        CK_EXPRESSION,                    ///< Cost of an expression
        CK_STATEMENT,                     ///< Cost of a statement
        CK_CALL,                          ///< Cost of any other call (includes operators)

        CK_CALL_RES_ISVALID,              ///< Cost of calls to resource isvalid functions

        CK_CALL_TEX_INFO,                 ///< Cost of calls to texture info functions
        CK_CALL_TEX_LOOKUP,               ///< Cost of calls to texture lookup functions
        CK_CALL_TEX_TEXEL,                ///< Cost of calls to texture texel functions

        CK_CALL_LIGHT_PROFILE_INFO,       ///< Cost of calls to light profile access functions

        CK_CALL_STATE_ACCESS,             ///< Cost of calls to state access functions

        CK_CALL_STATE_TRANSFORM,          ///< Cost of calls to state::transform*
        CK_CALL_STATE_TRANSFORM_SCALE,    ///< Cost of calls to state::transform_scale

        CK_CALL_MATH_ABS,                 ///< Cost of calls to math::abs
        CK_CALL_MATH_ACOS,                ///< Cost of calls to math::acos
        CK_CALL_MATH_ASIN,                ///< Cost of calls to math::asin
        CK_CALL_MATH_ATAN,                ///< Cost of calls to math::atan
        CK_CALL_MATH_ATAN2,               ///< Cost of calls to math::atan2
        CK_CALL_MATH_AVERAGE,             ///< Cost of calls to math::average
        CK_CALL_MATH_BLACKBODY,           ///< Cost of calls to math::blackbody
        CK_CALL_MATH_COS,                 ///< Cost of calls to math::cos
        CK_CALL_MATH_COSH,                ///< Cost of calls to math::cosh
        CK_CALL_MATH_CROSS,               ///< Cost of calls to math::cross
        CK_CALL_MATH_DISTANCE,            ///< Cost of calls to math::distance
        CK_CALL_MATH_DOT,                 ///< Cost of calls to math::dot
        CK_CALL_MATH_EMISSION_COLOR_N,    ///< Cost of math::emission_color(float[<N>], float[<N>])
        CK_CALL_MATH_EXP,                 ///< Cost of calls to math::exp
        CK_CALL_MATH_EXP2,                ///< Cost of calls to math::exp2
        CK_CALL_MATH_FMOD,                ///< Cost of calls to math::fmod
        CK_CALL_MATH_FRAC,                ///< Cost of calls to math::frac
        CK_CALL_MATH_LENGTH,              ///< Cost of calls to math::length
        CK_CALL_MATH_LERP,                ///< Cost of calls to math::lerp
        CK_CALL_MATH_LOG,                 ///< Cost of calls to math::log
        CK_CALL_MATH_LOG2,                ///< Cost of calls to math::log2
        CK_CALL_MATH_LOG10,               ///< Cost of calls to math::log10
        CK_CALL_MATH_LUMINANCE,           ///< Cost of calls to math::luminance
        CK_CALL_MATH_NORMALIZE,           ///< Cost of calls to math::normalize
        CK_CALL_MATH_POW,                 ///< Cost of calls to math::pow
        CK_CALL_MATH_RSQRT,               ///< Cost of calls to math::rsqrt
        CK_CALL_MATH_SIGN,                ///< Cost of calls to math::sign
        CK_CALL_MATH_SIN,                 ///< Cost of calls to math::sin
        CK_CALL_MATH_SINCOS,              ///< Cost of calls to math::sincos
        CK_CALL_MATH_SINH,                ///< Cost of calls to math::sinh
        CK_CALL_MATH_SMOOTHSTEP,          ///< Cost of calls to math::smoothstep
        CK_CALL_MATH_SQRT,                ///< Cost of calls to math::sqrt
        CK_CALL_MATH_TAN,                 ///< Cost of calls to math::tan
        CK_CALL_MATH_TANH,                ///< Cost of calls to math::tanh
        CK_CALL_MATH_TRANSPOSE,           ///< Cost of calls to math::transpose

        CK_CALL_DAG_SPECTRAL_CONVERSION,  ///< Cost of calls to spectral conversion functions

        CK_ACCESS_TEX_RESULT_PER_FLOAT,   ///< Costs of accessing a texture result (per float)

        CK_MIN_STORE_RESULT_COST,         ///< Minimum cost needed for allowing to store the result
                                          ///< in the texture results or the local lambda results

        CK_NUM_KINDS
    };

private:
    /// Helper class to calculate the costs of a function.
    class Function_cost_calculator : public Module_visitor
    {
    private:
        /// Constructor.
        ///
        /// \param cost_provider  the cost provider
        /// \param module         the owner module of function to process
        Function_cost_calculator(
            Cost_provider &cost_provider,
            Module const *module)
        : m_cost_provider(cost_provider)
        , m_module(module)
        , m_cost(0u)
        {
        }

    public:
        /// Calculates the cost for the given definition.
        ///
        /// \param cost_provider  the cost provider
        /// \param module         the owner module of the definition
        /// \param def            the definition for which the costs shall be calculated
        static unsigned int get_cost(
            Cost_provider &cost_provider,
            Module const *module,
            IDefinition const *def)
        {
            Module const *owner = module;
            if (def->get_property(IDefinition::DP_IS_IMPORTED)) {
                def = module->get_original_definition(def, owner);
            }

            mi::base::Handle<Module const> h_owner(owner, mi::base::DUP_INTERFACE);
            def = skip_presets(def, h_owner);

            Function_cost_calculator calculator(cost_provider, owner);

            mi::mdl::IDeclaration_function const *func_decl =
                cast<mi::mdl::IDeclaration_function>(def->get_declaration());

            IStatement const *stmt = func_decl->get_body();
            calculator.visit(stmt);

            return calculator.m_cost;
        }

        /// Fallback pre_visit function for non implemented expression types.
        bool pre_visit(IExpression *expr) MDL_FINAL
        {
            // add generic expression costs
            // TODO: should be per atomic (float3 -> + 3*COST_EXPRESSION)
            m_cost += m_cost_provider.get_cost(CK_EXPRESSION);
            return true;
        }

        /// Pre-visitor for a literal expression.
        bool pre_visit(IExpression_literal *expr) MDL_FINAL
        {
            // no costs to be added
            return true;
        }

        /// Pre-visitor for a reference expression.
        bool pre_visit(IExpression_reference *expr) MDL_FINAL
        {
            // no costs to be added
            return true;
        }

        /// Pre-visitor for a binary expression.
        bool pre_visit(IExpression_binary *expr) MDL_FINAL
        {
            // treat select expressions as free
            if (expr->get_operator() == IExpression_binary::OK_SELECT) {
                return true;
            }

            // fallback to generic expression case
            return pre_visit(static_cast<IExpression *>(expr));
        }

        /// Pre-visitor for a call expression.
        bool pre_visit(IExpression_call *call) MDL_FINAL
        {
            IExpression_reference const *ref = cast<IExpression_reference>(call->get_reference());
            if (ref->is_array_constructor()) {
                m_cost += m_cost_provider.get_cost(CK_EXPRESSION);
                return true;
            }

            m_cost += m_cost_provider.get_def_cost(m_module, ref->get_definition());

            return true;
        }

        /// Pre-visitor for a let expression.
        bool pre_visit(IExpression_let *expr) MDL_FINAL
        {
            // no costs to be added
            return true;
        }

        /// Fallback pre_visit function for non implemented statement types.
        bool pre_visit(IStatement *stmt) MDL_FINAL
        {
            // add generic statement costs
            m_cost += m_cost_provider.get_cost(CK_STATEMENT);
            return true;
        }

        /// Pre-visitor for a compound statement.
        bool pre_visit(IStatement_compound *expr) MDL_FINAL
        {
            // no costs to be added
            return true;
        }

        /// Pre-visitor for a declaration statement.
        bool pre_visit(IStatement_declaration *expr) MDL_FINAL
        {
            // no costs to be added, expect the variable to be assigned to a register
            return true;
        }

        /// Pre-visitor for a expression statement.
        bool pre_visit(IStatement_expression *expr) MDL_FINAL
        {
            // no costs to be added (will be handled by the expression itself)
            return true;
        }

    private:
        /// The cost provider.
        Cost_provider &m_cost_provider;

        /// The current MDL module.
        Module const *m_module;

        /// The current aggregated cost.
        unsigned int m_cost;
    };

public:
    /// Constructor.
    ///
    /// \param alloc                   the allocator
    /// \param resolver                the call name resolver for processing costs of user-defined functions
    /// \param target_language         the target language for which code is generated
    /// \param target_type_properties  the target type properties
    Cost_provider(
        IAllocator                       *alloc,
        ICall_name_resolver const        *resolver,
        ICode_generator::Target_language  target_language,
        ITarget_type_properties const    &target_type_properties)
    : m_resolver(resolver)
    , m_func_cost_map(alloc)
    , m_target_language(target_language)
    , m_target_type_properties(target_type_properties)
    {
        // costs correspond to number of SASS instructions determined from dumps of the DXR example

        m_costs[CK_INVALID] = 0;
        m_costs[CK_EXPRESSION] = 1;
        m_costs[CK_STATEMENT] = 1;
        m_costs[CK_CALL] = 1;

        m_costs[CK_CALL_RES_ISVALID] = 1;         // DXR example runtime cost

        m_costs[CK_CALL_TEX_INFO] = 55;           // DXR example runtime cost
        m_costs[CK_CALL_TEX_LOOKUP] = 99;         // DXR example runtime cost
        m_costs[CK_CALL_TEX_TEXEL] = 68;          // DXR example runtime cost

        m_costs[CK_CALL_LIGHT_PROFILE_INFO] = 6;  // DXR example runtime cost

        // direct state access is for free for most fields
        m_costs[CK_CALL_STATE_ACCESS] = 0;

        m_costs[CK_CALL_STATE_TRANSFORM] = 9;
        m_costs[CK_CALL_STATE_TRANSFORM_SCALE] = 15;

        m_costs[CK_CALL_MATH_ABS] = 0;            // abs is free
        m_costs[CK_CALL_MATH_ACOS] = 29;
        m_costs[CK_CALL_MATH_ASIN] = 29;
        m_costs[CK_CALL_MATH_ATAN] = 20;
        m_costs[CK_CALL_MATH_ATAN2] = 33;
        m_costs[CK_CALL_MATH_AVERAGE] = 2;
        m_costs[CK_CALL_MATH_BLACKBODY] = 32;
        m_costs[CK_CALL_MATH_COS] = 2;
        m_costs[CK_CALL_MATH_COSH] = 18;
        m_costs[CK_CALL_MATH_CROSS] = 6;
        m_costs[CK_CALL_MATH_DISTANCE] = 7;
        m_costs[CK_CALL_MATH_DOT] = 3;

        // expensive function implemented in libmdlrt
        m_costs[CK_CALL_MATH_EMISSION_COLOR_N] = 500;   // 488 - 530 SASS instructions seen for N=11

        m_costs[CK_CALL_MATH_EXP] = 2;
        m_costs[CK_CALL_MATH_EXP2] = 1;
        m_costs[CK_CALL_MATH_FMOD] = 7;
        m_costs[CK_CALL_MATH_FRAC] = 2;
        m_costs[CK_CALL_MATH_LENGTH] = 4;
        m_costs[CK_CALL_MATH_LERP] = 2;
        m_costs[CK_CALL_MATH_LOG] = 2;
        m_costs[CK_CALL_MATH_LOG2] = 1;
        m_costs[CK_CALL_MATH_LOG10] = 2;
        m_costs[CK_CALL_MATH_LUMINANCE] = 3;
        m_costs[CK_CALL_MATH_LOG] = 2;
        m_costs[CK_CALL_MATH_NORMALIZE] = 7;
        m_costs[CK_CALL_MATH_POW] = 3;
        m_costs[CK_CALL_MATH_RSQRT] = 1;
        m_costs[CK_CALL_MATH_SIGN] = 10;
        m_costs[CK_CALL_MATH_SIN] = 2;
        m_costs[CK_CALL_MATH_SINCOS] = 3;
        m_costs[CK_CALL_MATH_SINH] = 32;
        m_costs[CK_CALL_MATH_SMOOTHSTEP] = 20;
        m_costs[CK_CALL_MATH_SQRT] = 1;
        m_costs[CK_CALL_MATH_TAN] = 5;
        m_costs[CK_CALL_MATH_TANH] = 1;

        m_costs[CK_CALL_DAG_SPECTRAL_CONVERSION] = 50;  // TODO SPECTRAL: extract value from dump when example is available

        // transpose just reorders registers, so it should be for free
        m_costs[CK_CALL_MATH_TRANSPOSE] = 0;

        m_costs[CK_ACCESS_TEX_RESULT_PER_FLOAT] = 1;

        m_costs[CK_MIN_STORE_RESULT_COST] = 10;
    }

    /// Get the cost for the given cost kind.
    unsigned int get_cost(Cost_kind cost_kind)
    {
        if (cost_kind < 0 || cost_kind >= CK_NUM_KINDS) {
            MDL_ASSERT(!"Invalid cost kind");
            return 0;
        }

        return m_costs[cost_kind];
    }

    /// Get the cost of reading a texture result of the given size.
    unsigned int get_texture_result_cost(unsigned size)
    {
        return (size + 3) / 4 * get_cost(Cost_provider::CK_ACCESS_TEX_RESULT_PER_FLOAT);
    }

    /// Get the cost of a parameter access.
    unsigned int get_parameter_cost(IType const *param_type)
    {
        unsigned param_size = unsigned(m_target_type_properties.get_store_size(param_type));

        // more exact calculation for HLSL based on DXR example
        if (m_target_language == ICode_generator::TL_HLSL) {
            return 3 + 2 * ((param_size + 3) / 4);
        } else {
            // rough estimation
            return 1 + (param_size + 3) / 4;
        }
    }

    /// Get the cost for a function call.
    ///
    /// \param sema               the semantic of the function call
    /// \param num_params         the number of parameters of the function call
    /// \param first_param_type   the MDL type of the first parameter or nullptr if not available
    /// \param second_param_type  the MDL type of the second parameter or nullptr if not available
    unsigned int get_function_cost(
        IDefinition::Semantics sema,
        int num_params,
        IType const *first_param_type,
        IType const *second_param_type)
    {
        unsigned int type_factor = 1;
        if (first_param_type != nullptr) {
            if (IType_vector const *vt = as<IType_vector>(first_param_type)) {
                type_factor = vt->get_size();
            }
        }

        if (semantic_is_operator(sema)) {
            switch (semantic_to_operator(sema)) {
            case IExpression::OK_MULTIPLY:
                if (is<IType_matrix>(first_param_type) || is<IType_matrix>(second_param_type)) {
                    IType_matrix const *mt = as<IType_matrix>(first_param_type);
                    if (mt == nullptr) {
                        mt = cast<IType_matrix>(second_param_type);
                    }

                    unsigned rows = unsigned(mt->get_element_type()->get_size());
                    unsigned cols = unsigned(mt->get_columns());

                    // expect rows times cols fmul/ffma operations
                    return rows * cols * get_cost(CK_EXPRESSION);
                }
                return get_cost(CK_CALL);

            default:
                break;
            }
            return get_cost(CK_CALL);
        }

        switch (sema) {
        case IDefinition::DS_COPY_CONSTRUCTOR:
            // treat as free, is used with every variable initialization
            return 0;

        case IDefinition::DS_INTRINSIC_STATE_POSITION:
        case IDefinition::DS_INTRINSIC_STATE_NORMAL:
        case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_NORMAL:
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_SPACE_MAX:
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_COORDINATE:
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U:
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_V:
        case IDefinition::DS_INTRINSIC_STATE_DIRECTION:
        case IDefinition::DS_INTRINSIC_STATE_ANIMATION_TIME:
        case IDefinition::DS_INTRINSIC_STATE_WAVELENGTH_BASE:
        case IDefinition::DS_INTRINSIC_STATE_METERS_PER_SCENE_UNIT:
        case IDefinition::DS_INTRINSIC_STATE_SCENE_UNITS_PER_METER:
        case IDefinition::DS_INTRINSIC_STATE_OBJECT_ID:
        case IDefinition::DS_INTRINSIC_STATE_WAVELENGTH_MIN:
        case IDefinition::DS_INTRINSIC_STATE_WAVELENGTH_MAX:
            return get_cost(CK_CALL_STATE_ACCESS);

        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
            return get_cost(CK_CALL_STATE_TRANSFORM);

        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
            return get_cost(CK_CALL_STATE_TRANSFORM_SCALE);

        case IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID:
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
        case IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
            return get_cost(CK_CALL_RES_ISVALID);

        case IDefinition::DS_INTRINSIC_TEX_WIDTH:
        case IDefinition::DS_INTRINSIC_TEX_HEIGHT:
        case IDefinition::DS_INTRINSIC_TEX_DEPTH:
        case IDefinition::DS_INTRINSIC_TEX_WIDTH_OFFSET:
        case IDefinition::DS_INTRINSIC_TEX_HEIGHT_OFFSET:
        case IDefinition::DS_INTRINSIC_TEX_DEPTH_OFFSET:
        case IDefinition::DS_INTRINSIC_TEX_FIRST_FRAME:
        case IDefinition::DS_INTRINSIC_TEX_LAST_FRAME:
            return get_cost(CK_CALL_TEX_INFO);

        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_COLOR:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT2:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT3:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT4:
            return get_cost(CK_CALL_TEX_LOOKUP);

        case IDefinition::DS_INTRINSIC_TEX_TEXEL_COLOR:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT2:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT3:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT4:
            return get_cost(CK_CALL_TEX_TEXEL);

        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_POWER:
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM:
            return get_cost(CK_CALL_LIGHT_PROFILE_INFO);

        case IDefinition::DS_INTRINSIC_MATH_ABS:
            return type_factor * get_cost(CK_CALL_MATH_ABS);

        case IDefinition::DS_INTRINSIC_MATH_ACOS:
            return type_factor * get_cost(CK_CALL_MATH_ACOS);

        case IDefinition::DS_INTRINSIC_MATH_ASIN:
            return type_factor * get_cost(CK_CALL_MATH_ASIN);

        case IDefinition::DS_INTRINSIC_MATH_ATAN:
            return type_factor * get_cost(CK_CALL_MATH_ATAN);

        case IDefinition::DS_INTRINSIC_MATH_ATAN2:
            return type_factor * get_cost(CK_CALL_MATH_ATAN2);

        case IDefinition::DS_INTRINSIC_MATH_AVERAGE:
            return type_factor * get_cost(CK_CALL_MATH_AVERAGE);

        case IDefinition::DS_INTRINSIC_MATH_BLACKBODY:
            return get_cost(CK_CALL_MATH_BLACKBODY);

        case IDefinition::DS_INTRINSIC_MATH_COS:
            return type_factor * get_cost(CK_CALL_MATH_COS);

        case IDefinition::DS_INTRINSIC_MATH_COSH:
            return type_factor * get_cost(CK_CALL_MATH_COSH);

        case IDefinition::DS_INTRINSIC_MATH_CROSS:
            return get_cost(CK_CALL_MATH_CROSS);

        case IDefinition::DS_INTRINSIC_MATH_DISTANCE:
            return get_cost(CK_CALL_MATH_DISTANCE);

        case IDefinition::DS_INTRINSIC_MATH_DOT:
            return get_cost(CK_CALL_MATH_DOT);

        case IDefinition::DS_INTRINSIC_MATH_EMISSION_COLOR:
            // check for expensive spectral variant
            if (num_params == 2) {
                return get_cost(CK_CALL_MATH_EMISSION_COLOR_N);
            } else {
                // simple variant for free, just returns the argument
                return 0;
            }

        case IDefinition::DS_INTRINSIC_MATH_EXP:
            return type_factor * get_cost(CK_CALL_MATH_EXP);

        case IDefinition::DS_INTRINSIC_MATH_EXP2:
            return type_factor * get_cost(CK_CALL_MATH_EXP2);

        case IDefinition::DS_INTRINSIC_MATH_FMOD:
            return type_factor * get_cost(CK_CALL_MATH_FMOD);

        case IDefinition::DS_INTRINSIC_MATH_FRAC:
            return type_factor * get_cost(CK_CALL_MATH_FRAC);

        case IDefinition::DS_INTRINSIC_MATH_LENGTH:
            return type_factor * get_cost(CK_CALL_MATH_LENGTH);

        case IDefinition::DS_INTRINSIC_MATH_LERP:
            return type_factor * get_cost(CK_CALL_MATH_LERP);

        case IDefinition::DS_INTRINSIC_MATH_LOG:
            return type_factor * get_cost(CK_CALL_MATH_LOG);

        case IDefinition::DS_INTRINSIC_MATH_LOG2:
            return type_factor * get_cost(CK_CALL_MATH_LOG2);

        case IDefinition::DS_INTRINSIC_MATH_LOG10:
            return type_factor * get_cost(CK_CALL_MATH_LOG10);

        case IDefinition::DS_INTRINSIC_MATH_LUMINANCE:
            return get_cost(CK_CALL_MATH_LUMINANCE);

        case IDefinition::DS_INTRINSIC_MATH_NORMALIZE:
            return get_cost(CK_CALL_MATH_NORMALIZE);

        case IDefinition::DS_INTRINSIC_MATH_POW:
            return type_factor * get_cost(CK_CALL_MATH_POW);

        case IDefinition::DS_INTRINSIC_MATH_RSQRT:
            return type_factor * get_cost(CK_CALL_MATH_RSQRT);

        case IDefinition::DS_INTRINSIC_MATH_SIGN:
            return type_factor * get_cost(CK_CALL_MATH_SIGN);

        case IDefinition::DS_INTRINSIC_MATH_SIN:
            return type_factor * get_cost(CK_CALL_MATH_SIN);

        case IDefinition::DS_INTRINSIC_MATH_SINCOS:
            return type_factor * get_cost(CK_CALL_MATH_SINCOS);

        case IDefinition::DS_INTRINSIC_MATH_SINH:
            return type_factor * get_cost(CK_CALL_MATH_SINH);

        case IDefinition::DS_INTRINSIC_MATH_SMOOTHSTEP:
            return type_factor * get_cost(CK_CALL_MATH_SMOOTHSTEP);

        case IDefinition::DS_INTRINSIC_MATH_SQRT:
            return type_factor * get_cost(CK_CALL_MATH_SQRT);

        case IDefinition::DS_INTRINSIC_MATH_TAN:
            return type_factor * get_cost(CK_CALL_MATH_TAN);

        case IDefinition::DS_INTRINSIC_MATH_TANH:
            return type_factor * get_cost(CK_CALL_MATH_TANH);

        case IDefinition::DS_INTRINSIC_MATH_TRANSPOSE:
            return get_cost(CK_CALL_MATH_TRANSPOSE);

        case IDefinition::DS_INTRINSIC_DAG_RGB_TO_SPECTRAL_IOR:
        case IDefinition::DS_INTRINSIC_DAG_RGB_TO_SPECTRAL_REFLECTANCE:
        case IDefinition::DS_INTRINSIC_DAG_RGB_TO_SPECTRAL_LUMINANCE:
        case IDefinition::DS_INTRINSIC_DAG_RGB_TO_SPECTRAL_VOLUME_COEFFICIENT:
            return get_cost(CK_CALL_DAG_SPECTRAL_CONVERSION);

        default:
            break;
        }

        return get_cost(CK_CALL);
    }

    /// Get the cost for the given DAG call.
    /// If the cost is not known, yet, it is calculated and stored in the given map.
    ///
    /// \param sema              the semantics of the function
    /// \param num_params        the number of parameters of the function
    /// \param first_param_type  the type of the first parameter
    unsigned int get_dag_call_cost(DAG_call const *call)
    {
        IDefinition::Semantics sema = call->get_semantic();

        // known function?
        if (sema != IDefinition::DS_UNKNOWN) {
            int n_args = call->get_argument_count();
            IType const *first_param_type = nullptr;
            IType const *second_param_type = nullptr;
            if (n_args > 0) {
                first_param_type = call->get_argument(0)->get_type();
                if (n_args > 1) {
                    second_param_type = call->get_argument(1)->get_type();
                }
            }
            return get_function_cost(
                sema, call->get_argument_count(), first_param_type, second_param_type);
        }

        char const *signature = call->get_name();
        if (signature[0] == '#') {
            // skip prefix for derivative variants
            ++signature;
        }
        mi::base::Handle<Module const> mod(
            impl_cast<Module>(m_resolver->get_owner_module(signature)));
        if (!mod) {
            // resolving failed, add cost for unknown call
            MDL_ASSERT(!"get_owner_module should not fail for user-defined function");
            return get_cost(CK_CALL);
        }

        Module const *module = mod.get();
        IDefinition const *def = module->find_signature(signature, /*only_exported=*/ false);
        if (def == nullptr) {
            // definition not found, add cost for unknown call
            MDL_ASSERT(!"find_signature should not fail for user-defined function");
            return get_cost(CK_CALL);
        }

        def = skip_presets(def, mod);
        if (def->get_kind() != IDefinition::DK_FUNCTION) {
            // not a function (maybe a constructor?)
            return get_cost(CK_CALL);
        }

        // check function cost map
        auto it = m_func_cost_map.find(def);
        if (it != m_func_cost_map.end()) {
            return it->second;
        }

        // function not seen, yet, so process it now

        unsigned cost = Function_cost_calculator::get_cost(*this, module, def);

        m_func_cost_map[def] = cost;

#ifdef DEBUG_NEW_SCHEDULER
        printf("cost(\"%s\") = %u\n", def->get_symbol()->get_name(), cost);
#endif

        return cost;
    }

    /// Get the cost for the given function definition.
    /// If the cost is not known, yet, it is calculated and stored in the given map.
    ///
    /// \param module  the module of the definition
    /// \param def     the function definition for which the costs shall be calculated
    unsigned int get_def_cost(Module const *module, IDefinition const *def)
    {
        IDefinition::Semantics sema = def->get_semantics();

        // known function?
        if (sema != IDefinition::DS_UNKNOWN) {
            IType_function const *func_type = cast<mi::mdl::IType_function>(def->get_type());
            int n_params = func_type->get_parameter_count();
            IType const *first_param_type = nullptr;
            IType const *second_param_type = nullptr;
            if (n_params > 0) {
                ISymbol const *param_name = nullptr;
                func_type->get_parameter(0, first_param_type, param_name);
                if (n_params > 1) {
                    func_type->get_parameter(1, second_param_type, param_name);
                }
            }
            return get_function_cost(sema, n_params, first_param_type, second_param_type);
        }

        // check function cost map
        auto it = m_func_cost_map.find(def);
        if (it != m_func_cost_map.end()) {
            return it->second;
        }

        // function not seen, yet, so process it now

        unsigned cost = Function_cost_calculator::get_cost(*this, module, def);

        m_func_cost_map[def] = cost;

#ifdef DEBUG_NEW_SCHEDULER
        printf("cost(\"%s\") = %u\n", def->get_symbol()->get_name(), cost);
#endif

        return cost;
    }

private:
    /// The call name resolver.
    ICall_name_resolver const *m_resolver;

    /// Map from function definitions to cost.
    Function_cost_map m_func_cost_map;

    /// The target language for which code is generated.
    ICode_generator::Target_language m_target_language;

    /// The target type properties.
    ITarget_type_properties const &m_target_type_properties;

    /// The list of costs per cost kind.
    unsigned int m_costs[CK_NUM_KINDS];
};

/// Helper class to calculate the local costs of all nodes.
class Node_local_cost_calculator : public IDAG_ir_visitor
{
public:
    /// The constructor.
    ///
    /// \param cost_provider  the cost provider
    /// \param node_info_map  the node info map where the calculated costs will be stored
    Node_local_cost_calculator(
        Cost_provider &cost_provider,
        Node_info_map &node_info_map)
    : m_cost_provider(cost_provider)
    , m_node_info_map(node_info_map)
    {
    }

    /// Post-visit a Constant.
    ///
    /// \param cnst  the constant that is visited
    void visit(DAG_constant *cnst) MDL_FINAL
    {
        // do nothing. costs of resources are handled by the function calls
    }

    /// Post-visit a Temporary.
    ///
    /// \param tmp  the temporary that is visited
    void visit(DAG_temporary *tmp) MDL_FINAL
    {
        // do nothing, but should not happen here
        MDL_ASSERT(!"temporaries should not occur here");
    }

    /// Post-visit a call.
    ///
    /// \param call  the call that is visited
    void visit(DAG_call *call) MDL_FINAL
    {
        Node_info &info = m_node_info_map[call];

        info.local_cost += m_cost_provider.get_dag_call_cost(call);
    }

    /// Post-visit a Parameter.
    ///
    /// \param param  the parameter that is visited
    void visit(DAG_parameter *param) MDL_FINAL
    {
        Node_info &info = m_node_info_map[param];
        info.local_cost = m_cost_provider.get_parameter_cost(param->get_type());
    }

    /// Post-visit a temporary initializer.
    ///
    /// \param index  the index of the temporary
    /// \param init   the initializer expression of this temporary
    void visit(int index, DAG_node *init) MDL_FINAL
    {
        // should never be called
        MDL_ASSERT(!"temporary initializers should not occur here");
    }

    /// Calculate the cost of evaluating the given node.
    ///
    /// \param alloc          the allocator
    /// \param cost_provider  the cost provider
    /// \param node_info_map  the node info map
    /// \param root           the DAG node for which the cost should be calculated
    static void calc_cost(
        IAllocator     *alloc,
        Cost_provider  &cost_provider,
        Node_info_map  &node_info_map,
        DAG_node const *root)
    {
        DAG_ir_walker walker(alloc);
        Node_local_cost_calculator calculator(cost_provider, node_info_map);

        walker.walk_node(const_cast<DAG_node *>(root), &calculator);
    }

private:
    /// The cost provider
    Cost_provider &m_cost_provider;

    /// Map from DAG nodes to node infos.
    Node_info_map &m_node_info_map;
};

/// Helper structure for managing candidates for texture results.
/// Used to cut the graph into parts.
struct Result_candidate_info {
    typedef ptr_hash_set<Result_candidate_info>::Type Candidate_set;

    /// The DAG node for this result.
    DAG_node const *node;

    /// Bit flags representing the special kinds of this node (1 << kind).
    unsigned special_kinds;

    /// True, if the result was explicitly requested by the renderer or implicitly by the
    /// used distribution functions.
    bool is_requested;

    /// True, if the result is already scheduled and thus already available in mdl_init.
    bool is_scheduled;

    /// True, if the result depends on the evaluation state, i.e. does change when evaluating
    /// before or after evaluating geometry normal.
    bool is_eval_state_dependent;

    /// The evaluation state, when this result is to be calculated,
    /// if \ref is_eval_state_dependent is true.
    Distribution_function::Eval_state eval_state;

    /// Factor to be applied to the cost, if this is not precalculated in mdl_init
    /// because of multiple usages.
    unsigned usage_factor;

    /// Cost for calculating this DAG node without dependencies on other result candidates.
    unsigned local_cost;

    /// Cost for calculating this DAG node without dependencies on other result candidates
    /// without dynamic modifications due to flattening or storing in texture results.
    unsigned orig_local_cost;

    /// Cost for calculating this DAG node including the cost of dependencies on other result
    /// candidates.
    unsigned total_cost;

    /// The size in bytes of the result.
    unsigned size;

    /// The alignment required for the result.
    unsigned align;

    /// The texture result offset, if one has been assigned, or ~0.
    size_t texture_result_offset;

    /// Index of the texture result in the texture result struct.
    int texture_result_index;

    /// Version number for this info. When ever the graph is updated, the current version is
    /// increased, invalidating dependencies and total_cost.
    unsigned info_version;

    /// Number of direct usages. Will be updated by update_info and thus depends on
    /// whether texture results hide usage.
    unsigned direct_usage_count;

    /// List of direct users of this result candidate.
    mi::mdl::vector<Result_candidate_info *>::Type direct_users_list;

    /// List of direct dependencies on other result candidates.
    mi::mdl::vector<Result_candidate_info *>::Type direct_dependencies_list;

    /// Set of direct dependencies on other result candidates.
    Candidate_set direct_dependencies_set;

    /// Transitive set of dependencies on other result candidates.
    Candidate_set transitive_dependencies;

    /// Constructor.
    ///
    /// \param alloc                    the allocator
    /// \param node                     the DAG node for this result
    /// \param size                     the size in bytes of this result
    /// \param align                    the alignment required for this result
    /// \param is_requested             whether the result was directly or indirectly requested
    ///                                 by the renderer
    /// \param is_eval_state_dependent  whether the result depends on the evaluation state
    /// \param eval_state               the evaluation state when this result is calculated
    /// \param usage_factor             factor to be applied to the cost
    Result_candidate_info(
        mi::mdl::IAllocator               *alloc,
        DAG_node const                    *node,
        unsigned                          size,
        unsigned                          align,
        bool                              is_requested,
        bool                              is_eval_state_dependent,
        Distribution_function::Eval_state eval_state,
        unsigned                          usage_factor)
    : node(node)
    , special_kinds()
    , is_requested(is_requested)
    , is_scheduled(false)
    , is_eval_state_dependent(is_eval_state_dependent)
    , eval_state(eval_state)
    , usage_factor(usage_factor)
    , local_cost(0)
    , orig_local_cost(0)
    , total_cost(0)
    , size(size)
    , align(align)
    , texture_result_offset(~0)
    , texture_result_index(-1)
    , info_version(0)
    , direct_usage_count(0)
    , direct_users_list(alloc)
    , direct_dependencies_list(alloc)
    , direct_dependencies_set(alloc)
    , transitive_dependencies(alloc)
    {
    }

    /// Add the given special kind to this result candidate.
    void add_special_kind(Distribution_function::Special_kind special_kind)
    {
        MDL_ASSERT(special_kind != Distribution_function::SK_INVALID);

        special_kinds |= 1 << unsigned(special_kind);
    }

    /// Returns true, if this result candidate has the given special kind.
    bool has_special_kind(Distribution_function::Special_kind special_kind) const
    {
        MDL_ASSERT(special_kind != Distribution_function::SK_INVALID);

        return (special_kinds & (1 << unsigned(special_kind))) != 0;
    }

    /// Add another result candidate info as a direct dependency.
    ///
    /// \param dep  The other result candidate info, this result candidate depends on
    void add_direct_dependency(Result_candidate_info *dep)
    {
        direct_dependencies_set.insert(dep);
        direct_dependencies_list.push_back(dep);

        dep->direct_users_list.push_back(this);
    }

    /// Returns true, if the result is already stored in texture results or state normal,
    /// so the result is also available nearly for free outside of the init function.
    bool is_stored() const
    {
        return texture_result_offset != ~0 ||
            (is_scheduled &&
                has_special_kind(Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL));
    }

    /// Update transitive dependency and total cost info.
    /// Should be called, when ever the graph or the costs are changed.
    /// The direct usage count must be reset to zero before calling this function.
    void update_info(unsigned cur_graph_version, Cost_provider &cost_provider) {
        if (info_version == cur_graph_version) {
            return;
        }

        info_version = cur_graph_version;

        transitive_dependencies.clear();

        if (is_stored()) {
            if (texture_result_offset != ~0) {
                local_cost = cost_provider.get_texture_result_cost(size);
            } else {
                // stored in state normal
                local_cost = cost_provider.get_cost(Cost_provider::CK_CALL_STATE_ACCESS);
            }
            total_cost = local_cost;
            return;
        }

        total_cost = local_cost;

        // process all direct dependencies
        for (Result_candidate_info *dep : direct_dependencies_list) {
            ++dep->direct_usage_count;

            // not seen, yet?
            if (transitive_dependencies.insert(dep).second == true) {
                // recursively update the info of the direct dependency
                dep->update_info(cur_graph_version, cost_provider);
                total_cost += dep->local_cost;

                // add all transitive dependencies of direct dependency
                // note: we iterate over a pointer hash set, here, but that's OK
                for (Result_candidate_info *dep_dep : dep->transitive_dependencies) {
                    if (transitive_dependencies.insert(dep_dep).second == true) {
                        total_cost += dep_dep->local_cost;
                    }
                }
            }
        }
    }

    /// "Flattens" the graph by moving local costs up to the parent, if direct usage is 1
    /// for non-requested nodes.
    void flatten_graph(unsigned cur_graph_version) {
        if (info_version == cur_graph_version) {
            return;
        }

        info_version = cur_graph_version;

        if (is_stored()) {
            return;
        }

        for (Result_candidate_info *dep : direct_dependencies_list) {
            // recursively update direct dependencies first
            dep->flatten_graph(cur_graph_version);

            // note: don't move costs from candidates which are already stored,
            //       because the costs will always be restored during the update
            if (dep->direct_usage_count == 1 &&
                    !dep->is_requested &&
                    !is_stored() &&
                    dep->local_cost > 0) {
#ifdef DEBUG_NEW_SCHEDULER
                printf("Move %u cost from %u to %u\n",
                    dep->local_cost, unsigned(dep->node->get_id()), unsigned(node->get_id()));
#endif
                local_cost += dep->local_cost;
                dep->local_cost = 0;
            }
        }
    }
};


/// Implementation of the ITarget_type_properties interface for LLVM based types.
class LLVM_type_helper final : public ITarget_type_properties
{
    typedef ITarget_type_properties Base;
public:
    /// Constructor.
    ///
    /// \param data_layout  the LLVM data layout
    /// \param type_mapper  the type mapper
    LLVM_type_helper(
        llvm::DataLayout const *data_layout,
        Type_mapper            &type_mapper)
    : m_data_layout(data_layout)
    , m_type_mapper(type_mapper)
    {
    }

    /// Return the storage size of a type in bytes.
    size_t get_store_size(IType const *type) const final
    {
        llvm::Type *llvm_type = Base::get_target_type<llvm::Type>(type);
        return m_data_layout->getTypeStoreSize(llvm_type);
    }

    /// Return the allocation size of a type in bytes.
    size_t get_alloc_size(IType const *type) const final
    {
        llvm::Type *llvm_type = Base::get_target_type<llvm::Type>(type);
        return m_data_layout->getTypeAllocSize(llvm_type);
    }

    /// Return the ABI alignment of a type in bytes.
    size_t get_ABI_alignment(IType const *type) const final
    {
        llvm::Type *llvm_type = Base::get_target_type<llvm::Type>(type);
        return m_data_layout->getABITypeAlignment(llvm_type);
    }

    /// Return the target type of a type.
    ///
    /// \param type  the type
    /// \return      the target type
    void *get_target_type(IType const *type) const final
    {
        return m_type_mapper.lookup_type(
            m_type_mapper.get_llvm_context(), type);
    }

    /// Create the texture results type after scheduling was done.
    ///
    /// As a side effect, the texture result index fields in the schedule are adjusted to
    /// match the corresponding field indices in the LLVM type.
    ///
    /// \param alloc            the allocator
    /// \param llvm_context     the LLVM context
    /// \param results_schedule the list of result candidates scheduled for the init function
    ///
    /// \return                 the texture results type
    llvm::StructType *create_texture_results_type(
        IAllocator                                           *alloc,
        llvm::LLVMContext                                    &llvm_context,
        mi::mdl::vector<Schedule_entry>::Type                &schedule
    ) const
    {
        mi::mdl::vector<Schedule_entry *>::Type sorted_schedule_entries(
            alloc);
        for (Schedule_entry &entry : schedule) {
            sorted_schedule_entries.push_back(&entry);
        }
        std::sort(
            sorted_schedule_entries.begin(), sorted_schedule_entries.end(),
            [](Schedule_entry const *a, Schedule_entry const *b)
            {
                return a->texture_result_offset < b->texture_result_offset;
            }
        );
        mi::mdl::vector<llvm::Type *>::Type texture_result_types(alloc);
        texture_result_types.reserve(sorted_schedule_entries.size());
        int i = 0;
        for (Schedule_entry *schedule_entry : sorted_schedule_entries) {
            if (schedule_entry->texture_result_offset == ~0) {
                continue;
            }
            schedule_entry->texture_result_index = i++;
            llvm::Type *llvm_type = Base::get_target_type<llvm::Type>(
                schedule_entry->node->get_type());
            texture_result_types.push_back(llvm_type);
        }

        return llvm::StructType::create(
            llvm_context,
            texture_result_types,
            "struct.Texture_result_types",
            /*is_packed=*/ false);
    }

private:
    llvm::DataLayout const *m_data_layout;
    Type_mapper            &m_type_mapper;
};


#if DEBUG_INIT_LOOP_SCHEDULER
static int cntr = 0;
#endif

class Init_loop_scheduler {

public:
    Init_loop_scheduler(
        IAllocator                                                         *alloc,
        bool                                                               init_loop_enabled,
        bool                                                               target_is_structured_language,
        ICall_name_resolver const                                          *resolver,
        mi::mdl::vector<Schedule_entry>::Type const                        &schedule,
        mi::mdl::vector<std::pair<DAG_node const *, unsigned>>::Type const &sorted_nodes,
        Node_info_map const                                                &node_info_map,
        ITarget_type_properties const                                      &target_type_properties)
    : m_alloc(alloc)
    , m_resolver(resolver)
    , m_schedule(schedule)
    , m_sorted_nodes(sorted_nodes)
    , m_node_info_map(node_info_map)
    , m_target_type_properties(target_type_properties)
#if DEBUG_INIT_LOOP_SCHEDULER
    , m_dbg(std::string("mdl_init.debug-") + std::to_string(cntr++) + ".log")
#endif
    , m_target_is_structured_language(target_is_structured_language)
{
        auto handle_env_var = [](char const *env_name, int default_val, int mn, int mx) -> int {
            char const *env_val = std::getenv(env_name);
            if (!env_val) {
                return default_val;
            }
            char *endp = nullptr;
            long result = std::strtol(env_val, &endp, 10);
            if (!endp || *endp != '\0') {
                return default_val;
            }
            if (result < mn || result > mx) {
                return default_val;
            }
            return static_cast<int>(result);
            };

        // Take setting from backend option, can be overridden by environment variable.
        m_gen_schedule_loop = handle_env_var("MDL_INIT_LOOP_SCHEDULING", init_loop_enabled, 0, 1) != 0;

        m_gen_evaluate_sequentially = handle_env_var("MDL_INIT_LOOP_EVALUATE_SEQUENTIALLY", 0, 0, 1) != 0;
        m_gen_common_parameters = handle_env_var("MDL_INIT_LOOP_COMMON_PARAMETERS", 1, 0, 1) != 0;
        m_gen_parameter_offsets = handle_env_var("MDL_INIT_LOOP_PARAM_PARAMETERS", 1, 0, 1) != 0;
        m_gen_local_texres = handle_env_var("MDL_INIT_LOOP_LOCAL_TEXRES", 1, 0, 1) != 0;
        m_gen_reorder_arguments = handle_env_var("MDL_INIT_LOOP_REORDER_ARGUMENTS", 1, 0, 1) != 0;
        m_gen_expensive_function_cost_limit = handle_env_var("MDL_INIT_LOOP_EXP_FUNC_COST_LIMIT", 40, 1, 10000);
        m_gen_expensive_function_limit = handle_env_var("MDL_INIT_LOOP_EXP_FUNC_LIMIT", 20, 1, 1000);

        if (!m_target_is_structured_language) {
            m_gen_parameter_offsets = false;
            m_gen_local_texres = false;
        }

#if DEBUG_INIT_LOOP_SCHEDULER
        m_dbg << "init loop: generate loop: " << m_gen_schedule_loop << "\n";
        m_dbg << "init loop: generate sequentially: " << m_gen_evaluate_sequentially << "\n";
        m_dbg << "init loop: common parameters: " << m_gen_common_parameters << "\n";
        m_dbg << "init loop: parameter offsets: " << m_gen_parameter_offsets << "\n";
        m_dbg << "init loop: local texres: " << m_gen_local_texres << "\n";
        m_dbg << "init loop: reorder arguments: " << m_gen_reorder_arguments << "\n";
        m_dbg << "init loop: expensive function cost limit: " << m_gen_expensive_function_cost_limit << "\n";
        m_dbg << "init loop: expensive function limit: " << m_gen_expensive_function_limit << "\n";
#endif
    }

    void schedule(Loop_schedule &loop_schedule) {
        loop_schedule.schedule_loop = m_gen_schedule_loop;
        loop_schedule.evaluate_sequentially = m_gen_evaluate_sequentially;

        Loop_schedule::Node_set visited(m_alloc);
        Loop_schedule::Node_ptr_vector roots(m_alloc);
        Loop_schedule::Node_set expensive_nodes(m_alloc);
        Loop_schedule::Node_ptr_vector expensive_nodes_vec(m_alloc);

        { // Restrict scope for temporaries.

            // Calculate the expensive call nodes in the sub-graph by traversing the
            // sorted list of nodes. We first count the number of calls per function
            // within the sub-graph...
            Loop_schedule::Node_set exp_nodes(m_alloc);

            map<char const *, size_t>::Type exp_call_counters(m_alloc);
            map<char const *, size_t>::Type exp_call_costs(m_alloc);
            for (auto &sn : m_sorted_nodes) {
                // Quit looking for expensive nodes when cost is too low.
                if (sn.second < (size_t)m_gen_expensive_function_cost_limit) {
                    break;
                }
                // Only call nodes are considered expensive.
                if (sn.first->get_kind() != DAG_node::EK_CALL) {
                    continue;
                }
                exp_nodes.insert(sn.first);
                char const *name = cast<DAG_call>(sn.first)->get_name();
                exp_call_counters[name] += 1;
                exp_call_costs[name] = sn.second;
            }
            vector<std::pair<char const *, size_t>>::Type sort_vec(m_alloc);
            for (auto &name_cost : exp_call_costs) {
                if (exp_call_counters[name_cost.first] >= 2) {
                    sort_vec.push_back(name_cost);
                }
            }
            std::sort(sort_vec.begin(), sort_vec.end(),
                [](std::pair<char const *, size_t> &a, std::pair<char const *, size_t> &b) {
                    return a.second > b.second;
                });

            if (sort_vec.size() > m_gen_expensive_function_limit) {
                sort_vec.resize(m_gen_expensive_function_limit);
            }
#if DEBUG_INIT_LOOP_SCHEDULER
            m_dbg << "expensive call nodes:\n";
            for (auto const &p : sort_vec) {
                m_dbg << "  " << p.first << ", cost: " << p.second << ", count: " << exp_call_counters[p.first] << "\n";
            }
#endif

            for (auto &n : exp_nodes.nodes()) {
                char const *name = cast<DAG_call>(n)->get_name();
                if (std::find_if(sort_vec.begin(), sort_vec.end(), [&](std::pair<char const *, size_t> a) {
                    return !strcmp(a.first, name);
                    })
                    != sort_vec.end())
                {
                    expensive_nodes.insert(n);
                }
            }
        }
#if DEBUG_INIT_LOOP_SCHEDULER
        m_dbg << "expensive call count: " << expensive_nodes.nodes().size() << "\n";

        m_dbg << "expensive nodes: ";
        loop_schedule.print_debug_node_set(m_dbg, expensive_nodes);
        m_dbg << "\n";
#endif

        Loop_schedule::Evaluation_vector evaluations(m_alloc);

        schedule_evaluations(loop_schedule, expensive_nodes, evaluations);
    }

private:
    typedef ptr_hash_map<DAG_node const, size_t>::Type Node_uses_map;
    typedef hash_map<size_t, Loop_schedule::Node_ptr_vector>::Type Node_id_node_vec_map;
    typedef hash_map<size_t, DAG_node const *>::Type Node_id_node_ptr_map;

    struct Texture_write_info {
        size_t texture_result_index;
        size_t texture_result_offset;
    };

    typedef ptr_hash_map<DAG_node const, Texture_write_info>::Type Texture_write_map;

    /// Description of a temporary value lifetime. Used to determine which texture result
    /// slots can be used for allocating values that are saved/reloaded.
    struct Temporary_lifetime {
        size_t lifetime_index;
        size_t start;
        size_t end;
        DAG_node const *node;
        size_t local_index;
        size_t allocated_at;
    };
    /// Vector of the above.
    typedef vector<Temporary_lifetime>::Type Temporary_lifetime_vector;

    size_t get_type_size(IType const *type) const {
        return m_target_type_properties.get_store_size(type);
    }

    void print_evaluations(
        Loop_schedule const &schedule,
        Loop_schedule::Evaluation_vector const &evaluations) {
#if DEBUG_INIT_LOOP_SCHEDULER
        auto print_evaluation = [this, &schedule](size_t idx, Loop_schedule::Evaluation const &eval) {
            m_dbg << std::setw(3) << idx << " ";
            if (eval.from_schedule) {
                m_dbg << "  SCH";
            } else {
                m_dbg << "     ";
            }
            if (eval.state_dep) {
                m_dbg << " ST ";
            } else {
                m_dbg << "    ";
            }
            if (eval.expensive) {
                m_dbg << " EXP ";
            } else {
                m_dbg << "     ";
            }
            switch (eval.kind) {
            case Loop_schedule::Evaluation::Kind::EK_CASE_SPLIT:
                m_dbg << "==============================================\n";
                break;
            case Loop_schedule::Evaluation::Kind::EK_EXP_PARAM_SPLIT:
                m_dbg << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n";
                break;
            case Loop_schedule::Evaluation::Kind::EK_EXP_RED_SPLIT:
                m_dbg << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
                break;
            case Loop_schedule::Evaluation::Kind::EK_EVAL:
                m_dbg << "    eval " << eval.node->get_id();
                if (eval.node->get_kind() == DAG_node::EK_CALL) {
                    DAG_call const *call = cast<DAG_call>(eval.node);
                    m_dbg << "[" << call->get_name() << "]";
                    m_dbg << "(";
                    for (size_t i = 0, n = call->get_argument_count(); i < n; ++i) {
                        if (i > 0) {
                            m_dbg << ", ";
                        }
                        m_dbg << call->get_argument(i)->get_id();
                    }
                    m_dbg << ")";
                }
                m_dbg << "\n";
                if (eval.is_geom_normal) {
                    m_dbg << "                      -> store normal\n";
                }
                break;
            case Loop_schedule::Evaluation::Kind::EK_WRITE_TEXTURE:
                m_dbg << "    write_texture " << eval.node->get_id();
                if (eval.node->get_kind() == DAG_node::EK_CALL) {
                    DAG_call const *call = cast<DAG_call>(eval.node);
                    m_dbg << "[" << call->get_name() << "]";
                    m_dbg << "(";
                    for (size_t i = 0, n = call->get_argument_count(); i < n; ++i) {
                        if (i > 0) {
                            m_dbg << ", ";
                        }
                        m_dbg << call->get_argument(i)->get_id();
                    }
                    m_dbg << ") -> index: " << eval.texture_result_index << ", offset: " << eval.texture_result_offset;
                }
                m_dbg << "\n";
                if (eval.is_geom_normal) {
                    m_dbg << "                      -> store normal\n";
                }
                break;
            case Loop_schedule::Evaluation::Kind::EK_SET_NORMAL:
                m_dbg << "    -> set normal, invalidating:";
                for (auto n : eval.to_invalidate) {
                    m_dbg << " " << n->get_id();
                }
                m_dbg << "\n";
                break;
            case Loop_schedule::Evaluation::Kind::EK_SAVE:
                m_dbg << "    -> save:  ";
                if (eval.tmp_map.empty()) {
                    for (auto n : eval.values.nodes()) {
                        m_dbg << " " << n->get_id();
                    }
                } else {
                    for (auto const &p : eval.tmp_map) {
                        if (p.tex_result) {
                            m_dbg << " " << p.node->get_id() << "->tex[" << p.tex_result_offset << "]";
                        } else {
                            m_dbg << " " << p.node->get_id() << "->l[" << p.index << "]";
                        }
                    }
                }
                m_dbg << "\n";
                break;
            case Loop_schedule::Evaluation::Kind::EK_RELOAD:
                m_dbg << "    -> reload:";
                if (eval.tmp_map.empty()) {
                    for (auto n : eval.values.nodes()) {
                        m_dbg << " " << n->get_id();
                    }
                } else {
                    for (auto const &p : eval.tmp_map) {
                        if (p.tex_result) {
                            m_dbg << " " << p.node->get_id() << "<-tex[" << p.tex_result_offset << "]";
                        } else {
                            m_dbg << " " << p.node->get_id() << "<-l[" << p.index << "]";
                        }
                    }
                }
                m_dbg << "\n";
                break;
            }
            };

        m_dbg << "[**] evaluations:\n";
        size_t idx = 0;
        for (auto const &eval : evaluations) {
            print_evaluation(idx, eval);
            ++idx;
        }
        m_dbg << "[**]\n";
#endif
    }

    class Local_allocator {
    public:
        struct Local_entry {
            size_t index;
            IType const *type;
        };

        typedef vector<Local_entry>::Type Local_list;

        Local_allocator(IAllocator *alloc)
            : m_alloc(alloc)
            , m_locals(alloc)
            , m_nodes(alloc) {}

        /// Allocate a local variable for the given node.
        size_t allocate(DAG_node const *node) {
            size_t reg = ~0;
            for (size_t i = 0; i < m_nodes.size(); ++i) {
                if (m_nodes[i] == nullptr && m_locals[i].type == node->get_type()) {
                    reg = i;
                    m_nodes[reg] = node;
                    break;
                }
            }
            if (reg == ~0) {
                reg = m_locals.size();
                m_locals.push_back({ reg, node->get_type() });
                m_nodes.push_back(node);
            }
            MDL_ASSERT(reg != ~0);
            MDL_ASSERT(m_locals[reg].index == reg);
            MDL_ASSERT(m_locals[reg].type == node->get_type());
            MDL_ASSERT(m_nodes[reg] == node);
            return reg;
        }

        /// Free the given local variable.
        void free(size_t local) {
            m_nodes[local] = nullptr;
        }

        /// Free the given local variable for node `n`.
        void free(DAG_node const *n) {
            size_t reg = local(n);
            MDL_ASSERT(reg != ~0);
            free(reg);
        }

        /// Return the index of the local variable currently allocated for the given node.
        size_t local(DAG_node const *node) {
            for (size_t i = 0, n = m_nodes.size(); i < n; ++i) {
                if (m_nodes[i] == node) {
                    return m_locals[i].index;
                }
            }
            return ~0;
        }

        DAG_node const *node(size_t index) {
            return m_nodes[index];
        }

        /// Return the number of locals allocated.
        size_t count() const {
            return m_locals.size();
        }

        Local_list const &locals() const {
            return m_locals;
        }

    private:
        IAllocator *m_alloc;
        Local_list m_locals;
        vector<DAG_node const *>::Type m_nodes;
    };

    string instantiation_suffix(DAG_node const *node) {
        string ret(m_alloc);
        char buf[32];

        IDefinition const *def = find_definition(node);
        if (!def) {
            return ret;
        }
        IType const *ty = def->get_type();
        if (ty->get_kind() != IType::TK_FUNCTION) {
            return ret;
        }
        DAG_call const *call = cast<DAG_call>(node);
        uint64_t hash = 0;
        IType_function const *fty = cast<IType_function>(ty);
        for (size_t i = 0, n = fty->get_parameter_count(); i < n; ++i) {
            IType const *pty;
            ISymbol const *psym;
            fty->get_parameter(i, pty, psym);
            if (pty->get_kind() == IType::TK_ARRAY) {
                IType_array const *aty = cast<IType_array>(pty);
                if (!aty->is_immediate_sized()) {
                    DAG_node const *arg = call->get_argument(i);
                    IType const *arg_ty = arg->get_type();
                    MDL_ASSERT(arg_ty->get_kind() == IType::TK_ARRAY);
                    size_t isize = cast<IType_array>(arg_ty)->get_size();
                    hash = 5 * hash + isize;
                    snprintf(buf, sizeof(buf) - 1, "_%d", int(isize));
                    buf[sizeof(buf) - 1] = '\0';
                    ret += buf;
                }
            }
        }
        return ret;
    }

    /// Return the original definition of the funcion called by the DAG call
    /// node `node`. If it is not a call node, or its definition is not a function,
    /// or if it cannot be found, return nullptr.
    IDefinition const *find_definition(DAG_node const *node) {
        if (node->get_kind() != DAG_node::EK_CALL) {
            return nullptr;
        }
        DAG_call const *call = cast<DAG_call>(node);
        char const *signature = call->get_name();
        if (signature[0] == '#') {
            // skip prefix for derivative variants
            ++signature;
        }
        mi::base::Handle<Module const> mod(
            impl_cast<Module>(m_resolver->get_owner_module(signature)));
        if (!mod) {
            return nullptr;
        }

        Module const *module = mod.get();
        IDefinition const *def = module->find_signature(signature, /*only_exported=*/ false);
        if (def == nullptr) {
            return nullptr;
        }

        def = skip_presets(def, mod);
        if (def->get_kind() != IDefinition::DK_FUNCTION) {
            return nullptr;
        }
        return def;
    }

    void schedule_args(
        Loop_schedule &schedule,
        DAG_node const *node,
        Loop_schedule::Node_set const &expensive_nodes,
        Loop_schedule::Evaluation_vector &evaluations,
        Loop_schedule::Node_ptr_vector &cur_eval_state_results,
        Loop_schedule::Node_set &visited) {
        if (visited.contains(node)) {
            return;
        }
        visited.insert(node);
        if (node->get_kind() == DAG_node::EK_CALL) {
            DAG_call const *call = cast<DAG_call>(node);
            size_t         n_args = call->get_argument_count();
            Small_VLA<DAG_node const *, 8> arguments(m_alloc, n_args);
            for (size_t i = 0; i < n_args; ++i) {
                DAG_node const *arg = call->get_argument(i);
                arguments[i] = arg;
            }
            if (m_gen_reorder_arguments) {
                auto has_call_arg = [](DAG_node const *a) {
                    if (a->get_kind() == DAG_node::EK_CALL) {
                        DAG_call const *c = cast<DAG_call>(a);
                        for (size_t i = 0, n = c->get_argument_count(); i < n; ++i) {
                            DAG_node const *arg = c->get_argument(i);
                            if (arg->get_kind() == DAG_node::EK_CALL) {
                                return true;
                            }
                        }
                    }
                    return false;
                    };

                auto sort_args = [has_call_arg](DAG_node const *a, DAG_node const *b) {
                    bool a_has_call = has_call_arg(a);
                    bool b_has_call = has_call_arg(b);

                    return a_has_call && b_has_call
                        ? a->get_id() < b->get_id()
                        : a_has_call;
                    };
                std::sort(arguments.begin(), arguments.end(), sort_args);
            }
            for (size_t i = 0; i < n_args; ++i) {
                DAG_node const *arg = arguments[i];
                if (arg->get_kind() == DAG_node::EK_CALL) {
                    if (visited.contains(arg)) {
                        continue;
                    }
                    schedule_args(schedule, arg, expensive_nodes, evaluations, cur_eval_state_results, visited);
                    evaluations.emplace_back(m_alloc, arg, false,
                        expensive_nodes.contains(arg));
                    auto it = m_node_info_map.find(arg);
                    if (it != m_node_info_map.end()) {
                        Node_info const &ni = it->second;
                        if (ni.is_eval_state_dependent) {
                            evaluations.back().state_dep = true;
                            cur_eval_state_results.push_back(arg);
                        }
                    }
                }
            }
        }
    }

    /// Move splits between expensive calls to better positions (moving evaluations
    /// after an expensive call to the start of the next one, if it seems better).
    /// The heuristic aims at reducing the number of values to be saved/reloaded between
    /// cases.
    void move_splits(
        Loop_schedule::Evaluation_vector &evaluations)
    {
        auto reduces = [this](DAG_call const *exp_func, DAG_node const *following) {
            if (DAG_call const *following_call = as<DAG_call>(following)) {
                for (size_t i = 0, n = following_call->get_argument_count(); i < n; ++i) {
                    if (following_call->get_argument(i) == exp_func) {
                        if (get_type_size(following_call->get_type()) < get_type_size(exp_func->get_type())) {
                            return true;
                        }
                    }
                }
            }
            return false;
            };

        size_t current = evaluations.size();
        while (current > 1) { // If there are any reductions...
            while (current > 1
                && evaluations[current - 1].kind != Loop_schedule::Evaluation::Kind::EK_CASE_SPLIT) {
                current -= 1;
            }
            if (current < 2) {
                break;
            }

            size_t prev_exp = current;
            while (prev_exp > 0 && !evaluations[prev_exp - 1].expensive) {
                prev_exp -= 1;
            }
            if (prev_exp == 0) {
                break;
            }
            size_t new_split = current - 1;
            auto const &exp_eval = evaluations[prev_exp - 1];
            while (new_split > prev_exp) {
                auto const &eval = evaluations[new_split - 1];
                if (eval.kind != Loop_schedule::Evaluation::Kind::EK_EVAL) {
                    break;
                }
                if (reduces(cast<DAG_call>(exp_eval.node), eval.node)) {
                    break;
                }
                new_split -= 1;
            }
            if (new_split + 1 < current) {
                evaluations.insert(evaluations.begin() + new_split, Loop_schedule::Evaluation::case_split(m_alloc));
                evaluations[new_split].live_in = evaluations[new_split - 1].live_out.clone();
                if (new_split + 1 < evaluations.size()) {
                    evaluations[new_split].live_out = evaluations[new_split + 1].live_in.clone();
                }
                evaluations.erase(evaluations.begin() + current);
            }
            current = prev_exp - 1;
        }
    }

    /// Insert splits before expensive functions.
    void insert_splits(
        Loop_schedule::Evaluation_vector &evaluations)
    {
        size_t i = 0;
        bool first = true;
        while (i < evaluations.size()) {
            Loop_schedule::Evaluation &eval = evaluations[i];
            switch (eval.kind) {
            case Loop_schedule::Evaluation::Kind::EK_EVAL:
            case Loop_schedule::Evaluation::Kind::EK_WRITE_TEXTURE:
                if (eval.expensive) {
                    if (!first) {
                        evaluations.insert(evaluations.begin() + i, Loop_schedule::Evaluation::case_split(m_alloc));
                        evaluations[i].live_out = evaluations[i + 1].live_in.clone();
                        if (i > 0) {
                            evaluations[i].live_in = evaluations[i - 1].live_out.clone();
                        }
                        i += 1;
                    }
                    first = false;
                    evaluations.insert(evaluations.begin() + i, Loop_schedule::Evaluation::exp_param_split(m_alloc));
                    evaluations[i].live_out = evaluations[i + 1].live_in.clone();
                    if (i > 0) {
                        evaluations[i].live_in = evaluations[i - 1].live_out.clone();
                    }
                    evaluations.insert(evaluations.begin() + (i + 2), Loop_schedule::Evaluation::exp_red_split(m_alloc));
                    if (i + 3 < evaluations.size()) {
                        evaluations[i + 2].live_out = evaluations[i + 3].live_in.clone();
                    }
                    evaluations[i + 2].live_in = evaluations[i + 1].live_out.clone();
                    i += 2;
                }
                ++i;
                break;
            default:
                ++i;
                break;
            }
        }
    }

    /// Insert saves of values that were defined before splits and expensive calls and
    /// are live afterwards.
    /// This is done in a forwards scan of the evaluations.
    void insert_saves(
        Loop_schedule::Evaluation_vector &evaluations)
    {
        size_t i = 0;
        Loop_schedule::Node_set defined(m_alloc);
        Loop_schedule::Node_set live(m_alloc);
        while (i < evaluations.size()) {
            Loop_schedule::Evaluation &eval = evaluations[i];
            switch (eval.kind) {
            case Loop_schedule::Evaluation::Kind::EK_EVAL:

            case Loop_schedule::Evaluation::Kind::EK_WRITE_TEXTURE:
                defined.insert(eval.node);
                ++i;
                break;

            case Loop_schedule::Evaluation::Kind::EK_CASE_SPLIT:
            case Loop_schedule::Evaluation::Kind::EK_EXP_PARAM_SPLIT:
            case Loop_schedule::Evaluation::Kind::EK_EXP_RED_SPLIT:
                if (i + 1 < evaluations.size()) {
                    live = evaluations[i + 1].live_in.clone();
                } else {
                    live.clear();
                }
                live.intersect_with(defined);
                evaluations.insert(evaluations.begin() + i, Loop_schedule::Evaluation::save(m_alloc, live));
                if (i > 0) {
                    evaluations[i].live_in = evaluations[i - 1].live_out.clone();
                }
                evaluations[i].live_out = evaluations[i + 1].live_out.clone();
                i += 2;
                defined.clear();
                break;

            default:
                ++i;
                break;
            }
        }
    }

    /// Insert reloads after splits and expensive calls.
    /// This is done in a backwards scan of the evaluations.
    void insert_reloads(
        Loop_schedule::Evaluation_vector &evaluations)
    {
        size_t i = evaluations.size();
        Loop_schedule::Node_set used(m_alloc);
        Loop_schedule::Node_set live(m_alloc);
        while (i > 0) {
            Loop_schedule::Evaluation &eval = evaluations[i - 1];
            switch (eval.kind) {
            case Loop_schedule::Evaluation::Kind::EK_EVAL:
            case Loop_schedule::Evaluation::Kind::EK_WRITE_TEXTURE:
                if (eval.node->get_kind() == DAG_node::EK_CALL) {
                    DAG_call const *call = cast<DAG_call>(eval.node);
                    for (size_t j = 0, n = call->get_argument_count(); j < n; ++j) {
                        DAG_node const *arg = call->get_argument(j);
                        if (arg->get_kind() == DAG_node::EK_CALL) {
                            used.insert(arg);
                        }
                    }
                }
                --i;
                break;

            case Loop_schedule::Evaluation::Kind::EK_CASE_SPLIT:
            case Loop_schedule::Evaluation::Kind::EK_EXP_PARAM_SPLIT:
            case Loop_schedule::Evaluation::Kind::EK_EXP_RED_SPLIT:
                if (i >= 3) {
                    live = evaluations[i - 3].live_out.clone(); // take into account split + save slots
                    live.intersect_with(used);
                } else {
                    live.clear();
                }
                evaluations.insert(evaluations.begin() + i, Loop_schedule::Evaluation::reload(m_alloc, live));
                evaluations[i].live_in = evaluations[i - 1].live_out.clone();
                if (i + 1 < evaluations.size()) {
                    evaluations[i].live_out = evaluations[i + 1].live_in.clone();
                }
                used.clear();
                --i;
                break;

            default:
                --i;
                break;
            }
        }
    }

    void spill(Loop_schedule::Evaluation_vector &evaluations, Loop_schedule::Node_set &loop_vars) {
        insert_saves(evaluations);
        insert_reloads(evaluations);
        for (auto const &eval : evaluations) {
            switch (eval.kind) {
            case Loop_schedule::Evaluation::Kind::EK_SAVE:
            case Loop_schedule::Evaluation::Kind::EK_RELOAD:
                for (auto const n : eval.values.nodes()) {
                    loop_vars.insert(n);
                }
                break;

            default:
                break;
            }
        }
    }

    void allocate_locals(
        Loop_schedule const &schedule,
        Loop_schedule::Evaluation_vector &evaluations,
        Loop_schedule::Node_allocation_vector &local_types,
        Temporary_lifetime_vector &temp_lifetimes)
    {
        Local_allocator local_alloc(m_alloc);
        Loop_schedule::Node_set to_remove(m_alloc);
        Loop_schedule::Node_ptr_vector nodes(m_alloc);
        size_t idx = 0;
        size_t lt_idx = 0;
        auto node_cmp = [](DAG_node const *a, DAG_node const *b) {
            return a->get_id() < b->get_id();
            };

        auto start_lifetime = [&](DAG_node const *n) {
            size_t local_index = local_alloc.local(n);
            size_t end = ~0;
            temp_lifetimes.push_back({ lt_idx, idx, end, n, local_index, size_t(~0)});
            lt_idx += 1;
            };

        auto update_lifetime = [&](DAG_node const *n) {
            size_t local_index = local_alloc.local(n);
            for (size_t x = 0; x < temp_lifetimes.size(); ++x) {
                auto &lifetime = temp_lifetimes[x];
                if (lifetime.node == n && lifetime.local_index == local_index) {
                    lifetime.end = idx;
                }
            }
            };

        for (auto it = evaluations.begin(), end_it = evaluations.end(); it != end_it; ++it, ++idx) {
            auto &eval = *it;
            switch (eval.kind) {
            case Loop_schedule::Evaluation::Kind::EK_SAVE:
            {
                to_remove.clear();
                for (auto const &la : local_alloc.locals()) {
                    DAG_node const *n = local_alloc.node(la.index);
                    if (n != nullptr && !(it - 1)->live_out.contains(n)) {
                        to_remove.insert(n);
                    }
                }
                for (auto n : to_remove.nodes()) {
                    local_alloc.free(n);
                }
                Loop_schedule::Evaluation::Tmp_map tmp_map(m_alloc);
                nodes.clear();
                for (auto n : eval.values.nodes()) {
                    nodes.push_back(n);
                }
                std::sort(nodes.begin(), nodes.end(), node_cmp);
                for (auto n : nodes) {
                    size_t local = local_alloc.allocate(n);
                    start_lifetime(n);
                    tmp_map.push_back({ false, local, n, size_t(~0) });
                }
                std::swap(eval.tmp_map, tmp_map);
                break;
            }
            case Loop_schedule::Evaluation::Kind::EK_RELOAD:
            {
                Loop_schedule::Evaluation::Tmp_map tmp_map(m_alloc);
                nodes.clear();
                for (auto n : eval.values.nodes()) {
                    nodes.push_back(n);
                }
                std::sort(nodes.begin(), nodes.end(), node_cmp);
                for (auto n : nodes) {
                    update_lifetime(n);
                    size_t local = local_alloc.local(n);
                    tmp_map.push_back({ false, local, n, size_t(~0) });
                }
                std::swap(eval.tmp_map, tmp_map);
                break;
            }
            default:
                break;
            }
        }

        for (auto const &la : local_alloc.locals()) {
            local_types.push_back({ la.index, la.type });
        }
        auto loc_ty_cmp = [](std::pair<size_t, IType const *> &a, std::pair<size_t, IType const *> &b) {
            return a.first < b.first;
            };
        std::sort(local_types.begin(), local_types.end(), loc_ty_cmp);

        for (auto &lt : temp_lifetimes) {
            if (lt.end == ~0) {
                lt.end = evaluations.size() - 1;
            }
        }
        auto lt_cmp = [](Temporary_lifetime const &a, Temporary_lifetime const &b) {
            size_t a_len = a.end - a.start;
            size_t b_len = b.end - b.start;
            if (a_len == b_len) {
                if (a.end == b.end) {
                    if (a.start == b.start) {
                        return a.local_index < b.local_index;
                    }
                    return a.start < b.start;
                }
                return a.end > b.end;
            }
            return a_len > b_len;
            };
        std::sort(temp_lifetimes.begin(), temp_lifetimes.end(), lt_cmp);
#if DEBUG_INIT_LOOP_SCHEDULER && 0
        for (auto &lt : temp_lifetimes) {
            m_dbg << "lt: " << lt.lifetime_index << ", node: " << lt.node->get_id() << ", local: " << lt.local_index
                << ", start: " << lt.start << ", end: " << lt.end << ", length: " << (lt.end - lt.start)
                << ", allocated at: " <<lt.allocated_at << "\n";
        }
#endif
    }

    /// Calculate the live values before and after each evaluation.
    /// This is done in a backwards scan of the evaluations.
    void liveness(
        Loop_schedule &schedule,
        Loop_schedule::Evaluation_vector &evaluations)
    {
        size_t eval_idx = evaluations.size();
        Loop_schedule::Node_set live(m_alloc);
        for (auto it = evaluations.rbegin(), end_it = evaluations.rend(); it != end_it; ++it) {
            eval_idx -= 1;
            it->live_out = live.clone();
            switch (it->kind) {
            case Loop_schedule::Evaluation::Kind::EK_EVAL:
                live.erase(it->node);
                if (it->node->get_kind() == DAG_node::EK_CALL) {
                    DAG_call const *call = cast<DAG_call>(it->node);
                    for (size_t i = 0, n = call->get_argument_count(); i < n; ++i) {
                        DAG_node const *arg = call->get_argument(i);
                        if (arg->get_kind() == DAG_node::EK_CALL) {
                            live.insert(call->get_argument(i));
                        }
                    }
                }
                break;
            case Loop_schedule::Evaluation::Kind::EK_WRITE_TEXTURE:
                live.erase(it->node);
                if (it->node->get_kind() == DAG_node::EK_CALL) {
                    DAG_call const *call = cast<DAG_call>(it->node);
                    for (size_t i = 0, n = call->get_argument_count(); i < n; ++i) {
                        DAG_node const *arg = call->get_argument(i);
                        if (arg->get_kind() == DAG_node::EK_CALL) {
                            live.insert(call->get_argument(i));
                        }
                    }
                }
                break;
            default:
                break;
            }
            it->live_in = live.clone();
        }
    }

    size_t collect_expensive_call_sites(
        Loop_schedule::Evaluation_vector &evaluations,
        Loop_schedule::Expensive_call_site_map &function_to_expensive_call_sites)
    {
        size_t case_start = 0;
        size_t last_expensive = ~0;
        size_t case_cnt = 0;
        string suffix(m_alloc);
        DAG_call const *last_call = nullptr;

        // Append case `num' for evaluations `start' to `end', inclusive.
        auto append = [this, &function_to_expensive_call_sites, &suffix, &last_call](size_t num, size_t start, size_t end) {
            Loop_schedule::Expensive_call_site_map::key_type key(last_call->get_name(), suffix.c_str());
            auto const result = function_to_expensive_call_sites.try_emplace(
                key, m_alloc, last_call->get_argument_count());
            auto const it = result.first;
            it->second.cases.emplace_back(num, start, end, last_call);
        };

        for (size_t i = 0, n = evaluations.size(); i < n; ++i) {
            auto const &eval = evaluations[i];
            switch (eval.kind) {
            case Loop_schedule::Evaluation::Kind::EK_CASE_SPLIT:
                if (last_expensive != ~0) {
                    size_t case_end = i - 1;
                    append(case_cnt, case_start, case_end);

                    case_cnt += 1;
                    last_expensive = ~0;
                }
                case_start = i + 1;
                break;
            case Loop_schedule::Evaluation::Kind::EK_EVAL:
            case Loop_schedule::Evaluation::Kind::EK_WRITE_TEXTURE:
                if (eval.expensive) {
                    last_expensive = i;
                    last_call = cast<DAG_call>(eval.node);
                    suffix = instantiation_suffix(eval.node);
                }
                break;
            default:
                break;
            }
        }
        if (last_expensive != ~0) {
            size_t case_end = evaluations.size() - 1;
            append(case_cnt, case_start, case_end);
            case_cnt += 1;
        }
#if DEBUG_INIT_LOOP_SCHEDULER
        for (auto const &p : function_to_expensive_call_sites) {
            auto const &ec = p.second;
            m_dbg << "[>>] " << p.first.first << "/" << p.first.second << ":";
            for (auto const &t : ec.cases) {
                m_dbg << " case " << t.index << "/" << t.first << "--" << t.last;
            }
            m_dbg << "\n";
        }
        m_dbg << "\n";
#endif
        return case_cnt;
    }

    void analyze_parameters(
        Loop_schedule::Evaluation_vector const &evaluations,
        Loop_schedule::Expensive_call_site_map &function_to_expensive_call_sites)
    {
        if (!m_gen_common_parameters) {
            return;
        }

        for (auto &p : function_to_expensive_call_sites) {
            Loop_schedule::Expensive_call &ec = p.second;
            Loop_schedule::Expensive_call_site &first_ecs = ec.cases[0];
            size_t param_count = first_ecs.call->get_argument_count();

            Bitset equal_params(m_alloc, param_count);
            Bitset material_param_params(m_alloc, param_count);
            Bitset trivial_params(m_alloc, param_count);

            equal_params.set_bits();
            if (m_gen_parameter_offsets) {
                material_param_params.set_bits();
            }
            trivial_params.set_bits();

            for (auto &ecs : ec.cases) {
                for (size_t i = 0; i < param_count; ++i) {
                    DAG_node const *first_arg = first_ecs.call->get_argument(i);
                    DAG_node const *arg = ecs.call->get_argument(i);
                    if (arg->get_kind() != DAG_node::EK_PARAMETER) {
                        material_param_params.clear_bit(i);
                    }
                    if (arg->get_kind() != DAG_node::EK_CONSTANT) {
                        trivial_params.clear_bit(i);
                    }
                    if (first_arg != arg || arg->get_kind() == DAG_node::EK_CALL) {
                        equal_params.clear_bit(i);
                    } else {
                        auto ni = m_node_info_map.find(arg);
                        if (ni != m_node_info_map.end()) {
                            if (ni->second.is_eval_state_dependent) {
                                equal_params.clear_bit(i);
                            }
                        }
                    }
                }
            }
            ec.equal_params.copy_data(equal_params);
            ec.material_param_params.copy_data(material_param_params);
            ec.trivial_params.copy_data(trivial_params);
        }
    }

    size_t find_slot(IType const *ty, Bitset const &allocated) {
        size_t ty_size = get_type_size(ty);
        for (size_t ofs = 0; ofs + ty_size < allocated.get_size(); ofs += 4) {
            bool found = true;
            for (size_t byte_ofs = 0; byte_ofs < ty_size; ++byte_ofs) {
                if (allocated.test_bit(ofs + byte_ofs)) {
                    found = false;
                    break;
                }
            }
            if (found) {
                return ofs;
            }
        }
        return ~0;
    }

    void allocate_locals_to_texture_results(
        Loop_schedule &schedule,
        Loop_schedule::Node_allocation_vector &local_types,
        Loop_schedule::Evaluation_vector &evaluations,
        Temporary_lifetime_vector &temp_lifetimes)
    {
        // Whenever a texture result slot is written, we free the storage it occupies in our backwards
        // traversal through the evaluations. The freeing is delayed until the split before the texture
        // write, to avoid that an overwritten temporary in a texture result is erroneously reused.
        // This structure holds the offset and size of the space to be freed.
        struct Pending_frees {
            size_t offset;
            size_t size;
        };

        // List of delayed freeing operations. On each split operation encountered, all the storage
        // described by the entries in this list are marked free and the list is cleared.
        vector<Pending_frees>::Type pending_frees(m_alloc);

        // Store the given offset in the lifetime and mark all words as occupied.
        auto allocate_at = [&](Temporary_lifetime &lt, size_t offset, size_t size, Bitset &allocated_texture_results) {
            lt.allocated_at = offset;
            for (size_t j = offset, m = offset + size; j < m; ++j) {
                allocated_texture_results.set_bit(j);
            }
            };

        // Find the index of a lifetime that corresponds to the given temporary location and that includes the
        // given evaluation index.
        auto find_lifetime = [&](Loop_schedule::Evaluation::Tmp_location const &tmp_loc, size_t eval_idx) -> size_t {
            size_t lt_idx = 0;
            for (auto const &lt : temp_lifetimes) {
                if (eval_idx >= lt.start && eval_idx <= lt.end && lt.node == tmp_loc.node && lt.local_index == tmp_loc.index) {
                    return lt_idx;
                }
                lt_idx += 1;
            }
            return ~0;
            };

        Bitset allocated_texture_results(m_alloc, schedule.total_texture_bytes);
        allocated_texture_results.set_bits();

        size_t eval_idx = evaluations.size();
        for (auto it = evaluations.rbegin(), end_it = evaluations.rend(); it != end_it; ++it) {
            eval_idx -= 1;
            switch (it->kind) {
            case Loop_schedule::Evaluation::Kind::EK_WRITE_TEXTURE:
                if (it->node->get_kind() == DAG_node::EK_CALL) {
                    DAG_call const *call = cast<DAG_call>(it->node);
                    IType const *ty = call->get_type();
                    size_t result_size = get_type_size(ty);
                    size_t result_offset = it->texture_result_offset;
                    pending_frees.push_back({ result_offset, result_size });
                }
#if DEBUG_INIT_LOOP_SCHEDULER && 0
                m_dbg << "[$]" << eval_idx << " allocated texture results:";
                for (size_t i = 0; i < schedule.total_texture_bytes; ++i) {
                    if (i % 4 == 0) {
                        m_dbg << " " << i << ":";
                    }
                    if (allocated_texture_results.test_bit(i)) {
                        m_dbg << "1";
                    } else {
                        m_dbg << "0";
                    }
                }
                m_dbg << "\n";
#endif
                break;

            case Loop_schedule::Evaluation::Kind::EK_SAVE:
                for (auto &tmp_loc : it->tmp_map) {
                    size_t lt_idx = find_lifetime(tmp_loc, eval_idx);
                    MDL_ASSERT(lt_idx != ~0);
                    auto &lt = temp_lifetimes[lt_idx];
                    if (lt.allocated_at != ~0) {
                        tmp_loc.tex_result = true;
                        tmp_loc.tex_result_offset = lt.allocated_at;
                        if (lt.start == eval_idx) {
                            DAG_call const *lt_call = cast<DAG_call>(lt.node);
                            IType const *lt_ty = lt_call->get_type();
                            size_t lt_size = get_type_size(lt_ty);
                            size_t lt_offset = lt.allocated_at;
                            for (size_t j = lt_offset, m = lt_offset + lt_size; j < m; ++j) {
                                allocated_texture_results.clear_bit(j);
                            }
                        }
                    }
                }
                break;

            case Loop_schedule::Evaluation::Kind::EK_RELOAD:
                for (auto &tmp_loc : it->tmp_map) {
                    size_t lt_idx = find_lifetime(tmp_loc, eval_idx);
                    MDL_ASSERT(lt_idx != ~0);
                    auto &lt = temp_lifetimes[lt_idx];
                    if (lt.allocated_at != ~0) {
                        tmp_loc.tex_result = true;
                        tmp_loc.tex_result_offset = lt.allocated_at;
                    } else if (lt.end == eval_idx) {
                        // Lifetime of value ends here, try to allocate it to a texture result slot.
                        DAG_call const *lt_call = cast<DAG_call>(lt.node);
                        IType const *lt_ty = lt_call->get_type();
                        size_t lt_size = get_type_size(lt_ty);
                        size_t last_gap_size = 0;
                        size_t last_gap_start = 0;
                        for (size_t j = 0; j < allocated_texture_results.get_size(); j += 4) {
                            if (!allocated_texture_results.test_bit(j)) {
                                if (last_gap_size == 0) {
                                    last_gap_start = j;
                                }
                                last_gap_size += 4;
                                if (last_gap_size >= lt_size) {
                                    allocate_at(lt, last_gap_start, lt_size, allocated_texture_results);
                                    tmp_loc.tex_result = true;
                                    tmp_loc.tex_result_offset = lt.allocated_at;
                                    break;
                                }
                            } else {
                                last_gap_size = 0;
                            }
                        }
                    }
                }
                break;

            case Loop_schedule::Evaluation::Kind::EK_CASE_SPLIT:
            case Loop_schedule::Evaluation::Kind::EK_EXP_RED_SPLIT:
            case Loop_schedule::Evaluation::Kind::EK_EXP_PARAM_SPLIT:
                for (auto const &free : pending_frees) {
                    for (size_t j = free.offset, m = free.offset + free.size; j < m; ++j) {
                        allocated_texture_results.clear_bit(j);
                    }
                }
                pending_frees.clear();
                break;

            default:
                break;
            }
        }
#if DEBUG_INIT_LOOP_SCHEDULER
        Bitset used_locals(m_alloc, local_types.size());
        for (auto const &eval : evaluations) {
            switch (eval.kind) {
            case Loop_schedule::Evaluation::Kind::EK_SAVE:
            case Loop_schedule::Evaluation::Kind::EK_RELOAD:
                for (auto const &tmp_loc : eval.tmp_map) {
                    if (!tmp_loc.tex_result) {
                        used_locals.set_bit(tmp_loc.index);
                    }
                }
                break;
            default:
                break;
            }
        }
        m_dbg << "[?] used locals: ";
        for (size_t j = 0; j < local_types.size(); ++j) {
            m_dbg << (used_locals.test_bit(j) ? "1" : "0");
        }
        m_dbg << "\n";
#if 0
        for (auto &lt : temp_lifetimes) {
            m_dbg << "lt: " << lt.lifetime_index << ", node: " << lt.node->get_id() << ", local: " << lt.local_index
                << ", start: " << lt.start << ", end: " << lt.end << ", length: " << (lt.end - lt.start)
                << ", allocated at: " << lt.allocated_at << "\n";
        }
#endif
#endif
    }

    void schedule_evaluations(
        Loop_schedule &schedule,
        Loop_schedule::Node_set const &expensive_nodes,
        Loop_schedule::Evaluation_vector &evaluations)
    {
        Loop_schedule::Node_ptr_vector cur_eval_state_results(m_alloc);
        Loop_schedule::Node_set visited(m_alloc);

        size_t i;
        size_t n = m_schedule.size();
        size_t max_index_before_normal_update = ~0;
        size_t total_texture_bytes = 0;

        // if displacement, cutout opacity or geometry.normal are part of the schedule,
        // they and all scheduled nodes before them need to be translated first.
        // find the maximum schedule index, which must be translated before updating the normal.
        // note: displacement and/or cutout opacity may depend on the result of the DAG node
        //       of geometry.normal
        for (i = 0; i < n; ++i) {
            Schedule_entry const &entry = m_schedule[i];
            if (entry.has_special_kind(
                Distribution_function::SK_MATERIAL_GEOMETRY_DISPLACEMENT) ||
                entry.has_special_kind(
                    Distribution_function::SK_MATERIAL_GEOMETRY_CUTOUT_OPACITY) ||
                entry.has_special_kind(
                    Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL))
            {
                max_index_before_normal_update = i;
            }
            if (entry.is_stored_in_texture_results() && entry.node->get_kind() == DAG_node::EK_CALL) {
                DAG_call const *call = cast<DAG_call>(entry.node);
                IType const *ty = call->get_type();
                size_t result_size = get_type_size(ty);
                size_t result_offset = entry.texture_result_offset;
                total_texture_bytes = std::max(total_texture_bytes, result_offset + result_size);
            }
        }

        schedule.total_texture_bytes = total_texture_bytes;

        Allocator_builder builder(m_alloc);

        // reset i to the start of the schedule again
        i = 0;

        bool normal_stored = false;

        if (max_index_before_normal_update != ~0) {
            for (; i <= max_index_before_normal_update; ++i) {
                Schedule_entry const &entry = m_schedule[i];

                bool is_geom_normal = false;
                if (entry.has_special_kind(Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL)) {
                    is_geom_normal = true;
                }
                if (entry.is_stored_in_texture_results()) {
                    schedule_args(schedule, entry.node, expensive_nodes, evaluations, cur_eval_state_results, visited);
                    visited.insert(entry.node);
                    evaluations.emplace_back(
                        m_alloc,
                        entry.node,
                        entry.texture_result_index,
                        entry.texture_result_offset,
                        total_texture_bytes,
                        is_geom_normal,
                        expensive_nodes.contains(entry.node));
                    evaluations.back().from_schedule = true;
                    evaluations.back().state_dep = entry.is_eval_state_dependent;
                } else {
                    schedule_args(schedule, entry.node, expensive_nodes, evaluations, cur_eval_state_results, visited);
                    visited.insert(entry.node);
                    evaluations.emplace_back(m_alloc, entry.node, is_geom_normal,
                        expensive_nodes.contains(entry.node));
                    evaluations.back().from_schedule = true;
                    evaluations.back().state_dep = entry.is_eval_state_dependent;
                }
                if (!normal_stored && is_geom_normal) {
                    normal_stored = true;
                }

                // if the node is evaluation state dependent, we need to remove it from the manual node
                // value map later again.
                // Support for evaluation state dependency allows reusing independent node results after
                // geometry.normal updates state.normal.
                // Example: ::nvidia::vMaterials::AEC::Masonry::CMU_Running_Half_Bond_Splitface
                if (entry.is_eval_state_dependent) {
                    cur_eval_state_results.push_back(entry.node);
                }
            }
        }

        // set normal now, if necessary
        if (normal_stored) {
            for (auto remove : cur_eval_state_results) {
                visited.erase(remove);
            }
            evaluations.emplace_back(m_alloc, std::move(cur_eval_state_results));
        }

        // translate the remaining scheduled nodes
        for (; i < n; ++i) {
            Schedule_entry const &entry = m_schedule[i];

            if (entry.is_stored_in_texture_results()) {
                schedule_args(schedule, entry.node, expensive_nodes, evaluations, cur_eval_state_results, visited);
                visited.insert(entry.node);
                evaluations.emplace_back(
                    m_alloc,
                    entry.node,
                    entry.texture_result_index,
                    entry.texture_result_offset,
                    total_texture_bytes,
                    false,
                    expensive_nodes.contains(entry.node));
                evaluations.back().from_schedule = true;
                evaluations.back().state_dep = entry.is_eval_state_dependent;
            }
        }

        // Calculate liveness of all values.
        liveness(schedule, evaluations);

        // Split evaluations at expensive calls.
        insert_splits(evaluations);

        // Move splits backwards, if possible, to reduce the number of values that
        // need to be stored in local variables.
        move_splits(evaluations);

        // Insert save and reload instructions before and after splits. Also calculate
        // the set of values that are stored in local variables.
        Loop_schedule::Node_set loop_vars(m_alloc);
        spill(evaluations, loop_vars);

        // Allocate saved/reloaded values to local variables.
        Loop_schedule::Node_allocation_vector local_types(m_alloc);
        Temporary_lifetime_vector temp_lifetimes(m_alloc);
        allocate_locals(schedule, evaluations, local_types, temp_lifetimes);
        if (m_gen_local_texres) {
            allocate_locals_to_texture_results(schedule, local_types, evaluations, temp_lifetimes);
        }

#if DEBUG_INIT_LOOP_SCHEDULER
        print_evaluations(schedule, evaluations);
#endif

        // Combine calls to the same expensive functions in a mapping, so that
        // they can be processed together during code generation.
        Loop_schedule::Expensive_call_site_map function_to_expensive_call_sites(m_alloc);
        size_t iterations = collect_expensive_call_sites(evaluations, function_to_expensive_call_sites);
        analyze_parameters(evaluations, function_to_expensive_call_sites);

#if DEBUG_INIT_LOOP_SCHEDULER
        size_t saves = 0;
        size_t reloads = 0;
        for (auto const &eval : evaluations) {
            switch (eval.kind) {
            case Loop_schedule::Evaluation::Kind::EK_SAVE:
                saves += 1;
                break;
            case Loop_schedule::Evaluation::Kind::EK_RELOAD:
                reloads += 1;
                break;
            default:
                break;
            }
        }
        m_dbg << "[:] saves:   " << saves << "\n";
        m_dbg << "[:] reloads: " << reloads << "\n";

        for (auto const &p : function_to_expensive_call_sites) {
            char const *name = p.first.first;
            auto const &ec = p.second;
            Bitset const &equal_params = ec.equal_params;
            Bitset const &material_param_params = ec.material_param_params;
            Bitset const &trivial_params = ec.trivial_params;
            m_dbg << "[-] " << name << "\n[.] equal_params: ";
            for (size_t i = 0; i < ec.cases[0].call->get_argument_count(); ++i) {
                if (equal_params.test_bit(i)) {
                    m_dbg << "1";
                } else {
                    m_dbg << "0";
                }
            }
            m_dbg << "\n[.] material_param_params: ";
            for (size_t i = 0; i < ec.cases[0].call->get_argument_count(); ++i) {
                if (material_param_params.test_bit(i)) {
                    m_dbg << "1";
                } else {
                    m_dbg << "0";
                }
            }
            m_dbg << "\n[.] trivial_params: ";
            for (size_t i = 0; i < ec.cases[0].call->get_argument_count(); ++i) {
                if (trivial_params.test_bit(i)) {
                    m_dbg << "1";
                } else {
                    m_dbg << "0";
                }
            }
            m_dbg << "\n";
        }

        size_t local_size = 0;
        size_t local_count = 0;
        for (size_t i = 0; i < local_types.size(); ++i) {
            IType const *mdl_type = local_types[i].second;
            size_t this_size = get_type_size(mdl_type);
            local_size += this_size;
            local_count += 1;
        }
        m_dbg << "[:] local count: " << local_count << "\n";
        m_dbg << "[:] local size: " << local_size << "\n";
#endif

        std::swap(schedule.evaluations, evaluations);
        std::swap(schedule.call_site_map, function_to_expensive_call_sites);
        std::swap(schedule.local_types, local_types);
        schedule.iterations = iterations;
    }

    IAllocator *m_alloc;
    ICall_name_resolver const *m_resolver;
    mi::mdl::vector<Schedule_entry>::Type const &m_schedule;
    mi::mdl::vector<std::pair<DAG_node const *, unsigned>>::Type const &m_sorted_nodes;
    Node_info_map const &m_node_info_map;
    ITarget_type_properties const &m_target_type_properties;
#if DEBUG_INIT_LOOP_SCHEDULER
    std::ofstream m_dbg;
#endif
    bool m_target_is_structured_language;
    bool m_gen_schedule_loop{ true };
    bool m_gen_evaluate_sequentially{ false };
    bool m_gen_common_parameters{ true };
    bool m_gen_parameter_offsets{ true };
    bool m_gen_local_texres{ true };
    bool m_gen_reorder_arguments{ true };
    int m_gen_expensive_function_cost_limit{ 40 };
    int m_gen_expensive_function_limit{ 20 };
};

/// Helper class to assign expressions to slots in the texture results
/// and determine the order of evaluation.
class Expression_scheduler
{
private:
    /// Helper class for allocating texture result entries.
    class Texture_result_allocator
    {
    public:
        // Constructor.
        Texture_result_allocator(
            IAllocator *alloc,
            size_t      texture_result_size)
        : m_alloc(alloc)
        , m_texture_result_size(texture_result_size)
        , m_free_1byte_list(alloc)
        , m_free_4byte_list(alloc)
        , m_free_16byte_list(alloc)
        , m_num_overaligned_results(0)
        , m_allocated_result_candidates(alloc)
        , m_repack_failed_result_candidates(alloc)
        {
            // initialize 16-byte free list
            m_free_16byte_list.reserve(texture_result_size / 16);
            for (size_t i = 0, n = texture_result_size / 16, offs = (n - 1) * 16;
                    i < n; ++i, offs -= 16) {
                m_free_16byte_list.push_back(offs);
            }
        }

    private:
        /// Get the offset of an unused 4-byte aligned 4-byte block.
        size_t get_4byte()
        {
            size_t new_offs;

            if (m_free_4byte_list.empty()) {
                if (m_free_16byte_list.empty()) {
                    return ~0;
                }
                new_offs = m_free_16byte_list.back();
                m_free_16byte_list.pop_back();

                m_free_4byte_list.push_back(new_offs + 12);
                m_free_4byte_list.push_back(new_offs + 8);
                m_free_4byte_list.push_back(new_offs + 4);
            } else {
                new_offs = m_free_4byte_list.back();
                m_free_4byte_list.pop_back();
            }
            return new_offs;
        }

        /// Get the offset of an unused 1-byte block.
        size_t get_1byte()
        {
            size_t new_offs;

            if (m_free_1byte_list.empty()) {
                new_offs = get_4byte();
                if (new_offs == ~0) {
                    return ~0;
                }

                m_free_1byte_list.push_back(new_offs + 3);
                m_free_1byte_list.push_back(new_offs + 2);
                m_free_1byte_list.push_back(new_offs + 1);
            } else {
                new_offs = m_free_1byte_list.back();
                m_free_1byte_list.pop_back();
            }
            return new_offs;
        }

        /// Pop the given number of elements from the list at the given index.
        void pop_at(mi::mdl::vector<size_t>::Type &list, size_t index, size_t num)
        {
            list.erase(
                list.begin() + index + 1 - num,
                list.begin() + index + 1);
        }

        /// Get the offset of an unused 16-byte aligned block of the given size.
        size_t get_16byte_aligned_block(size_t size)
        {
            size_t remaining_4byte_blocks = (size + 3) / 4;
            if (m_free_16byte_list.size() * 4 < remaining_4byte_blocks) {
                return ~0;
            }

            size_t new_offs = m_free_16byte_list.back();
            size_t remaining_full_16byte_blocks = size / 16;
            pop_at(m_free_16byte_list, m_free_16byte_list.size() - 1, remaining_full_16byte_blocks);

            remaining_4byte_blocks -= remaining_full_16byte_blocks * 4;

            if (remaining_4byte_blocks > 0) {
                size_t cur_offs = m_free_16byte_list.back();
                m_free_16byte_list.pop_back();

                // we need to add the rest of the blocks to the free 4-byte blocks
                for (int i = 3; i >= remaining_4byte_blocks; --i) {
                    m_free_4byte_list.push_back(cur_offs + size_t(i * 4));
                }
            }
            return new_offs;
        }

        /// Get the offset of an unused 4-byte aligned block of the given size.
        size_t get_4byte_aligned_block(size_t size)
        {
            size_t num_blocks = (size + 3) / 4;

            // quick rough check for available size
            if (m_free_4byte_list.size() + 4 * m_free_16byte_list.size() < num_blocks) {
                return ~0;
            }

            size_t found_consecutive_blocks = 0;
            int start_block = -1;
            size_t last_offs = ~0;

            for (int i = int(m_free_4byte_list.size() - 1); i >= 0; --i) {
                size_t cur_offs = m_free_4byte_list[i];
                if (last_offs + 4 == cur_offs) {
                    last_offs += 4;
                    ++found_consecutive_blocks;

                    // found enough blocks?
                    if (found_consecutive_blocks == num_blocks) {
                        // take the blocks and return the start offset
                        size_t new_offs = m_free_4byte_list[start_block];
                        pop_at(m_free_4byte_list, start_block, num_blocks);
                        return new_offs;
                    }
                } else {
                    // there's a hole (or this is the first block), start from the beginning
                    start_block = i;
                    last_offs = cur_offs;
                    found_consecutive_blocks = 1;
                }
            }

            // we didn't find enough blocks, yet
            // is the next 16-byte block consecutive to the last 4-byte block?
            if (!m_free_16byte_list.empty() && last_offs + 4 == m_free_16byte_list.back()) {
                // are there enough blocks available?
                // note: 16-byte blocks are always consecutive (we never add them back to the list)
                size_t remaining_4byte_blocks = num_blocks - found_consecutive_blocks;
                if (m_free_16byte_list.size() * 4 < remaining_4byte_blocks) {
                    return ~0;
                }

                // take the blocks (from the end of the list) and return the start offset
                size_t new_offs = m_free_4byte_list[start_block];
                pop_at(m_free_4byte_list, start_block, found_consecutive_blocks);

                size_t remaining_offs = get_16byte_aligned_block(remaining_4byte_blocks * 4);
                MDL_ASSERT(new_offs + 4 * found_consecutive_blocks == remaining_offs);
                (void) remaining_offs;
                return new_offs;
            }

            // we need to start with a new 16-byte block
            return get_16byte_aligned_block(size);
        }

        /// Get the alignment of an offset.
        size_t get_offset_alignment(size_t offs)
        {
            if ((offs & 15) == 0) {
                return 16;
            } else if ((offs & 3) == 0) {
                return 4;
            }
            return 1;
        }

        /// Allocate a (movable) slot in the texture results.
        ///
        /// \returns false when no suitable slot is available anymore
        bool alloc_impl(Result_candidate_info *result_candidate)
        {
            size_t size = result_candidate->size;
            size_t align = result_candidate->align;

            MDL_ASSERT(align <= 16 && "Unexpected alignment");

            size_t new_offs;

            if (size == 1) {
                new_offs = get_1byte();
            } else if (size <= 4) {
                new_offs = get_4byte();
            } else if (align <= 4) {
                new_offs = get_4byte_aligned_block(size);
            } else {
                new_offs = get_16byte_aligned_block(size);
            }

            // no slot found?
            if (new_offs == ~0) {
                return false;
            }

            size_t offs_align = get_offset_alignment(new_offs);
            if (offs_align > align) {
                ++m_num_overaligned_results;
            }

            result_candidate->texture_result_offset = new_offs;
            m_allocated_result_candidates.push_back(result_candidate);

            return true;
        }

    public:
        /// Allocate a (movable) slot in the texture results.
        ///
        /// \returns false when no suitable slot is available anymore
        bool alloc(Result_candidate_info *result_candidate)
        {
            if (alloc_impl(result_candidate)) {
                return true;
            }

            // allocation failed, check total available size
            unsigned available_size =
                m_free_1byte_list.size() +
                m_free_4byte_list.size() * 4 +
                m_free_16byte_list.size() * 16;
            if (result_candidate->size > available_size) {
                // not enough space left
                return false;
            }

            // result may still fit. add it to the result candidates and repack all
            m_allocated_result_candidates.push_back(result_candidate);
            if (repack()) {
                return true;
            }

            // repacking failed, so the result candidate does not fit anymore. undo what we did

            // remove the result candidate again
            result_candidate->texture_result_offset = ~0;
            result_candidate->texture_result_index = -1;

            auto it = std::find(
                m_allocated_result_candidates.begin(),
                m_allocated_result_candidates.end(),
                result_candidate);
            if (it != m_allocated_result_candidates.end()) {
                m_allocated_result_candidates.erase(it);
            }

            // re-add result candidates which could not be re-allocated during repacking
            for (Result_candidate_info *failed_cand : m_repack_failed_result_candidates) {
                // skip the result candidate which we wanted to add
                if (failed_cand == result_candidate) {
                    continue;
                }

                m_allocated_result_candidates.push_back(failed_cand);
            }

            // repack the original result candidates again
            bool undo_success = repack();
            MDL_ASSERT(undo_success && "Repacking the original result candidates should succeed");
            (void) undo_success;

            return false;
        }

        /// Free the texture results slot allocated by the given result candidate.
        void free(Result_candidate_info *result_candidate)
        {
            MDL_ASSERT(result_candidate->texture_result_offset != ~0);

            result_candidate->texture_result_offset = ~0;
            result_candidate->texture_result_index = -1;
            auto it = std::find(
                m_allocated_result_candidates.begin(),
                m_allocated_result_candidates.end(),
                result_candidate);
            if (it != m_allocated_result_candidates.end()) {
                m_allocated_result_candidates.erase(it);
            }

            repack();
        }

        /// Repack the result candidates in the texture results to use less memory
        /// or to fit more result candidates.
        ///
        /// \returns True, if all result candidates could be repacked.
        ///    False otherwise. In this case m_repack_failed_result_candidates contains the
        ///    result candidates, which did not fit.
        bool repack()
        {
            mi::mdl::vector<Result_candidate_info *>::Type old_result_candidates(
                m_allocated_result_candidates, m_alloc);

            // reset all lists
            m_free_1byte_list.clear();
            m_free_4byte_list.clear();
            m_free_16byte_list.clear();
            m_num_overaligned_results = 0;
            m_allocated_result_candidates.clear();
            m_repack_failed_result_candidates.clear();

            // initialize 16-byte free list
            m_free_16byte_list.reserve(m_texture_result_size / 16);
            for (size_t i = 0, n = m_texture_result_size / 16, offs = (n - 1) * 16;
                    i < n; ++i, offs -= 16) {
                m_free_16byte_list.push_back(offs);
            }

            // start repacking result candidates with aligns >= 16
            for (auto it = old_result_candidates.begin(); it != old_result_candidates.end(); ) {
                if ((*it)->align >= 16) {
                    if (!alloc_impl(*it)) {
                        m_repack_failed_result_candidates.push_back(*it);
                    }
                    it = old_result_candidates.erase(it);
                } else {
                    ++it;
                }
            }

            // repack result candidates with aligns > 4
            for (auto it = old_result_candidates.begin(); it != old_result_candidates.end(); ) {
                if ((*it)->align > 4) {
                    if (!alloc_impl(*it)) {
                        m_repack_failed_result_candidates.push_back(*it);
                    }
                    it = old_result_candidates.erase(it);
                } else {
                    ++it;
                }
            }

            // repack result candidates with aligns == 4
            for (auto it = old_result_candidates.begin(); it != old_result_candidates.end(); ) {
                if ((*it)->align == 4) {
                    if (!alloc_impl(*it)) {
                        m_repack_failed_result_candidates.push_back(*it);
                    }
                    it = old_result_candidates.erase(it);
                } else {
                    ++it;
                }
            }

            // allocate the rest
            for (auto it = old_result_candidates.begin(); it != old_result_candidates.end(); ++it) {
                if (!alloc_impl(*it)) {
                    m_repack_failed_result_candidates.push_back(*it);
                }
            }

            return m_repack_failed_result_candidates.empty();
        }

    private:
        /// The allocator.
        mi::mdl::IAllocator *m_alloc;

        /// The total available texture result size in bytes.
        size_t m_texture_result_size;

        /// A list of free 1 byte blocks.
        mi::mdl::vector<size_t>::Type m_free_1byte_list;

        /// A list of free aligned 4 byte blocks.
        mi::mdl::vector<size_t>::Type m_free_4byte_list;

        /// A list of free aligned 16 byte blocks.
        mi::mdl::vector<size_t>::Type m_free_16byte_list;

        /// Number of results, which have a higher alignment than needed.
        unsigned m_num_overaligned_results;

        /// List of result candidates, for which texture result slots were allocated.
        /// Will be used, if the offsets need to be reordered in order to make room for
        /// results which need a higher alignment.
        mi::mdl::vector<Result_candidate_info *>::Type m_allocated_result_candidates;

        /// List of result candidates which could not be allocated during repacking.
        mi::mdl::vector<Result_candidate_info *>::Type m_repack_failed_result_candidates;
    };

public:
    /// Constructor.
    Expression_scheduler(
        IAllocator                             *alloc,
        ITarget_type_properties                &target_type_properties,
        ICode_generator::Target_language       target_language,
        bool                                   init_loop_enabled,
        bool                                   target_is_structured_language,
        unsigned                               num_texture_results,
        ICall_name_resolver const              *resolver,
        Distribution_function const            &dist_func,
        mi::mdl::vector<Schedule_entry>::Type  &schedule,
        Loop_schedule                          &loop_schedule)
        : m_alloc(alloc)
        , m_builder(alloc)
        , m_target_language(target_language)
        , m_target_type_properties(target_type_properties)
        , m_num_texture_results(num_texture_results)
        , m_resolver(resolver)
        , m_dist_func(dist_func)
        , m_cost_provider(
            m_alloc,
            m_resolver,
            m_target_language,
            m_target_type_properties)
        , m_result_candidate_map(alloc)
        , m_result_candidates(alloc)
        , m_results_schedule(alloc)
        , m_schedule(schedule)
        , m_loop_schedule(loop_schedule)
        , m_text_results_alloc(alloc, num_texture_results * 16)
        , m_node_info_map(alloc)
        , m_used_node_list(alloc)
        , m_init_loop_enabled(init_loop_enabled)
        , m_target_is_structured_language(target_is_structured_language)
    {
    }

    /// Destructor.
    ~Expression_scheduler()
    {
        for (Result_candidate_info *result_candidate : m_result_candidates) {
            m_builder.destroy(result_candidate);
        }
    }

    /// Get the schedule.
    mi::mdl::vector<Schedule_entry>::Type &get_schedule() const
    {
        return m_schedule;
    }

    /// Determines whether the called function is depending on the evaluation state
    /// not considering the arguments.
    ///
    /// \param call  the call to check
    bool is_eval_state_dependent_direct(DAG_call const *call)
    {
        // operators and special DAG nodes never depend on the state normal and get_owner_module
        // or find_signature may fail for them
        Definition::Semantics sema = call->get_semantic();
        if (semantic_is_operator(sema) || is_DAG_semantics(sema)) {
            return false;
        }

        char const *signature = call->get_name();
        if (signature[0] == '#') {
            // skip prefix for derivative variants
            ++signature;
        }
        mi::base::Handle<Module const> mod(
            impl_cast<Module>(m_resolver->get_owner_module(signature)));
        if (!mod.is_valid_interface()) {
            MDL_ASSERT(!"get_owner_module should not fail for non-operator call");
            return false;
        }

        Module const *module = mod.get();

        IDefinition const *def = module->find_signature(signature, /*only_exported=*/false);
        if (def == nullptr) {
            MDL_ASSERT(!"find_signature should not fail for non-operator call");
            return false;
        }

        // skip presets
        def = skip_presets(def, mod);

        // as we divide only in "before state::normal()" and "after state::normal()", we
        // just check for this property here
        return def->get_property(IDefinition::DP_USES_NORMAL);
    }

    /// Checks whether the type is a DF type or contains a DF type.
    ///
    /// \param type  the type to check
    static bool contains_df_type(IType const *type)
    {
        type = type->skip_type_alias();
        switch (type->get_kind()) {
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
            return true;
        case IType::TK_ARRAY:
            return contains_df_type(as<IType_array>(type)->get_element_type());
        case IType::TK_STRUCT:
            {
                IType_compound const *comp_type = as<IType_compound>(type);
                for (int i = 0, n = comp_type->get_compound_size(); i < n; ++i) {
                    if (contains_df_type(comp_type->get_compound_type(i))) {
                        return true;
                    }
                }
                return false;
            }
        default:
            return false;
        }
    }

    /// Walk the expression to collect the used nodes, how many times they are used,
    /// and determine, whether the expression is evaluation state dependent.
    /// If so, the function returns true.
    /// The req_eval_state is only relevant for the used nodes counter, not for the visited status,
    /// as the req_eval_state depends on the top-level request, which does not change during a walk.
    bool calc_eval_state_dep_and_usage_counts(
        DAG_node const                    *expr,
        Distribution_function::Eval_state  req_eval_state,
        unsigned                          &walk_id)
    {
        Node_info &info = m_node_info_map[expr];

        // stop when already visited in this walk
        if (info.already_visited(walk_id)) {
            return info.is_eval_state_dependent;
        }

        // first time ever seen?
        if (info.last_walk_id == 0) {
            // add to used nodes list
            m_used_node_list.push_back(expr);
        }

        // stop if a non-DF node was already seen in another walk for this eval state
        if (!contains_df_type(expr->get_type()) && info.get_count(req_eval_state) > 0) {
            info.inc_count(req_eval_state, walk_id);
            return info.is_eval_state_dependent;
        }

        unsigned node_walk_id = walk_id;

        bool res = false;
        switch (expr->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                res = calc_eval_state_dep_and_usage_counts(expr, req_eval_state, walk_id);
                break;
            }
        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            // note: parameters cannot be evaluation state dependent. If state::normal()
            //    was used as an argument during material instantiation, the
            //    corresponding parameter would not be a parameter anymore (but inlined).
            break;
        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(expr);
                IDefinition::Semantics sema = call->get_semantic();

                bool is_df_sema = is_df_semantics(sema);

                int n_args = call->get_argument_count();
                for (int i = 0; i < n_args; ++i) {
                    DAG_node const *arg = call->get_argument(i);

                    // new walks start at non-DFish arguments of DFs
                    if (is_df_sema && !contains_df_type(arg->get_type())) {
                        ++walk_id;
                    }

                    res |= calc_eval_state_dep_and_usage_counts(arg, req_eval_state, walk_id);
                }

                // only check this call, if we haven't found a state dependent call, yet
                if (!res) {
                    res = is_eval_state_dependent_direct(call);
                }

                break;
            }
        }

        info.is_eval_state_dependent = res;

        // update counter now, that evaluation state dependence is known
        info.inc_count(req_eval_state, node_walk_id);

        return res;
    }

    /// Add a result candidate.
    Result_candidate_info *add_result_candidate(
        DAG_node const                    *expr,
        bool                               is_requested,
        Distribution_function::Eval_state  eval_state,
        unsigned                           usage_factor)
    {
        Node_info &info = m_node_info_map[expr];
        IType const *type = expr->get_type();

        unsigned size = unsigned(m_target_type_properties.get_store_size(type));

        // HLSL and GLSL use float4 arrays with elementwise assignments
        unsigned align = m_target_is_structured_language ? 4
            : m_target_type_properties.get_ABI_alignment(type);

        m_result_candidates.emplace_back(
            m_builder.create<Result_candidate_info>(
                m_alloc, expr, size, align,
                is_requested, info.is_eval_state_dependent, eval_state, usage_factor));

        set_result_candidate(m_result_candidates.back());

        return m_result_candidates.back();
    }

    /// Add result candidates for an array or constructor element by element to avoid
    /// storing constants and parameters or cheap values in texture results.
    void add_result_candidates_elementwise(
        DAG_node const                    *expr,
        bool                               is_requested,
        Distribution_function::Eval_state  eval_state,
        unsigned                           usage_factor)
    {
        DAG_call const *call = cast<DAG_call>(expr);
        for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
            DAG_node const *arg = call->get_argument(i);
            switch (arg->get_kind()) {
            case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                add_result_candidates_elementwise(expr, is_requested, eval_state, usage_factor);
                break;
            }
            case DAG_node::EK_CONSTANT:
            case DAG_node::EK_PARAMETER:
                // no result candidates for constants and parameters
                break;
            case DAG_node::EK_CALL:
                // for arrays, go deeper
                if (is_array_or_elem_constructor(arg)) {
                    add_result_candidates_elementwise(arg, is_requested, eval_state, usage_factor);
                }
                // no array, just add it if suitable
                else if (can_be_result_candidate(arg)) {
                    add_result_candidate(arg, is_requested, eval_state, usage_factor);
                }
                break;
            }
        }
    }

    /// Change a result candidate to point to a different DAG node.
    void change_result_candidate_node(
        Result_candidate_info *result_candidate,
        DAG_node const        *new_node)
    {
        Node_info &info = m_node_info_map[new_node];
        IType const *node_type = new_node->get_type();

        unsigned size = unsigned(m_target_type_properties.get_store_size(node_type));

        // HLSL and GLSL use float4 arrays with elementwise assignments
        unsigned align = m_target_is_structured_language ? 4
            : m_target_type_properties.get_ABI_alignment(node_type);

        result_candidate->node = new_node;
        result_candidate->is_eval_state_dependent = info.is_eval_state_dependent;
        result_candidate->size = size;
        result_candidate->align = align;
    }

    /// Add result candidates for a requested DF.
    void add_requested_df_result_candidates(
        DAG_node const                    *expr,
        Distribution_function::Eval_state  eval_state,
        unsigned                           walk_id)
    {
        Node_info &info = m_node_info_map[expr];

        // Stop when already visited in this walk
        if (info.already_visited(walk_id, eval_state)) {
            return;
        }

        info.mark_visited(walk_id, eval_state);

        // found non-DF node, add this as result candidate and stop walk
        if (!contains_df_type(expr->get_type())) {
            // for arrays, add suitable array elements
            if (is_array_or_elem_constructor(expr)) {
                add_result_candidates_elementwise(expr, true, eval_state, 2);
            }
            // only add, if suitable for result candidates
            else if (can_be_result_candidate(expr)) {
                // add as requested with 2 usages (at least sample and evaluate)
                add_result_candidate(expr, true, eval_state, 2);
            }
            return;
        }

        switch (expr->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                add_requested_df_result_candidates(expr, eval_state, walk_id);
                break;
            }
        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            // no need to store in texture results
            break;
        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(expr);

                int n_args = call->get_argument_count();
                for (int i = 0; i < n_args; ++i) {
                    DAG_node const *arg = call->get_argument(i);
                    add_requested_df_result_candidates(arg, eval_state, walk_id);
                }
                break;
            }
        }
    }

    /// Walk the graph of a result candidate with other result candidates as borders.
    /// Calculate the local cost to calculate this subgraph and collect the dependencies
    /// on other result candidates.
    void calc_result_candidate_deps_and_local_costs(
        Result_candidate_info *result_candidate,
        DAG_node const        *cur_node,
        unsigned              &walk_id)
    {
        Node_info &info = m_node_info_map[cur_node];

        // Stop when already visited in this walk
        if (info.already_visited(walk_id)) {
            return;
        }

        info.mark_visited(walk_id);

        // is current node another result candidate?
        if (cur_node != result_candidate->node) {
            if (Result_candidate_info *other = get_result_candidate(
                    cur_node, result_candidate->eval_state)) {
                result_candidate->add_direct_dependency(other);
                return;
            }
        }

        result_candidate->local_cost += info.local_cost;
        result_candidate->orig_local_cost += info.local_cost;

        switch (cur_node->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            MDL_ASSERT(!"no temporaries allowed at this point");
            return;

        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            return;

        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(cur_node);

                int n_args = call->get_argument_count();
                for (int i = 0; i < n_args; ++i) {
                    DAG_node const *arg = call->get_argument(i);
                    calc_result_candidate_deps_and_local_costs(
                        result_candidate, arg, walk_id);
                }
                return;
            }
        }

        MDL_ASSERT(!"Unexpected node kind");
    }

    /// Returns true, if the given node is allowed to be stored as texture result.
    /// So it is not a constant, nor a parameter nor part of the state.
    ///
    /// We assume here, that the state is precalculated and at most as expensive to access
    /// as a texture result. So it doesn't make sense to store them in a texture result.
    /// If the state is remapped or implemented via a renderer state module, state accesses
    /// may be more expensive, though.
    bool can_be_result_candidate(DAG_node const *node)
    {
        switch (node->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            MDL_ASSERT(!"no temporaries allowed at this point");
            return false;

        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            return false;

        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(node);

                IDefinition::Semantics sema = call->get_semantic();
                return !is_state_semantics(sema);
            }
        }

        MDL_ASSERT(!"unexpected node kind");
        return false;
    }

#ifdef DEBUG_NEW_SCHEDULER
    /// Walk the expression to calculate the cost of the node and its dependencies.
    /// Returns the total cost. Only used for printing statistics.
    unsigned calc_node_cost(
        DAG_node const                     *expr,
        ptr_hash_set<DAG_node const>::Type &texture_result_nodes,
        unsigned                           &walk_id)
    {
        Node_info &info = m_node_info_map[expr];

        // Stop when already visited in this walk, result is already available in this function
        if (info.already_visited(walk_id)) {
            return 0;
        }

        info.mark_visited(walk_id);

        // Available as texture result? Use it
        if (texture_result_nodes.count(expr) != 0) {
            unsigned result_size = info.get_result_size(expr, m_data_layout, m_type_mapper);
            return m_cost_provider.get_texture_result_cost(result_size);
        }

        switch (expr->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                return calc_node_cost(expr, texture_result_nodes, walk_id);
            }

        case DAG_node::EK_CONSTANT:
            return 0;  // TODO: matrices may have relevant cost

        case DAG_node::EK_PARAMETER:
            return info.local_cost;

        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(expr);

                unsigned res = info.local_cost;

                int n_args = call->get_argument_count();
                for (int i = 0; i < n_args; ++i) {
                    DAG_node const *arg = call->get_argument(i);
                    res += calc_node_cost(arg, texture_result_nodes, walk_id);
                }
                return res;
            }
        }

        MDL_ASSERT(!"Unexpected node kind");
        return 0;
    }

    /// Calculate the cost of a special node if used.
    /// Only used for printing statistics.
    unsigned get_special_node_cost(
        Distribution_function::Special_kind   special_kind,
        ptr_hash_set<DAG_node const>::Type   &texture_result_nodes,
        unsigned                              walk_id)
    {
        size_t index = m_dist_func.get_special_node_index(special_kind);
        if (index == ~0) {
            return 0;
        }

        DAG_node const *node = m_dist_func.get_requested_node(index)->node;
        return calc_node_cost(node, texture_result_nodes, walk_id);
    }

    /// Print the schedule and calculate the cost of requested nodes.
    void print_schedule(unsigned &walk_id)
    {
        printf("\nSchedule:\n");
        size_t texture_result_size = 0;
        for (Result_candidate_info *scheduled_result : m_results_schedule) {
            bool bad_texture_result = false;

            // no texture result should be set for non-requested nodes, unless they are
            // used by requested nodes without texture result (non-scheduled result candidate)
            if (!scheduled_result->is_requested && scheduled_result->texture_result_offset != ~0) {
                bool found_valid_user = false;
                for (Result_candidate_info *user : m_result_candidates) {
                    if (user->is_scheduled || !user->is_requested) {
                        continue;
                    }
                    if (user->transitive_dependencies.count(scheduled_result) > 0) {
                        found_valid_user = true;
                        break;
                    }
                }

                if (!found_valid_user) {
                    bad_texture_result = true;
                }
            }

            printf(
                " - %c %3u: orig local: %4u, local: %4u, result offs: %4d, size: %3u, align: %2u, "
                " eval_state_dep: %s, eval_state: %d, special kinds: %x%s\n",
                scheduled_result->is_requested ? '*' : ' ',
                unsigned(scheduled_result->node->get_id()),
                scheduled_result->orig_local_cost,
                scheduled_result->local_cost,
                int(scheduled_result->texture_result_offset),
                scheduled_result->size,
                scheduled_result->align,
                scheduled_result->is_eval_state_dependent ? "true" : "false",
                int(scheduled_result->eval_state),
                int(scheduled_result->special_kinds),
                bad_texture_result ? "     ### unused texture result!" : "");

            if (scheduled_result->texture_result_offset != ~0) {
                size_t end_offs = scheduled_result->texture_result_offset + scheduled_result->size;
                if (end_offs > texture_result_size) {
                    texture_result_size = end_offs;
                }
            }
        }

        printf("\nSchedule texture result size: %u\n", unsigned(texture_result_size));
        unsigned schedule_cost = 0;
        ptr_hash_set<DAG_node const>::Type texture_result_nodes(m_alloc);

        // first the cost of mdl_init.
        // Mark all DAG nodes stored in the texture results as calculated.
        ++walk_id;
        for (Schedule_entry const &entry : m_schedule) {
            Node_info &info = m_node_info_map[entry.node];
            schedule_cost += calc_node_cost(entry.node, texture_result_nodes, walk_id);

            unsigned result_size = info.get_result_size(entry.node, m_data_layout, m_type_mapper);
            schedule_cost += m_cost_provider.get_texture_result_cost(result_size);

            // stored in texture results?
            if (entry.texture_result_offset != ~0) {
                texture_result_nodes.insert(entry.node);
            }
        }

        printf("\nCosts of schedule:\n      %-30s = %u\n", "mdl_init", schedule_cost);

        size_t normal_index = m_dist_func.get_special_node_index(
            Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL);

        // then add the cost for each function directly requested by the renderer separately,
        // counting DFs twice (at least sample and evaluate).
        // For DFs, this assumes, that all nodes are only calculated once per generated
        // top-level function (by using "lambda results" or by inlining all code into one function)
        for (size_t i = 0, n = m_dist_func.get_total_requested_node_count(); i < n; ++i) {
            Distribution_function::Requested_node const *req =
                m_dist_func.get_requested_node(i);

            ++walk_id;
            unsigned cost;
            if (i == normal_index) {
                cost = 0;  // the geometry.normal is already calculated in the init function
            } else {
                cost = calc_node_cost(req->node, texture_result_nodes, walk_id);
            }
            if (contains_df_type(req->node->get_type())) {
                cost *= 2;
            }
            printf("  %3d %-30s = %u\n", unsigned(req->node->get_id()), req->path, cost);
            schedule_cost += cost;

            // TODO: for nodes implicitly using special nodes, costs of those nodes must be added
            //    for now, assume that all requested BSDF nodes use thin_walled and ior if present
            if (is<IType_bsdf>(req->node->get_type())) {
                schedule_cost += get_special_node_cost(
                    Distribution_function::SK_MATERIAL_THIN_WALLED, texture_result_nodes, walk_id);
                schedule_cost += get_special_node_cost(
                    Distribution_function::SK_MATERIAL_IOR, texture_result_nodes, walk_id);
            }
        }

        printf("      %-30s = %u\n", "total", schedule_cost);
    }

    /// Dump the distribution function with multiple used nodes marked to a .gv file.
    void dump_with_marked_multiuse_nodes()
    {
        mi::base::Handle<const Lambda_function> root_lambda(m_dist_func.get_root_lambda());

        static int dumpid = 0;
        std::string dumpname("matinst");
        dumpname += std::to_string(dumpid++);
        dumpname += "_full_with_requests-2-scheduler.gv";

        char const *fname = root_lambda->get_dag_unit().get_fname(0);
        if (fname == nullptr) {
            fname = "<unknown file>";
        }
        printf("Dumping material from \"%s\" as %s\n", fname, dumpname.c_str());

        if (FILE *f = fopen(dumpname.c_str(), "w")) {
            Allocator_builder builder(m_alloc);
            mi::base::Handle<File_Output_stream> out(
                builder.create<File_Output_stream>(m_alloc, f, /*close_at_destroy=*/true));

            Distribution_function_dumper::Node_color_map node_color_map(m_alloc);
            for (size_t i = 0, n = m_dist_func.get_total_requested_node_count(); i < n; ++i) {
                Distribution_function::Requested_node const *req =
                    m_dist_func.get_requested_node(i);
                node_color_map[req->node] = "\"#E29245\"";  // orange: requested node
            }

            for (int i = 0; i < Distribution_function::SK_NUM_KINDS; ++i) {
                size_t special_node_index = m_dist_func.get_special_node_index(
                    Distribution_function::Special_kind(i));
                if (special_node_index == ~0) {
                    continue;
                }

                Distribution_function::Requested_node const *special_node =
                    m_dist_func.get_requested_node(special_node_index);

                // only mark special node, if they were not requested specifically
                if (node_color_map.find(special_node->node) == node_color_map.end()) {
                    // green: special nodes, indirectly requested node
                    node_color_map[special_node->node] = "\"#92E245\"";
                }
            }

            // mark all nodes which are used multiple times
            for (DAG_node const *node : m_used_node_list) {
                // skip nodes which are already marked
                if (node_color_map.find(node) != node_color_map.end()) {
                    continue;
                }

                // skip constants, parameters and df nodes
                if (is<DAG_constant>(node) ||
                        is<DAG_parameter>(node) ||
                        contains_df_type(node->get_type())) {
                    continue;
                }

                for (unsigned i = 0; i <= Distribution_function::ES_LAST; ++i) {
                    if (m_node_info_map[node].count[i] > 1) {
                        node_color_map[node] = "\"#E23785\"";  // dark pink: "multiply used"
                    }
                }
            }

            Distribution_function_dumper dumper(m_alloc, out.get(), &m_dist_func, node_color_map);
            dumper.dump();
        }
    }

    /// Print the result candidates for debugging.
    void print_result_candidates(
        mi::mdl::vector<Result_candidate_info *>::Type &sorted_result_candidates)
    {
        printf("\n\nTotal cost of used result candidates without texture results:\n");
        for (Result_candidate_info *result_candidate : sorted_result_candidates) {
            // already stored (in texture results or state normal)? ignore
            if (result_candidate->is_stored()) {
                continue;
            }

            // not used anymore? ignore
            if (!result_candidate->is_requested && result_candidate->direct_usage_count == 0) {
                continue;
            }

            char const *name = "";
            if (DAG_call const *call = as<DAG_call>(result_candidate->node)) {
                name = call->get_name();
            }
            unsigned size = m_node_info_map[result_candidate->node].get_result_size(
                result_candidate->node, m_data_layout, m_type_mapper);
            printf(" - %c %3u %-60s: %u (local: %u, result size: %u, eval state: %d, "
                "direct usage: %u, usage factor: %u",
                result_candidate->is_requested ? '*' : ' ',
                unsigned(result_candidate->node->get_id()),
                name,
                result_candidate->total_cost,
                result_candidate->local_cost,
                size,
                result_candidate->is_eval_state_dependent ? int(result_candidate->eval_state) : -1,
                result_candidate->direct_usage_count,
                result_candidate->usage_factor);

            if (!result_candidate->direct_dependencies_list.empty()) {
                printf(", ddeps: ");
                bool first = true;
                for (Result_candidate_info *dep : result_candidate->direct_dependencies_list) {
                    if (first) {
                        first = false;
                    } else {
                        printf(", ");
                    }
                    printf("%3u", unsigned(dep->node->get_id()));
                }

                printf(", tdeps: ");
                first = true;
                for (Result_candidate_info *tdep : result_candidate->transitive_dependencies) {
                    if (first) {
                        first = false;
                    } else {
                        printf(", ");
                    }
                    printf("%3u", unsigned(tdep->node->get_id()));
                }
            }
            printf(")\n");
        }
    }
#endif

    /// Add a result candidate and its dependencies to the schedule in topological order.
    void add_to_schedule(Result_candidate_info *result_candidate)
    {
        if (result_candidate->is_scheduled) {
            return;
        }
        result_candidate->is_scheduled = true;

        for (Result_candidate_info *dep : result_candidate->direct_dependencies_list) {
            add_to_schedule(dep);
        }

        m_results_schedule.push_back(result_candidate);
    }

    /// Update total costs, direct usage count and transitive dependencies of result candidates.
    /// Also sorts the result candidates again.
    void update_and_sort_result_candidates(
        mi::mdl::vector<Result_candidate_info *>::Type &sorted_result_candidates,
        unsigned &graph_version)
    {
        ++graph_version;

        // need to first clear direct usage count
        for (Result_candidate_info *result_candidate : m_result_candidates) {
            result_candidate->direct_usage_count = 0;
        }
        // update transitive dependencies and total costs, starting at requested result candidates
        for (Result_candidate_info *result_candidate : m_result_candidates) {
            if (result_candidate->is_requested) {
                result_candidate->update_info(graph_version, m_cost_provider);
            }
        }

        // check for unused texture results of non-requested nodes
        bool freed_unused_texture_results = false;
        for (Result_candidate_info *result_candidate : m_result_candidates) {
            if (!result_candidate->is_requested && result_candidate->texture_result_offset != ~0) {
                bool found_valid_user = false;
                for (Result_candidate_info *user : m_result_candidates) {
                    // only consider requested nodes as valid users
                    if (!user->is_requested) {
                        continue;
                    }
                    // result is already stored, so result_candidate is not needed anymore by user?
                    if (user->is_stored()) {
                        continue;
                    }
                    if (user->transitive_dependencies.count(result_candidate) > 0) {
                        found_valid_user = true;
                        break;
                    }
                }

                if (!found_valid_user) {
                    m_text_results_alloc.free(result_candidate);
                    freed_unused_texture_results = true;
                }
            }
        }

        // if texture results were freed, we need to restart the update
        if (freed_unused_texture_results) {
            update_and_sort_result_candidates(
                sorted_result_candidates, graph_version);
            return;
        }

        ++graph_version;

        // "flatten" graph by moving up local cost, if direct usage is 1 and !is_requested,
        // starting at requested result candidates
        for (Result_candidate_info *result_candidate : m_result_candidates) {
            if (result_candidate->is_requested) {
                result_candidate->flatten_graph(graph_version);
            }
        }

        // sort the result candidates by decreasing local cost per byte.
        std::sort(
            sorted_result_candidates.begin(), sorted_result_candidates.end(),
            [](Result_candidate_info const *a, Result_candidate_info const *b)
            {
                return a->local_cost * 16 / std::max(a->size, 1u) >
                    b->local_cost * 16 / std::max(b->size, 1u);
            }
        );

#ifdef DEBUG_NEW_SCHEDULER
        print_result_candidates(sorted_result_candidates);
#endif
    }

    /// Returns true if the given node is an array constructor or elemental constructor call.
    bool is_array_or_elem_constructor(DAG_node const *node) {
        if (DAG_call const *call = as<DAG_call>(node)) {
            IDefinition::Semantics sema = call->get_semantic();
            return sema == IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR
                || sema == IDefinition::DS_ELEM_CONSTRUCTOR;
        }
        return false;
    }

    /// Collect all texture result candidates by processing the special and requested nodes
    /// of the distribution function.
    void collect_result_candidates(unsigned &walk_id)
    {
        ++walk_id;

        // process special nodes as requested result candidates
        MDL_ASSERT(Distribution_function::SK_MATERIAL_GEOMETRY_DISPLACEMENT <
            Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL &&
            "Expect displacement to be checked first in the loop");
        for (int i = 0; i < Distribution_function::SK_NUM_KINDS; ++i) {
            size_t special_node_index = m_dist_func.get_special_node_index(
                Distribution_function::Special_kind(i));
            if (special_node_index == ~0) {
                continue;
            }

            Distribution_function::Requested_node const *special_node =
                m_dist_func.get_requested_node(special_node_index);

            // skip constants, parameters and state calls, unless it is geometry.normal,
            // displacement or cutout opacity, which need to be handled specially before
            // state.normal is updated
            if (i != Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL &&
                    i != Distribution_function::SK_MATERIAL_GEOMETRY_DISPLACEMENT &&
                    i != Distribution_function::SK_MATERIAL_GEOMETRY_CUTOUT_OPACITY &&
                    !can_be_result_candidate(special_node->node)) {
                continue;
            }

            Node_info &info = m_node_info_map[special_node->node];
            if (info.already_visited(walk_id, special_node->eval_state)) {
                // already visited, but still add special kind to existing result candidate.
                // this should only happen for geometry.normal after displacement has been visited
                MDL_ASSERT(i == Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL);

                Result_candidate_info *special_result_candidate = get_result_candidate(
                    special_node->node, special_node->eval_state);
                special_result_candidate->add_special_kind(Distribution_function::Special_kind(i));
                continue;
            }

            info.mark_visited(walk_id, special_node->eval_state);

            unsigned usage_factor;
            if (i == Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL ||
                    i == Distribution_function::SK_MATERIAL_GEOMETRY_DISPLACEMENT ||
                    i == Distribution_function::SK_MATERIAL_GEOMETRY_CUTOUT_OPACITY) {
                usage_factor = 1;
            } else {
                // non-geometry special nodes are at least used twice (sample and evaluate)
                usage_factor = 2;
            }
            Result_candidate_info *special_result_candidate = add_result_candidate(
                special_node->node, true, special_node->eval_state, usage_factor);
            special_result_candidate->add_special_kind(Distribution_function::Special_kind(i));
        }

        // process the (directly and indirectly) requested nodes
        for (size_t i = 0, n = m_dist_func.get_total_requested_node_count(); i < n; ++i) {
            Distribution_function::Requested_node const *req =
                m_dist_func.get_requested_node(i);

            // skip constants, parameters and state calls
            if (!can_be_result_candidate(req->node)) {
                continue;
            }

            Node_info &info = m_node_info_map[req->node];
            if (info.already_visited(walk_id, req->eval_state)) {
                continue;
            }

            // for requested DFs, we need to walk the graph and collect the non-DF nodes as
            // result candidates. They will be marked as double uses (at least sample and evaluate)
            if (contains_df_type(req->node->get_type())) {
                add_requested_df_result_candidates(req->node, req->eval_state, walk_id);
            } else if (is_array_or_elem_constructor(req->node)) {
                info.mark_visited(walk_id, req->eval_state);
                add_result_candidates_elementwise(req->node, true, req->eval_state, 1);
            } else {
                // just a non-DF node, add directly
                info.mark_visited(walk_id, req->eval_state);
                add_result_candidate(req->node, true, req->eval_state, 1);
            }
        }

        // the candidates added here are only temporary results for other candidates.
        // They are used multiple times, so if the cost for this node is spent, all users
        // become cheaper
        for (DAG_node const *node : m_used_node_list) {
            Node_info &info = m_node_info_map[node];
            if (info.already_visited(walk_id, Distribution_function::ES_AFTER_GEOMETRY_NORMAL)) {
                continue;
            }

            // skip constants, parameters, state calls and DF nodes
            if (!can_be_result_candidate(node) || contains_df_type(node->get_type())) {
                continue;
            }

            // We don't store any temporary results which are only valid before geometry.normal
            unsigned count = info.get_count(Distribution_function::ES_AFTER_GEOMETRY_NORMAL);
            if (count > 1) {
                add_result_candidate(
                    node, false, Distribution_function::ES_AFTER_GEOMETRY_NORMAL, count);
            }
        }
    }

    /// Get the result candidate for a given DAG node and an evaluation state, if it exists,
    /// or nullptr otherwise.
    Result_candidate_info *get_result_candidate(
        DAG_node const *node,
        Distribution_function::Eval_state eval_state)
    {
        auto it = m_result_candidate_map.find(node);
        if (it == m_result_candidate_map.end()) {
            return nullptr;
        }

        if (eval_state == Distribution_function::ES_BEGIN_STATE) {
            return it->second.first;
        } else {
            return it->second.second;
        }
    }

    /// Remove the given result candidate from the result candidate map.
    void unset_result_candidate(
        Result_candidate_info *result_candidate)
    {
        auto &res = m_result_candidate_map[result_candidate->node];

        if (result_candidate->is_eval_state_dependent) {
            if (result_candidate->eval_state == Distribution_function::ES_BEGIN_STATE) {
                res.first = nullptr;
            } else {
                res.second = nullptr;
            }
        } else {
            res.first = nullptr;
            res.second = nullptr;
        }
    }

    /// Set the result candidate for a given DAG node and evaluation state.
    void set_result_candidate(
        Result_candidate_info *result_candidate)
    {
        auto &res = m_result_candidate_map[result_candidate->node];

        if (result_candidate->is_eval_state_dependent) {
            if (result_candidate->eval_state == Distribution_function::ES_BEGIN_STATE) {
                res.first = result_candidate;
            } else {
                res.second = result_candidate;
            }
        } else {
            res.first = result_candidate;
            res.second = result_candidate;
        }
    }

    /// Try to reduce the size of the result candidates.
    void optimize_result_candidates()
    {
        // reduce up (struct -> tuple of used struct fields or extract values and call new
        // constructor with only used fields and rest nulled) or down (rematerialize to reduce
        // size of data in texture results)
        // TODO: implement reduce up for example for structs, where only some fields are used
        // TODO: rematerialization cost calculation should also count in the number of uses

        mi::mdl::vector<Result_candidate_info *>::Type candidates_to_remove(m_alloc);

        // try to rematerialize result candidates to decrease the size of their results.
        // only considers calls with a single argument
        for (Result_candidate_info *result_candidate : m_result_candidates) {
            // skip processing special nodes, these need to be handled as they are
            if (result_candidate->special_kinds != 0) {
                continue;
            }

            DAG_node const *cur_node = result_candidate->node;
            unsigned cur_size = result_candidate->size;

            DAG_node const *best_node = cur_node;
            unsigned best_size = cur_size;

            unsigned remat_cost = 0;
            while (true) {
                DAG_call const *call = as<DAG_call>(cur_node);

                // not a call? we're done
                if (call == nullptr) {
                    break;
                }

                // we only follow calls with one argument
                if (call->get_argument_count() != 1) {
                    break;
                }

                Node_info &info = m_node_info_map[call];

                // rematerialization is getting too expensive?
                if (remat_cost + info.local_cost > m_cost_provider.get_cost(
                        Cost_provider::CK_MIN_STORE_RESULT_COST)) {
                    break;
                }

                DAG_node const *arg = call->get_argument(0);
                Node_info &arg_info = m_node_info_map[arg];
                unsigned arg_size = arg_info.get_result_size(arg, m_target_type_properties);

                // result is getting bigger? abort
                if (arg_size > cur_size) {
                    break;
                }

                // we can follow the argument
                remat_cost += info.local_cost;

                // found a smaller result?
                if (arg_size < cur_size) {
                    best_node = arg;
                    best_size = arg_size;
                }

                cur_node = arg;
                cur_size = arg_size;
            }

            // if we found a smaller result, update the result candidate
            if (best_size < result_candidate->size) {
                // does the best node already exist as result candidate or may not be
                // a result candidate?
                if (get_result_candidate(best_node, result_candidate->eval_state) != nullptr ||
                        !can_be_result_candidate(best_node)) {
                    // mark this result candidate to be removed
                    candidates_to_remove.push_back(result_candidate);
                } else {
                    // no, update this candidate to point to the best node
                    unset_result_candidate(result_candidate);

                    change_result_candidate_node(result_candidate, best_node);

                    set_result_candidate(result_candidate);
                }
            }
        }

        // remove result candidates not needed anymore
        for (Result_candidate_info *result_candidate : candidates_to_remove) {
            auto it = std::find(
                m_result_candidates.begin(), m_result_candidates.end(), result_candidate);
            m_result_candidates.erase(it);
            unset_result_candidate(result_candidate);
            m_builder.destroy(result_candidate);
        }
    }

    /// Schedule the expressions and assign them to texture result slots if necessary.
    bool schedule_expressions()
    {
        mi::base::Handle<const Lambda_function> root_lambda(m_dist_func.get_root_lambda());

        // calculate costs per DAG node
        Node_local_cost_calculator::calc_cost(
            m_alloc,
            m_cost_provider,
            m_node_info_map,
            root_lambda->get_body());

        // Costs for one node is local cost + sum of local costs over all unique transitive
        // dependencies. The cost of a node can be reduced to a small constant, when the result is
        // made available via texture results. The dependencies of that node are then irrelevant.

        // Find DAG nodes which are used by multiple requested nodes.
        // Depending on whether a node is used by multiple requested nodes or expected to be
        // used in a generated function which is called multiple times by the renderer,
        // it may make sense to store the result in a texture result slot.
        // Currently, we assume, that for DF nodes, at least sample and evaluate are called,
        // so DF nodes are expected to be "called twice".

        unsigned walk_id = 0;

        // determine whether the nodes are evaluation state dependent and calculate the usage
        // counts of all nodes reachable via requested nodes
        for (size_t i = 0, n = m_dist_func.get_total_requested_node_count(); i < n; ++i) {
            Distribution_function::Requested_node const *req_node =
                m_dist_func.get_requested_node(i);

            calc_eval_state_dep_and_usage_counts(req_node->node, req_node->eval_state, ++walk_id);

            // for requested DF nodes, add a second use (assume at least sample + evaluate)
            if (contains_df_type(req_node->node->get_type())) {
                calc_eval_state_dep_and_usage_counts(
                    req_node->node, req_node->eval_state, ++walk_id);
            }
        }

#ifdef DEBUG_NEW_SCHEDULER
        dump_with_marked_multiuse_nodes();
#endif

        // collect and optimize result candidates
        collect_result_candidates(walk_id);
        optimize_result_candidates();

        // calculate dependencies between result candidates and their local costs
        for (Result_candidate_info *result_candidate : m_result_candidates) {
            ++walk_id;
            calc_result_candidate_deps_and_local_costs(
                result_candidate,
                result_candidate->node,
                walk_id);
        }

        // init sorted list of result candidates
        mi::mdl::vector<Result_candidate_info *>::Type sorted_result_candidates(m_alloc);
        sorted_result_candidates.reserve(m_result_candidates.size());
        for (Result_candidate_info *result_candidate : m_result_candidates) {
            sorted_result_candidates.push_back(result_candidate);
        }

        // initialize total costs, direct usage counts and transitive dependencies
        unsigned graph_version = 1;  // update whenever the graph or any costs are changed
        update_and_sort_result_candidates(sorted_result_candidates, graph_version);

        mi::mdl::vector<std::pair<DAG_node const *, unsigned>>::Type sorted_nodes(m_alloc);
        sorted_nodes.reserve(m_used_node_list.size());
        for (DAG_node const *node : m_used_node_list) {
            sorted_nodes.emplace_back(node, m_node_info_map[node].local_cost);
        }

        // sort the DAG nodes by decreasing local costs
        std::sort(
            sorted_nodes.begin(), sorted_nodes.end(),
            [](std::pair<DAG_node const *, unsigned> const &a,
                std::pair<DAG_node const *, unsigned> const &b)
            {
                // compare local costs of the nodes
                return a.second > b.second;
            }
        );

#ifdef DEBUG_NEW_SCHEDULER
        printf("\nTop 10 local cost of DAG nodes:\n");
        for (size_t i = 0, n = sorted_nodes.size(); i < 10 && i < n; ++i) {
            auto &pair = sorted_nodes[i];

            char const *name = "";
            if (DAG_call const *call = as<DAG_call>(pair.first)) {
                name = call->get_name();
            }
            printf(" - %3u = %u: %-40s\n",
                unsigned(pair.first->get_id()), pair.second, name);
        }
#endif

        // Select geometry displacement and cutout_opacity results if present as special nodes.
        // In this case, they depend on the original normal and the normal is modified by
        // geometry.normal. So the results must be calculated (and stored) before
        // the normal is modified
        bool changed = false;
        Result_candidate_info *normal_res = nullptr;
        for (Result_candidate_info *result_candidate : m_result_candidates) {
            // remember geometry.normal result candidate
            if (result_candidate->has_special_kind(
                    Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL)) {
                MDL_ASSERT(normal_res == nullptr &&
                    "Only one result candidate may represent the geometry.normal");
                normal_res = result_candidate;
            }

            // not geometry displacement or cutout opacity? skip
            if (!result_candidate->has_special_kind(
                        Distribution_function::SK_MATERIAL_GEOMETRY_DISPLACEMENT) &&
                    !result_candidate->has_special_kind(
                        Distribution_function::SK_MATERIAL_GEOMETRY_CUTOUT_OPACITY)) {
                continue;
            }

            // allocate a texture result
            if (!m_text_results_alloc.alloc(result_candidate)) {
                // result doesn't fit into texture results anymore
                return false;
            }

            // add all transitive dependencies and the result candidate to schedule
            add_to_schedule(result_candidate);
            changed = true;
        }

        // add geometry.normal to the schedule (if not already scheduled)
        if (normal_res != nullptr) {
            add_to_schedule(normal_res);
            changed = true;
        }

        // update costs and choose next result candidates until nothing changes anymore.
        // total costs get cheaper, when dependencies are stored as texture results.
        // local costs become more expensive, if a dependency is "merged" into its
        // only remaining user.
        // TODO: updates could be cheaper, if result candidates knew their users
        // TODO: total costs are currently not used at all
        do {
            if (changed) {
                update_and_sort_result_candidates(sorted_result_candidates, graph_version);
                changed = false;
            }

            for (Result_candidate_info *result_candidate : sorted_result_candidates) {
                // ignore unused temporary result candidates
                if (!result_candidate->is_requested && result_candidate->direct_usage_count == 0) {
                    continue;
                }

                // stored in texture results or state normal already? ignore
                if (result_candidate->is_stored()) {
                    continue;
                }

                // too cheap?
                if (result_candidate->local_cost < m_cost_provider.get_cost(
                        Cost_provider::CK_MIN_STORE_RESULT_COST)) {
                    continue;
                }

                // TODO: if the current result_candidate is the last user of a texture result,
                //   and with that slot, there would be enough space, the texture result should
                //   first be freed

                if (!m_text_results_alloc.alloc(result_candidate)) {
                    // result doesn't fit into texture results anymore
                    continue;
                }

                // add all transitive dependencies and the result candidate to schedule
#ifdef DEBUG_NEW_SCHEDULER
                printf("\nChosen result candidate for texture results: %u (eval_state: %d)\n",
                    unsigned(result_candidate->node->get_id()), int(result_candidate->eval_state));
#endif
                add_to_schedule(result_candidate);
                changed = true;
                break;
            }
        } while (changed);

        // repack texture results
        m_text_results_alloc.repack();

        // create output schedule
        m_schedule.reserve(m_results_schedule.size());
        for (Result_candidate_info *scheduled_result : m_results_schedule) {
            m_schedule.emplace_back(
                scheduled_result->node,
                scheduled_result->special_kinds,
                scheduled_result->is_eval_state_dependent,
                scheduled_result->eval_state,
                scheduled_result->texture_result_index,
                scheduled_result->texture_result_offset);
        }

#ifdef DEBUG_NEW_SCHEDULER
        // only for debugging and showing schedule costs
        update_and_sort_result_candidates(sorted_result_candidates, graph_version);

        print_schedule(walk_id);
#endif

        // avoid inlining of multiply called expensive functions by using a loop
        // -> prerequisite DAG nodes, [params, call, result DAG node, keep or store]
        // does loop stuff influence which nodes to store in texture results?
        //   if expensive functions remain, we blow up the code...
        //   OmniSurface has 35 texture lookups
        //   -> not all may fit
        //   -> determine reduced sizes of all results which would be stored in texture results
        //   -> fill with smallest results while maintaining dependencies

        // influence on order of evaluation or storing in texture results:
        //  - dependencies
        //  - avoiding unused gaps in texture results
        //  - reduce register pressure when there are common dependencies?
        //    A needs B and C, D only needs B. Then first evaluate A to get rid of C.

        // when to execute for-loop? some results may be calculated before already (geometry.normal)

#if 0
        Init_loop_scheduler init_lp_sched(
            m_alloc,
            m_init_loop_enabled,
            m_target_is_structured_language,
            m_resolver,
            m_schedule,
            sorted_nodes,
            m_node_info_map,
            m_target_type_properties);
        init_lp_sched.schedule(m_loop_schedule);
#endif
        return true;
    }

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The allocator builder.
    Allocator_builder m_builder;

    /// The target language for which code is generated.
    ICode_generator::Target_language m_target_language;

    /// The target type properties.
    ITarget_type_properties const &m_target_type_properties;

    /// The number of texture result entries.
    unsigned m_num_texture_results;

    /// The resolver for calls.
    ICall_name_resolver const *m_resolver;

    /// The distribution function.
    Distribution_function const &m_dist_func;

    /// The cost provider.
    Cost_provider m_cost_provider;

    /// Map from DAG nodes to result candidates which have the DAG node as node field.
    /// The first result candidate of the pair is for the begin evaluation state, the second for the
    /// after geometry normal evaluation state. If the node is evaluation state independent,
    /// then both point to the same result candidate.
    ptr_hash_map<DAG_node const, std::pair<Result_candidate_info *, Result_candidate_info *>>::Type
        m_result_candidate_map;

    /// Vector to hold all allocated result candidates.
    mi::mdl::vector<Result_candidate_info *>::Type m_result_candidates;

    /// List of result candidates in the scheduled order.
    mi::mdl::vector<Result_candidate_info *>::Type m_results_schedule;

    /// List of schedule entries, which will be produced by this class.
    mi::mdl::vector<Schedule_entry>::Type &m_schedule;

    /// Loop schedule for init funciton, which will be filled in by this class.
    Loop_schedule &m_loop_schedule;

    /// The texture result allocator.
    Texture_result_allocator m_text_results_alloc;

    /// Map from DAG nodes to Node_info objects.
    Node_info_map m_node_info_map;

    /// List of used nodes. Needed to deterministically iterate over all nodes.
    mi::mdl::vector<DAG_node const *>::Type m_used_node_list;

    /// Whether the init loop is enabled.
    bool m_init_loop_enabled;

    /// Whether the target is a structured language.
    bool m_target_is_structured_language;
};

} // anonymous namespace


/// Helper class for code generation of df::bsdf_component elements.
class Df_component_info
{
public:
    /// Constructor.
    ///
    /// \param code_gen  The code generator.
    Df_component_info(
        LLVM_code_generator &code_gen,
        IType::Kind         kind)
    : m_code_gen(code_gen)
    , m_df_funcs { NULL }
    , m_kind(kind)
    {
    }

    /// Add a BSDF node of a component or a constant BSDF_component node.
    void add_component_df(DAG_node const *node)
    {
        m_component_dfs.push_back(node);
    }

    /// Returns true, if the functions returned by get_df_function() are switch functions.
    bool is_switch_function() const
    {
        return !m_component_dfs.empty();
    }

    /// Get the BSDF function for the given state.
    llvm::Function *get_df_function(
        Function_context                                 &caller_ctx,
        LLVM_code_generator::Distribution_function_state state)
    {
        // no components registered -> black_bsdf()
        if (m_component_dfs.empty()) {
            char const *f_name = NULL;
            switch (m_kind) {
            case IType::TK_BSDF:
            case IType::TK_HAIR_BSDF:
                f_name = "gen_black_bsdf";
                break;

            case IType::TK_EDF:
                f_name = "gen_black_edf";
                break;

            default:
                MDL_ASSERT(!"Invalid distribution kind for getting a DF function");
                return NULL;
            }

            string func_name(f_name, m_code_gen.get_allocator());

            func_name += LLVM_code_generator::get_dist_func_state_suffix(state);
            llvm::Function *black_bsdf_func =
                m_code_gen.get_llvm_module()->getFunction(func_name.c_str());
            return black_bsdf_func;
        }

        size_t index;
        switch (state) {
        case LLVM_code_generator::Distribution_function_state::DFSTATE_SAMPLE:    index = 0; break;
        case LLVM_code_generator::Distribution_function_state::DFSTATE_EVALUATE:  index = 1; break;
        case LLVM_code_generator::Distribution_function_state::DFSTATE_PDF:       index = 2; break;
        case LLVM_code_generator::Distribution_function_state::DFSTATE_AUXILIARY: index = 3; break;
        default:
            MDL_ASSERT(!"Invalid state for getting a DF function");
            return NULL;
        }

        // LLVM function already generated?
        if (m_df_funcs[index] != NULL) {
            return m_df_funcs[index];
        }

        // no, temporarily set given state as current and instantiate the BSDFs
        llvm::SmallVector<llvm::Function *, 8> comp_funcs;
        {
            Store<LLVM_code_generator::Distribution_function_state> state_store(
                m_code_gen.m_dist_func_state, state);

            for (DAG_node const *node : m_component_dfs) {
                comp_funcs.push_back(m_code_gen.instantiate_df(caller_ctx, node));
            }
        }
        // generate and remember switch function for generated DF functions
        llvm::Function *df_switch_func = generate_df_switch_func(comp_funcs);
        m_df_funcs[index] = df_switch_func;
        return df_switch_func;
    }

    /// Generates a switch function calling the DF function identified by the last parameter
    /// with the provided arguments.
    ///
    /// Note: We don't use function pointers to be compatible with OptiX.
    ///
    /// \param funcs  the function array
    ///
    /// \returns the generated switch function
    llvm::Function *generate_df_switch_func(
        llvm::ArrayRef<llvm::Function *> const &funcs)
    {
        llvm::LLVMContext &llvm_context = m_code_gen.get_llvm_context();
        size_t num_funcs = funcs.size();
        llvm::Type *int_type = m_code_gen.get_type_mapper().get_int_type();

        llvm::FunctionType *bsdf_func_type = funcs[0]->getFunctionType();

        llvm::SmallVector<llvm::Type *, 8> arg_types;
        arg_types.append(bsdf_func_type->param_begin(), bsdf_func_type->param_end());
        arg_types.push_back(int_type);

        llvm::FunctionType *switch_func_type = llvm::FunctionType::get(
            llvm::Type::getVoidTy(llvm_context), arg_types, false);

        // note: we don't update m_curr_bb as the DAG node cache is not used here
        llvm::Function *switch_func = llvm::Function::Create(
            switch_func_type,
            llvm::GlobalValue::InternalLinkage,
            "switch_func",
            m_code_gen.get_llvm_module());
        m_code_gen.m_state_usage_analysis.register_function(switch_func);
        m_code_gen.set_llvm_function_attributes(switch_func, /*mark_noinline=*/false);

        llvm::DISubprogram *di_func = nullptr;
        if (llvm::DIBuilder *di_builder = m_code_gen.get_debug_info_builder()) {
            llvm::DIFile *di_file = di_builder->createFile("<generated>", "");

            di_func = di_builder->createFunction(
                /*Scope=*/ di_file,
                /*Name=*/ switch_func->getName(),
                /*LinkageName=*/ switch_func->getName(),
                /*File=*/ di_file,
                1,
                m_code_gen.get_type_mapper().get_debug_info_type(
                    di_builder, di_file, switch_func_type),
                1,
                llvm::DINode::FlagPrototyped,
                llvm::DISubprogram::toSPFlags(
                    /*IsLocalToUnit=*/true,
                    /*IsDefinition=*/true,
                    /*IsOptimized=*/m_code_gen.is_optimized()
                ));
            switch_func->setSubprogram(di_func);
        }

        llvm::BasicBlock *start_block =
            llvm::BasicBlock::Create(llvm_context, "start", switch_func);
        llvm::BasicBlock *end_block = llvm::BasicBlock::Create(llvm_context, "end", switch_func);

        llvm::IRBuilder<> builder(start_block);
        if (di_func) {
            builder.SetCurrentDebugLocation(llvm::DILocation::get(
                di_func->getContext(), 1, 0, di_func));
        }
        llvm::SwitchInst *switch_inst =
            builder.CreateSwitch(switch_func->arg_end() - 1, end_block, num_funcs);

        // collect the arguments for the DF functions to be called (without the index argument)
        llvm::SmallVector<llvm::Value *, 8> arg_values;
        for (llvm::Function::arg_iterator ai = switch_func->arg_begin(),
                ae = switch_func->arg_end() - 1; ai != ae; ++ai)
        {
            arg_values.push_back(ai);
        }

        // generate the switch cases with the calls to the corresponding DF function
        for (size_t i = 0; i < num_funcs; ++i) {
            llvm::BasicBlock *case_block =
                llvm::BasicBlock::Create(llvm_context, "case", switch_func);
            switch_inst->addCase(
                llvm::ConstantInt::get(llvm_context, llvm::APInt(32, uint64_t(i))),
                case_block);
            builder.SetInsertPoint(case_block);
            m_code_gen.m_state_usage_analysis.add_call(switch_func, funcs[i]);
            builder.CreateCall(funcs[i], arg_values);
            builder.CreateBr(end_block);
        }

        builder.SetInsertPoint(end_block);
        builder.CreateRetVoid();

        // optimize function to improve inlining
        m_code_gen.optimize(switch_func);

        return switch_func;
    }

private:
    /// The code generator.
    LLVM_code_generator &m_code_gen;

    /// A list of component DF or constant DF_component DAG nodes.
    llvm::SmallVector<DAG_node const *, 8> m_component_dfs;

    /// The on-demand generated LLVM DF functions for sample, evaluate and pdf.
    llvm::Function *m_df_funcs[4];

    /// Kind of distribution function.
    IType::Kind m_kind;
};


/// The different kinds of functions in the BSDF/EDF struct in libbsdf_internal.h.
enum Libbsdf_DF_func_kind
{
    LDFK_INVALID,
    LDFK_SAMPLE,
    LDFK_EVALUATE,
    LDFK_PDF,
    LDFK_AUXILIARY,
    LDFK_IS_BLACK,
    LDFK_IS_DEFAULT_DIFFUSE_REFLECTION,
    LDFK_HAS_ALLOWED_COMPONENTS
};

/// Get the kind of BSDF/EDF function call for a constant BSDF field index in libbsdf.
static Libbsdf_DF_func_kind get_libbsdf_df_func_kind(llvm::ConstantInt *bsdf_field_index)
{
    switch (bsdf_field_index->getValue().getZExtValue()) {
    case 0: return LDFK_SAMPLE;
    case 1: return LDFK_EVALUATE;
    case 2: return LDFK_PDF;
    case 3: return LDFK_AUXILIARY;
    case 4: return LDFK_IS_BLACK;
    case 5: return LDFK_IS_DEFAULT_DIFFUSE_REFLECTION;
    case 6: return LDFK_HAS_ALLOWED_COMPONENTS;
    default:
        MDL_ASSERT(!"Unknown DF struct index");
        return LDFK_INVALID;
    }
}

/// Get the kind of BSDF/EDF function call for a member call for an BSDF/EDF object in libbsdf.
static Libbsdf_DF_func_kind get_libbsdf_df_func_kind(llvm::CallInst *call)
{
    // Match this code fragment and extract <idx> as the function kind:
    //   %51 = getelementptr inbounds %struct.BSDF, %struct.BSDF* %bsdf_arg3, i32 0, i32 <idx>
    //   %52 = load i1 ()*, i1 ()** %51, align 4, !tbaa !6
    //   %53 = tail call zeroext i1 %52(), !libbsdf.bsdf_param !11

    llvm::Value *callee = call->getCalledOperand();
    if (llvm::LoadInst *load = llvm::dyn_cast<llvm::LoadInst>(callee)) {
        if (llvm::GetElementPtrInst *gep =
            llvm::dyn_cast<llvm::GetElementPtrInst>(load->getPointerOperand()))
        {
            if (llvm::StructType *df_type = llvm::dyn_cast<llvm::StructType>(
                gep->getPointerOperandType()->getPointerElementType()))
            {
                (void) df_type;  // avoid warning for non-debug builds
                MDL_ASSERT(
                    df_type->getName() == "struct.BSDF" || df_type->getName() == "struct.EDF");
                MDL_ASSERT(gep->getNumOperands() == 3 && "Unknown DF struct access");
                llvm::Value *bsdf_field_index = gep->getOperand(2);
                llvm::ConstantInt *bsdf_field_index_const =
                    llvm::dyn_cast<llvm::ConstantInt>(bsdf_field_index);
                MDL_ASSERT(bsdf_field_index_const);
                return get_libbsdf_df_func_kind(bsdf_field_index_const);
            }
        }
    }
    MDL_ASSERT(!"Unknown DF call");
    return LDFK_INVALID;
}

static LLVM_code_generator::Distribution_function_state convert_to_df_state(
    Libbsdf_DF_func_kind df_func_kind)
{
    switch (df_func_kind) {
    case LDFK_SAMPLE:    return LLVM_code_generator::DFSTATE_SAMPLE;
    case LDFK_EVALUATE:  return LLVM_code_generator::DFSTATE_EVALUATE;
    case LDFK_PDF:       return LLVM_code_generator::DFSTATE_PDF;
    case LDFK_AUXILIARY: return LLVM_code_generator::DFSTATE_AUXILIARY;
    default:
        MDL_ASSERT(!"Unexpected df call kind");
        return LLVM_code_generator::DFSTATE_NONE;
    }
}

// Create the BSDF function types using the BSDF data types from the already linked libbsdf
// module.
void LLVM_code_generator::create_bsdf_function_types()
{
    // fetch the BSDF data types from the already linked libbsdf

    m_type_bsdf_sample_data = llvm::StructType::getTypeByName(
        m_llvm_context, "struct.BSDF_sample_data");
    m_type_bsdf_evaluate_data = llvm::StructType::getTypeByName(
        m_llvm_context, "struct.BSDF_evaluate_data");
    m_type_bsdf_pdf_data = llvm::StructType::getTypeByName(
        m_llvm_context, "struct.BSDF_pdf_data");
    m_type_bsdf_auxiliary_data = llvm::StructType::getTypeByName(
        m_llvm_context, "struct.BSDF_auxiliary_data");

    // create function types for the BSDF functions

    llvm::Type *ret_tp = m_type_mapper.get_void_type();
    llvm::Type *second_param_type;
    if (target_supports_lambda_results_parameter()) {
        second_param_type = m_type_mapper.get_exec_ctx_ptr_type();
    } else {
        second_param_type = m_type_mapper.get_state_ptr_type(m_state_mode);
    }
    llvm::Type *float3_struct_ptr_type = Type_mapper::get_ptr(m_float3_struct_type);
    llvm::Type *spectral_sample_type = m_type_mapper.get_spectral_sample_type();
    llvm::Type *spectral_sample_ptr_type = m_type_mapper.get_spectral_sample_ptr_type();

    // BSDF_API void diffuse_reflection_bsdf_sample(
    //     BSDF_sample_data *data, Execution_context *ctx, float3 *inherited_normal)

    llvm::Type *arg_types_sample[] = {
        Type_mapper::get_ptr(m_type_bsdf_sample_data),
        second_param_type,
        float3_struct_ptr_type
    };

    m_type_bsdf_sample_func = llvm::FunctionType::get(ret_tp, arg_types_sample, false);

    // BSDF_API void diffuse_reflection_bsdf_evaluate(
    //     BSDF_evaluate_data *data, Execution_context *ctx, float3 *inherited_normal,
    //     spectral_sample *inherited_weight)

    llvm::Type *arg_types_eval[] = {
        Type_mapper::get_ptr(m_type_bsdf_evaluate_data),
        second_param_type,
        float3_struct_ptr_type,
        spectral_sample_ptr_type
    };

    m_type_bsdf_evaluate_func = llvm::FunctionType::get(ret_tp, arg_types_eval, false);

    // BSDF_API spectral_sample thin_film_bsdf_get_factor(
    //     BSDF_evaluate_data *data, Execution_context *ctx, float3 *inherited_normal)

    llvm::Type *arg_types_get_factor[] = {
        Type_mapper::get_ptr(m_type_bsdf_evaluate_data),
        second_param_type,
        float3_struct_ptr_type
    };

    m_type_bsdf_get_factor_func = llvm::FunctionType::get(
        spectral_sample_type, arg_types_get_factor, false);

    // BSDF_API void diffuse_reflection_bsdf_pdf(
    //     BSDF_pdf_data *data, Execution_context *ctx, float3 *inherited_normal)

    llvm::Type *arg_types_pdf[] = {
        Type_mapper::get_ptr(m_type_bsdf_pdf_data),
        second_param_type,
        float3_struct_ptr_type
    };

    m_type_bsdf_pdf_func = llvm::FunctionType::get(ret_tp, arg_types_pdf, false);

    // BSDF_API void diffuse_reflection_bsdf_auxiliary(
    //     BSDF_auxiliary_data *data, Execution_context *ctx, float3 *inherited_normal,
    //     spectral_sample *inherited_weight)

    llvm::Type *arg_types_auxiliary[] = {
        Type_mapper::get_ptr(m_type_bsdf_auxiliary_data),
        second_param_type,
        float3_struct_ptr_type,
        spectral_sample_ptr_type
    };

    m_type_bsdf_auxiliary_func = llvm::FunctionType::get(ret_tp, arg_types_auxiliary, false);
}


// Create the EDF function types using the EDF data types from the already linked libbsdf
// module.
void LLVM_code_generator::create_edf_function_types()
{
    // fetch the EDF data types from the already linked libbsdf

    m_type_edf_sample_data    = llvm::StructType::getTypeByName(
        m_llvm_context, "struct.EDF_sample_data");
    m_type_edf_evaluate_data  = llvm::StructType::getTypeByName(
        m_llvm_context, "struct.EDF_evaluate_data");
    m_type_edf_pdf_data       = llvm::StructType::getTypeByName(
        m_llvm_context, "struct.EDF_pdf_data");
    m_type_edf_auxiliary_data = llvm::StructType::getTypeByName(
        m_llvm_context, "struct.EDF_auxiliary_data");

    // create function types for the EDF functions

    llvm::Type *ret_tp = m_type_mapper.get_void_type();
    llvm::Type *second_param_type;
    if (target_supports_lambda_results_parameter()) {
        second_param_type = m_type_mapper.get_exec_ctx_ptr_type();
    } else {
        second_param_type = m_type_mapper.get_state_ptr_type(m_state_mode);
    }
    llvm::Type *float3_struct_ptr_type = Type_mapper::get_ptr(m_float3_struct_type);
    llvm::Type *spectral_sample_type = m_type_mapper.get_spectral_sample_type();
    llvm::Type *spectral_sample_ptr_type = m_type_mapper.get_spectral_sample_ptr_type();

    // BSDF_API void diffuse_edf_sample(
    //     EDF_sample_data *data, Execution_context *ctx, float3 *inherited_normal)

    llvm::Type *arg_types_sample[] = {
        Type_mapper::get_ptr(m_type_edf_sample_data),
        second_param_type,
        float3_struct_ptr_type
    };

    m_type_edf_sample_func = llvm::FunctionType::get(ret_tp, arg_types_sample, false);

    // BSDF_API void diffuse_edf_evaluate(
    //     EDF_evaluate_data *data, Execution_context *ctx, float3 *inherited_normal,
    //     spectral_sample *inherited_weight)

    llvm::Type *arg_types_eval[] = {
        Type_mapper::get_ptr(m_type_edf_evaluate_data),
        second_param_type,
        float3_struct_ptr_type,
        spectral_sample_ptr_type
    };

    m_type_edf_evaluate_func = llvm::FunctionType::get(ret_tp, arg_types_eval, false);

    // BSDF_API spectral_sample tint_edf_get_factor(
    //     EDF_evaluate_data *data, Execution_context *ctx, float3 *inherited_normal)

    llvm::Type *arg_types_get_factor[] = {
        Type_mapper::get_ptr(m_type_edf_evaluate_data),
        second_param_type,
        float3_struct_ptr_type
    };

    m_type_edf_get_factor_func = llvm::FunctionType::get(
        spectral_sample_type, arg_types_get_factor, false);

    // BSDF_API void diffuse_edf_pdf(
    //     EDF_pdf_data *data, Execution_context *ctx, float3 *inherited_normal)

    llvm::Type *arg_types_pdf[] = {
        Type_mapper::get_ptr(m_type_edf_pdf_data),
        second_param_type,
        float3_struct_ptr_type
    };

    m_type_edf_pdf_func = llvm::FunctionType::get(ret_tp, arg_types_pdf, false);

    // BSDF_API void diffuse_edf_auxiliary(
    //     EDF_auxiliary_data *data, Execution_context *ctx, float3 *inherited_normal,
    //     spectral_sample *inherited_weight)

    llvm::Type *arg_types_auxiliary[] = {
        Type_mapper::get_ptr(m_type_edf_auxiliary_data),
        second_param_type,
        float3_struct_ptr_type,
        spectral_sample_ptr_type
    };

    m_type_edf_auxiliary_func = llvm::FunctionType::get(ret_tp, arg_types_auxiliary, false);
}


// Compile a distribution function into an LLVM Module and return the LLVM module.
llvm::Module *LLVM_code_generator::compile_distribution_function(
    bool                        incremental,
    Distribution_function const &dist_func,
    ICall_name_resolver const   *resolver,
    Function_vector             &llvm_funcs,
    size_t                      next_arg_block_index,
    size_t                      *req_func_indices)
{
    Store<Distribution_function const *> dist_func_store(m_dist_func, &dist_func);
    m_cur_resolver = resolver;

#if 0
    static int dumpid = 0;
    std::string dumpname("df");
    dumpname += std::to_string(dumpid++);
    dumpname += ".gv";
    m_dist_func->dump(dumpname.c_str());
#endif

    mi::base::Handle<const Lambda_function> root_lambda_handle(m_dist_func->get_root_lambda());
    Lambda_function const *root_lambda = root_lambda_handle.get();

    create_captured_argument_struct(m_llvm_context, *root_lambda);

    // must be done before load_and_link_libbsdf() because of calls to texture runtime
    Store<Derivative_infos const *> deriv_infos_store(
        m_deriv_infos,
        m_texruntime_with_derivs ? m_dist_func->get_derivative_infos() : nullptr);

    // create a module for the functions
    if (m_module == NULL) {
        create_module("lambda_mod", NULL);

        // initialize the module with user code
        if (!init_user_modules()) {
            // drop the module and give up
            drop_llvm_module(m_module);
            m_cur_resolver = NULL;
            return NULL;
        }

        if (target_is_structured_language()) {
            init_sl_code_gen();
        }
    }

    // load libbsdf into the current module, if it was not initialized, yet
    if (m_type_bsdf_sample_data == NULL &&
        !load_and_link_libbsdf(m_link_libbsdf_df_handle_slot_mode))
    {
        // drop the module and give up
        drop_llvm_module(m_module);
        m_cur_resolver = NULL;
        return NULL;
    }

    create_resource_tables(*root_lambda);

    // determine which expressions will be put into the init function (texture results)

#ifdef DEBUG_NEW_SCHEDULER
    auto t1 = std::chrono::steady_clock::now();
#endif

    IAllocator *alloc = get_allocator();

    mi::mdl::vector<Schedule_entry>::Type schedule(alloc);
    Loop_schedule loop_schedule(alloc);

    LLVM_type_helper LLVM_type_properties(
        &m_data_layout,
        m_type_mapper);

    Expression_scheduler expr_sched(
        get_allocator(),
        LLVM_type_properties,
        m_target_lang,
        init_loop_enabled(),
        target_is_structured_language(),
        m_num_texture_results,
        m_cur_resolver,
        *m_dist_func,
        schedule,
        loop_schedule);
    if (!expr_sched.schedule_expressions()) {
        // scheduling failed with an error, drop the module and give up
        error(NOT_ENOUGH_TEXTURE_RESULTS_FOR_PRE_NORMAL_EVALUATIONS,
            Error_params(alloc));

        drop_llvm_module(m_module);
        m_cur_resolver = NULL;
        return NULL;
    }

#ifdef DEBUG_NEW_SCHEDULER
    auto t2 = std::chrono::steady_clock::now();

    typedef std::chrono::duration<double, std::milli> durationMs;
    printf("\nNew scheduler time: %.1f ms\n", durationMs(t2 - t1).count());
#endif

    m_texture_results_struct_type = LLVM_type_properties.create_texture_results_type(
        get_allocator(), m_llvm_context, expr_sched.get_schedule());
#if 0
    loop_schedule.update_texres_indices(expr_sched.get_schedule());
#endif
    reset_lambda_state();

    // let return type decide to allow init function without structure return parameter
    m_lambda_force_sret         = false;

    // distribution functions always includes a render state in its interface
    m_lambda_force_render_state = true;

    // the BSDF API functions create the lambda results they use, so no lambda results parameter
    m_lambda_force_no_lambda_results = true;

    // create init function

    {
        Function_instance inst(get_allocator(), root_lambda, target_supports_storage_spaces());

        m_dist_func_state = Distribution_function_state(DFSTATE_INIT);

        // we cannot use get_or_create_context_data here, because we need to force the creation of
        // a new function here, as the (const) root_lambda cannot be changed to reflect the
        // different states
        LLVM_context_data *ctx_data = declare_lambda(root_lambda);
        m_context_data[inst] = ctx_data;

        llvm::Function *func = ctx_data->get_function();
        llvm_funcs.push_back(func);
        unsigned flags = ctx_data->get_function_flags();

        // set function name as requested by user
        func->setName(root_lambda->get_name());

        add_generated_attributes(func);

        // remember function as an exported function
        IGenerated_code_executable::Function_kind func_kind =
            IGenerated_code_executable::FK_DF_INIT;

        IGenerated_code_executable::Distribution_kind dist_kind =
            IGenerated_code_executable::DK_NONE;

        if (req_func_indices != NULL) {
            req_func_indices[0] = m_exported_func_list.size();
        }

        m_exported_func_list.push_back(
            Exported_function(
                get_allocator(),
                func,
                dist_kind,
                func_kind,
                m_captured_args_type != NULL ? next_arg_block_index : ~0));

        // Add all referenced DF-handles to init function for backward compatibility
        // (should only be associated with distribution functions)
        Exported_function &exp_func = m_exported_func_list.back();
        for (size_t i = 0, n = m_dist_func->get_df_handle_count(); i < n; ++i) {
            exp_func.add_df_handle(m_dist_func->get_df_handle(i));
        }

        Function_context context(alloc, *this, inst, func, flags);

        translate_distribution_function_init(schedule, loop_schedule);

        context.create_void_return();

#if DEBUG_INIT_LOOP_SCHEDULER
        static int cntr = 0;
        std::error_code EC;
        llvm::raw_fd_ostream out(std::string("mdl_init-") + std::to_string(cntr++) + ".ll", EC, llvm::sys::fs::OF_None);
        context.get_function()->print(out);
#endif
    }

    // initialize texture result map after init function has been generated
    for (Schedule_entry &entry : schedule) {
        if (entry.texture_result_offset == ~0) {
            continue;
        }

        if (!entry.is_eval_state_dependent ||
                entry.eval_state == Distribution_function::ES_BEGIN_STATE) {
            m_texture_result_map[Distribution_function::ES_BEGIN_STATE][entry.node] =
                Texture_result_slot(
                    entry.texture_result_index,
                    entry.texture_result_offset);
        }
        if (!entry.is_eval_state_dependent ||
            entry.eval_state == Distribution_function::ES_AFTER_GEOMETRY_NORMAL) {
            m_texture_result_map[Distribution_function::ES_AFTER_GEOMETRY_NORMAL][entry.node] =
                Texture_result_slot(
                    entry.texture_result_index,
                    entry.texture_result_offset);
        }
    }

    for (size_t req_node_idx = 0, n_req_nodes = m_dist_func->get_explicit_requested_node_count();
        req_node_idx < n_req_nodes; ++req_node_idx)
    {
        Distribution_function::Requested_node const *req_node =
            m_dist_func->get_requested_node(req_node_idx);
        MDL_ASSERT(req_node->function_name != NULL &&
            "explicitly requested nodes must have a function name");

        // We create a new top-level function, so clear the DAG node map.
        // While recursively instantiating the DFs, we keep the DAG node map, as we will
        // continue generating more code within one function after instantiating other DFs.
        clear_dag_node_map();

        m_cur_req_node = req_node;

        DAG_node const *node = req_node->node;
        mi::mdl::IType const *node_type = node->get_type()->skip_type_alias();

        IGenerated_code_executable::Distribution_kind dist_kind =
            IGenerated_code_executable::DK_NONE;
        switch (node_type->get_kind()) {
        case IType::TK_BSDF:      dist_kind = IGenerated_code_executable::DK_BSDF;      break;
        case IType::TK_HAIR_BSDF: dist_kind = IGenerated_code_executable::DK_HAIR_BSDF; break;
        case IType::TK_EDF:       dist_kind = IGenerated_code_executable::DK_EDF;       break;
        default:
            break;
        }

        if (req_func_indices != NULL) {
            req_func_indices[req_node_idx + 1] = m_exported_func_list.size();
        }

        llvm::Twine base_name(req_node->function_name);
        Function_instance inst(
            get_allocator(), m_dist_func, req_node_idx, target_supports_storage_spaces());

        // Don't allow returning structs at ABI level, even in value mode
        m_lambda_force_sret = m_lambda_return_mode == Return_mode::RETMODE_SRET
            || (m_lambda_return_mode == Return_mode::RETMODE_VALUE &&
                is<mi::mdl::IType_struct>(node_type));

        // only force, when actually supported by backend
        m_lambda_force_sret &= target_supports_sret_for_lambda();

        // non-distribution function?
        if (dist_kind == IGenerated_code_executable::DK_NONE) {
            m_dist_func_state = Distribution_function_state(DFSTATE_NONE);

            //LLVM_context_data *ctx_data = get_or_create_context_data(&lambda);

            LLVM_context_data *ctx_data = declare_requested_node(m_dist_func, req_node_idx);
            m_context_data[inst] = ctx_data;

            llvm::Function    *func     = ctx_data->get_function();
            unsigned          flags     = ctx_data->get_function_flags();

            add_generated_attributes(func);

            m_exported_func_list.push_back(
                Exported_function(
                    get_allocator(),
                    func,
                    IGenerated_code_executable::DK_NONE,
                    IGenerated_code_executable::FK_LAMBDA,
                    m_captured_args_type != NULL ? next_arg_block_index : ~0));

            Function_context context(alloc, *this, inst, func, flags);

            // translate function body
            Expression_result res = translate_node(node, resolver);

            context.create_return(res.as_value(context));

            continue;
        }

        // a distribution function

        llvm::GlobalVariable *mat_data_global = NULL;

        // create one LLVM function for each distribution function state
        for (int state = DFSTATE_SAMPLE; state < DFSTATE_END_STATE; ++state) {
            // skip get_factor function
            if (state == DFSTATE_GET_FACTOR) {
                continue;
            }

            m_dist_func_state = Distribution_function_state(state);

            // we cannot use get_or_create_context_data here, because we need to force the creation
            // of a new function here, as the (const) root_lambda cannot be changed to reflect the
            // different states
            LLVM_context_data *ctx_data = declare_requested_node(m_dist_func, req_node_idx);
            m_context_data[inst] = ctx_data;

            llvm::Function *func = ctx_data->get_function();
            llvm_funcs.push_back(func);
            unsigned flags = ctx_data->get_function_flags();

            // set proper function name according to distribution function state
            func->setName(base_name + get_dist_func_state_suffix());

            add_generated_attributes(func);

            // remember function as an exported function
            IGenerated_code_executable::Function_kind func_kind =
                IGenerated_code_executable::FK_INVALID;
            switch (Distribution_function_state(state)) {
            case DFSTATE_SAMPLE:    func_kind = IGenerated_code_executable::FK_DF_SAMPLE;    break;
            case DFSTATE_EVALUATE:  func_kind = IGenerated_code_executable::FK_DF_EVALUATE;  break;
            case DFSTATE_PDF:       func_kind = IGenerated_code_executable::FK_DF_PDF;       break;
            case DFSTATE_AUXILIARY: func_kind = IGenerated_code_executable::FK_DF_AUXILIARY; break;
            default:
                MDL_ASSERT(!"Unexpected DF state");
                break;
            }

            // skip the auxiliary functions if deactivated
            if (!m_enable_auxiliary && state == DFSTATE_AUXILIARY) {
                continue;
            }

            // skip the PDF functions if deactivated
            if (!m_enable_pdf && state == DFSTATE_PDF) {
                continue;
            }

            m_exported_func_list.push_back(
                Exported_function(
                    get_allocator(),
                    func,
                    dist_kind,
                    func_kind,
                    m_captured_args_type != NULL ? next_arg_block_index : ~0));

            Exported_function &exp_func = m_exported_func_list.back();
            for (const char *handle : req_node->df_handles) {
                exp_func.add_df_handle(handle);
            }

            Function_context context(alloc, *this, inst, func, flags);

            // translate the distribution function
            translate_distribution_function(req_node->node, mat_data_global);
            context.create_void_return();
        }
    }

    // if we are compiling with derivatives, all waiting functions need to be compiled now,
    // to give them access to the derivative infos
    if (m_deriv_infos != NULL) {
        compile_waiting_functions();
    }

    // reset some fields
    m_scatter_components_map.clear();
    m_texture_result_map[Distribution_function::ES_BEGIN_STATE].clear();
    m_texture_result_map[Distribution_function::ES_AFTER_GEOMETRY_NORMAL].clear();
    m_cur_resolver = NULL;
    m_cur_req_node = NULL;
    for (size_t i = 0, n = m_instantiated_dfs.size(); i < n; ++i) {
        m_instantiated_dfs[i].clear();
    }

    if (!incremental) {
        // finalize the module and store it
        if (llvm::Module *module = finalize_module()) {
            return module;
        }
        return NULL;
    }
    return m_module;
}

// Returns the BSDF function name suffix for the current distribution function state.
char const *LLVM_code_generator::get_dist_func_state_suffix(Distribution_function_state state)
{
    switch (state) {
    case DFSTATE_INIT:        return "_init";
    case DFSTATE_SAMPLE:      return "_sample";
    case DFSTATE_EVALUATE:    return "_evaluate";
    case DFSTATE_PDF:         return "_pdf";
    case DFSTATE_AUXILIARY:   return "_auxiliary";
    case DFSTATE_GET_FACTOR:  return "_get_factor";
    default:
        MDL_ASSERT(!"Invalid distribution function state");
        return "";
    }
}

// Returns the distribution function state requested by the given call.
LLVM_code_generator::Distribution_function_state
LLVM_code_generator::get_dist_func_state_from_call(llvm::CallInst *call)
{
    llvm::FunctionType *func_tp = llvm::cast<llvm::FunctionType>(
        call->getCalledOperand()->getType()->getPointerElementType());
    llvm::Type *df_data_tp =
        func_tp->getParamType(0)->getPointerElementType();

    if (df_data_tp == m_type_bsdf_sample_data || df_data_tp == m_type_edf_sample_data) {
        return DFSTATE_SAMPLE;
    } else if (df_data_tp == m_type_bsdf_evaluate_data || df_data_tp == m_type_edf_evaluate_data) {
        return DFSTATE_EVALUATE;
    } else if (df_data_tp == m_type_bsdf_pdf_data || df_data_tp == m_type_edf_pdf_data) {
        return DFSTATE_PDF;
    } else if (df_data_tp == m_type_bsdf_auxiliary_data ||
        df_data_tp == m_type_edf_auxiliary_data) {
        return DFSTATE_AUXILIARY;
    }

    MDL_ASSERT(!"Invalid distribution function type called");
    return DFSTATE_NONE;
}

// Get the BSDF function for the given semantics and the current distribution function state
// from the BSDF library.
llvm::Function *LLVM_code_generator::get_libbsdf_function(
    DAG_call const *dag_call,
    char const     *prefix)
{
    IDefinition::Semantics sema = dag_call->get_semantic();
    IType::Kind kind = dag_call->get_type()->get_kind();

    if (prefix == NULL) {
        prefix = "";
    }

    string func_name(prefix, get_allocator());
    string suffix(get_allocator());

    // check for tint(color, color, bsdf) overload
    if (sema == IDefinition::DS_INTRINSIC_DF_TINT && dag_call->get_argument_count() == 3) {
        suffix = "_rt";
    }

    switch (kind) {
    case IType::Kind::TK_BSDF:      suffix += "_bsdf"; break;
    case IType::Kind::TK_HAIR_BSDF: suffix += "_hair_bsdf"; break;
    case IType::Kind::TK_EDF:       suffix += "_edf"; break;
    default: break;
    }


    #define SEMA_CASE(val, name)  case IDefinition::val: func_name += name; break;

    switch (sema) {
    SEMA_CASE(DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF,
                "diffuse_reflection_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_DUSTY_DIFFUSE_REFLECTION_BSDF,
                "dusty_diffuse_reflection_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF,
                "diffuse_transmission_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_SPECULAR_BSDF,
                "specular_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF,
                "simple_glossy_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF,
                "backscattering_glossy_reflection_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_SHEEN_BSDF,
                "sheen_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_MICROFLAKE_SHEEN_BSDF,
                "microflake_sheen_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_MEASURED_BSDF,
                "measured_bsdf")

    SEMA_CASE(DS_INTRINSIC_DF_DIFFUSE_EDF,
                "diffuse_edf")
    SEMA_CASE(DS_INTRINSIC_DF_MEASURED_EDF,
                "measured_edf")
    SEMA_CASE(DS_INTRINSIC_DF_SPOT_EDF,
                "spot_edf")

    // Unsupported: DS_INTRINSIC_DF_ANISOTROPIC_VDF
    // Unsupported: DS_INTRINSIC_DF_FOG_VDF

    SEMA_CASE(DS_INTRINSIC_DF_NORMALIZED_MIX,
                "normalized_mix" + suffix)
    SEMA_CASE(DS_INTRINSIC_DF_CLAMPED_MIX,
                "clamped_mix" + suffix)
    SEMA_CASE(DS_INTRINSIC_DF_UNBOUNDED_MIX,
                "unbounded_mix" + suffix)
    SEMA_CASE(DS_INTRINSIC_DF_WEIGHTED_LAYER,
                "weighted_layer")
    SEMA_CASE(DS_INTRINSIC_DF_FRESNEL_LAYER,
                "fresnel_layer")
    SEMA_CASE(DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER,
                "custom_curve_layer")
    SEMA_CASE(DS_INTRINSIC_DF_MEASURED_CURVE_LAYER,
                "measured_curve_layer")
    SEMA_CASE(DS_INTRINSIC_DF_THIN_FILM,
                "thin_film")
    SEMA_CASE(DS_INTRINSIC_DF_TINT,
                "tint" + suffix)
    SEMA_CASE(DS_INTRINSIC_DF_DIRECTIONAL_FACTOR,
                "directional_factor" + suffix)
    SEMA_CASE(DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR,
                "measured_curve_factor")
    SEMA_CASE(DS_INTRINSIC_DF_MEASURED_FACTOR,
                "measured_factor")
    SEMA_CASE(DS_INTRINSIC_DF_COAT_ABSORPTION_FACTOR,
                "coat_absorption_factor")

    // Not a DF: DS_INTRINSIC_DF_LIGHT_PROFILE_POWER
    // Not a DF: DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM
    // Not a DF: DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID
    // Not a DF: DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID

    SEMA_CASE(DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF,
                "microfacet_beckmann_smith_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF,
                "microfacet_ggx_smith_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF,
                "microfacet_beckmann_vcavities_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF,
                "microfacet_ggx_vcavities_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF,
                "ward_geisler_moroder_bsdf")
    SEMA_CASE(DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
                "color_normalized_mix" + suffix)
    SEMA_CASE(DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
                "color_clamped_mix" + suffix)
    SEMA_CASE(DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX,
                "color_unbounded_mix" + suffix)
    SEMA_CASE(DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER,
                "color_weighted_layer")
    SEMA_CASE(DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER,
                "color_fresnel_layer")
    SEMA_CASE(DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER,
                "color_custom_curve_layer")
    SEMA_CASE(DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER,
                "color_measured_curve_layer")
    SEMA_CASE(DS_INTRINSIC_DF_FRESNEL_FACTOR,
                "fresnel_factor")

    SEMA_CASE(DS_INTRINSIC_DF_CHIANG_HAIR_BSDF,
                "chiang_hair_bsdf")

    default:
        return NULL;  // unsupported DF, should be mapped to black DF
    }

    #undef SEMA_CASE

    func_name = "gen_" + func_name + get_dist_func_state_suffix();
    llvm::Function *func = m_module->getFunction(func_name.c_str());
    MDL_ASSERT(func && "Function for supported DF not found in libbsdf");
    return func;
}

// Determines the semantics for a libbsdf df function name.
IDefinition::Semantics LLVM_code_generator::get_libbsdf_function_semantics(
    llvm::StringRef name)
{
    llvm::StringRef basename;
    if (name.endswith("_sample")) {
        basename = name.drop_back(7);
    } else if (name.endswith("_evaluate")) {
        basename = name.drop_back(9);
    } else if (name.endswith("_pdf")) {
        basename = name.drop_back(4);
    } else if (name.endswith("_auxiliary")) {
        basename = name.drop_back(10);
    } else if (name.endswith("_get_factor")) {
        basename = name.drop_back(11);
    } else {
        return IDefinition::DS_UNKNOWN;
    }

    if (basename.endswith("_mix_bsdf")) {
        basename = basename.drop_back(5);
    }
    if (basename.endswith("_mix_edf")) {
        basename = basename.drop_back(4);
    }

    if (basename == "black_bsdf") {
        return IDefinition::DS_INVALID_REF_CONSTRUCTOR;
    }
    if (basename == "black_edf") {
        return IDefinition::DS_INVALID_REF_CONSTRUCTOR;
    }

    // df::tint(color, color, bsdf) overload?
    if (basename == "tint_rt_bsdf") {
        return IDefinition::DS_INTRINSIC_DF_TINT;
    }

    // df::tint(color, edf) overload?
    if (basename == "tint_edf") {
        return IDefinition::DS_INTRINSIC_DF_TINT;
    }

    // df::tint(color, bsdf) overload?
    if (basename == "tint_bsdf") {
        return IDefinition::DS_INTRINSIC_DF_TINT;
    }

    // df::tint(color, hair_bsdf) overload?
    if (basename == "tint_hair_bsdf") {
        return IDefinition::DS_INTRINSIC_DF_TINT;
    }

    // df::directional_factor(color, color, float, edf) overload?
    if (basename == "directional_factor_edf") {
        return IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR;
    }

    // df::directional_factor(color, color, color, float, bsdf) overload?
    if (basename == "directional_factor_bsdf") {
        return IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR;
    }

    string builtin_name("::df::", get_allocator());
    builtin_name.append(basename.data(), basename.size());

    IDefinition::Semantics sema = m_compiler->get_builtin_semantic(builtin_name.c_str());
    if (sema == IDefinition::DS_UNKNOWN && name.startswith("thin_film_")) {
        // check if this is a modifier prefix
        return get_libbsdf_function_semantics(name.drop_front(10));
    }
    return sema;
}

// Check whether the given parameter of the given df function is an array parameter.
bool LLVM_code_generator::is_libbsdf_array_parameter(IDefinition::Semantics sema, int df_param_idx)
{
    switch (sema) {
    case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER:
    case IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX:
    case IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX:
    case IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
    case IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
    case IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER:
    case IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX:
    case IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX:
        return df_param_idx == 0;

    default:
        return false;
    }
}

// Translates a potential runtime call in a libbsdf function to a call to the according
// intrinsic, converting the arguments as necessary.
bool LLVM_code_generator::translate_libbsdf_runtime_call(
    llvm::CallInst             *call,
    llvm::BasicBlock::iterator &ii)
{
    Function_context &ctx = *m_ctx;

    unsigned num_params_eaten = 0;

    llvm::Function *called_func = call->getCalledFunction();
    if (called_func == NULL) {
        return true;   // ignore indirect function invocation
    }

    llvm::StringRef func_name = called_func->getName();

    if (func_name == "is_bsdf_flags_enabled") {
        call->replaceAllUsesWith(
            llvm::ConstantInt::get(
                llvm::IntegerType::get(m_llvm_context, 1),
                m_libbsdf_flags_in_bsdf_data ? 1 : 0));

        // Remove old call and let iterator point to instruction before old call
        ii = --ii->getParent()->getInstList().erase(call);
        return true;
    }

    if (!func_name.startswith("_Z") || !called_func->isDeclaration()) {
        return true;   // ignore non-mangled functions and functions with definitions
    }

    // try to resolve the function name to the LLVM function of an intrinsic

    string demangled_name(get_allocator());
    MDL_name_mangler mangler(get_allocator(), demangled_name);
    if (!mangler.demangle(func_name.data(), func_name.size())) {
        demangled_name.assign(func_name.data(), func_name.size());
    }

    // replace "::State::" by "::state::"
    bool use_state_from_this = false;
    if (demangled_name.compare(0, 9, "::State::") == 0) {
        demangled_name[2] = 's';

        // get the state from the "this" object, which is actually an Execution_context
        use_state_from_this = true;
        ++num_params_eaten;
    }

    llvm::Function *func = NULL;
    LLVM_context_data *p_data = NULL;
    unsigned ret_array_size = 0;
    Internal_function *internal_func = NULL;

    if (demangled_name.compare(0, 9, "::state::") == 0) {
        // special case of an internal function not available in MDL?
        if (demangled_name == "::state::set_normal(float3)") {
            internal_func = m_int_func_state_set_normal;
        } else if (demangled_name == "::state::get_texture_results()") {
            internal_func = m_int_func_state_get_texture_results;
        } else if (demangled_name == "::state::get_arg_block()") {
            internal_func = m_int_func_state_get_arg_block;
        } else if (demangled_name == "::state::call_lambda_float(int)") {
            internal_func = m_int_func_state_call_lambda_float;
        } else if (demangled_name == "::state::call_lambda_float3(int)") {
            internal_func = m_int_func_state_call_lambda_float3;
        } else if (demangled_name == "::state::call_lambda_uint(int)") {
            internal_func = m_int_func_state_call_lambda_uint;
        } else if (demangled_name == "::state::get_arg_block_float(int)") {
            internal_func = m_int_func_state_get_arg_block_float;
        } else if (demangled_name == "::state::get_arg_block_float3(int)") {
            internal_func = m_int_func_state_get_arg_block_float3;
        } else if (demangled_name == "::state::get_arg_block_uint(int)") {
            internal_func = m_int_func_state_get_arg_block_uint;
        } else if (demangled_name == "::state::get_arg_block_bool(int)") {
            internal_func = m_int_func_state_get_arg_block_bool;
        } else if (demangled_name == "::state::get_measured_curve_value(int,int)") {
            internal_func = m_int_func_state_get_measured_curve_value;
        } else if (demangled_name == "::state::adapt_microfacet_roughness(float2)") {
            internal_func = m_int_func_state_adapt_microfacet_roughness;
        } else if (demangled_name == "::state::adapt_normal(float3)") {
            internal_func = m_int_func_state_adapt_normal;
        } else if (demangled_name == "::state::rgb_to_spectral_ior(float3)") {
            internal_func = m_int_func_state_rgb_to_spectral_ior;
        } else if (demangled_name == "::state::rgb_to_spectral_reflectance(float3)") {
            internal_func = m_int_func_state_rgb_to_spectral_reflectance;
        } else if (demangled_name == "::state::rgb_to_spectral_luminance(float3)") {
            internal_func = m_int_func_state_rgb_to_spectral_luminance;
        } else if (demangled_name == "::state::rgb_to_spectral_volume_coefficient(float3)") {
            internal_func = m_int_func_state_rgb_to_spectral_volume_coefficient;
        } else if (demangled_name == "::state::get_wavelengths()") {
            internal_func = m_int_func_state_get_wavelengths;
        } else if (demangled_name == "::state::bsdf_measurement_resolution(int,int)") {
            internal_func = m_int_func_df_bsdf_measurement_resolution;
        } else if (demangled_name == "::state::bsdf_measurement_evaluate(int,float2,float2,int)") {
            internal_func = m_int_func_df_bsdf_measurement_evaluate;
        } else if (demangled_name == "::state::bsdf_measurement_sample(int,float2,float3,int)") {
            internal_func = m_int_func_df_bsdf_measurement_sample;
        } else if (demangled_name == "::state::bsdf_measurement_pdf(int,float2,float2,int)") {
            internal_func = m_int_func_df_bsdf_measurement_pdf;
        } else if (demangled_name == "::state::bsdf_measurement_albedos(int,float2)") {
            internal_func = m_int_func_df_bsdf_measurement_albedos;
        } else if (demangled_name == "::state::light_profile_evaluate(int,float2)") {
            internal_func = m_int_func_df_light_profile_evaluate;
        } else if (demangled_name == "::state::light_profile_sample(int,float3)") {
            internal_func = m_int_func_df_light_profile_sample;
        } else if (demangled_name == "::state::light_profile_pdf(int,float2)") {
            internal_func = m_int_func_df_light_profile_pdf;
        } else {
            // remap to different functions
            if (demangled_name == "::state::tex_resolution_2d(int)") {
                demangled_name = "::tex::resolution(texture_2d)";
            } else if (demangled_name == "::state::tex_is_valid_2d(int)") {
                demangled_name = "::tex::is_valid(texture_2d)";
            } else if (demangled_name ==
                "::state::tex_lookup_float3_2d(int,float2,int,int,float2,float2,float)")
            {
                demangled_name = "::tex::lookup_float3(texture_2d,float2,"
                    "::tex::wrap_mode,::tex::wrap_mode,float2,float2,float)";
            } else if (demangled_name ==
                "::state::tex_lookup_float_3d(int,float3,int,int,int,float2,float2,float2,float)")
            {
                demangled_name = "::tex::lookup_float(texture_3d,float3,::tex::wrap_mode,"
                    "::tex::wrap_mode,::tex::wrap_mode,float2,float2,float2,float)";
            } else if (demangled_name ==
                "::state::tex_lookup_float3_3d(int,float3,int,int,int,float2,float2,float2,float)")
            {
                demangled_name = "::tex::lookup_float3(texture_3d,float3,::tex::wrap_mode,"
                    "::tex::wrap_mode,::tex::wrap_mode,float2,float2,float2,float)";
            } else if (demangled_name == "::state::get_bsdf_data_texture_id(Bsdf_data_kind)") {
                // will be handled by finalize_module() when all resources of
                // the link unit are known
                return true;
            }
        }
    }

    unsigned promote = PR_NONE;

    if (internal_func != NULL) {
        func = get_internal_function(internal_func);

        Function_instance inst(get_allocator(),
            reinterpret_cast<size_t>(internal_func),
            target_supports_storage_spaces());
        p_data = get_context_data(inst);
    } else {
        // find last "::" before the parameters
        size_t parenpos = demangled_name.find('(');
        size_t colonpos = demangled_name.rfind("::", parenpos);
        if (colonpos == string::npos || colonpos == 0) {
            return true;  // not in a module, maybe a builtin function
        }

        string module_name = demangled_name.substr(0, colonpos);
        string signature = demangled_name.substr(colonpos + 2);
        IDefinition const *def = m_compiler->find_stdlib_signature(
            module_name.c_str(), signature.c_str());
        if (def == NULL) {
            return true;  // not one of our modules, maybe a builtin function
        }

        State_usage usage;
        switch (def->get_semantics()) {
        case IDefinition::DS_INTRINSIC_STATE_POSITION:
            usage = IGenerated_code_executable::SU_POSITION;
            break;
        case IDefinition::DS_INTRINSIC_STATE_NORMAL:
            usage = IGenerated_code_executable::SU_NORMAL;
            break;
        case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_NORMAL:
            usage = IGenerated_code_executable::SU_GEOMETRY_NORMAL;
            break;
        case IDefinition::DS_INTRINSIC_STATE_MOTION:
            usage = IGenerated_code_executable::SU_MOTION;
            break;
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_COORDINATE:
            usage = IGenerated_code_executable::SU_TEXTURE_COORDINATE;
            break;
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U:
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_V:
            usage = IGenerated_code_executable::SU_TEXTURE_TANGENTS;
            break;
        case IDefinition::DS_INTRINSIC_STATE_TANGENT_SPACE:
            usage = IGenerated_code_executable::SU_TANGENT_SPACE;
            break;
        case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_U:
        case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_V:
            usage = IGenerated_code_executable::SU_GEOMETRY_TANGENTS;
            break;
        case IDefinition::DS_INTRINSIC_STATE_DIRECTION:
            usage = IGenerated_code_executable::SU_DIRECTION;
            break;
        case IDefinition::DS_INTRINSIC_STATE_ANIMATION_TIME:
            usage = IGenerated_code_executable::SU_ANIMATION_TIME;
            break;
        case IDefinition::DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL:
            usage = IGenerated_code_executable::SU_ROUNDED_CORNER_NORMAL;
            break;
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
            usage = IGenerated_code_executable::SU_TRANSFORMS;
            break;
        case IDefinition::DS_INTRINSIC_STATE_OBJECT_ID:
            usage = IGenerated_code_executable::SU_OBJECT_ID;
            break;
        default:
            usage = 0;
            break;
        }

        if (usage != 0) {
            m_state_usage_analysis.add_state_usage(ctx.get_function(), usage);
        }

        if (target_is_structured_language()) {
            IDefinition const *latest_def = promote_to_highest_version(def, promote);
            if (promote != PR_NONE) {
                def = latest_def;
            }

            func = get_sl_intrinsic_function(def, /*return_derivs=*/false);
        }

        if (func == NULL) {
            func = get_intrinsic_function(def, /*return_derivs=*/false);
        }

        // check for MDL function with array return and retrieve array size
        MDL_ASSERT(def->get_type()->get_kind() == IType::TK_FUNCTION);
        IType_function const *mdl_func_type = static_cast<IType_function const *>(def->get_type());
        IType const *mdl_ret_type = mdl_func_type->get_return_type();
        if (mdl_ret_type->get_kind() == IType::TK_ARRAY) {
            IType_array const *mdl_array_type = static_cast<IType_array const *>(mdl_ret_type);
            MDL_ASSERT(mdl_array_type->is_immediate_sized());
            ret_array_size = unsigned(mdl_array_type->get_size());
        }

        Function_instance inst(
            get_allocator(), def, /*return_derivs=*/ false, target_supports_storage_spaces());
        p_data = get_context_data(inst);
    }
    if (func == NULL) {
        MDL_ASSERT(!"Unsupported runtime function");
        return false;
    }

    // replace the call by a call to the intrinsic function adapting the arguments and
    // providing additional arguments as requested by the intrinsic

    llvm::SmallVector<llvm::Value *, 8> llvm_args;

    // For the return value, we have 5 different cases:
    //    original call         runtime reality
    //      res = f(a,b)          res = f_r(a,b)
    //      res = f(a,b)          f_r(&res,a,b)
    //      f(&res,a,b)           res = f_r(a,b)
    //      f(&res,a,b)           f_r(&res,a,b)
    //      f(a,b,&res1,&res2)    f_r(&res,a,b) with res being an array

    llvm::Type *orig_res_type = called_func->getReturnType();
    llvm::Value *orig_res_ptr = NULL;
    llvm::Value *runtime_res_ptr = NULL;

    // insert new code before the old call
    ctx->SetInsertPoint(call);

    // Original call case: f(&res,a,b)?
    if (ret_array_size == 0 && orig_res_type == m_type_mapper.get_void_type()) {
        orig_res_ptr = call->getArgOperand(0);
        orig_res_type = llvm::cast<llvm::PointerType>(orig_res_ptr->getType())->getElementType();
        ++num_params_eaten;
    }

    // Runtime call case: f_r(&res,a,b)?
    if (p_data->is_sret_return()) {
        runtime_res_ptr = ctx.create_local(p_data->get_return_type(), "runtime_call_result");
        llvm_args.push_back(runtime_res_ptr);
    }

    llvm::Value *exec_ctx = NULL;
    if (use_state_from_this) {
        // first arg may be return value pointer
        exec_ctx = ctx->CreateBitCast(
            call->getArgOperand(num_params_eaten - 1),
                m_type_mapper.get_exec_ctx_ptr_type());
    }

    if (p_data->has_exec_ctx_param()) {
        // pass execution context parameter
        llvm_args.push_back(exec_ctx);
    } else {
        if (p_data->has_state_param()) {
            // pass state parameter
            llvm::Value *state = ctx.get_state_parameter(exec_ctx);
            if (use_state_from_this) {
                state = ctx->CreateBitCast(state, m_type_mapper.get_state_ptr_type(m_state_mode));
            }
            llvm_args.push_back(state);
        }

        if (p_data->has_resource_data_param()) {
            // pass resource_data parameter
            llvm_args.push_back(ctx.get_resource_data_parameter(exec_ctx));
        }

        if (target_uses_exception_state_parameter() && p_data->has_exc_state_param()) {
            // pass exc_state_param parameter
            llvm_args.push_back(ctx.get_exc_state_parameter(exec_ctx));
        }

        if (p_data->has_captured_args_param()) {
            // pass captured_arguments parameter
            llvm_args.push_back(ctx.get_cap_args_parameter(exec_ctx));
        }
    }

    if (p_data->has_object_id_param()) {
        // should not happen, as we always require the render state
        MDL_ASSERT(!"Object ID parameter not supported, yet");
        return false;
    }

    if (p_data->has_transform_params()) {
        // should not happen, as we always require the render state
        MDL_ASSERT(!"Transform parameters not supported, yet");
        return false;
    }

    llvm::FunctionType *func_type = func->getFunctionType();

    // handle all remaining arguments (except for array return arguments)
    unsigned n_args = call->getNumArgOperands();
    for (unsigned i = num_params_eaten; i < n_args - ret_array_size; ++i) {
        llvm::Value *arg        = call->getArgOperand(i);
        llvm::Type  *arg_type   = arg->getType();
        llvm::Type  *param_type = func_type->getParamType(llvm_args.size());

        if (arg_type == param_type) {
            llvm_args.push_back(arg);
            continue;
        }

        // normalize argument to a value
        if (llvm::isa<llvm::PointerType>(arg_type)) {
            arg = ctx->CreateLoad(arg);
            arg_type = arg->getType();
        }

        llvm::Type *param_elem_type = param_type;
        if (llvm::isa<llvm::PointerType>(param_type)) {
            param_elem_type = param_type->getPointerElementType();
        }

        // need to convert to a derivative value?
        // can happen for 2D texture access in libbsdf for measured_factor()
        if (!ctx.is_deriv_type(arg_type) && ctx.is_deriv_type(param_elem_type)) {
            arg = ctx.get_dual(arg);
            arg_type = arg->getType();
        }

        if (arg_type == param_type) {
            llvm_args.push_back(arg);
            continue;
        }

        // conversion required
        llvm::Value *convert_tmp_ptr = ctx.create_local(param_elem_type, "convert_tmp");
        ctx.convert_and_store(arg, convert_tmp_ptr);

        // function expects a pointer
        if (llvm::isa<llvm::PointerType>(param_type)) {
            llvm_args.push_back(convert_tmp_ptr);
            continue;
        }

        // function expects a value
        arg = ctx->CreateLoad(convert_tmp_ptr);
        llvm_args.push_back(arg);
    }

    add_promoted_arguments(promote, llvm_args);

    llvm::Value *res = ctx->CreateCall(func, llvm_args);

    // Runtime call case: f_r(&res,a,b)?
    if (runtime_res_ptr != NULL) {
        if (ret_array_size != 0) {
            res = ctx->CreateLoad(runtime_res_ptr);
        } else {
            res = ctx.load_and_convert(orig_res_type, runtime_res_ptr);
        }
    } else if (ret_array_size == 0) {
        // Case: res = f_r(a,b)
        if (res->getType() != orig_res_type) {
            // conversion to bool? -> avoid tmp var
            if (llvm::isa<llvm::IntegerType>(res->getType()) &&
                    orig_res_type == llvm::IntegerType::get(m_llvm_context, 1)) {
                res = ctx->CreateICmpNE(res, llvm::ConstantInt::getNullValue(res->getType()));
            } else {
                llvm::Value *convert_tmp_ptr = ctx.create_local(res->getType(), "convert_tmp");
                ctx->CreateStore(res, convert_tmp_ptr);
                res = ctx.load_and_convert(orig_res_type, convert_tmp_ptr);
            }
        }
    }

    // Original call case: f(&res,a,b)?
    if (orig_res_ptr != NULL) {
        ctx->CreateStore(res, orig_res_ptr);
    } else if (ret_array_size != 0) {
        // Case: f(a,b,&res1,&res2)
        // Copy the result from the array into the single result arguments
        for (unsigned i = 0; i < ret_array_size; ++i) {
            uint32_t idx[1] = { i };
            llvm::Value *res_elem = ctx->CreateExtractValue(res, idx);
            ctx.convert_and_store(res_elem, call->getArgOperand(n_args - ret_array_size + i));
        }
    } else {
        // Case: res = f(a,b)
        call->replaceAllUsesWith(res);
    }

    // Remove old call and let iterator point to instruction before old call
    ii = --ii->getParent()->getInstList().erase(call);
    return true;
}

// Transitively walk over the uses of the given argument and mark any calls as BSDF calls,
// storing the provided parameter index as "libbsdf.bsdf_param" metadata.
void LLVM_code_generator::mark_df_calls(
    llvm::Argument *arg,
    int            df_param_idx,
    IType::Kind    kind)
{
    llvm::SmallPtrSet<llvm::Value *, 16> visited;
    llvm::SmallVector<llvm::Value *, 16> worklist;

    llvm::Type *int_type = m_type_mapper.get_int_type();

    worklist.push_back(arg);
    while (!worklist.empty()) {
        llvm::Value *cur = worklist.pop_back_val();
        if (visited.count(cur)) {
            continue;
        }
        visited.insert(cur);

        unsigned num_stores = 0;
        for (auto user : cur->users()) {
            if (llvm::StoreInst *store = llvm::dyn_cast<llvm::StoreInst>(user)) {
                // for stores, also follow the variable which is written
                worklist.push_back(store->getPointerOperand());
                ++num_stores;
            } else if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(user)) {
                // found a call, store the parameter index as metadata
                llvm::Metadata *param_idx = llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(int_type, df_param_idx));
                llvm::MDNode *md = llvm::MDNode::get(m_llvm_context, param_idx);

                switch (kind) {
                case IType::TK_BSDF:
                case IType::TK_HAIR_BSDF:
                    call->setMetadata(m_bsdf_param_metadata_id, md);
                    break;
                case IType::TK_EDF:
                    call->setMetadata(m_edf_param_metadata_id, md);
                    break;
                default:
                    MDL_ASSERT(!"Invalid kind of distribution");
                }
            } else {
                // for all other uses, just follow the use
                worklist.push_back(user);
            }
        }

        // if we have more than one store to the same variable, the code is probably not supported
        MDL_ASSERT(num_stores <= 1);
    }
}

// Returns the set of context data flags to use for functions used with distribution functions.
LLVM_context_data::Flags LLVM_code_generator::get_df_function_flags(const llvm::Function *func)
{
    LLVM_context_data::Flags flags = LLVM_context_data::FL_HAS_STATE;

    // DF functions always use a data struct as first parameter
    // (treat as sret, even if function returns something. A return instruction will not
    // be generated by the context, as for functions returning something, we only modify them)
    flags |= LLVM_context_data::FL_SRET;

    if (target_uses_resource_data_parameter()) {
        flags |= LLVM_context_data::FL_HAS_RES;
    }
    if (target_uses_exception_state_parameter()) {
        flags |= LLVM_context_data::FL_HAS_EXC;
    }
    if (target_supports_captured_argument_parameter()) {
        flags |= LLVM_context_data::FL_HAS_CAP_ARGS;
    }
    if (target_supports_lambda_results_parameter()) {
        flags |= LLVM_context_data::FL_HAS_EXEC_CTX | LLVM_context_data::FL_HAS_LMBD_RES;
    }
    return flags;
}

// Load and link libbsdf into the current LLVM module.
bool LLVM_code_generator::load_and_link_libbsdf(mdl::Df_handle_slot_mode hsm)
{
    std::unique_ptr<llvm::Module> libbsdf(load_libbsdf(m_llvm_context, hsm));
    MDL_ASSERT(libbsdf != NULL);

    // clear target triple to avoid LLVM warning on console about mixing different targets
    // when linking libbsdf ("x86_x64-pc-win32") with libdevice ("nvptx-unknown-unknown").
    // Using an nvptx target for libbsdf would cause struct parameters to be split, which we
    // try to avoid.
    libbsdf->setTargetTriple("");

    // also avoid LLVM warning on console about mixing different data layouts
    libbsdf->setDataLayout(m_module->getDataLayout());

    // remove all comdat infos from functions in the libbsdf module,
    // as this is not used by us and not supported on MacOS
    for (llvm::Function &f : libbsdf->functions()) {
        f.setComdat(nullptr);
    }

    // temporarily create a global to properly link the spectral_sample struct type,
    // which is either struct.Spectral_sample_struct or struct.float3 depending on spectral
    // rendering being enabled or not
    llvm::GlobalVariable *tmp_spectral_sample_global = new llvm::GlobalVariable(
        *m_module,
        m_type_mapper.get_spectral_sample_type(),
        true,
        llvm::GlobalValue::InternalLinkage,
        llvm::Constant::getNullValue(m_type_mapper.get_spectral_sample_type()),
        "tmp_spectral_sample_global");

    // collect all functions available before linking
    // note: we cannot use the function pointers, as linking removes some function declarations and
    //       may reuse the old pointers
    hash_set<string, string_hash<string> >::Type old_func_names(get_allocator());
    for (llvm::Function &f : m_module->functions()) {
        if (!f.isDeclaration()) {
            old_func_names.insert(string(f.getName().begin(), f.getName().end(), get_allocator()));
        }
    }

    if (llvm::Linker::linkModules(*m_module, std::move(libbsdf))) {
        // true means linking has failed
        error(LINKING_LIBBSDF_FAILED, "unknown linker error");
        MDL_ASSERT(!"Linking libbsdf failed");
        return false;
    }

    // remove the temporary global after linking
    tmp_spectral_sample_global->eraseFromParent();

    m_float3_struct_type = llvm::StructType::getTypeByName(
        m_llvm_context, "struct.float3");
    if (m_float3_struct_type == NULL) {
        // name was lost during linking? get it from
        //    void @black_bsdf_sample(
        //        %struct.BSDF_sample_data* nocapture %data,
        //        %class.State* nocapture readnone %state,
        //        %struct.float3* nocapture readnone %inherited_normal)

        llvm::Function *func = m_module->getFunction("black_bsdf_sample");
        MDL_ASSERT(func != NULL);
        llvm::FunctionType *func_type = func->getFunctionType();
        m_float3_struct_type = llvm::cast<llvm::StructType>(
            func_type->getParamType(2)->getPointerElementType());
        MDL_ASSERT(m_float3_struct_type != NULL);
    }


    create_bsdf_function_types();
    create_edf_function_types();

    // get the unique IDs for two metadata we will use
    m_bsdf_param_metadata_id = m_llvm_context.getMDKindID("libbsdf.bsdf_param");
    m_edf_param_metadata_id  = m_llvm_context.getMDKindID("libbsdf.edf_param");

    llvm::Type *int_type = m_type_mapper.get_int_type();
    unsigned alloca_addr_space = m_module->getDataLayout().getAllocaAddrSpace();

    // find all functions which were added by linking the libbsdf module,
    // collect in vector as module functions will be modified, later
    vector<llvm::Function *>::Type libbsdf_funcs(get_allocator());
    for (llvm::Function &f : m_module->functions()) {
        // just a declaration or did already exist before linking? -> skip
        if (f.isDeclaration() || old_func_names.count(
                string(f.getName().begin(), f.getName().end(), get_allocator())) != 0)
        {
            continue;
        }

        // Found a libbsdf function
        libbsdf_funcs.push_back(&f);
    }

    // iterate over all functions added from the libbsdf module
    for (llvm::Function *func : libbsdf_funcs) {
        // remove "target-features" attribute to avoid warnings about unsupported PTX features
        // for non-PTX backends
        func->removeFnAttr("target-features");

        // make all functions from libbsdf internal to allow global dead code elimination
        func->setLinkage(llvm::GlobalValue::InternalLinkage);

        // set always inline if necessary (this also handles non-instantiated functions
        // like Fresnel_function_coated::eval())
        if (is_always_inline_enabled() && !func->hasFnAttribute(llvm::Attribute::NoInline)) {
            func->addFnAttr(llvm::Attribute::AlwaysInline);
        }

        m_state_usage_analysis.register_function(func);

        // translate all runtime calls
        {
            Function_context ctx(
                get_allocator(),
                *this,
                func,
                get_df_function_flags(func),  // note, the lambda results are not really used
                false);  // don't optimize, because of parameter handling via uninitialized allocas

            // search for all CallInst instructions and link runtime function calls to the
            // corresponding intrinsics
            for (llvm::Function::iterator BI = func->begin(), BE = func->end(); BI != BE; ++BI) {
                for (llvm::BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
                    if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(II)) {
                        if (!translate_libbsdf_runtime_call(call, II)) {
                            return false;
                        }
                    }
                }
            }
        }

        // check whether this is a BSDF API function, for which we need to update the prototype
        if (func->arg_size() >= 3) {
            llvm::Function::arg_iterator func_arg_it = func->arg_begin();
            llvm::Value *first_arg = func_arg_it++;

            // is the type of the first parameter one of the BSDF data types?
            if (llvm::PointerType *df_data_ptr_type =
                    llvm::dyn_cast<llvm::PointerType>(first_arg->getType()))
            {
                llvm::Type *df_data_type = df_data_ptr_type->getElementType();

                llvm::FunctionType *new_func_type;
                IType::Kind df_kind = IType::TK_ERROR;
                bool has_inherited_weight = false;

                // bsdf
                if (df_data_type == m_type_bsdf_sample_data) {
                    new_func_type = m_type_bsdf_sample_func;
                    df_kind = IType::TK_BSDF; // or TK_HAIR_BSDF
                } else if (df_data_type == m_type_bsdf_evaluate_data) {
                    // *_get_factor() functions use evaluate data struct, but not inherited_weight
                    if (func->getName().endswith("_get_factor")) {
                        new_func_type = m_type_bsdf_get_factor_func;
                        has_inherited_weight = false;
                    } else {
                        new_func_type = m_type_bsdf_evaluate_func;
                        has_inherited_weight = true;
                    }
                    df_kind = IType::TK_BSDF; // or TK_HAIR_BSDF
                } else if (df_data_type == m_type_bsdf_pdf_data) {
                    new_func_type = m_type_bsdf_pdf_func;
                    df_kind = IType::TK_BSDF; // or TK_HAIR_BSDF
                } else if (df_data_type == m_type_bsdf_auxiliary_data) {
                    new_func_type = m_type_bsdf_auxiliary_func;
                    df_kind = IType::TK_BSDF; // or TK_HAIR_BSDF
                    has_inherited_weight = true;
                }
                // edf
                else if (df_data_type == m_type_edf_sample_data) {
                    new_func_type = m_type_edf_sample_func;
                    df_kind = IType::TK_EDF;
                } else if (df_data_type == m_type_edf_evaluate_data) {
                    // *_get_factor() functions use evaluate data struct, but not inherited_weight
                    if (func->getName().endswith("_get_factor")) {
                        new_func_type = m_type_edf_get_factor_func;
                        has_inherited_weight = false;
                    } else {
                        new_func_type = m_type_edf_evaluate_func;
                        has_inherited_weight = true;
                    }
                    df_kind = IType::TK_EDF;
                } else if (df_data_type == m_type_edf_pdf_data) {
                    new_func_type = m_type_edf_pdf_func;
                    df_kind = IType::TK_EDF;
                } else if (df_data_type == m_type_edf_auxiliary_data) {
                    new_func_type = m_type_edf_auxiliary_func;
                    df_kind = IType::TK_EDF;
                    has_inherited_weight = true;
                } else {
                    new_func_type = NULL;
                }

                char const *df_arg = "";
                char const *df_arg_var = "";
                char const *df_struct_name = "";
                switch (df_kind) {
                case IType::TK_BSDF:
                case IType::TK_HAIR_BSDF:
                    df_arg = "bsdf_arg";
                    df_arg_var = "bsdf_arg_var";
                    df_struct_name = "struct.BSDF";
                    break;

                case IType::TK_EDF:
                    df_arg = "edf_arg";
                    df_arg_var = "edf_arg_var";
                    df_struct_name = "struct.EDF";
                    break;
                default:
                    break;
                }

                // for HLSL and GLSL check for interpreter functions
                if (target_is_structured_language() && func->getName().startswith("mdl_bsdf_")) {
                    llvm::FunctionType *old_func_type = func->getFunctionType();
                    if (df_data_type == m_type_bsdf_sample_data) {
                        MDL_ASSERT(func->getName() == "mdl_bsdf_sample");
                    } else if (df_data_type == m_type_bsdf_evaluate_data) {
                        MDL_ASSERT(func->getName() == "mdl_bsdf_evaluate");
                    } else if (df_data_type == m_type_bsdf_pdf_data) {
                        MDL_ASSERT(func->getName() == "mdl_bsdf_pdf");
                    } else {
                        MDL_ASSERT(!"Unexpected function starting with \"mdl_bsdf_\"");
                    }

                    llvm::Function::arg_iterator old_arg_it = func->arg_begin();
                    llvm::Value *old_df_data_param       = old_arg_it++;
                    llvm::Value *old_state_param         = old_arg_it++;
                    llvm::Value *old_scratch_space_param = old_arg_it++;
                    llvm::Value *old_material_param      = old_arg_it++;

                    llvm::Type *arg_types[] = {
                        Type_mapper::get_ptr(df_data_type),
                        m_type_mapper.get_state_ptr_type(m_state_mode),
                        old_func_type->getParamType(2),
                        old_func_type->getParamType(3)
                    };

                    llvm::FunctionType *new_interpreter_func_type = llvm::FunctionType::get(
                        old_func_type->getReturnType(), arg_types, false);

                    llvm::Function *new_func = llvm::Function::Create(
                        new_interpreter_func_type,
                        llvm::GlobalValue::InternalLinkage,
                        "",
                        m_module);
                    m_state_usage_analysis.register_cloned_function(new_func, func);
                    llvm::DISubprogram *di_func = func->getSubprogram();
                    new_func->setSubprogram(di_func);
                    set_llvm_function_attributes(new_func, /*mark_noinline=*/false);
                    new_func->takeName(func);
                    new_func->getBasicBlockList().splice(
                        new_func->begin(), func->getBasicBlockList());

                    // make sure we don't introduce initialization code before alloca instructions
                    llvm::BasicBlock::iterator param_init_insert_point = new_func->front().begin();
                    while (llvm::isa<llvm::AllocaInst>(param_init_insert_point))
                        ++param_init_insert_point;

                    llvm::Function::arg_iterator new_arg_it = new_func->arg_begin();
                    llvm::Value *new_df_data_param       = new_arg_it++;
                    llvm::Value *new_state_param         = new_arg_it++;
                    llvm::Value *new_scratch_space_param = new_arg_it++;
                    llvm::Value *new_material_param      = new_arg_it++;

                    llvm::Instruction *state_cast = new llvm::BitCastInst(
                        new_state_param,
                        old_state_param->getType(),
                        "state_cast",
                        &*param_init_insert_point);
                    if (di_func) {
                        state_cast->setDebugLoc(llvm::DILocation::get(
                            di_func->getContext(), di_func->getLine(), 0, di_func));
                    }

                    // replace all uses of parameters
                    old_df_data_param->replaceAllUsesWith(new_df_data_param);
                    old_state_param->replaceAllUsesWith(state_cast);
                    old_scratch_space_param->replaceAllUsesWith(new_scratch_space_param);
                    old_material_param->replaceAllUsesWith(new_material_param);

                    func->eraseFromParent();
                    continue;
                }

                IDefinition::Semantics sema = get_libbsdf_function_semantics(func->getName());

                if (new_func_type != NULL && (is_df_semantics(sema) ||
                    sema == IDefinition::DS_INVALID_REF_CONSTRUCTOR))
                {
                    // this is a BSDF API function

                    // For DF instantiation, any DF parameters (like tint, roughness or layer) are
                    // replaced by local variable placeholders. These will be replaced by the real
                    // values or calls to other instantiated DF functions during instantiation.
                    // The local variables will be placed at the beginning of the entry block.

                    llvm::Function *old_func = func;

                    llvm::Function::arg_iterator old_arg_it  = old_func->arg_begin();
                    llvm::Function::arg_iterator old_arg_end = old_func->arg_end();
                    llvm::Value *df_data          = old_arg_it++;
                    llvm::Value *exec_ctx         = old_arg_it++;
                    llvm::Value *inherited_normal = old_arg_it++;
                    llvm::Value *inherited_weight = NULL;
                    if (has_inherited_weight) {
                        inherited_weight = old_arg_it++;
                    }

                    llvm::Function *new_func = llvm::Function::Create(
                        new_func_type,
                        llvm::GlobalValue::InternalLinkage,
                        "",
                        m_module);
                    m_state_usage_analysis.register_cloned_function(new_func, func);
                    llvm::DISubprogram *di_func = old_func->getSubprogram();
                    new_func->setSubprogram(di_func);
                    set_llvm_function_attributes(new_func, /*mark_noinline=*/false);
                    new_func->setName("gen_" + func->getName());
                    new_func->getBasicBlockList().splice(
                        new_func->begin(), old_func->getBasicBlockList());

                    // the exec_ctx parameter (or state parameter if lambda results are not
                    // supported) does not alias and is not captured
                    new_func->addParamAttr(1, llvm::Attribute::NoAlias);
                    new_func->addParamAttr(1, llvm::Attribute::NoCapture);

                    // the inherited normal does not alias and is not captured
                    new_func->addParamAttr(2, llvm::Attribute::NoAlias);
                    new_func->addParamAttr(2, llvm::Attribute::NoCapture);

                    m_libbsdf_template_funcs.push_back(new_func);

                    // make sure we don't introduce initialization code before alloca instructions
                    llvm::BasicBlock::iterator param_init_insert_point = new_func->front().begin();
                    while (llvm::isa<llvm::AllocaInst>(param_init_insert_point))
                        ++param_init_insert_point;

                    // tell context where to find the state parameters
                    llvm::Function::arg_iterator arg_it = new_func->arg_begin();
                    llvm::Value *data_param             = arg_it++;
                    llvm::Value *exec_ctx_param         = arg_it++;
                    llvm::Value *inherited_normal_param = arg_it++;
                    llvm::Value *inherited_weight_param = NULL;
                    if (has_inherited_weight) {
                        inherited_weight_param = arg_it++;
                    }

                    llvm::DILocation *start_loc = NULL;

                    llvm::Instruction *exec_ctx_cast = new llvm::BitCastInst(
                        exec_ctx_param,
                        exec_ctx->getType(),
                        "exec_ctx_cast",
                        &*param_init_insert_point);
                    if (di_func) {
                        start_loc = llvm::DILocation::get(
                            di_func->getContext(), di_func->getLine(), 0, di_func);
                        exec_ctx_cast->setDebugLoc(start_loc);
                    }

                    // replace all uses of parameters which will not be removed
                    df_data->replaceAllUsesWith(data_param);
                    exec_ctx->replaceAllUsesWith(exec_ctx_cast);
                    inherited_normal->replaceAllUsesWith(inherited_normal_param);
                    if (has_inherited_weight) {
                        inherited_weight->replaceAllUsesWith(inherited_weight_param);
                    }

                    // introduce local variables at the beginning of the entry block for all used
                    // DF parameters
                    bool skipped_df_idx_inc = false;
                    for (int df_idx = 0; old_arg_it != old_arg_end; ++old_arg_it) {
                        int cur_df_idx = df_idx;

                        // Determine parameter index for next iteration
                        if (skipped_df_idx_inc) {
                            skipped_df_idx_inc = false;
                            ++df_idx;
                        } else if (is_libbsdf_array_parameter(sema, cur_df_idx)) {
                            // array parameters consist of a pointer and a length in libbsdf
                            // and both get the same associated df parameter index
                            skipped_df_idx_inc = true;
                        } else {
                            ++df_idx;
                        }

                        if (old_arg_it->use_empty()) {
                            continue;
                        }

                        llvm::AllocaInst *arg_var;
                        llvm::Instruction *arg_val;
                        if (llvm::PointerType *ptr_type = llvm::dyn_cast<llvm::PointerType>(
                            old_arg_it->getType()))
                        {
                            llvm::Type *elem_type = ptr_type->getElementType();

                            arg_val = arg_var = new llvm::AllocaInst(
                                elem_type,
                                alloca_addr_space,
                                df_arg,
                                &*new_func->getEntryBlock().begin());

                            if (elem_type->isStructTy() &&
                                !llvm::cast<llvm::StructType>(elem_type)->isLiteral() &&
                                elem_type->getStructName() == df_struct_name)
                            {
                                // for *DF typed parameters, we mark the calls to the DF methods
                                // with metadata additionally to the local variables.
                                // We still need the meta data on the local variables,
                                // as the select methods have two DF parameters.
                                // The argument value is not necessary, but we keep it, in case
                                // the uses are not optimized away.
                                // Note: we don't do this for the DFs inside *DF_component!
                                mark_df_calls(old_arg_it, cur_df_idx, df_kind);
                            }
                        } else {
                            // for non-pointer types we also need to load the value
                            // and replace the argument by the load, not the alloca
                            arg_var = new llvm::AllocaInst(
                                old_arg_it->getType(),
                                alloca_addr_space,
                                df_arg_var,
                                &*new_func->getEntryBlock().begin());
                            arg_var->setDebugLoc(start_loc);
                            arg_val = new llvm::LoadInst(
                                old_arg_it->getType(),
                                arg_var,
                                df_arg,
                                &*param_init_insert_point);
                            arg_val->setDebugLoc(start_loc);
                        }

                        // set metadata on the local variables
                        llvm::ConstantAsMetadata *param_idx = llvm::ConstantAsMetadata::get(
                            llvm::ConstantInt::get(int_type, cur_df_idx));
                        llvm::MDNode *md = llvm::MDNode::get(m_llvm_context, param_idx);

                        switch (df_kind) {
                        case IType::TK_BSDF:
                        case IType::TK_HAIR_BSDF:
                            arg_var->setMetadata(m_bsdf_param_metadata_id, md);
                            break;
                        case IType::TK_EDF:
                            arg_var->setMetadata(m_edf_param_metadata_id, md);
                            break;
                        default:
                            MDL_ASSERT(!"Linking libbsdf failed");
                            return false;
                        }

                        old_arg_it->replaceAllUsesWith(arg_val);
                    }

                    old_func->eraseFromParent();
                }
            }
        }
    }

    return true;
}

// Store a value inside a float4 array at the given byte offset, updating the offset.
void LLVM_code_generator::store_to_float4_array_impl(
    llvm::Value      *val,
    llvm::Value      *dest,
    unsigned         &dest_offs)
{
    Function_context &ctx = *m_ctx;

    llvm::Type *val_type = val->getType();

    if (llvm::IntegerType *it = llvm::dyn_cast<llvm::IntegerType>(val_type)) {
        if (it->getBitWidth() > 8) {
            dest_offs = (dest_offs + 3) & ~3;
        }

        llvm::Value *access[] = {
            ctx.get_constant(int(0)),
            ctx.get_constant(int(dest_offs >> 4)),     // float4 index
            ctx.get_constant(int(dest_offs >> 2) & 3)  // float index within float4
        };

        llvm::Value *ptr = ctx->CreateInBoundsGEP(dest, access);

        // store i1 and i8 in one byte per value, as specified by the data layout
        if (it->getBitWidth() <= 8) {
            // only modify the bits corresponding to the data offset
            llvm::IntegerType *i32_type = llvm::IntegerType::get(m_llvm_context, 32);
            val = ctx->CreateZExt(val, i32_type);
            ptr = ctx->CreatePointerCast(ptr, i32_type->getPointerTo());
            llvm::Value *data = ctx->CreateLoad(ptr);
            data = ctx->CreateAnd(
                data,
                ctx.get_constant(int(~(0xff << ((dest_offs & 3) * 8)))));
            if ((dest_offs & 3) != 0) {
                val = ctx->CreateShl(val, (dest_offs & 3) * 8);
            }
            data = ctx->CreateOr(data, val);
            ctx->CreateStore(data, ptr);
            ++dest_offs;
            return;
        }

        ptr = ctx->CreatePointerCast(ptr, it->getPointerTo());
        ctx->CreateStore(val, ptr);
        dest_offs += 4;
        return;
    }

    if (val_type->isFloatTy()) {
        dest_offs = (dest_offs + 3) & ~3;
        llvm::Value *access[] = {
            ctx.get_constant(int(0)),
            ctx.get_constant(int(dest_offs >> 4)),     // float4 index
            ctx.get_constant(int(dest_offs >> 2) & 3)  // float index within float4
        };

        llvm::Value *ptr = ctx->CreateInBoundsGEP(dest, access);
        ctx->CreateStore(val, ptr);
        dest_offs += 4;
        return;
    }

    if (llvm::isa<llvm::StructType>(val_type) || llvm::isa<llvm::FixedVectorType>(val_type)
            || llvm::isa<llvm::ArrayType>(val_type)) {
        size_t size = size_t(
            ctx.get_code_gen().get_target_layout_data()->getTypeAllocSize(val_type));
        unsigned compound_start_offs = dest_offs;

        uint64_t n;
        if (llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(val_type)) {
            n = st->getNumElements();
        } else if (llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(val_type)){
            n = vt->getNumElements();
        } else {
            n = val_type->getArrayNumElements();
        }

        for (uint64_t i = 0; i < n; ++i) {
            llvm::Value *elem = ctx.create_extract(val, unsigned(i));
            store_to_float4_array_impl(elem, dest, dest_offs);
        }

        // compound values might have an higher alignment then the sum of its components
        dest_offs = compound_start_offs + size;
        return;
    }

    // TODO: bool, enum, double (, string?)
    MDL_ASSERT(!"not supported");
}

// Store a value inside a float4 array at the given byte offset.
void LLVM_code_generator::store_to_float4_array(
    llvm::Value *val,
    llvm::Value *dest,
    unsigned dest_offs)
{
    // call wrapped function with a copy of the offset, so only the copy is changed
    store_to_float4_array_impl(val, dest, dest_offs);
}

// Load a value inside a float4 array at the given byte offset, updating the offset.
llvm::Value *LLVM_code_generator::load_from_float4_array_impl(
    llvm::Type       *val_type,
    llvm::Value      *src,
    unsigned         &src_offs)
{
    Function_context &ctx = *m_ctx;

    if (llvm::IntegerType *it = llvm::dyn_cast<llvm::IntegerType>(val_type)) {
        if (it->getBitWidth() > 8) {
            src_offs = (src_offs + 3) & ~3;
        }

        llvm::Value *access[] = {
            ctx.get_constant(int(0)),
            ctx.get_constant(int(src_offs >> 4)),     // float4 index
            ctx.get_constant(int(src_offs >> 2) & 3)  // float index within float4
        };

        llvm::Value *ptr = ctx->CreateInBoundsGEP(src, access);

        // load i1 and i8 from one byte per value, as specified by the data layout
        if (it->getBitWidth() <= 8) {
            llvm::IntegerType *i32_type = llvm::IntegerType::get(m_llvm_context, 32);
            ptr = ctx->CreatePointerCast(ptr, i32_type->getPointerTo());
            llvm::Value *val = ctx->CreateLoad(ptr);

            if ((src_offs & 3) != 0) {
                val = ctx->CreateLShr(val, (src_offs & 3) * 8);
            }
            val = ctx->CreateTrunc(val, it);
            ++src_offs;
            return val;
        }

        ptr = ctx->CreatePointerCast(ptr, it->getPointerTo());
        llvm::Value *elem = ctx->CreateLoad(ptr);
        src_offs += 4;
        return elem;
    }

    if (val_type->isFloatTy()) {
        src_offs = (src_offs + 3) & ~3;
        llvm::Value *access[] = {
            ctx.get_constant(int(0)),
            ctx.get_constant(int(src_offs >> 4)),     // float4 index
            ctx.get_constant(int(src_offs >> 2) & 3)  // float index within float4
        };

        llvm::Value *ptr = ctx->CreateInBoundsGEP(src, access);
        llvm::Value *elem = ctx->CreateLoad(ptr);
        src_offs += 4;
        return elem;
    }

    if (llvm::isa<llvm::StructType>(val_type) || llvm::isa<llvm::FixedVectorType>(val_type)
            || llvm::isa<llvm::ArrayType>(val_type)) {
        size_t size = size_t(
            ctx.get_code_gen().get_target_layout_data()->getTypeAllocSize(val_type));
        unsigned compound_start_offs = src_offs;

        uint64_t n;
        if (llvm::StructType* st = llvm::dyn_cast<llvm::StructType>(val_type)) {
            n = st->getNumElements();
        } else if (llvm::FixedVectorType* vt = llvm::dyn_cast<llvm::FixedVectorType>(val_type)) {
            n = vt->getNumElements();
        } else {
            n = val_type->getArrayNumElements();
        }

        llvm::Value *res = llvm::UndefValue::get(val_type);
        for (uint64_t i = 0; i < n; ++i) {
            llvm::Value *elem = load_from_float4_array_impl(
                llvm::GetElementPtrInst::getTypeAtIndex(val_type, unsigned(i)), src, src_offs);
            res = ctx.create_insert(res, elem, unsigned(i));
        }

        // compound values might have an higher alignment then the sum of its components
        src_offs = compound_start_offs + size;
        return res;
    }

    // TODO: bool, enum, double (, string?)
    MDL_ASSERT(!"not supported");
    return llvm::UndefValue::get(val_type);
}

// Load a value inside a float4 array at the given byte offset.
llvm::Value *LLVM_code_generator::load_from_float4_array(
    llvm::Type       *val_type,
    llvm::Value      *src,
    unsigned         src_offs)
{
    // call wrapped function with a copy of the offset, so only the copy is changed
    return load_from_float4_array_impl(val_type, src, src_offs);
}

// Translate a DAG call argument which may be a precalculated lambda function to LLVM IR.
Expression_result LLVM_code_generator::translate_call_arg(
    DAG_node const   *arg,
    llvm::Type       *expected_type)
{
    Function_context &ctx = *m_ctx;

    // ensure that we are at the end of a block as we will call translate_node().
    MDL_ASSERT(ctx->GetInsertPoint() == ctx->GetInsertBlock()->end());

    Expression_result res = translate_node(arg, m_cur_resolver);

    // type doesn't matter or fits already?
    if (expected_type == NULL || res.get_value_type() == expected_type) {
        return res;
    }

    // convert to expected type
    return Expression_result::value(ctx.load_and_convert(expected_type, res.as_ptr(ctx)));
}

// Returns true, if a texture result is stored for this node.
bool LLVM_code_generator::has_texture_result(DAG_node const *node) const
{
    if (m_cur_req_node == nullptr) {
        return false;
    }
    return m_texture_result_map[m_cur_req_node->eval_state].find(node) !=
        m_texture_result_map[m_cur_req_node->eval_state].end();
}

/// Returns the texture result for the given node in the evaluation state of the current
/// requested node, or nullptr if there is none.
///
/// \param node  the DAG node
LLVM_code_generator::Texture_result_slot const *LLVM_code_generator::get_texture_result_slot(
    DAG_node const *node) const
{
    if (m_cur_req_node == nullptr) {
        return nullptr;
    }

    auto it = m_texture_result_map[m_cur_req_node->eval_state].find(node);
    if (it == m_texture_result_map[m_cur_req_node->eval_state].end()) {
        return nullptr;
    }

    return &it->second;
}

// Translate a DAG node which may be a precalculated lambda function to LLVM IR
// at the current insert point.
Expression_result LLVM_code_generator::translate_node_at_insert_point(
    DAG_node const   *node,
    llvm::Type       *expected_type)
{
    Function_context &ctx = *m_ctx;

    Expression_result res;

    // skip temporaries
    while (is<DAG_temporary>(node))
        node = as<DAG_temporary>(node)->get_expr();

    // Translate constants, parameters and nodes with texture results without extra blocks
    if (is<DAG_constant>(node) || is<DAG_parameter>(node) || has_texture_result(node)) {
        res = translate_node(node, m_cur_resolver);

        // convert to expected type, if necessary
        if (res.get_value_type() != expected_type) {
            res = Expression_result::value(ctx.load_and_convert(expected_type, res.as_ptr(ctx)));
        }
    } else {
        // We need to do this in a separate block without a terminator, because translate_node
        // expects to insert code at the end of a block.

        llvm::BasicBlock *insert_bb = ctx->GetInsertBlock();
        llvm::BasicBlock *after_bb = insert_bb->splitBasicBlock(
            ctx->GetInsertPoint(), "after_split");
        insert_bb->getTerminator()->eraseFromParent();

        ctx->SetInsertPoint(insert_bb);

        res = translate_node(node, m_cur_resolver);

        // convert to expected type, if necessary
        if (res.get_value_type() != expected_type) {
            res = Expression_result::value(ctx.load_and_convert(expected_type, res.as_ptr(ctx)));
        }

        ctx->CreateBr(after_bb);
        ctx->SetInsertPoint(&after_bb->front());
    }

    return res;
}

// Get the BSDF parameter ID metadata for an instruction.
int LLVM_code_generator::get_metadata_df_param_id(
    llvm::Instruction *inst,
    IType::Kind       kind)
{
    if (inst == NULL) {
        return -1;
    }

    llvm::MDNode *md = NULL;
    switch (kind) {
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
        md = inst->getMetadata(m_bsdf_param_metadata_id);
        break;

    case IType::TK_EDF:
        md = inst->getMetadata(m_edf_param_metadata_id);
        break;

    default:
        MDL_ASSERT(!"Invalid DF alloca parameter metadata");
        return -1;
    }

    if (md == NULL) {
        return -1;
    }

    llvm::ConstantInt *param_idx_val =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(md->getOperand(0));
    if (param_idx_val == NULL) {
        MDL_ASSERT(!"Invalid BSDF alloca parameter metadata");
        return -1;
    }
    return int(param_idx_val->getValue().getZExtValue());
}

// Rewrite the address of a memcpy from a color_bsdf_component to the given weight array.
bool LLVM_code_generator::rewrite_weight_memcpy_addr(
    llvm::Value                                *weight_array,
    llvm::BitCastInst                          *addr_bitcast,
    llvm::Value                                *index,
    llvm::SmallVector<llvm::Instruction *, 16> &delete_list)
{
    // check for
    //   <C> = bitcast %struct.color_xDF_component* <X> to i8*
    //   call void @llvm.memcpy.p0i8.p0i8.i64(i8* <Y>, i8* <C>, i64 12, i32 4, i1 false)

    Function_context &ctx = *m_ctx;

    // ensure, that all usages of this cast are memcpys of a weight
    for (auto cast_user : addr_bitcast->users()) {
        llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(cast_user);
        if (call == NULL) {
            MDL_ASSERT(
                !"Unsupported usage of color_xDF_component parameter with bitcast");
            return false;
        }
        llvm::Function *called_func = call->getCalledFunction();
        if (!called_func->getName().startswith("llvm.memcpy.")) {
            MDL_ASSERT(
                !"Unsupported usage of color_xDF_component parameter with bitcast/call");
            return false;
        }
        if (call->getNumArgOperands() != 5 ||
                call->getArgOperand(1) != addr_bitcast ||             // source is cast
                !ctx.is_constant_value(call->getArgOperand(2), 12)) { // size of float3
            MDL_ASSERT(
                !"Unsupported usage of color_xDF_component parameter with memcpy");
            return false;
        }
    }

    // rewrite cast to use pointer to index'th weight in weight array
    llvm::Value *null_val = llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type());
    llvm::Value *idxs[] = { null_val, index };
    llvm::GetElementPtrInst *weight_ptr = llvm::GetElementPtrInst::Create(
        nullptr, weight_array, idxs, "", addr_bitcast);
    weight_ptr->setDebugLoc(addr_bitcast->getDebugLoc());

    llvm::Instruction *new_cast = llvm::BitCastInst::Create(
        llvm::Instruction::BitCast, weight_ptr, addr_bitcast->getType(), "", weight_ptr);
    new_cast->setDebugLoc(addr_bitcast->getDebugLoc());

    addr_bitcast->replaceAllUsesWith(new_cast);
    delete_list.push_back(addr_bitcast);

    return true;
}

// Rewrite all usages of a BSDF component variable using the given weight array and the
// BSDF function, which can either be a switch function depending on the array index
// or the same function for all indices.
void LLVM_code_generator::rewrite_df_component_usages(
    llvm::AllocaInst                           *inst,
    llvm::Value                                *weight_array,
    llvm::Value                                *df_flags_array,
    Df_component_info                          &comp_info,
    llvm::SmallVector<llvm::Instruction *, 16> &delete_list)
{
    Function_context &ctx = *m_ctx;

    // These rewrites are performed:
    //  - bsdf_component[i].weight -> weights[i]
    //  - bsdf_component[i].component.sample() -> df_func(...) or df_func(..., i)
    for (auto user : inst->users()) {
        llvm::GetElementPtrInst *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(user);
        if (gep == NULL) {
            // check for
            //   <C> = bitcast %struct.color_BSDF_component* <X> to i8*
            //   call void @llvm.memcpy.p0i8.p0i8.i64(i8* <Y>, i8* <C>, i64 12, i32 4, i1 false)
            llvm::BitCastInst *cast = llvm::dyn_cast<llvm::BitCastInst>(user);
            if (cast == NULL) {
                MDL_ASSERT(!"Unsupported usage of color_xDF_component parameter");
                continue;
            }

            llvm::Value *null_val = llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type());
            rewrite_weight_memcpy_addr(weight_array, cast, null_val, delete_list);
            continue;
        }

        llvm::Value *component_idx_val = gep->getOperand(1);
        if (gep->getNumOperands() == 2) {
            // check for
            //   <X> = getelementptr inbounds %struct.color_BSDF_component* %bsdf_arg, i64 <I>
            //   <C> = bitcast %struct.color_BSDF_component* <X> to i8*
            //   call void @llvm.memcpy.p0i8.p0i8.i64(i8* <Y>, i8* <C>, i64 12, i32 4, i1 false)

            for (auto gep_user : gep->users()) {
                llvm::BitCastInst *cast = llvm::dyn_cast<llvm::BitCastInst>(gep_user);
                if (cast == NULL) {
                    MDL_ASSERT(!"Unsupported gep usage of color_xDF_component parameter");
                    continue;
                }
                rewrite_weight_memcpy_addr(weight_array, cast, component_idx_val, delete_list);
            }
            delete_list.push_back(gep);
            continue;
        }

        llvm::Value *struct_idx_val = gep->getOperand(2);
        MDL_ASSERT(struct_idx_val);
        llvm::ConstantInt *struct_idx_const =
            llvm::dyn_cast<llvm::ConstantInt>(struct_idx_val);
        MDL_ASSERT(struct_idx_const);
        unsigned struct_idx = unsigned(struct_idx_const->getValue().getZExtValue());

        // access to weight?
        if (struct_idx == 0) {
            llvm::Instruction *new_gep;

            // check whether this is actually
            //   color_df_component[i].weight.x/y/z -> color_weights[i].x/y/z
            if (gep->getNumOperands() == 4) {
                // replace by access to same color component on same index of color array
                llvm::Value *col_comp_idx_val = gep->getOperand(3);
                llvm::Value *idxs[] = {
                    llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type()),
                    component_idx_val,
                    col_comp_idx_val
                };
                new_gep = llvm::GetElementPtrInst::Create(nullptr, weight_array, idxs, "", gep);
            }
            else if (gep->getNumOperands() == 5) {
                // spectral case:
                // color_df_component[i].weight.values.x/y/z -> color_weights[i].values.x/y/z
                // replace by access to same color component on same index of color array
                llvm::Value *struct_field_idx_val = gep->getOperand(3);
                llvm::Value *col_comp_idx_val = gep->getOperand(4);
                llvm::Value *idxs[] = {
                    llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type()),
                    component_idx_val,
                    struct_field_idx_val,
                    col_comp_idx_val
                };
                new_gep = llvm::GetElementPtrInst::Create(nullptr, weight_array, idxs, "", gep);
            } else {
                // replace by access on same index of weight array (can be float or color)
                llvm::Value *idxs[] = {
                    llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type()),
                    component_idx_val
                };
                new_gep = llvm::GetElementPtrInst::Create(nullptr, weight_array, idxs, "", gep);
            }
            new_gep->setDebugLoc(gep->getDebugLoc());
            gep->replaceAllUsesWith(new_gep);
            continue;
        }

        // access to component?
        if (struct_idx == 1) {
            // We have to rewrite all accesses.
            // The code we search for should look like this:
            //  - %elemptr = getelementptr %components, %i, 1, bsdf_field_index
            //  - %funcptr = load %elemptr
            //  - call %funcptr
            // So iterate over all usages of the gep and the loads
            MDL_ASSERT(gep->getNumOperands() == 4);
            llvm::Value *bsdf_field_index = gep->getOperand(3);
            llvm::ConstantInt *bsdf_field_index_const =
                llvm::dyn_cast<llvm::ConstantInt>(bsdf_field_index);
            MDL_ASSERT(bsdf_field_index_const);
            Libbsdf_DF_func_kind df_func_kind = get_libbsdf_df_func_kind(bsdf_field_index_const);

            for (auto gep_user : gep->users()) {
                llvm::LoadInst *load = llvm::dyn_cast<llvm::LoadInst>(gep_user);
                MDL_ASSERT(load);

                for (auto load_user : load->users()) {
                    llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(load_user);
                    MDL_ASSERT(call);

                    if (df_func_kind == LDFK_HAS_ALLOWED_COMPONENTS) {
                        if (m_libbsdf_flags_in_bsdf_data) {
                            auto oldIP = ctx->saveIP();
                            ctx->SetInsertPoint(call);

                            llvm::Value *comp_val = ctx->CreateLoad(
                                ctx.create_simple_gep_in_bounds(df_flags_array, component_idx_val),
                                "df_flags");
                            llvm::Value *allowed_val = call->getArgOperand(0);
                            llvm::Value *union_val = ctx->CreateAnd(comp_val, allowed_val);
                            llvm::Value *comp = ctx->CreateICmpNE(union_val, ctx.get_constant(0));
                            call->replaceAllUsesWith(comp);
                            ctx->restoreIP(oldIP);
                        } else {
                            // no flags available -> no restriction on allowed components -> true
                            call->replaceAllUsesWith(
                                llvm::ConstantInt::get(
                                    llvm::IntegerType::get(m_llvm_context, 1), 1));
                        }
                        delete_list.push_back(call);
                        continue;
                    }

                    MDL_ASSERT((df_func_kind == LDFK_SAMPLE || df_func_kind == LDFK_EVALUATE
                        || df_func_kind == LDFK_PDF || df_func_kind == LDFK_AUXILIARY) &&
                        "bsdfs in bsdf_component currently only support has_allowed_components() "
                        "and sample/evaluate/pdf/auxiliary()");

                    Distribution_function_state call_state = convert_to_df_state(df_func_kind);
                    llvm::Function *df_func = comp_info.get_df_function(ctx, call_state);
                    m_state_usage_analysis.add_call(ctx.get_function(), df_func);

                    // convert 64-bit index to 32-bit index
                    llvm::Value *idx_val = component_idx_val;
                    if (idx_val->getType() != m_type_mapper.get_int_type()) {
                        idx_val = new llvm::TruncInst(
                            component_idx_val,
                            m_type_mapper.get_int_type(),
                            "",
                            call);
                    }

                    // call it with state parameters added
                    llvm::SmallVector<llvm::Value *, 5> llvm_args;
                    llvm_args.push_back(call->getArgOperand(0));      // res_pointer
                    llvm_args.push_back(ctx.has_exec_ctx_parameter() ?
                        ctx.get_exec_ctx_parameter() : ctx.get_state_parameter());
                    llvm_args.push_back(call->getArgOperand(2));      // inherited_normal param
                    if (df_func_kind == LDFK_EVALUATE || df_func_kind == LDFK_AUXILIARY) {
                        llvm_args.push_back(call->getArgOperand(3));  // inherited_weight param
                    }
                    if (comp_info.is_switch_function()) {
                        llvm_args.push_back(idx_val);                 // BSDF function index
                    }
                    llvm::CallInst *new_call = llvm::CallInst::Create(df_func, llvm_args, "", call);
                    new_call->setDebugLoc(call->getDebugLoc());
                    delete_list.push_back(call);
                }
                delete_list.push_back(load);
            }
            continue;
        }

        MDL_ASSERT(!"Invalid access to BSDF_component structure");
    }
}

// Get the array index accessor function for the spectral sample array.
// Note: For now, the array cannot contain spectral data, so the conversion to spectral
//       happens at the end of the function.
llvm::Function *LLVM_code_generator::get_measured_curve_array_index_accessor(
    DAG_node const *arg)
{
    Measured_curve_array_index_accessor_map::iterator it =
        m_measured_curve_array_index_accessor_map.find(arg);
    if (it != m_measured_curve_array_index_accessor_map.end()) {
        return it->second;
    }

    // collect argument types
    mi::mdl::vector<llvm::Type *>::Type arg_types(get_allocator());
    arg_types.push_back(m_type_mapper.get_spectral_sample_ptr_type());  // sret
    if (target_supports_lambda_results_parameter()) {
        arg_types.push_back(m_type_mapper.get_exec_ctx_ptr_type());
    } else {
        arg_types.push_back(m_type_mapper.get_state_ptr_type(m_state_mode));
        if (target_uses_resource_data_parameter()) {
            arg_types.push_back(m_type_mapper.get_res_data_pair_ptr_type());
        }
        if (target_uses_exception_state_parameter()) {
            arg_types.push_back(m_type_mapper.get_exc_state_ptr_type());
        }
        if (target_supports_captured_argument_parameter()) {
            arg_types.push_back(m_type_mapper.get_char_ptr_type());
        }
    }
    arg_types.push_back(m_type_mapper.get_int_type());  // array index

    // create array index accessor function
    llvm::Function *func = llvm::Function::Create(
        llvm::FunctionType::get(m_type_mapper.get_void_type(), arg_types, false),
        llvm::Function::InternalLinkage,
        "get_measured_curve_array_element",
        m_module);
    m_state_usage_analysis.register_function(func);
    set_llvm_function_attributes(func, false);
    add_generated_attributes(func);
    m_measured_curve_array_index_accessor_map[arg] = func;
    BB_store func_chain(m_curr_bb, get_next_bb());

    if (m_di_builder) {
        llvm::DIFile *di_file = m_di_builder->createFile("<generated>", "");

        llvm::DISubprogram *di_func = m_di_builder->createFunction(
            /*Scope=*/ di_file,
            /*Name=*/ func->getName(),
            /*LinkageName=*/ func->getName(),
            /*File=*/ di_file,
            1,
            m_type_mapper.get_debug_info_type(
                m_di_builder, di_file, func->getFunctionType()),
            1,
            llvm::DINode::FlagPrototyped,
            llvm::DISubprogram::toSPFlags(
                /*IsLocalToUnit=*/true,
                /*IsDefinition=*/true,
                /*IsOptimized=*/is_optimized()
            ));
        func->setSubprogram(di_func);
    }

    {
        // context needs a non-empty start block, so create a jump to a second block
        llvm::BasicBlock *start_bb = llvm::BasicBlock::Create(m_llvm_context, "start", func);
        llvm::BasicBlock *body_bb  = llvm::BasicBlock::Create(m_llvm_context, "body", func);
        start_bb->getInstList().push_back(llvm::BranchInst::Create(body_bb));

        Function_context ctx(
            get_allocator(),
            *this,
            func,
            get_df_function_flags(func),
            true);

        ctx->SetInsertPoint(body_bb);

        llvm::Value *sret_ptr = func->arg_begin();
        llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();
        llvm::Value *array_index = arg_it++;

        // float3 type used by the conversion function
        llvm::Type *float3_type = m_type_mapper.get_float3_type();

        // constant or parameter array?
        if (is<DAG_parameter>(arg) || is<DAG_constant>(arg)) {
            Expression_result array_result = translate_node(arg, m_cur_resolver);
            Expression_result array_index_result = translate_index_expression(
                arg->get_type(),
                array_result,
                array_index,
                nullptr);

            llvm::Value *rgb_result = array_index_result.as_value(ctx);

            // call the conversion function on the RGB result
            llvm::Function *conv_func = get_internal_function(
                m_int_func_state_rgb_to_spectral_reflectance);
            llvm::SmallVector<llvm::Value *, 3> args;
            args.push_back(ctx.get_state_parameter());
            if (target_uses_resource_data_parameter()) {
                args.push_back(ctx.get_resource_data_parameter());
            }
            args.push_back(rgb_result);
            llvm::Value *spectral_sample_result = call_rt_func(conv_func, args);
            ctx->CreateStore(spectral_sample_result, sret_ptr);
            ctx->CreateRetVoid();
            return func;
        }

        // the array is not a constant, check whether there are enough constant elements
        // to still make it worthwhile to generate a global constant
        IType_array const *arg_type = mi::mdl::cast<IType_array>(arg->get_type());
        MDL_ASSERT(arg_type->is_immediate_sized() && "array type must be instantiated");
        int elem_count = arg_type->get_size();
        int num_const_elems = 0;

        DAG_call const *arg_call = mi::mdl::cast<DAG_call>(arg);
        for (int i = 0; i < elem_count; ++i) {
            DAG_node const *elem_node = arg_call->get_argument(i);
            if (is<DAG_constant>(elem_node)) {
                ++num_const_elems;
            }
        }

        // create a global constant if there are enough constant elements,
        // use zero placeholders for non-constant elements
        llvm::Value *array_global = nullptr;
        if (num_const_elems > 9) {
            llvm::ArrayType *color_array_type =
                llvm::ArrayType::get(float3_type, elem_count);
            llvm::SmallVector<llvm::Constant *, 8> elems(elem_count);
            for (int i = 0; i < elem_count; ++i) {
                DAG_node const *elem_node = arg_call->get_argument(i);
                if (!is<DAG_constant>(elem_node)) {
                    // non-constant element -> use zero value
                    elems[i] = llvm::ConstantAggregateZero::get(float3_type);
                    continue;
                }

                DAG_constant const *elem_const = mi::mdl::cast<DAG_constant>(elem_node);
                mi::mdl::IValue const *elem_val = elem_const->get_value();
                MDL_ASSERT(elem_val->get_kind() == mi::mdl::IValue::VK_RGB_COLOR);
                mi::mdl::IValue_compound const *color =
                    mi::mdl::cast<mi::mdl::IValue_compound>(elem_val);
                elems[i] = llvm::cast<llvm::Constant>(ctx.get_constant(color));
            }
            llvm::Constant *array = llvm::ConstantArray::get(color_array_type, elems);
            array_global = new llvm::GlobalVariable(
                *m_module,
                color_array_type,
                /*isConstant=*/ true,
                llvm::GlobalValue::InternalLinkage,
                array,
                "_global_measured_curve_array_const");
        }

        // create a switch over all indices
        llvm::BasicBlock *end_bb = ctx.create_bb("end");
        llvm::BasicBlock *invalid_index_bb = ctx.create_bb("invalid_index");
        ctx->SetInsertPoint(invalid_index_bb);
        ctx->CreateBr(end_bb);

        // create end block returning the converted result of a PHI node
        ctx->SetInsertPoint(end_bb);
        llvm::PHINode *rgb_result = ctx->CreatePHI(
            float3_type,
            elem_count + 1); // +1 for invalid index case
        rgb_result->addIncoming(llvm::ConstantInt::getNullValue(float3_type), invalid_index_bb);

        // call the conversion function on the result
        llvm::Function *conv_func = get_internal_function(
            m_int_func_state_rgb_to_spectral_reflectance);
        llvm::SmallVector<llvm::Value *, 3> args;
        args.push_back(ctx.get_state_parameter());
        if (target_uses_resource_data_parameter()) {
            args.push_back(ctx.get_resource_data_parameter());
        }
        args.push_back(rgb_result);
        llvm::Value *spectral_sample_result = call_rt_func(conv_func, args);
        ctx->CreateStore(spectral_sample_result, sret_ptr);
        ctx->CreateRetVoid();

        // if a global constant was created for constant array entries, create
        // the case block for accessing the global constant
        llvm::BasicBlock *global_constant_case_bb = nullptr;
        if (array_global != nullptr) {
            global_constant_case_bb = ctx.create_bb("global_constant_case");
            ctx->SetInsertPoint(global_constant_case_bb);
            llvm::Value *gep = ctx.create_simple_gep_in_bounds(array_global, array_index);
            llvm::Value *global_result = ctx->CreateLoad(gep);
            rgb_result->addIncoming(global_result, global_constant_case_bb);
            ctx->CreateBr(end_bb);
        }

        ctx->SetInsertPoint(body_bb);
        llvm::SwitchInst *switch_inst = ctx->CreateSwitch(
            array_index, invalid_index_bb, elem_count);

        // cache the case blocks for the nodes to avoid duplicate translations
        typedef mi::mdl::ptr_hash_map<DAG_node const, llvm::BasicBlock *>::Type
            Node_to_case_map;
        Node_to_case_map node_to_case_map(
            0,
            Node_to_case_map::hasher(),
            Node_to_case_map::key_equal(),
            get_allocator());

        for (int i = 0; i < elem_count; ++i) {
            DAG_node const *elem_node = arg_call->get_argument(i);

            // already translated?
            Node_to_case_map::iterator it = node_to_case_map.find(elem_node);
            if (it != node_to_case_map.end()) {
                switch_inst->addCase(ctx.get_constant(i), it->second);
                continue;
            }

            // use global_constant_case for constant array entries if a global constant was created
            if (global_constant_case_bb != nullptr && is<DAG_constant>(elem_node)) {
                switch_inst->addCase(ctx.get_constant(i), global_constant_case_bb);
                continue;
            }

            // create new case block and translate the node
            char case_name[32];
            snprintf(case_name, sizeof(case_name), "case_%i", i);
            llvm::BasicBlock *case_bb = ctx.create_bb(case_name);
            switch_inst->addCase(ctx.get_constant(i), case_bb);

            BB_store case_chain(m_curr_bb, get_next_bb());
            ctx->SetInsertPoint(case_bb);
            Expression_result res = translate_call_arg(elem_node, float3_type);
            llvm::Value *value = res.as_value(ctx);
            rgb_result->addIncoming(value, case_bb);
            ctx->CreateBr(end_bb);

            node_to_case_map[elem_node] = case_bb;
        }
    }

    return func;
}

// Handle BSDF array parameter during BSDF instantiation.
void LLVM_code_generator::handle_df_array_parameter(
    Function_context                           &ctx,
    IDefinition::Semantics                     sema,
    llvm::AllocaInst                           *inst,
    DAG_node const                             *arg,
    llvm::SmallVector<llvm::Instruction *, 16> &delete_list)
{
    // ensure that we are at the end of a block as we will call translate_node().
    MDL_ASSERT(ctx->GetInsertPoint() == ctx->GetInsertBlock()->end());

    llvm::Type *elem_type = inst->getAllocatedType();

    // is it an array size parameter? -> replace by the number of elements
    if (elem_type == m_type_mapper.get_int_type()) {
        int elem_count;
        if (arg->get_kind() == DAG_node::EK_CONSTANT) {
            DAG_constant const *arg_const = mi::mdl::cast<DAG_constant>(arg);
            mi::mdl::IValue_array const *arg_array =
                mi::mdl::cast<mi::mdl::IValue_array>(arg_const->get_value());
            elem_count = arg_array->get_component_count();
        } else {
            IType_array const *arg_type = mi::mdl::cast<IType_array>(arg->get_type());
            MDL_ASSERT(arg_type->is_immediate_sized() && "array type must be instantiated");
            elem_count = arg_type->get_size();
        }

        Expression_result res = Expression_result::value(
            llvm::ConstantInt::get(m_type_mapper.get_int_type(), elem_count));
        inst->replaceAllUsesWith(res.as_ptr(ctx));
        return;
    }

    // For the measured curve BSDFs, the arrays are expected to contain around 50 colors.
    // So always converting all values to spectral_sample is not feasible.
    // Instead, we use an array index accessor function to calculate and
    // convert the values on demand.
    if (m_enable_libbsdf_spectral && (
            sema == IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR ||
            sema == IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER ||
            sema == IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER)) {
        // get array index accessor function for spectral sample array
        llvm::Function *array_index_accessor = get_measured_curve_array_index_accessor(arg);

        // rewrite all usages of the spectral sample array
        //  - curve_values[i] -> array_index_accessor(i)
        //  - curve_values[i].values[j] -> tmp = array_index_accessor(i); tmp.values[j]

        // Collect GEP users per block and sort each block's GEPs by instruction order
        // (top-to-bottom) so that the array_index_accessor call is inserted before the
        // first GEP for each (block, index). comesBefore() is only valid within the same block.
        llvm::DenseMap<llvm::BasicBlock *, llvm::SmallVector<llvm::GetElementPtrInst *, 8>> geps_per_block;
        for (auto user : inst->users()) {
            if (llvm::GetElementPtrInst *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(user)) {
                geps_per_block[gep->getParent()].push_back(gep);
            }
        }
        for (auto &kv : geps_per_block) {
            llvm::SmallVectorImpl<llvm::GetElementPtrInst *> &geps = kv.second;
            std::sort(geps.begin(), geps.end(),
                [](llvm::GetElementPtrInst *a, llvm::GetElementPtrInst *b) {
                    return a->comesBefore(b);
                });
        }

        llvm::Value *last_curve_value = nullptr;
        llvm::Value *last_curve_value_index = nullptr;
        llvm::BasicBlock *last_curve_value_bb = nullptr;
        for (auto &kv : geps_per_block) {
            for (llvm::GetElementPtrInst *gep : kv.second) {
                // curve_values[i] -> array_index_accessor(i)
                // %206 = getelementptr inbounds %struct.Spectral_sample_struct,
                //    %struct.Spectral_sample_struct* %bsdf_arg, i64 %i
                if (gep->getNumOperands() == 2) {
                    llvm::Value *llvm_args[2];
                    llvm_args[0] = ctx.has_exec_ctx_parameter()
                        ? ctx.get_exec_ctx_parameter() : ctx.get_state_parameter();
                    llvm_args[1] = gep->getOperand(1);
                    llvm::CallInst *call = llvm::CallInst::Create(
                        array_index_accessor, llvm_args, "", gep);
                    call->setDebugLoc(gep->getDebugLoc());
                    gep->replaceAllUsesWith(call);
                    delete_list.push_back(gep);
                    continue;
                }

                // curve_values[i].values[j] -> tmp = array_index_accessor(i); tmp.values[j]
                // %206 = getelementptr inbounds %struct.Spectral_sample_struct,
                //    %struct.Spectral_sample_struct* %bsdf_arg, i64 %i, i32 0, i64 j
                if (gep->getNumOperands() == 4) {
                    llvm::Value *curve_value_index = gep->getOperand(1);
                    MDL_ASSERT(llvm::dyn_cast<llvm::ConstantInt>(gep->getOperand(2))->isNullValue());
                    llvm::Value *sample_index = gep->getOperand(3);

                    // skip sext from i32 to i64 if present
                    // %218 = sext i32 %201 to i64
                    if (llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(curve_value_index)) {
                        curve_value_index = sext->getOperand(0);
                    }

                    // did we already get this array index?
                    if (curve_value_index != last_curve_value_index ||
                            gep->getParent() != last_curve_value_bb) {
                        // no, we need to get the value for this index

                        if (last_curve_value == nullptr) {
                            last_curve_value = ctx.create_local(
                                m_type_mapper.get_spectral_sample_type(), "curve_value");
                        }

                        llvm::Value *llvm_args[3];
                        llvm_args[0] = last_curve_value;
                        llvm_args[1] = ctx.has_exec_ctx_parameter()
                            ? ctx.get_exec_ctx_parameter() : ctx.get_state_parameter();
                        llvm_args[2] = curve_value_index;
                        llvm::CallInst *call = llvm::CallInst::Create(
                            array_index_accessor, llvm_args, "", gep);
                        call->setDebugLoc(gep->getDebugLoc());

                        last_curve_value_index = curve_value_index;
                        last_curve_value_bb = gep->getParent();
                    }

                    // extract the value for this index from the last_curve_value
                    llvm::ConstantInt *zero = ctx.get_constant(int(0));
                    llvm::GetElementPtrInst *new_gep = llvm::GetElementPtrInst::Create(
                        nullptr, last_curve_value, { zero, zero, sample_index }, "", gep);
                    new_gep->setDebugLoc(gep->getDebugLoc());
                    gep->replaceAllUsesWith(new_gep);
                    delete_list.push_back(gep);
                    continue;
                }
                MDL_ASSERT(!"Unsupported usage of spectral sample array");
                continue;
            }
        }
        return;
    }

    // special handling for constant array parameters
    // TODO SPECTRAL: For spectral, no globals can be generated for colors, as they depend on the
    //    sampled spectral wavelengths. So the DAG rebuilder should already insert
    //    the upsampling calls
    if (arg->get_kind() == DAG_node::EK_CONSTANT) {
        DAG_constant const *arg_const = mi::mdl::cast<DAG_constant>(arg);
        mi::mdl::IValue const *arg_val = arg_const->get_value();
        MDL_ASSERT(arg_val->get_kind() == mi::mdl::IValue::VK_ARRAY);
        mi::mdl::IValue_array const *arg_array = mi::mdl::cast<mi::mdl::IValue_array>(arg_val);
        int elem_count = arg_array->get_component_count();

        // is it a float3 array? (this should be a RGB color array)
        // TODO SPECTRAL: This is probably for measured_curve_factor with constant curve_value array
        //   The BSDF expects spectral_sample array...
        if (elem_type == m_float3_struct_type) {
            MDL_ASSERT(!m_enable_libbsdf_spectral &&
                "For spectral, colors should be upsampled and thus not be constant anymore");

            // create a global constant of float3 structs with the corresponding color values
            llvm::ArrayType *color_array_type =
                llvm::ArrayType::get(m_float3_struct_type, elem_count);
            llvm::SmallVector<llvm::Constant *, 8> elems(elem_count);
            for (int i = 0; i < elem_count; ++i) {
                MDL_ASSERT(arg_array->get_value(i)->get_kind() == mi::mdl::IValue::VK_RGB_COLOR);
                mi::mdl::IValue_compound const *color =
                    mi::mdl::cast<mi::mdl::IValue_compound>(arg_array->get_value(i));
                llvm::Constant *color_vals[3];
                for (int j = 0; j < 3; ++j) {
                    color_vals[j] = llvm::cast<llvm::Constant>(
                        ctx.get_constant(color->get_value(j)));
                }
                elems[i] = llvm::ConstantStruct::get(m_float3_struct_type, color_vals);
            }

            llvm::Constant *array = llvm::ConstantArray::get(color_array_type, elems);
            llvm::Value *cv = new llvm::GlobalVariable(
                *m_module,
                color_array_type,
                /*isConstant=*/ true,
                llvm::GlobalValue::InternalLinkage,
                array,
                "_global_libbsdf_const");
            llvm::Value *casted_val = ctx->CreateBitCast(cv, m_float3_struct_type->getPointerTo());
            inst->replaceAllUsesWith(casted_val);
            return;
        }

        // is it a constant BSDF_component array?
        bool color_df_component = false;
        if (elem_type->isStructTy() && !llvm::cast<llvm::StructType>(elem_type)->isLiteral() && (
                elem_type->getStructName() == "struct.BSDF_component" ||
                elem_type->getStructName() == "struct.EDF_component" ||
                (color_df_component = (
                    elem_type->getStructName() == "struct.color_BSDF_component" ||
                    elem_type->getStructName() == "struct.color_EDF_component")))) {

            MDL_ASSERT((!color_df_component || !m_enable_libbsdf_spectral) &&
                "For spectral, colors should be upsampled and thus not be constant anymore");

            llvm::Type *weight_type = color_df_component ?
                m_float3_struct_type : m_type_mapper.get_float_type();

            // create a global constant weight array
            llvm::SmallVector<llvm::Constant *, 8> elems(elem_count);
            for (int i = 0; i < elem_count; ++i) {
                MDL_ASSERT(arg_array->get_value(i)->get_kind() == mi::mdl::IValue::VK_STRUCT);
                mi::mdl::IValue_struct const *comp_val =
                    mi::mdl::cast<mi::mdl::IValue_struct>(arg_array->get_value(i));
                if (color_df_component) {
                    MDL_ASSERT(
                        comp_val->get_field("weight")->get_kind() == mi::mdl::IValue::VK_RGB_COLOR);
                    mi::mdl::IValue_rgb_color const *weight_val =
                        mi::mdl::cast<mi::mdl::IValue_rgb_color>(comp_val->get_field("weight"));
                    llvm::Constant *color_vals[3];
                    for (int j = 0; j < 3; ++j) {
                        color_vals[j] = llvm::cast<llvm::Constant>(
                            ctx.get_constant(weight_val->get_value(j)));
                    }
                    elems[i] = llvm::ConstantStruct::get(m_float3_struct_type, color_vals);
                } else {
                    MDL_ASSERT(
                        comp_val->get_field("weight")->get_kind() == mi::mdl::IValue::VK_FLOAT);
                    mi::mdl::IValue_float const *weight_val =
                        mi::mdl::cast<mi::mdl::IValue_float>(comp_val->get_field("weight"));
                    elems[i] = llvm::ConstantFP::get(
                        m_llvm_context, llvm::APFloat(weight_val->get_value()));
                }
            }

            llvm::ArrayType *weight_array_type = llvm::ArrayType::get(weight_type, elem_count);
            llvm::Constant *array = llvm::ConstantArray::get(weight_array_type, elems);
            llvm::Value *weight_array_global = new llvm::GlobalVariable(
                *m_module,
                weight_array_type,
                /*isConstant=*/ true,
                llvm::GlobalValue::InternalLinkage,
                array,
                "_global_libbsdf_weights_const");

            llvm::Value *df_flags_array_global = nullptr;
            if (m_libbsdf_flags_in_bsdf_data) {
                llvm::ArrayType *df_array_type =
                    llvm::ArrayType::get(m_type_mapper.get_int_type(), elem_count);
                df_flags_array_global = new llvm::GlobalVariable(
                    *m_module,
                    weight_array_type,
                    /*isConstant=*/ true,
                    llvm::GlobalValue::InternalLinkage,
                    llvm::ConstantAggregateZero::get(df_array_type),
                    "_global_libbsdf_df_flags_const");
            }

            const IType_array* array_type = as<IType_array>(arg->get_type());
            const IType_struct* element_type = as<IType_struct>(array_type->get_element_type());
            IType::Kind df_kind = element_type->get_compound_type(1)->get_kind();

            // only "xdf()" can be part of a constant, so use an empty component info
            Df_component_info comp_info(*this, df_kind);

            // rewrite all usages of the components variable
            rewrite_df_component_usages(
                inst,
                weight_array_global,
                df_flags_array_global,
                comp_info,
                delete_list);
            return;
        }

        MDL_ASSERT(!"Unsupported constant array parameter type");
        return;
    }

    IType_array const *arg_type = mi::mdl::cast<IType_array>(arg->get_type());
    MDL_ASSERT(arg_type->is_immediate_sized() && "array type must be instantiated");
    int elem_count = arg_type->get_size();

    // is it a spectral_sample array? (this should be a color array)
    if (elem_type == m_type_mapper.get_spectral_sample_type()) {
        llvm::ArrayType *color_array_type = llvm::ArrayType::get(
            m_type_mapper.get_spectral_sample_type(), elem_count);

        Expression_result array_res = translate_call_arg(arg, color_array_type);

        llvm::Value *color_array = array_res.as_ptr(ctx);
        llvm::Value *casted_array = ctx->CreateBitCast(
            color_array, m_type_mapper.get_spectral_sample_ptr_type());
        inst->replaceAllUsesWith(casted_array);
        return;
    }

    // is it a non-constant BSDF_component array?
    bool color_df_component = false;
    if (elem_type->isStructTy() && !llvm::cast<llvm::StructType>(elem_type)->isLiteral() && (
            elem_type->getStructName() == "struct.BSDF_component" ||
            elem_type->getStructName() == "struct.EDF_component" ||
            (color_df_component = (
                elem_type->getStructName() == "struct.color_BSDF_component" ||
                elem_type->getStructName() == "struct.color_EDF_component")))) {

        llvm::Type *weight_type = color_df_component ?
            m_type_mapper.get_spectral_sample_type() : m_type_mapper.get_float_type();

        // create local weight and Df_flags array and instantiate all BSDF components
        llvm::ArrayType *weight_array_type = llvm::ArrayType::get(weight_type, elem_count);
        llvm::Value *weight_array = ctx.create_local(weight_array_type, "weights");

        llvm::Value *df_flags_array = nullptr;
        if (m_libbsdf_flags_in_bsdf_data) {
            llvm::ArrayType *df_flags_array_type =
                llvm::ArrayType::get(m_type_mapper.get_int_type(), elem_count);
            df_flags_array = ctx.create_local(df_flags_array_type, "df_flags");
        }

        // get df kind
        IType_array const  *array_type   = cast<IType_array>(arg->get_type());
        IType_struct const *element_type = cast<IType_struct>(array_type->get_element_type());
        IType::Kind df_kind = element_type->get_compound_type(1)->get_kind();

        Df_component_info comp_info(*this, df_kind);

        DAG_call const *arg_call = mi::mdl::cast<DAG_call>(arg);

        for (int i = 0; i < elem_count; ++i) {
            DAG_node const *elem_node = arg_call->get_argument(i);

            Expression_result weight_res;
            llvm::Value *df_flags_val = nullptr;

            // is the i-th element a BSDF_component constant?
            if (elem_node->get_kind() == DAG_node::EK_CONSTANT) {
                DAG_constant const *constant = mi::mdl::cast<DAG_constant>(elem_node);
                mi::mdl::IValue_struct const *value =
                    mi::mdl::cast<IValue_struct>(constant->get_value());
                mi::mdl::IValue const *weight_val = value->get_field("weight");
                weight_res = translate_value(weight_val);

                if (weight_res.get_value_type() != weight_type) {
                    weight_res = Expression_result::value(
                        ctx.load_and_convert(weight_type, weight_res.as_ptr(ctx)));
                }

                // only "bsdf()" can be part of a constant
                MDL_ASSERT(value->get_field("component")->get_kind() ==
                    mi::mdl::IValue::VK_INVALID_REF);
                comp_info.add_component_df(elem_node);

                if (m_libbsdf_flags_in_bsdf_data) {
                    df_flags_val = ctx.get_constant(int(DF_FLAGS_NONE));
                }
            } else {
                // should be a BSDF_component constructor call
                MDL_ASSERT(elem_node->get_kind() == DAG_node::EK_CALL);
                DAG_call const *elem_call = mi::mdl::cast<DAG_call>(elem_node);
                DAG_node const *weight_node = elem_call->get_argument("weight");
                weight_res = translate_call_arg(weight_node, weight_type);

                // instantiate BSDF for component parameter of the constructor
                DAG_node const *component_node = elem_call->get_argument("component");
                comp_info.add_component_df(component_node);

                if (m_libbsdf_flags_in_bsdf_data) {
                    df_flags_val = ctx.get_constant(
                        int(get_bsdf_scatter_components(component_node)));
                }
            }

            // store results in arrays
            ctx->CreateStore(weight_res.as_value(ctx),
                ctx.create_simple_gep_in_bounds(weight_array, unsigned(i)));
            if (m_libbsdf_flags_in_bsdf_data) {
                ctx->CreateStore(df_flags_val,
                    ctx.create_simple_gep_in_bounds(df_flags_array, unsigned(i)));
            }
        }

        // rewrite all usages of the components variable
        rewrite_df_component_usages(
            inst,
            weight_array,
            df_flags_array,
            comp_info,
            delete_list);
        return;
    }

    MDL_ASSERT(!"Unsupported array parameter type");
}

// Returns the base BSDF of the given node, if the node is a factor BSDF, otherwise NULL.
DAG_node const *LLVM_code_generator::get_factor_base_bsdf(DAG_node const *node)
{
    DAG_call const *call = as<DAG_call>(node);
    if (call == NULL) {
        return NULL;
    }

    // return the base BSDF for factor BSDFs
    switch (call->get_semantic()) {
    case IDefinition::DS_INTRINSIC_DF_TINT:
    case IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR:
    case IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_FACTOR:
    case IDefinition::DS_INTRINSIC_DF_COAT_ABSORPTION_FACTOR:
        {
            DAG_node const *base = call->get_argument("base");
            MDL_ASSERT(base != NULL && "base parameter missing for factor BSDF");
            return base;
        }

    default:
        return NULL;
    }
}

/// Returns the common node, if both nodes are either the common node or a factor
/// BSDF of the common node, otherwise NULL.
DAG_node const *LLVM_code_generator::matches_factor_pattern(
    DAG_node const *left,
    DAG_node const *right)
{
    DAG_node const *left_factor_base  = get_factor_base_bsdf(left);
    DAG_node const *right_factor_base = get_factor_base_bsdf(right);

    if (left_factor_base != NULL) {
        // factor(bsdf), bsdf
        // factor_1(bsdf), factor_2(bsdf)
        if (left_factor_base == right || left_factor_base == right_factor_base) {
            return left_factor_base;
        }
    }
    if (right_factor_base != NULL) {
        // bsdf, factor(bsdf)
        if (left == right_factor_base) {
            return right_factor_base;
        }
    }
    return NULL;
}

/// Recursively instantiate a ternary operator of type BSDF.
llvm::Function *LLVM_code_generator::instantiate_ternary_df(
    Function_context &caller_ctx,
    DAG_call const *dag_call)
{
    // optimize thin_film special case:
    //      cond ? thin_film(ior, thickness, base) : base
    //   -> thin_film(ior, cond ? thickness : 0, base)
    //
    //      cond ? base : thin_film(ior, thickness, base)
    //   -> thin_film(ior, cond ? 0 : thickness, base)
    if (is<IExpression::OK_TERNARY>(dag_call) && is<IType_bsdf>(dag_call->get_type())) {
        DAG_call const *true_node  = as<DAG_call>(dag_call->get_argument(1));
        DAG_call const *false_node = as<DAG_call>(dag_call->get_argument(2));
        if (true_node && false_node) {
            if (is<IDefinition::DS_INTRINSIC_DF_THIN_FILM>(true_node)) {
                DAG_node const *base = true_node->get_argument("base");
                if (false_node == base) {
                    return instantiate_df(
                        caller_ctx,
                        true_node,
                        Instantiate_opt_context::opt_ternary_thin_film(
                            dag_call->get_argument(0),
                            /*thin_film_if_true=*/ true));
                }
            }
            if (is<IDefinition::DS_INTRINSIC_DF_THIN_FILM>(false_node)) {
                DAG_node const *base = true_node->get_argument("base");
                if (true_node == base) {
                    return instantiate_df(
                        caller_ctx,
                        false_node,
                        Instantiate_opt_context::opt_ternary_thin_film(
                            dag_call->get_argument(0),
                            /*thin_film_if_true=*/ false));
                }
            }
        }
    }

    // create a new function with the type for current distribution function state
    IType::Kind kind = dag_call->get_type()->get_kind();
    llvm::FunctionType *func_type;
    char const *operator_name = NULL;
    switch (kind) {
    case IType::Kind::TK_BSDF:
    case IType::Kind::TK_HAIR_BSDF:
        {
            switch (m_dist_func_state) {
            case DFSTATE_SAMPLE:    func_type = m_type_bsdf_sample_func; break;
            case DFSTATE_EVALUATE:  func_type = m_type_bsdf_evaluate_func; break;
            case DFSTATE_PDF:       func_type = m_type_bsdf_pdf_func; break;
            case DFSTATE_AUXILIARY: func_type = m_type_bsdf_auxiliary_func; break;
            default:
                MDL_ASSERT(!"Invalid bsdf distribution function state");
                return NULL;
            }
            if (kind == IType::Kind::TK_HAIR_BSDF) {
                operator_name = "ternary_hair_bsdf";
            } else {
                operator_name = "ternary_bsdf";
            }
        }
        break;

    case IType::Kind::TK_EDF:
        {
            switch (m_dist_func_state) {
            case DFSTATE_SAMPLE:    func_type = m_type_edf_sample_func; break;
            case DFSTATE_EVALUATE:  func_type = m_type_edf_evaluate_func; break;
            case DFSTATE_PDF:       func_type = m_type_edf_pdf_func; break;
            case DFSTATE_AUXILIARY: func_type = m_type_edf_auxiliary_func; break;
            default:
                MDL_ASSERT(!"Invalid edf distribution function state");
                return NULL;
            }
            operator_name = "ternary_edf";
        }
        break;

    default:
        MDL_ASSERT(!"Invalid distribution kind");
        return NULL;
    }

    llvm::Function *func = llvm::Function::Create(
        func_type,
        llvm::GlobalValue::InternalLinkage,
        operator_name,
        m_module);
    m_state_usage_analysis.register_function(func);
    set_llvm_function_attributes(func, /*mark_noinline=*/false);
    BB_store func_chain(m_curr_bb, get_next_bb());

    if (m_di_builder) {
        llvm::DIFile *di_file = m_di_builder->createFile("<generated>", "");

        llvm::DISubprogram *di_func = m_di_builder->createFunction(
            /*Scope=*/ di_file,
            /*Name=*/ operator_name,
            /*LinkageName=*/ operator_name,
            /*File=*/ di_file,
            1,
            m_type_mapper.get_debug_info_type(m_di_builder, di_file, func_type),
            1,
            llvm::DINode::FlagPrototyped,
            llvm::DISubprogram::toSPFlags(
                /*IsLocalToUnit=*/true,
                /*IsDefinition=*/true,
                /*IsOptimized=*/is_optimized()
            ));
        func->setSubprogram(di_func);
    }

    {
        // context needs a non-empty start block, so create a jump to a second block
        llvm::BasicBlock *start_bb = llvm::BasicBlock::Create(m_llvm_context, "start", func);
        llvm::BasicBlock *body_bb  = llvm::BasicBlock::Create(m_llvm_context, "body", func);
        start_bb->getInstList().push_back(llvm::BranchInst::Create(body_bb));

        Function_context ctx(
            get_allocator(),
            *this,
            func,
            get_df_function_flags(func),
            true);

        ctx->SetInsertPoint(body_bb);

        // generate code for the condition
        DAG_node const *cond = dag_call->get_argument(0);
        Expression_result res = translate_call_arg(cond, m_type_mapper.get_bool_type());

        // check for factor pattern
        DAG_node const *true_bsdf   = dag_call->get_argument(1);
        DAG_node const *false_bsdf  = dag_call->get_argument(2);
        DAG_node const *common_node = NULL;
        if (m_dist_func_state == DFSTATE_SAMPLE || m_dist_func_state == DFSTATE_EVALUATE) {
            common_node = matches_factor_pattern(true_bsdf, false_bsdf);
        }

        // collect function parameters
        llvm::Value *res_pointer = func->arg_begin();
        llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();
        llvm::Value *inherited_normal = arg_it;
        llvm::Value *inherited_weight = NULL;
        if (m_dist_func_state == DFSTATE_EVALUATE || m_dist_func_state == DFSTATE_AUXILIARY) {
            inherited_weight = ++arg_it;
        }

        // read inherited_weight here already, if we need it later
        llvm::Value *inherited_weight_val = NULL;
        if (common_node != NULL && inherited_weight != NULL) {
            inherited_weight_val = ctx->CreateLoad(inherited_weight);
        }

        // generate "if(cond)-then-else; return"

        llvm::BasicBlock *cond_true_bb  = ctx.create_bb("cond_true");
        llvm::BasicBlock *cond_false_bb = ctx.create_bb("cond_false");
        llvm::BasicBlock *end_bb        = ctx.create_bb("end");

        llvm::Value *cond_res  = res.as_value(ctx);
        llvm::Value *cond_bool = ctx->CreateICmpNE(
            cond_res,
            llvm::Constant::getNullValue(cond_res->getType()));
        llvm::Instruction *branch = ctx->CreateCondBr(cond_bool, cond_true_bb, cond_false_bb);

        ctx->SetInsertPoint(cond_true_bb);
        llvm::Instruction *true_term = ctx->CreateBr(end_bb);

        ctx->SetInsertPoint(cond_false_bb);
        llvm::Instruction *false_term = ctx->CreateBr(end_bb);

        ctx->SetInsertPoint(end_bb);
        llvm::Instruction *end_term = ctx->CreateRetVoid();

        // for sample with factor pattern, execute sample on common node before branch
        if (common_node != NULL && m_dist_func_state == DFSTATE_SAMPLE) {
            // sample(cond ? factor(common_node) : factor_2(common_node))
            // -> sample(common_node)
            //    if (cond) sample(factor(null))
            //    else sample(factor_2(null)

            // execute common sample code
            instantiate_and_call_df(
                ctx,
                common_node,
                m_dist_func_state,
                res_pointer,
                inherited_normal,
                NULL,
                branch);

            // handle true_bsdf, if it is a factor BSDF
            if (true_bsdf != common_node) {
                BB_store true_chain(m_curr_bb, get_next_bb());
                instantiate_and_call_df(
                    ctx,
                    true_bsdf,
                    m_dist_func_state,
                    res_pointer,
                    inherited_normal,
                    NULL,
                    true_term,
                    Instantiate_opt_context::skip_bsdf_call_ctx());
            }

            // handle false_bsdf, if it is a factor BSDF
            if (false_bsdf != common_node) {
                BB_store false_chain(m_curr_bb, get_next_bb());
                instantiate_and_call_df(
                    ctx,
                    false_bsdf,
                    m_dist_func_state,
                    res_pointer,
                    inherited_normal,
                    NULL,
                    false_term,
                    Instantiate_opt_context::skip_bsdf_call_ctx());
            }
        }
        // for evaluate with factor pattern, execute evaluate on common node after the if
        else if (common_node && m_dist_func_state == DFSTATE_EVALUATE) {
            // evaluate(cond ? factor(common_node) : factor_2(common_node), inherited_weight)
            // -> if (cond) factor_val = evaluate(factor(null, inherited_weight))
            //    else      factor_val = evaluate(factor_2(null, inherited_weight))
            //    evaluate(common_node, factor_val * inherited_weight)

            MDL_ASSERT(inherited_weight != NULL);
            MDL_ASSERT(inherited_weight_val != NULL);

            // handle true_bsdf, if it is a factor BSDF
            //   cond ? factor(common_node) : common_node
            //   cond ? factor(common_node) : factor_2(common_node)
            llvm::Value *true_inherited_weight_val = inherited_weight_val;
            if (true_bsdf != common_node) {
                BB_store true_chain(m_curr_bb, get_next_bb());
                llvm::Value *factor_val = instantiate_and_call_df(
                    ctx,
                    true_bsdf,
                    DFSTATE_GET_FACTOR,
                    res_pointer,
                    inherited_normal,
                    NULL,
                    true_term,
                    Instantiate_opt_context::skip_bsdf_call_ctx());

                // true_inherited_weight_val = factor_val * inherited_weight_val
                ctx->SetInsertPoint(true_term);
                true_inherited_weight_val = ctx.create_mul(
                    inherited_weight_val->getType(),
                    factor_val,
                    inherited_weight_val);
            }

            // handle false_bsdf, if it is a factor BSDF
            //   cond ? common_node : factor(common_node)
            //   cond ? factor(common_node) : factor_2(common_node)
            llvm::Value *false_inherited_weight_val = inherited_weight_val;
            if (false_bsdf != common_node) {
                BB_store false_chain(m_curr_bb, get_next_bb());
                llvm::Value *factor = instantiate_and_call_df(
                    ctx,
                    false_bsdf,
                    DFSTATE_GET_FACTOR,
                    res_pointer,
                    inherited_normal,
                    NULL,
                    false_term,
                    Instantiate_opt_context::skip_bsdf_call_ctx());

                // false_inherited_weight_val = factor_val * inherited_weight_val
                ctx->SetInsertPoint(false_term);

                false_inherited_weight_val = ctx.create_mul(
                    inherited_weight_val->getType(),
                    factor,
                    inherited_weight_val);
            }

            // execute common code in end block
            BB_store common_chain(m_curr_bb, get_next_bb());
            ctx->SetInsertPoint(end_term);

            // selected new inherited weight according to predecessor
            llvm::PHINode *phi = ctx->CreatePHI(inherited_weight_val->getType(), 2);
            phi->addIncoming(true_inherited_weight_val, cond_true_bb);
            phi->addIncoming(false_inherited_weight_val, cond_false_bb);

            // store it in a new local variable
            llvm::Value *new_inherited_weight =
                ctx.create_local(
                    inherited_weight->getType()->getPointerElementType(), "new_inherited_weight");
            ctx->CreateStore(phi, new_inherited_weight);

            // TODO: check for zero weight?
            //   (fresnel_factor, directional_factor, measured_curve absorbs in this case)

            // call common code with new inherited weight
            instantiate_and_call_df(
                ctx,
                common_node,
                m_dist_func_state,
                res_pointer,
                inherited_normal,
                new_inherited_weight,
                end_term);
        } else {
            // True case
            {
                BB_store true_chain(m_curr_bb, get_next_bb());
                instantiate_and_call_df(
                    ctx,
                    true_bsdf,
                    m_dist_func_state,
                    res_pointer,
                    inherited_normal,
                    inherited_weight,
                    true_term);
            }

            // False case
            {
                BB_store false_chain(m_curr_bb, get_next_bb());
                instantiate_and_call_df(
                    ctx,
                    false_bsdf,
                    m_dist_func_state,
                    res_pointer,
                    inherited_normal,
                    inherited_weight,
                    false_term);
            }
        }
    }

    // return the now finalized function
    return func;
}

// Returns true, if the given DAG node is a call to diffuse_reflection_bsdf(color(1), 0, color(0)).
bool LLVM_code_generator::is_default_diffuse_reflection(DAG_node const *node)
{
    // match diffuse_reflection_bsdf(*)
    DAG_call const *dag_call = as<DAG_call>(node);
    if (dag_call == NULL || !is<IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF>(dag_call)) {
        return false;
    }

    // match tint argument as color(1.0f)
    DAG_constant const *tint_arg = as<DAG_constant>(dag_call->get_argument(0));
    CHECK_PARAM_NAME(dag_call, 0, "tint");
    if (tint_arg == NULL) {
        return false;
    }

    IValue_rgb_color const *tint_value = as<IValue_rgb_color>(tint_arg->get_value());
    if (tint_value == NULL || !tint_value->is_one()) {
        return false;
    }

    // match roughness argument as 0.0f
    DAG_constant const *roughness_arg = as<DAG_constant>(dag_call->get_argument(1));
    CHECK_PARAM_NAME(dag_call, 1, "roughness");
    if (roughness_arg == NULL) {
        return false;
    }

    IValue_float const *roughness_value = as<IValue_float>(roughness_arg->get_value());
    if (roughness_value == NULL || !roughness_value->is_zero()) {
        return false;
    }

    // match multiscatter_tint argument as color(0.0f)
    DAG_constant const *multiscatter_arg = as<DAG_constant>(dag_call->get_argument(2));
    CHECK_PARAM_NAME(dag_call, 2, "multiscatter_tint");
    if (multiscatter_arg == NULL) {
        return false;
    }

    IValue_rgb_color const *multiscatter_value =
        as<IValue_rgb_color>(multiscatter_arg->get_value());
    if (multiscatter_value == NULL || !multiscatter_value->is_zero()) {
        return false;
    }

    // ignore handle argument

    // successfully matched
    return true;
}

// Returns the scatter components the given DAG node can return.
Df_flags LLVM_code_generator::get_bsdf_scatter_components(
    DAG_node const *node)
{
    DAG_call const *dag_call = as<DAG_call>(node);
    if (dag_call == NULL) {
        return DF_FLAGS_NONE;
    }

    Scatter_components_map::const_iterator it = m_scatter_components_map.find(dag_call);
    if (it != m_scatter_components_map.end()) {
        return it->second;
    }

    Df_flags res = DF_FLAGS_NONE;
    DAG_node const *scatter_mode_arg = nullptr;

    switch (unsigned(dag_call->get_semantic())) {
    case IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF:
    case IDefinition::DS_INTRINSIC_DF_DUSTY_DIFFUSE_REFLECTION_BSDF:
    case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
    case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
        res = DF_FLAGS_ALLOW_REFLECT;
        break;

    case IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF:
        res = DF_FLAGS_ALLOW_TRANSMIT;
        break;

    case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF:
    case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
        scatter_mode_arg = dag_call->get_argument("mode");
        MDL_ASSERT(scatter_mode_arg != nullptr && "mode parameter missing for BSDF");
        break;

    case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFLAKE_SHEEN_BSDF:
        {
            DAG_node const *multiscatter = dag_call->get_argument("multiscatter");
            MDL_ASSERT(multiscatter != nullptr && "multiscatter parameter missing for sheen BSDF");
            Df_flags multiscatter_comps = get_bsdf_scatter_components(multiscatter);
            res = Df_flags(int(DF_FLAGS_ALLOW_REFLECT) | multiscatter_comps);
            break;
        }

    case IDefinition::DS_INTRINSIC_DF_TINT:
    case IDefinition::DS_INTRINSIC_DF_THIN_FILM:
    case IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR:
    case IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_FACTOR:
    case IDefinition::DS_INTRINSIC_DF_COAT_ABSORPTION_FACTOR:
        {
            DAG_node const *base = dag_call->get_argument("base");
            MDL_ASSERT(base != nullptr && "base parameter missing for factor BSDF");
            res = get_bsdf_scatter_components(base);
        }
        break;

    case IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER:
    case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
    case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER:
    case IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER:
    case IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER:
    case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
    case IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER:
        {
            DAG_node const *layer = dag_call->get_argument("layer");
            MDL_ASSERT(layer != nullptr && "layer parameter missing for layer BSDF");
            DAG_node const *base = dag_call->get_argument("base");
            MDL_ASSERT(base != nullptr && "base parameter missing for layer BSDF");

            Df_flags layer_comps = get_bsdf_scatter_components(layer);
            Df_flags base_comps  = get_bsdf_scatter_components(base);
            res = Df_flags(int(layer_comps) | int(base_comps));
            break;
        }

    // case IDefinition::DS_INTRINSIC_DF_CHIANG_HAIR_BSDF:

    case IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX:
    case IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX:
    case IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
    case IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
    case IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX:
    case IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX:
        {
            // only argument is components array
            res = DF_FLAGS_NONE;
            DAG_call const *components_array = as<DAG_call>(dag_call->get_argument(0));
            if (components_array == nullptr) {
                MDL_ASSERT(is<DAG_constant>(dag_call->get_argument(0)) && "expected empty array");
                break;
            }
            for (int i = 0, n = components_array->get_argument_count(); i < n; ++i) {
                DAG_call const *comp_struct = as<DAG_call>(components_array->get_argument(i));
                if (comp_struct == nullptr) {
                    MDL_ASSERT(is<DAG_constant>(components_array->get_argument(i)) &&
                        "expected component with black BSDF");
                    continue;
                }
                DAG_node const *comp_bsdf = comp_struct->get_argument(1);
                Df_flags comp_res = get_bsdf_scatter_components(comp_bsdf);
                res = Df_flags(int(res) | int(comp_res));
            }
            break;
        }

    case IDefinition::Semantics(IDefinition::DS_OP_BASE + IExpression::OK_TERNARY):
        {
            Df_flags true_comps = get_bsdf_scatter_components(
                dag_call->get_argument(1));
            Df_flags false_comps = get_bsdf_scatter_components(
                dag_call->get_argument(2));
            res = Df_flags(int(true_comps) | int(false_comps));
            break;
        }

    default:
        MDL_ASSERT(!"Unexpected DAG call for get_bsdf_scatter_components");
        res = DF_FLAGS_NONE;
        break;
    }

    // if there is a scatter_mode argument, get it from BSDF node and
    // convert to Bsdf_scatter_components
    if (scatter_mode_arg != nullptr) {
        if (DAG_constant const *scatter_const = as<DAG_constant>(scatter_mode_arg)) {
            int scatter_const_int = cast<IValue_enum>(scatter_const->get_value())->get_value();
            res = Df_flags(scatter_const_int + 1);
        } else {
            // for now, if we get a material parameter or a DAG call here,
            // return a conservative answer
            res = DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
        }
    }

    m_scatter_components_map[dag_call] = res;
    return res;
}

// Instantiate a DF from the given DAG node and call the resulting function.
llvm::CallInst *LLVM_code_generator::instantiate_and_call_df(
    Function_context            &ctx,
    DAG_node const              *node,
    Distribution_function_state df_state,
    llvm::Value                 *res_pointer,
    llvm::Value                 *inherited_normal,
    llvm::Value                 *opt_inherited_weight,
    llvm::Instruction           *insertBefore,
    Instantiate_opt_context     opt_ctx)
{
    Store<Distribution_function_state> state_store(m_dist_func_state, df_state);

    llvm::Function *param_bsdf_func = instantiate_df(ctx, node, opt_ctx);

    m_state_usage_analysis.add_call(ctx.get_function(), param_bsdf_func);

    // call it with state parameters added
    llvm::SmallVector<llvm::Value *, 4> llvm_args;
    llvm_args.push_back(res_pointer);
    llvm_args.push_back(ctx.has_exec_ctx_parameter()
        ? ctx.get_exec_ctx_parameter() : ctx.get_state_parameter());
    llvm_args.push_back(inherited_normal);
    if (df_state == DFSTATE_EVALUATE || df_state == DFSTATE_AUXILIARY) {
        llvm_args.push_back(opt_inherited_weight);
    }
    llvm::CallInst *call = llvm::CallInst::Create(param_bsdf_func, llvm_args, "", insertBefore);
    ctx->SetInstDebugLocation(call);

    return call;
}

// Recursively instantiate a DF specified by the given DAG node from code in libbsdf
// according to the current distribution function state.
llvm::Function *LLVM_code_generator::instantiate_df(
    Function_context        &caller_ctx,
    DAG_node const          *node,
    Instantiate_opt_context opt_ctx)
{
    // handle ugly thin_film semantic
    DAG_call const *thin_film_node = NULL;
    DAG_call const *inner          = as<DAG_call>(node);
    DAG_node const *arg            = NULL;
    llvm::Function *df_lib_func    = NULL;

    while (inner !=  NULL && is<IDefinition::DS_INTRINSIC_DF_THIN_FILM>(inner)) {
        thin_film_node = inner;
        arg            = thin_film_node->get_argument(2);
        inner          = cast<DAG_call>(arg);

        CHECK_PARAM_NAME(thin_film_node, 2, "base");
    }

    if (thin_film_node != NULL) {
        // we found the inner thin_film() call (and skip all outer ones)
        if (is<DAG_call>(arg)) {
            // check if we have a combined implementation of thin_film and its argument
            df_lib_func = get_libbsdf_function(cast<DAG_call>(arg), "thin_film_");
        }

        if (df_lib_func == NULL) {
            // there is NO combined mode for this combination, skip thin_film() calls at all
            node           = arg;
            thin_film_node = NULL;
        }
    }

    // handle DF constant nodes (bsdf(), edf(), xdf_component(), color_xdf_component())
    if (DAG_constant const *c = as<DAG_constant>(node)) {
        IValue const *value = c->get_value();

        // check for "bsdf()" or "df::[color_]bsdf_component(weight, bsdf())" constant
        if ( (
                // "bsdf()"
                is<IValue_invalid_ref>(value) &&
                (is<IType_bsdf>(value->get_type()) || is<IType_hair_bsdf>(value->get_type()))
            ) || (
                // "df::bsdf_component(weight, bsdf())" / "df::color_bsdf_component(weight, bsdf())"
                is<IValue_struct>(value) &&
                (
                    strcmp(cast<IValue_struct>(value)->get_type()->get_symbol()->get_name(),
                        "::df::bsdf_component") == 0
                ||
                    strcmp(cast<IValue_struct>(value)->get_type()->get_symbol()->get_name(),
                        "::df::color_bsdf_component") == 0
                )
            ) )
        {
            mi::mdl::string func_name("gen_black_bsdf", get_allocator());
            func_name.append(get_dist_func_state_suffix());
            df_lib_func = m_module->getFunction(func_name.c_str());
            if (df_lib_func == NULL) {
                MDL_ASSERT(!"libbsdf is missing an implementation of bsdf(): black_bsdf_*");
                return NULL;
            }
            return df_lib_func;   // the black_bsdf needs no instantiation, return it directly
        }

        // check for "edf()" or "df::[color_]edf_component(weight, edf())" constant
        if ( (
                // "edf()"
                is<IValue_invalid_ref>(value) && is<IType_edf>(value->get_type())
            ) || (
                // "df::edf_component(weight, edf())" / "df::color_edf_component(weight, edf())"
                is<IValue_struct>(value) &&
                (
                    strcmp(cast<IValue_struct>(value)->get_type()->get_symbol()->get_name(),
                        "::df::edf_component") == 0
                ||
                    strcmp(cast<IValue_struct>(value)->get_type()->get_symbol()->get_name(),
                        "::df::color_edf_component") == 0
                )
            ) )
        {
            mi::mdl::string func_name("gen_black_edf", get_allocator());
            func_name.append(get_dist_func_state_suffix());
            df_lib_func = m_module->getFunction(func_name.c_str());
            if (df_lib_func == NULL) {
                MDL_ASSERT(!"libbsdf is missing an implementation of edf(): black_edf_*");
                return NULL;
            }
            return df_lib_func;   // the black_edf needs no instantiation, return it directly
        }
    }

    if (!is<DAG_call>(node)) {
        MDL_ASSERT(!"Unsupported DAG node");
        return NULL;
    }

    llvm::OptimizationRemarkEmitter ORE(caller_ctx.get_function());

    DAG_call const *dag_call =
        cast<DAG_call>(thin_film_node != NULL ? thin_film_node->get_argument(2) : node);

    // check if we already created code for this node and state
    Instantiated_df instantiated_df(
        thin_film_node != NULL ? cast<DAG_call>(thin_film_node) : dag_call,
        opt_ctx);

    Instantiated_dfs::const_iterator it =
        m_instantiated_dfs[m_dist_func_state].find(instantiated_df);
    if (it != m_instantiated_dfs[m_dist_func_state].end()) {
        ORE.emit([&]() {
            return llvm::OptimizationRemark(DEBUG_TYPE, "NoInstNeeded", it->second)
                << "BSDF " << dag_call->get_name() << " already instantiated: "
                << it->second->getName();
        });
        return it->second;
    }

    IDefinition::Semantics sema = dag_call->get_semantic();
    if (sema == operator_to_semantic(IExpression::OK_TERNARY)) {
        // handle ternary operators
        llvm::Function *res_func = instantiate_ternary_df(caller_ctx, dag_call);
        m_instantiated_dfs[m_dist_func_state][instantiated_df] = res_func;
        ORE.emit([&]() {
            return llvm::OptimizationRemark(DEBUG_TYPE, "Instantiation", res_func)
                << "BSDF " << dag_call->get_name() << " instantiated: " << res_func->getName();
        });

        return res_func;
    }

    bool is_elemental = is_elemental_df_semantics(sema);
    IType::Kind kind = dag_call->get_type()->get_kind();

    // get DF function according to semantics and current state, if we don't have one already
    // from the thin-film handling above

    if (df_lib_func == NULL) {
        // get the implementation if we do not have it already
        df_lib_func = get_libbsdf_function(dag_call, /*prefix=*/NULL);
    }

    if (df_lib_func == NULL) {
        char const *suffix;
        switch (kind) {
        case IType::Kind::TK_EDF:
            suffix = "_edf";
            break;

        case IType::Kind::TK_BSDF:
        case IType::Kind::TK_HAIR_BSDF:  // same prototype as BSDF variant
        default:
                suffix = "_bsdf";
                break;
        }

        mi::mdl::string func_name("gen_black", get_allocator());
        func_name.append(suffix);
        func_name.append(get_dist_func_state_suffix());

        df_lib_func = m_module->getFunction(func_name.c_str());
        if (df_lib_func == NULL) {
            MDL_ASSERT(!"libbsdf is missing an implementation of bsdf(): black_*");
            return NULL;
        }
        return df_lib_func;   // the black_bsdf needs no instantiation, return it directly
    }

    // clone the DF function into the current module
    llvm::ValueToValueMapTy ValueMap;
    llvm::Function *bsdf_func = llvm::CloneFunction(df_lib_func, ValueMap);
    add_generated_attributes(bsdf_func);
    if (m_enable_noinline && !is_always_inline_enabled()) {
        bsdf_func->addFnAttr(llvm::Attribute::NoInline);
    }
    m_state_usage_analysis.register_cloned_function(bsdf_func, df_lib_func);
    BB_store func_chain(m_curr_bb, get_next_bb());

    ORE.emit([&]() {
        return llvm::OptimizationRemark(DEBUG_TYPE, "Instantiation", bsdf_func)
            << "BSDF " << dag_call->get_name() << " instantiated: " << bsdf_func->getName();
    });

    Function_context ctx(
        get_allocator(),
        *this,
        bsdf_func,
        get_df_function_flags(bsdf_func),
        /*optimize_on_finalize=*/true);

    llvm::SmallVector<llvm::Instruction *, 16> delete_list;

    // Process all calls to BSDF parameter accessors.
    // We need to do this in a separate block without a terminator, because translate_node expects
    // to insert code at the end of a block.
    size_t n_args = dag_call->get_argument_count();
    ctx.move_to_body_start();
    llvm::BasicBlock *entry_bb = &bsdf_func->getEntryBlock();
    llvm::BasicBlock *args_bb = entry_bb->splitBasicBlock(ctx->GetInsertPoint(), "process_args");
    llvm::BasicBlock *after_args_bb = args_bb->splitBasicBlock(args_bb->begin(), "after_args");
    args_bb->getTerminator()->eraseFromParent();

    m_curr_bb = get_next_bb();
    ctx->SetInsertPoint(args_bb);

    // In load_and_link_libbsdf, we replaced all DF parameters by local variable placeholders.
    // Iterate over all non-DF typed DF parameter allocas and replace them with the real values.
    for (llvm::BasicBlock::iterator II = entry_bb->begin(); II != entry_bb->end(); ++II) {
        llvm::AllocaInst *inst = llvm::dyn_cast<llvm::AllocaInst>(II);
        if (inst == nullptr) {
            continue;
        }

        llvm::Type *elem_type = inst->getAllocatedType();

        // ignore BSDF and EDF struct allocas
        if (elem_type->isStructTy() &&
            !llvm::cast<llvm::StructType>(elem_type)->isLiteral() &&
            (elem_type->getStructName() == "struct.BSDF"
                || elem_type->getStructName() == "struct.EDF") ) {
            continue;
        }

        // get the DF parameter index of the alloca, if it is a DF parameter
        int param_idx = get_metadata_df_param_id(inst, kind);
        if (param_idx < 0) {
            // not a DF parameter, skip
            continue;
        }

        DAG_node const *arg = NULL;
        if (param_idx < n_args) {
            // get the parameter from the BSDF call
            arg = dag_call->get_argument(param_idx);
        } else {
            // get extra parameter from the thin_film modifier
            MDL_ASSERT(thin_film_node != NULL);
            arg = thin_film_node->get_argument(param_idx - n_args);
        }

        // special handling for array parameters
        if (is_libbsdf_array_parameter(sema, param_idx)) {
            handle_df_array_parameter(ctx, sema, inst, arg, delete_list);
            continue;
        }

        // special handling for handle parameters
        if (is_elemental && param_idx == dag_call->get_argument_count() - 1 &&
                strcmp(dag_call->get_parameter_name(param_idx), "handle") == 0) {
            MDL_ASSERT(is<DAG_constant>(arg) && "DF handle must be a constant");
            if (DAG_constant const *handle_const = as<DAG_constant>(arg)) {
                IValue const *handle_val = handle_const->get_value();
                IValue_string const *handle_str = as<IValue_string>(handle_val);
                MDL_ASSERT(handle_str && "DF handle must be string");
                if (handle_str != nullptr) {
                    char const *handle_name = handle_str->get_value();

                    // translate the handle name into a handle ID
                    int handle_id = -1;
                    for (size_t i = 0, n = m_cur_req_node->df_handles.size(); i < n; ++i) {
                        if (strcmp(handle_name, m_cur_req_node->df_handles[i]) == 0) {
                            handle_id = int(i);
                            break;
                        }
                    }

                    MDL_ASSERT(handle_id != -1 && "df handle name not registered");
                    Expression_result res = Expression_result::value(
                        ctx.get_constant(handle_id));
                    inst->replaceAllUsesWith(res.as_ptr(ctx));
                    continue;
                }
            }
        }

        // translate the argument to a value
        Expression_result res = translate_call_arg(arg, elem_type);

        // in "ternary operator with thin_film" optimization mode and current parameter
        // is the coating_thickness?
        if (opt_ctx.m_ternary_cond != NULL && param_idx == n_args) {
            // set thickness to 0, if condition says thin_film should be skipped
            Expression_result cond_res = translate_call_arg(
                opt_ctx.m_ternary_cond, m_type_mapper.get_bool_type());
            llvm::Value *cond_res_val = cond_res.as_value(ctx);
            if (cond_res_val->getType() != m_type_mapper.get_predicate_type()) {
                // map to predicate type
                cond_res_val = ctx->CreateICmpNE(cond_res_val, ctx.get_constant(false));
            }

            res = Expression_result::value(ctx->CreateSelect(
                cond_res_val,
                opt_ctx.m_thin_film_if_true ? res.as_value(ctx) : ctx.get_constant(0.f),
                opt_ctx.m_thin_film_if_true ? ctx.get_constant(0.f) : res.as_value(ctx)));
        }

        // replace all uses of the alloca with the translated value
        inst->replaceAllUsesWith(res.as_ptr(ctx));
    }

    // add jump to next block
    ctx->CreateBr(after_args_bb);

    // handle calls to DFs and special functions
    for (llvm::Function::iterator BI = bsdf_func->begin(), BE = bsdf_func->end(); BI != BE; ++BI) {
        m_curr_bb = get_next_bb();
        for (llvm::BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
            if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(II)) {
                // check for calls to DFs
                int param_idx = get_metadata_df_param_id(call, kind);
                if (param_idx >= 0) {
                    llvm::Function *called_func = call->getCalledFunction();
                    if (called_func != NULL) {
                        // check for BSDF::* functions
                        llvm::StringRef func_name = called_func->getName();
                        if (func_name.startswith("_ZN4BSDF")) {
                            Distribution_function_state new_state = DFSTATE_NONE;

                            // check for BSDF::select_sample(...)
                            if (func_name.startswith("_ZN4BSDF13select_sample")) {
                                new_state = DFSTATE_SAMPLE;
                            } else if (func_name.startswith("_ZN4BSDF10select_pdf")) {
                                // check for BSDF::select_pdf(...)
                                new_state = DFSTATE_PDF;
                            }

                            if (new_state != DFSTATE_NONE) {
                                llvm::Value *param_cond                   = call->getArgOperand(0);
                                llvm::Value *param_res_pointer            = call->getArgOperand(1);
                                llvm::Value *param_true_bsdf              = call->getArgOperand(3);
                                llvm::Value *param_true_inherited_normal  = call->getArgOperand(4);
                                llvm::Value *param_false_bsdf             = call->getArgOperand(5);
                                llvm::Value *param_false_inherited_normal = call->getArgOperand(6);

                                // get the DAG node for the true_bsdf argument
                                int true_param_idx = get_metadata_df_param_id(
                                    llvm::dyn_cast<llvm::Instruction>(param_true_bsdf), kind);
                                if (true_param_idx < 0) {
                                    continue;
                                }

                                DAG_node const *true_arg = dag_call->get_argument(true_param_idx);

                                // get the DAG node for the false_bsdf argument
                                int false_param_idx = get_metadata_df_param_id(
                                    llvm::dyn_cast<llvm::Instruction>(param_false_bsdf), kind);
                                if (false_param_idx < 0) {
                                    continue;
                                }

                                DAG_node const *false_arg = dag_call->get_argument(false_param_idx);

                                // true_bsdf and false_bsdf point to same DAG node?
                                if (true_arg == false_arg) {
                                    // instantiated df will be the same, so only one call needed
                                    //   select_sample/pdf(cond, bsdf, bsdf)
                                    //   -> sample/pdf(bsdf, normal(cond))
                                    llvm::Instruction *inherited_normal =
                                        llvm::SelectInst::Create(
                                            param_cond,
                                            param_true_inherited_normal,
                                            param_false_inherited_normal,
                                            "",
                                            call);
                                    inherited_normal->setDebugLoc(call->getDebugLoc());

                                    instantiate_and_call_df(
                                        ctx,
                                        true_arg,
                                        new_state,
                                        param_res_pointer,
                                        inherited_normal,
                                        nullptr,
                                        call);
                                } else if (DAG_node const *common_node =
                                        matches_factor_pattern(true_arg, false_arg)) {
                                    // one or both args are factor BSDFs of a common node
                                    //   select_sample/pdf(cond, factor_1(bsdf), factor_2(bsdf))
                                    //   -> sample/pdf(bsdf, normal(cond))
                                    //      if (cond)
                                    //         sample/pdf(factor_1(nullptr), normal(cond))
                                    //      else
                                    //         sample/pdf(factor_2(nullptr), normal(cond))
                                    //
                                    // the conditional sample/pdf is skipped, if the arg is not
                                    // a factor BSDF

                                    // get selected normal
                                    llvm::Instruction *inherited_normal =
                                        llvm::SelectInst::Create(
                                            param_cond,
                                            param_true_inherited_normal,
                                            param_false_inherited_normal,
                                            "",
                                            call);
                                    inherited_normal->setDebugLoc(call->getDebugLoc());

                                    // call common code only once
                                    instantiate_and_call_df(
                                        ctx,
                                        common_node,
                                        new_state,
                                        param_res_pointer,
                                        inherited_normal,
                                        nullptr,
                                        call);

                                    // call both factor code but without calling the base BSDFs
                                    llvm::Instruction *then_term;
                                    llvm::Instruction *else_term;
                                    llvm::SplitBlockAndInsertIfThenElse(
                                        param_cond,
                                        call,
                                        &then_term,
                                        &else_term);

                                    // handle true case if true_arg is a factor BSDF
                                    if (true_arg != common_node) {
                                        BB_store true_chain(m_curr_bb, get_next_bb());
                                        instantiate_and_call_df(
                                            ctx,
                                            true_arg,
                                            new_state,
                                            param_res_pointer,
                                            param_true_inherited_normal,
                                            nullptr,
                                            then_term,
                                            Instantiate_opt_context::skip_bsdf_call_ctx());
                                    }

                                    // handle false case if false_arg is a factor BSDF
                                    if (false_arg != common_node) {
                                        BB_store false_chain(m_curr_bb, get_next_bb());
                                        instantiate_and_call_df(
                                            ctx,
                                            false_arg,
                                            new_state,
                                            param_res_pointer,
                                            param_false_inherited_normal,
                                            nullptr,
                                            else_term,
                                            Instantiate_opt_context::skip_bsdf_call_ctx());
                                    }

                                    // fix iterators
                                    BI = call->getParent()->getIterator();
                                    BE = bsdf_func->end();
                                } else {
                                    // instantiated dfs will be different,
                                    // call them according to condition
                                    llvm::Instruction *then_term;
                                    llvm::Instruction *else_term;
                                    llvm::SplitBlockAndInsertIfThenElse(
                                        param_cond,
                                        call,
                                        &then_term,
                                        &else_term);

                                    // handle true case
                                    {
                                        BB_store true_chain(m_curr_bb, get_next_bb());
                                        instantiate_and_call_df(
                                            ctx,
                                            true_arg,
                                            new_state,
                                            param_res_pointer,
                                            param_true_inherited_normal,
                                            nullptr,
                                            then_term);
                                    }

                                    // handle false case
                                    {
                                        BB_store false_chain(m_curr_bb, get_next_bb());
                                        instantiate_and_call_df(
                                            ctx,
                                            false_arg,
                                            new_state,
                                            param_res_pointer,
                                            param_false_inherited_normal,
                                            nullptr,
                                            else_term);
                                    }

                                    // fix iterators
                                    BI = call->getParent()->getIterator();
                                    BE = bsdf_func->end();
                                }

                                // mark call instruction for deletion
                                delete_list.push_back(call);
                                continue;
                            }
                        }
                    }

                    // instantiate the BSDF function according to the DAG call argument
                    DAG_node const *arg = dag_call->get_argument(param_idx);
                    Libbsdf_DF_func_kind df_func_kind = get_libbsdf_df_func_kind(call);
                    llvm::Type *bool_type = llvm::IntegerType::get(m_llvm_context, 1);

                    // check for is_black() call
                    if (df_func_kind == LDFK_IS_BLACK) {
                        // replace is_black() by true, if the DAG call argument is a "*df()"
                        // constant, otherwise replace it by false
                        bool is_black = false;
                        if (is<DAG_constant>(arg)) {
                            IValue const *value = cast<DAG_constant>(arg)->get_value();
                            is_black =
                                is<IValue_invalid_ref>(value) && is<IType_df>(value->get_type());
                        }

                        call->replaceAllUsesWith(
                            llvm::ConstantInt::get(bool_type, is_black ? 1 : 0));
                    } else if (df_func_kind == LDFK_IS_DEFAULT_DIFFUSE_REFLECTION) {
                        // replace is_default_diffuse_reflection() by true, if the DAG call argument
                        // is a "diffuse_reflection_bsdf(color(1.0f), 0.0f, color(0.0f))"
                        // constant, otherwise replace it by false

                        call->replaceAllUsesWith(
                            llvm::ConstantInt::get(
                                bool_type,
                                is_default_diffuse_reflection(arg) ? 1 : 0));
                    } else if (df_func_kind == LDFK_HAS_ALLOWED_COMPONENTS) {
                        if (m_libbsdf_flags_in_bsdf_data) {
                            ctx->SetInsertPoint(call);

                            Df_flags components = get_bsdf_scatter_components(arg);
                            llvm::Value *comp_val = ctx.get_constant(int(components));
                            llvm::Value *allowed_val = call->getArgOperand(0);
                            llvm::Value *union_val = ctx->CreateAnd(comp_val, allowed_val);
                            llvm::Value *comp = ctx->CreateICmpNE(union_val, ctx.get_constant(0));
                            call->replaceAllUsesWith(comp);
                        } else {
                            // no flags available -> no restriction on allowed components
                            // only no allowed component, if the df is black
                            bool is_black = false;
                            if (is<DAG_constant>(arg)) {
                                IValue const *value = cast<DAG_constant>(arg)->get_value();
                                is_black =
                                    is<IValue_invalid_ref>(value) && is<IType_df>(value->get_type());
                            }

                            call->replaceAllUsesWith(
                                llvm::ConstantInt::get(bool_type, is_black ? 0 : 1));
                        }
                    } else if (!opt_ctx.m_skip_bsdf_call) {
                        Distribution_function_state new_state = convert_to_df_state(df_func_kind);

                        instantiate_and_call_df(
                            ctx,
                            arg,
                            new_state,
                            /*res_pointer=*/ call->getArgOperand(0),
                            /*inherited_normal=*/ call->getArgOperand(2),
                            /*opt_inherited_weight=*/
                                new_state == DFSTATE_EVALUATE || new_state == DFSTATE_AUXILIARY
                                    ? call->getArgOperand(3) : nullptr,
                            /*insertBefore=*/ call);
                    }

                    // mark call instruction for deletion
                    delete_list.push_back(call);
                    continue;
                }

                llvm::Function *called_func = call->getCalledFunction();
                if (called_func == NULL) {
                    // ignore indirect function invocation
                    continue;
                }

                // check for calls to special functions
                llvm::StringRef func_name = called_func->getName();
                if (!func_name.startswith("get_")) {
                    continue;
                }

                Distribution_function::Special_kind special_kind;
                if (func_name == "get_material_ior") {
                    special_kind = Distribution_function::SK_MATERIAL_IOR;
                } else if (func_name == "get_material_thin_walled") {
                    special_kind = Distribution_function::SK_MATERIAL_THIN_WALLED;
                } else if (func_name == "get_material_volume_absorption_coefficient") {
                    special_kind = Distribution_function::SK_MATERIAL_VOLUME_ABSORPTION;
                } else {
                    continue;
                }

                size_t index = m_dist_func->get_special_node_index(special_kind);
                MDL_ASSERT(index != ~0 && "Invalid special node");

                ctx->SetInsertPoint(call);

                // determine expected return type (either type of call or from first argument)
                llvm::Type *expected_type = call->getType();
                if (expected_type == llvm::Type::getVoidTy(m_llvm_context)) {
                    expected_type = call->getArgOperand(0)->getType()->getPointerElementType();
                }

                Expression_result res = translate_node_at_insert_point(
                    m_dist_func->get_requested_node(index)->node,
                    expected_type);

                // fix iterators, as translate_node_at_insert_point may have inserted new blocks
                BI = call->getParent()->getIterator();
                BE = bsdf_func->end();

                // void function with result pointer in first argument?
                if (call->getType() != expected_type) {
                    // yes, write result to the result pointer
                    ctx->CreateStore(res.as_value(ctx), call->getArgOperand(0));
                } else {
                    // no, function returns result directly, so replace the call
                    call->replaceAllUsesWith(res.as_value(ctx));
                }

                // mark call instruction for deletion
                delete_list.push_back(call);
                continue;
            }
        }
    }

    for (size_t i = 0, num = delete_list.size(); i < num; ++i) {
        delete_list[i]->eraseFromParent();
    }

    // optimize function to improve inlining
    m_func_pass_manager->run(*bsdf_func);

    m_instantiated_dfs[m_dist_func_state][instantiated_df] = bsdf_func;

    return bsdf_func;
}


// Translate a DAG node pointing to a DF to LLVM IR.
void LLVM_code_generator::translate_distribution_function(
    DAG_node const       *df_node,
    llvm::GlobalVariable *mat_data_global)
{
    Function_context &ctx = *m_ctx;

    MDL_ASSERT(
        is<IType_df>(df_node->get_type()->skip_type_alias())
        && (
        (
            is<DAG_call>(df_node) &&
            (
                is_df_semantics(cast<DAG_call>(df_node)->get_semantic())
                ||
                is<IExpression::OK_TERNARY>(cast<DAG_call>(df_node))
                )
            ) || (
                is<DAG_constant>(df_node) &&
                cast<DAG_constant>(df_node)->get_value()->get_kind() == IValue::VK_INVALID_REF
                )
            )
    );

    // TODO: remove
    //   lambda results are not supported by new scheduler
#if 0
    // allocate the lambda results struct and make it available in the context
    llvm::Value *lambda_results = NULL;
    if (target_supports_lambda_results_parameter()) {
        lambda_results = ctx.create_local(m_lambda_results_struct_type, "lambda_results");
        ctx.override_lambda_results(
            ctx->CreateBitCast(lambda_results, m_type_mapper.get_void_ptr_type()));
    }

    // calculate all required non-constant expression lambdas
    for (size_t i = 0, n = lambda_result_exprs.size(); i < n; ++i) {
        size_t expr_index   = lambda_result_exprs[i];
        size_t result_index = m_lambda_result_indices[expr_index];

        generate_expr_lambda_call(expr_index, lambda_results, result_index);
    }
#else
    if (target_supports_lambda_results_parameter()) {
        ctx.override_lambda_results(
            llvm::ConstantPointerNull::get(m_type_mapper.get_void_ptr_type()));
    }
#endif

    // get the current normal
    llvm::Value *normal_buf;
    {
        IDefinition const *def = m_compiler->find_stdlib_signature("::state", "normal()");
        llvm::Function *func = get_intrinsic_function(def, /*return_derivs=*/ false);
        llvm::Value *args[] = { ctx.get_state_parameter() };
        llvm::Value *normal = call_rt_func(func, args);

        // convert to type used in libbsdf
        normal_buf = ctx.create_local(m_float3_struct_type, "normal_buf");
        ctx.convert_and_store(normal, normal_buf);

        m_state_usage_analysis.add_state_usage(
            ctx.get_function(), IGenerated_code_executable::SU_NORMAL);
    }

    // initialize evaluate and auxiliary data
    mi::mdl::IType::Kind df_kind = df_node->get_type()->get_kind();
    llvm::Constant *zero = ctx.get_constant(0.0f);
    if (m_dist_func_state == DFSTATE_EVALUATE || m_dist_func_state == DFSTATE_AUXILIARY) {
        llvm::Constant *float3_zero =
            llvm::ConstantAggregateZero::get(m_float3_struct_type);
        llvm::Constant *spectral_sample_zero = ctx.get_constant_spectral_sample_zero();
        llvm::Value *data_ptr = ctx.get_function()->arg_begin();

        if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_NONE) {
            // no handles
            if (df_kind == mi::mdl::IType::TK_BSDF || df_kind == mi::mdl::IType::TK_HAIR_BSDF) {
                switch (m_dist_func_state) {
                case DFSTATE_EVALUATE:
                    {
                        // bsdf_diffuse
                        ctx->CreateStore(spectral_sample_zero, ctx->CreateStructGEP(data_ptr, 4));

                        // bsdf_glossy
                        ctx->CreateStore(spectral_sample_zero, ctx->CreateStructGEP(data_ptr, 5));
                    }
                    break;
                case DFSTATE_AUXILIARY:
                    {
                        // albedo_diffuse
                        ctx->CreateStore(spectral_sample_zero, ctx->CreateStructGEP(data_ptr, 3));

                        // albedo_glossy
                        ctx->CreateStore(spectral_sample_zero, ctx->CreateStructGEP(data_ptr, 4));

                        // normal
                        ctx->CreateStore(float3_zero, ctx->CreateStructGEP(data_ptr, 5));

                        // roughness
                        ctx->CreateStore(float3_zero, ctx->CreateStructGEP(data_ptr, 6));
                    }
                    break;
                default:
                    break;
                }
            } else if (df_kind == mi::mdl::IType::TK_EDF && m_dist_func_state == DFSTATE_EVALUATE) {
                // edf
                ctx->CreateStore(spectral_sample_zero, ctx->CreateStructGEP(data_ptr, 2));
            }
        } else {
            // fixed size array or user data
            // number of elements in the buffer/array
            llvm::Value *handle_count = NULL;
            if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_POINTER) { // DF_HSM_POINTER
                int handle_count_idx = -1;
                if (df_kind == mi::mdl::IType::TK_BSDF || df_kind == mi::mdl::IType::TK_HAIR_BSDF) {
                    handle_count_idx = m_dist_func_state == DFSTATE_EVALUATE ? 5 : 4;
                } else if (df_kind == mi::mdl::IType::TK_EDF) {
                    handle_count_idx = m_dist_func_state == DFSTATE_EVALUATE ? 2 : -1;
                }

                if (handle_count_idx >= 0) {
                    handle_count = ctx->CreateLoad(
                        ctx->CreateStructGEP(data_ptr, handle_count_idx));
                }
            } else {                                                            // DF_HSM_FIXED_X
                handle_count = ctx.get_constant(
                    static_cast<int>(m_link_libbsdf_df_handle_slot_mode));
            }

            if (handle_count != NULL) {
                // setup a block and index
                llvm::BasicBlock *loop_block = ctx.create_bb("init_loop");
                llvm::BasicBlock *loop_block_end = ctx.create_bb("init_loop_end");

                llvm::Value *index_ptr = ctx.create_local(
                    m_type_mapper.get_int_type(), "init_index");
                ctx->CreateStore(ctx.get_constant(int(0)), index_ptr);

                // start loop
                ctx->CreateBr(loop_block);
                ctx->SetInsertPoint(loop_block);
                llvm::Value *cur_index = ctx->CreateLoad(index_ptr);

                // git indices of the fields to initialize
                int value_0_idx = -1;
                int value_1_idx = -1;
                int value_2_idx = -1;
                int value_3_idx = -1;
                if (df_kind == mi::mdl::IType::TK_BSDF || df_kind == mi::mdl::IType::TK_HAIR_BSDF) {
                    value_0_idx =
                        m_dist_func_state == DFSTATE_EVALUATE ? 5 : 4; // bsdf_diffuse/albedo_diffuse
                    value_1_idx =
                        m_dist_func_state == DFSTATE_EVALUATE ? 6 : 5; // bsdf_glossy/albedo_glossy
                    value_2_idx =
                        m_dist_func_state == DFSTATE_EVALUATE ? -1 : 6; // normal
                    value_3_idx =
                        m_dist_func_state == DFSTATE_EVALUATE ? -1 : 7; // roughness
                } else if (df_kind == mi::mdl::IType::TK_EDF &&
                    m_dist_func_state == DFSTATE_EVALUATE) {
                    value_0_idx = 3;                                  // edf
                }

                // get pointer and write zeros
                if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_POINTER) {
                    // for user buffers there is an additional 'handle_count' -> +1

                    // bsdf_diffuse/albedo_diffuse
                    if (value_0_idx >= 0) {
                        llvm::Value *result_value_ptr = ctx->CreateLoad(
                            ctx.create_simple_gep_in_bounds(data_ptr, value_0_idx + 1));
                        result_value_ptr = ctx->CreateGEP(result_value_ptr, cur_index);
                        ctx->CreateStore(spectral_sample_zero, result_value_ptr);
                    }

                    // bsdf_glossy/albedo_glossy
                    if (value_1_idx >= 0) {
                        llvm::Value *result_value_ptr = ctx->CreateLoad(
                            ctx.create_simple_gep_in_bounds(data_ptr, value_1_idx + 1));
                        result_value_ptr = ctx->CreateGEP(result_value_ptr, cur_index);
                        ctx->CreateStore(spectral_sample_zero, result_value_ptr);
                    }

                    // normal
                    if (value_2_idx >= 0) {
                        llvm::Value* result_value_ptr = ctx->CreateLoad(
                            ctx.create_simple_gep_in_bounds(data_ptr, value_2_idx + 1));
                        result_value_ptr = ctx->CreateGEP(result_value_ptr, cur_index);
                        ctx->CreateStore(float3_zero, result_value_ptr);
                    }

                    // roughness
                    if (value_3_idx >= 0) {
                        llvm::Value* result_value_ptr = ctx->CreateLoad(
                            ctx.create_simple_gep_in_bounds(data_ptr, value_3_idx + 1));
                        result_value_ptr = ctx->CreateGEP(result_value_ptr, cur_index);
                        ctx->CreateStore(float3_zero, result_value_ptr);
                    }
                } else {
                    // m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_FIXED_X

                    // bsdf_diffuse/albedo_diffuse
                    if (value_0_idx >= 0) {
                        llvm::Value *result_value_ptr = ctx->CreateGEP(
                            data_ptr,
                            { ctx.get_constant(int(0)),
                              ctx.get_constant(int(value_0_idx)),
                              cur_index });
                        ctx->CreateStore(spectral_sample_zero, result_value_ptr);
                    }

                    // bsdf_glossy/albedo_glossy
                    if (value_1_idx >= 0) {
                        llvm::Value *result_value_ptr = ctx->CreateGEP(
                            data_ptr,
                            { ctx.get_constant(int(0)),
                              ctx.get_constant(int(value_1_idx)),
                              cur_index });
                        ctx->CreateStore(spectral_sample_zero, result_value_ptr);
                    }

                    // normal
                    if (value_2_idx >= 0) {
                        llvm::Value* result_value_ptr = ctx->CreateGEP(
                            data_ptr,
                            { ctx.get_constant(int(0)),
                              ctx.get_constant(int(value_2_idx)),
                              cur_index });
                        ctx->CreateStore(float3_zero, result_value_ptr);
                    }

                    // roughness
                    if (value_3_idx >= 0) {
                        llvm::Value* result_value_ptr = ctx->CreateGEP(
                            data_ptr,
                            { ctx.get_constant(int(0)),
                              ctx.get_constant(int(value_3_idx)),
                              cur_index });
                        ctx->CreateStore(float3_zero, result_value_ptr);
                    }
                }

                // increment index, next iteration or end of loop
                llvm::Value *new_index = ctx->CreateAdd(cur_index, ctx.get_constant(1));
                ctx->CreateStore(new_index, index_ptr);
                llvm::Value *cond = ctx->CreateICmpSLT(new_index, handle_count);
                ctx->CreateCondBr(cond, loop_block, loop_block_end);
                ctx->SetInsertPoint(loop_block_end);
            }
        }
    }

    // create and initialize execution context
    llvm::Value *exec_ctx = nullptr;

    // avoid warning about unused parameter
    (void) mat_data_global;

    if (target_supports_lambda_results_parameter()) {
        exec_ctx = ctx.create_local(
            m_type_mapper.get_exec_ctx_type(), "exec_ctx");
        ctx->CreateStore(
            ctx.get_state_parameter(),
            ctx.create_simple_gep_in_bounds(exec_ctx, 0u));
        ctx->CreateStore(
            ctx.get_resource_data_parameter(),
            ctx.create_simple_gep_in_bounds(exec_ctx, 1u));
        ctx->CreateStore(
            target_uses_exception_state_parameter()
                ? ctx.get_exc_state_parameter()
                : llvm::ConstantPointerNull::get(m_type_mapper.get_exc_state_ptr_type()),
            ctx.create_simple_gep_in_bounds(exec_ctx, 2u));
        ctx->CreateStore(
            ctx.get_cap_args_parameter(),
            ctx.create_simple_gep_in_bounds(exec_ctx, 3u));
        ctx->CreateStore(
            ctx.get_lambda_results_parameter(),  // actually our overridden local struct
            ctx.create_simple_gep_in_bounds(exec_ctx, 4u));
    }
    // recursively instantiate the DF
    llvm::Function *df_func = instantiate_df(ctx, df_node);
    if (df_func == NULL) {
        MDL_ASSERT(!"BSDF instantiation failed");
        return;
    }

    // call the instantiated distribution function
    llvm::Value *result_pointer = ctx.get_function()->arg_begin();
    llvm::SmallVector<llvm::Value *, 4> df_args;
    df_args.push_back(result_pointer);
    df_args.push_back(exec_ctx ? exec_ctx : ctx.get_state_parameter());
    df_args.push_back(normal_buf);
    if (m_dist_func_state == DFSTATE_EVALUATE || m_dist_func_state == DFSTATE_AUXILIARY) {
        // create the initial inherited_weight
        llvm::Value *weight_buf = ctx.create_local(
            m_type_mapper.get_spectral_sample_type(), "weight");
        ctx->CreateStore(ctx.get_constant_spectral_sample_one(), weight_buf);
        df_args.push_back(weight_buf);
    }
    m_state_usage_analysis.add_call(ctx.get_function(), df_func);
    ctx->CreateCall(df_func, df_args);

    // at the end of the sample function, call the pdf function to calculate the pdf result
    if (m_dist_func_state == DFSTATE_SAMPLE) {
        bool is_edf = df_kind == mi::mdl::IType::TK_EDF;

        // Depending on the size of spectral_sample, an additional padding field
        // may be added by Clang before the xi field, which appears before the pdf and the
        // libbsdf flags field
        unsigned padding_fields = 0;
        if (!is_edf && m_type_bsdf_sample_data->getStructElementType(4)->isArrayTy()) {
            padding_fields = 1;
        }

        // BSDF_pdf_data and EDF_pdf_data are declared __align__(16). Use explicit 16-byte
        // alignment here because LLVM's DataLayout does not see the C++ alignment attribute
        // and would otherwise under-align the alloca, causing vmovaps crashes in spectral mode.
        llvm::Value *pdf_data = ctx.create_local(
            is_edf ? m_type_edf_pdf_data : m_type_bsdf_pdf_data, llvm::Align(16), "pdf_data");
        llvm::Value *sample_data = result_pointer;

        // copy over the values from the sample data to a pdf data struct
        if (is_edf) {
            // only k1 needs to be copied
            llvm::Value *k1_val =
                ctx->CreateLoad(ctx->CreateStructGEP(sample_data, 1));
            ctx->CreateStore(k1_val, ctx->CreateStructGEP(pdf_data, 0));
        } else {
            // copy first 4 struct fields (ior1, ior2, k1, k2)
            for (unsigned i = 0; i < 4; ++i) {
                llvm::Value *data =
                    ctx->CreateLoad(ctx->CreateStructGEP(sample_data, i));
                ctx->CreateStore(data, ctx->CreateStructGEP(pdf_data, i));
            }

            // copy libbsdf flags if used
            if (m_libbsdf_flags_in_bsdf_data) {
                llvm::Value *flags =
                    ctx->CreateLoad(ctx->CreateStructGEP(sample_data, 9 + padding_fields));
                ctx->CreateStore(flags, ctx->CreateStructGEP(pdf_data, 5));
            }
        }

        // instantiate pdf function
        llvm::Function *param_bsdf_func = nullptr;
        {
            Store<Distribution_function_state> state_store(m_dist_func_state, DFSTATE_PDF);
            param_bsdf_func = instantiate_df(ctx, df_node);
        }
        m_state_usage_analysis.add_call(ctx.get_function(), param_bsdf_func);

        // call it
        llvm::SmallVector<llvm::Value *, 4> llvm_args;
        llvm_args.push_back(pdf_data);
        llvm_args.push_back(exec_ctx ? exec_ctx : ctx.get_state_parameter());
        llvm_args.push_back(normal_buf);  // inherited_normal
        ctx->CreateCall(param_bsdf_func, llvm_args);

        // write pdf value from pdf data to sample data
        llvm::Value *pdf_val = ctx->CreateLoad(
            ctx->CreateStructGEP(pdf_data, is_edf ? 1 : 4));
        ctx->CreateStore(
            pdf_val,
            ctx->CreateStructGEP(sample_data, is_edf ? 2 : 5 + padding_fields));
    }

    if ((df_kind == mi::mdl::IType::TK_BSDF || df_kind == mi::mdl::IType::TK_HAIR_BSDF) &&
        m_dist_func_state == DFSTATE_AUXILIARY)
    {
        // normalize function
        IDefinition const *norm_def = m_compiler->find_stdlib_signature(
            "::math", "normalize(float3)");
        llvm::Function *norm_func = get_intrinsic_function(norm_def, /*return_derivs=*/ false);

        if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_NONE) {
            // no handles

            {
                // normalize the normals
                // find normal in the data structure (element at index 5)
                llvm::Value *result_normal_ptr = ctx->CreateGEP(
                    ctx.get_function()->arg_begin(),  // result pointer
                    { ctx.get_constant(int(0)), ctx.get_constant(int(5)) });

                llvm::Value *result_normal = ctx.load_and_convert(
                    m_type_mapper.get_float3_type(), result_normal_ptr);

                llvm::Value *cond_x = ctx->CreateFCmpONE(ctx.create_extract(result_normal, 0), zero);
                llvm::Value *cond_y = ctx->CreateFCmpONE(ctx.create_extract(result_normal, 1), zero);
                llvm::Value *cond_z = ctx->CreateFCmpONE(ctx.create_extract(result_normal, 2), zero);
                llvm::Value *cond_normalize = ctx->CreateOr(cond_x, ctx->CreateOr(cond_y, cond_z));

                // setup a block and index
                llvm::BasicBlock *if_non_zero_block = ctx.create_bb("if_non_zero_normal");
                llvm::BasicBlock *if_non_zero_block_end = ctx.create_bb("if_non_zero_normal_end");

                ctx->CreateCondBr(cond_normalize, if_non_zero_block, if_non_zero_block_end);
                ctx->SetInsertPoint(if_non_zero_block);

                // if (cond_normalize)
                //     result_normalized = normalize(result_normalized)
                llvm::Value *result_normalized = call_rt_func(norm_func, {result_normal});
                ctx.convert_and_store(result_normalized, result_normal_ptr);
                ctx->CreateBr(if_non_zero_block_end);

                ctx->SetInsertPoint(if_non_zero_block_end);
            }
            {
                // apply weights to the roughness, i.e. divide the weighted sums by the summed weights
                llvm::Value *result_roughness_ptr = ctx->CreateGEP(
                    ctx.get_function()->arg_begin(),  // result pointer
                    { ctx.get_constant(int(0)), ctx.get_constant(int(6)) });

                llvm::Value *result_roughness = ctx.load_and_convert(
                    m_type_mapper.get_float3_type(), result_roughness_ptr);

                // condition for applying the wight is that the z component is not zero
                llvm::Value *roughness_u = ctx.create_extract(result_roughness, 0);
                llvm::Value *roughness_v = ctx.create_extract(result_roughness, 1);
                llvm::Value *summed_weights = ctx.create_extract(result_roughness, 2);
                llvm::Value *cond = ctx->CreateFCmpONE(summed_weights, zero);
                llvm::BasicBlock *if_non_zero_block = ctx.create_bb("if_non_zero_weight");
                llvm::BasicBlock *if_non_zero_block_end = ctx.create_bb("if_non_zero_weight_end");

                ctx->CreateCondBr(cond, if_non_zero_block, if_non_zero_block_end);
                ctx->SetInsertPoint(if_non_zero_block);

                // if (cond)
                //     rougness_u = rougness_u / summed_weights;
                //     rougness_v = rougness_v / summed_weights;
                roughness_u = ctx.create_fdiv(roughness_u->getType(), roughness_u, summed_weights);
                roughness_v = ctx.create_fdiv(roughness_v->getType(), roughness_v, summed_weights);
                result_roughness = ctx.create_insert(result_roughness, roughness_u, 0);
                result_roughness = ctx.create_insert(result_roughness, roughness_v, 1);

                ctx.convert_and_store(result_roughness, result_roughness_ptr);
                ctx->CreateBr(if_non_zero_block_end);

                ctx->SetInsertPoint(if_non_zero_block_end);
            }
            return;
        }

        // number of elements in the buffer/array
        llvm::Value *handle_count = NULL;
        if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_POINTER) {
            handle_count = ctx->CreateLoad(
                ctx.create_simple_gep_in_bounds(ctx.get_function()->arg_begin(), 4));
        } else { // m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_FIXED_X
            handle_count = ctx.get_constant(static_cast<int>(m_link_libbsdf_df_handle_slot_mode));
        }

        // setup a block and index
        llvm::BasicBlock *loop_block = ctx.create_bb("post_loop");
        llvm::BasicBlock *loop_block_end = ctx.create_bb("post_loop_end");

        llvm::Value *index_ptr = ctx.create_local(m_type_mapper.get_int_type(), "post_loop_index");
        ctx->CreateStore(ctx.get_constant(int(0)), index_ptr);

        // start loop
        ctx->CreateBr(loop_block);
        ctx->SetInsertPoint(loop_block);
        llvm::Value *cur_index = ctx->CreateLoad(index_ptr);

        // get a pointer to the normal at the current index
        llvm::Value *result_normal_ptr = NULL;
        llvm::Value *result_roughness_ptr = NULL;
        if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_POINTER) {
            llvm::Value *result_normal_ptr_ptr = ctx->CreateGEP(
                ctx.get_function()->arg_begin(),
                { ctx.get_constant(int(0)), ctx.get_constant(int(7)) });
            result_normal_ptr = ctx->CreateLoad(result_normal_ptr_ptr);
            result_normal_ptr = ctx->CreateGEP(result_normal_ptr, cur_index);

            llvm::Value *result_roughness_ptr_ptr = ctx->CreateGEP(
                ctx.get_function()->arg_begin(),
                { ctx.get_constant(int(0)), ctx.get_constant(int(8)) });
            result_roughness_ptr = ctx->CreateLoad(result_roughness_ptr_ptr);
            result_roughness_ptr = ctx->CreateGEP(result_roughness_ptr, cur_index);

        } else { // m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_FIXED_X
            result_normal_ptr = ctx->CreateGEP(
                ctx.get_function()->arg_begin(),
                { ctx.get_constant(int(0)), ctx.get_constant(int(6)), cur_index });
            result_roughness_ptr = ctx->CreateGEP(
                ctx.get_function()->arg_begin(),
                { ctx.get_constant(int(0)), ctx.get_constant(int(7)), cur_index });
        }

        {
            // normalize the normals
            // load, check if none-zero, normalize, store
            llvm::Value *result_normal = ctx.load_and_convert(
                m_type_mapper.get_float3_type(), result_normal_ptr);

            llvm::Value *cond_x = ctx->CreateFCmpONE(ctx.create_extract(result_normal, 0), zero);
            llvm::Value *cond_y = ctx->CreateFCmpONE(ctx.create_extract(result_normal, 1), zero);
            llvm::Value *cond_z = ctx->CreateFCmpONE(ctx.create_extract(result_normal, 2), zero);
            llvm::Value *cond_normalize = ctx->CreateOr(cond_x, ctx->CreateOr(cond_y, cond_z));

            // setup a block and index
            llvm::BasicBlock *if_non_zero_block = ctx.create_bb("if_non_zero");
            llvm::BasicBlock *if_non_zero_block_end = ctx.create_bb("if_non_zero_end");

            ctx->CreateCondBr(cond_normalize, if_non_zero_block, if_non_zero_block_end);
            ctx->SetInsertPoint(if_non_zero_block);

            // if (cond_normalize)
            //     result_normalized = normalize(result_normalized)
            llvm::Value *result_normalized = call_rt_func(norm_func, { result_normal });
            ctx.convert_and_store(result_normalized, result_normal_ptr);
            ctx->CreateBr(if_non_zero_block_end);

            ctx->SetInsertPoint(if_non_zero_block_end);
        }
        {
            // apply weights to the roughness, i.e. divide the weighted sums by the summed weights
            llvm::Value *result_roughness = ctx.load_and_convert(
                m_type_mapper.get_float3_type(), result_roughness_ptr);

            llvm::Value *roughness_u = ctx.create_extract(result_roughness, 0);
            llvm::Value *roughness_v = ctx.create_extract(result_roughness, 1);
            llvm::Value *summed_weights = ctx.create_extract(result_roughness, 2);
            llvm::Value *cond = ctx->CreateFCmpONE(summed_weights, zero);
            llvm::BasicBlock *if_non_zero_block = ctx.create_bb("if_non_zero_weight");
            llvm::BasicBlock *if_non_zero_block_end = ctx.create_bb("if_non_zero_weight_end");

            ctx->CreateCondBr(cond, if_non_zero_block, if_non_zero_block_end);
            ctx->SetInsertPoint(if_non_zero_block);

            // if (cond)
            //     rougness_u = rougness_u / summed_weights;
            //     rougness_v = rougness_v / summed_weights;
            roughness_u = ctx.create_fdiv(roughness_u->getType(), roughness_u, summed_weights);
            roughness_v = ctx.create_fdiv(roughness_v->getType(), roughness_v, summed_weights);
            result_roughness = ctx.create_insert(result_roughness, roughness_u, 0);
            result_roughness = ctx.create_insert(result_roughness, roughness_v, 1);

            ctx.convert_and_store(result_roughness, result_roughness_ptr);
            ctx->CreateBr(if_non_zero_block_end);

            ctx->SetInsertPoint(if_non_zero_block_end);
        }
        // increment index, next iteration or end of loop
        llvm::Value *new_index = ctx->CreateAdd(cur_index, ctx.get_constant(1));
        ctx->CreateStore(new_index, index_ptr);
        llvm::Value *cond = ctx->CreateICmpSLT(new_index, handle_count);
        ctx->CreateCondBr(cond, loop_block, loop_block_end);
        ctx->SetInsertPoint(loop_block_end);
    }
}

void LLVM_code_generator::translate_distribution_function_init_loop(Loop_schedule const &loop_schedule) {
    Function_context &ctx = *m_ctx;
    bool sets_normal = false;
    bool writes_textures = false;
    bool has_expensive = false;


    for (auto const &eval : loop_schedule.evaluations) {
        switch (eval.kind) {
        case Loop_schedule::Evaluation::Kind::EK_SET_NORMAL:
            sets_normal = true;
            break;
        case Loop_schedule::Evaluation::Kind::EK_WRITE_TEXTURE:
            writes_textures = true;
            if (eval.expensive) {
                has_expensive = true;
            }
            break;
        case Loop_schedule::Evaluation::Kind::EK_EVAL:
            if (eval.expensive) {
                has_expensive = true;
            }
            break;
        default:
            break;
        }
    }

    llvm::Value *texture_results = nullptr;
    if (writes_textures) {
        texture_results = get_texture_results();
        if (target_is_structured_language()) {
            llvm::Value *res_value = llvm::Constant::getNullValue(m_type_mapper.get_float_type());
            for (size_t i = 0; i < loop_schedule.total_texture_bytes; i += 4) {
                store_to_float4_array(
                    res_value,
                    texture_results,
                    i);
            }
        }
    }

    llvm::Type *normal_type = m_type_mapper.get_float3_type();
    llvm::Value *normal_variable = sets_normal ? ctx.create_local(normal_type, "normal") : nullptr;

    typedef vector<llvm::Value *>::Type Loop_variable_vector;
    Loop_variable_vector local_variable_list(get_allocator());
    Loop_variable_vector local_array_variable_list(get_allocator());

    for (size_t i = 0; i < loop_schedule.local_types.size(); ++i) {
        IType const *mdl_type = loop_schedule.local_types[i].second;
        llvm::Type *llvm_type = m_type_mapper.lookup_type(
            m_type_mapper.get_llvm_context(), mdl_type);
        llvm::Value *local = ctx.create_local(llvm_type, "local");
        local_variable_list.push_back(local);

    }

    auto write_texture = [&](Loop_schedule::Evaluation const &eval, Expression_result &res) {
        llvm::Value *res_value = res.as_value(ctx);
        if (target_is_structured_language()) {
            store_to_float4_array(
                res_value,
                texture_results,
                eval.texture_result_offset);
        } else {
            llvm::Value *ptr = ctx.create_simple_gep_in_bounds(
                texture_results, eval.texture_result_index);
            ctx.convert_and_store(res_value, ptr);
        }
        };

    auto store_normal = [&](Loop_schedule::Evaluation const &eval, Expression_result &res) {
        // type doesn't matter or fits already?
        if (res.get_value_type() != m_type_mapper.get_float3_type()) {
            // convert to expected type
            res = Expression_result::value(
                ctx.load_and_convert(m_type_mapper.get_float3_type(), res.as_ptr(ctx)));
        }

        llvm::Value *normal = res.as_value(ctx);
        ctx->CreateStore(normal, normal_variable);
        };

    auto generate_evaluation = [&](Loop_schedule::Evaluation const &eval, llvm::Value *loop_var) {
        switch (eval.kind) {
        case Loop_schedule::Evaluation::Kind::EK_EVAL:
        {
            Expression_result res = translate_node(
                eval.node,
                m_cur_resolver);
            m_manual_node_value_map[eval.node] = res;

            if (eval.is_geom_normal) {
                store_normal(eval, res);
            }
            break;
        }
        case Loop_schedule::Evaluation::Kind::EK_WRITE_TEXTURE:
        {
            Expression_result res = translate_node(
                eval.node,
                m_cur_resolver);
            m_manual_node_value_map[eval.node] = res;


            if (eval.is_geom_normal) {
                store_normal(eval, res);
            }

            write_texture(eval, res);

            break;
        }

        case Loop_schedule::Evaluation::Kind::EK_SET_NORMAL:
        {
            llvm::Value *normal = ctx->CreateLoad(normal_variable, "normal");

            // call state::adapt_normal(normal), if requested
            if (m_use_renderer_adapt_normal) {
                llvm::Function *adapt_normal = get_internal_function(m_int_func_state_adapt_normal);
                llvm::SmallVector<llvm::Value *, 3> args;
                args.push_back(ctx.get_state_parameter());
                if (target_uses_resource_data_parameter()) {
                    args.push_back(ctx.get_resource_data_parameter());
                }
                args.push_back(normal);
                normal = call_rt_func(adapt_normal, args);
            }

            // call state::set_normal(normal)
            llvm::Function *set_func = get_internal_function(m_int_func_state_set_normal);
            llvm::Value *set_normal_args[] = {
                ctx.get_state_parameter(),
                normal
            };
            call_rt_func_void(set_func, set_normal_args);

            // clear DAG results, as they may depend on the evaluation state.
            // Reused nodes, for which we know, that they don't depend on the evaluation state
            // will still be available in the manual node value map.
            clear_dag_node_map();

            // the evaluation state changes now, so remove all results depending on it
            // from the manual node value map
            for (auto node : eval.to_invalidate) {
                m_manual_node_value_map.erase(node);
            }
            break;
        }

        case Loop_schedule::Evaluation::Kind::EK_CASE_SPLIT:
        case Loop_schedule::Evaluation::Kind::EK_EXP_PARAM_SPLIT:
            clear_dag_node_map();
            m_manual_node_value_map.clear();
            break;

        case Loop_schedule::Evaluation::Kind::EK_EXP_RED_SPLIT:
            break;

        case Loop_schedule::Evaluation::Kind::EK_SAVE:
            for (auto const &p : eval.tmp_map) {
                if (p.tex_result) {
                    if (target_is_structured_language()) {
                        store_to_float4_array(
                            m_manual_node_value_map[p.node].as_value(ctx),
                            texture_results,
                            p.tex_result_offset);
                    } else {
                        MDL_ASSERT(false);
                    }
                } else {
                    ctx->CreateStore(m_manual_node_value_map[p.node].as_value(ctx), local_variable_list[p.index]);
                }
            }
            break;

        case Loop_schedule::Evaluation::Kind::EK_RELOAD:
            for (auto const &p : eval.tmp_map) {
                auto map_it = m_manual_node_value_map.find(p.node);
                if (map_it != m_manual_node_value_map.end()) {
                    continue;
                }
                if (p.tex_result) {
                    if (target_is_structured_language()) {
                        llvm::Type *llvm_type = m_type_mapper.lookup_type(
                            m_type_mapper.get_llvm_context(), p.node->get_type());
                        llvm::Value *val = load_from_float4_array(
                            llvm_type,
                            texture_results,
                            p.tex_result_offset);
                        m_manual_node_value_map[p.node] = Expression_result::value(val);
                    } else {
                        MDL_ASSERT(false);
                    }
                } else {
                    llvm::Value *val = ctx->CreateLoad(local_variable_list[p.index]);
                    m_manual_node_value_map[p.node] = Expression_result::value(val);
                }
            }
            break;

        default:
            MDL_ASSERT(!"unhandled case");
            break;
        }
        };

    if (loop_schedule.evaluate_sequentially || !has_expensive) {
        for (auto const &eval : loop_schedule.evaluations) {
            generate_evaluation(eval, nullptr);
        }
        // clear manual node value map again
        m_manual_node_value_map.clear();
        return;
    }

    // Constants for loop iteration count and the value `0`.
    llvm::Value *loop_iterations = ctx.get_constant(int(loop_schedule.iterations));
    llvm::Value *zero = ctx.get_constant(int(0));

    // Start of loop, jumped to at the end of pre-loop sequence, checks iteraration
    // variable against limit and quits loop at the end by jumping to `after_loop`.
    // Otherwise, jumps to `outer_switch`.
    llvm::BasicBlock *loop_start_bb = ctx.create_bb("loop_start");

    // End of loop. Jump increment loop variable and jump back to `loop_start`.
    llvm::BasicBlock *loop_end_bb = ctx.create_bb("loop_end");

    // End of loop. Jump increment loop variable and jump back to `loop_start`.
    llvm::BasicBlock *after_loop_bb = ctx.create_bb("after_loop");

    // Outer "switch" created with an if cascade. This is the loop body.
    // Control flow ends at `after_outer` after the outer switch.
    llvm::BasicBlock *outer_switch_bb = ctx.create_bb("outer_switch");

    // Create the case body blocks of the outer "switch".
    size_t num_outer_cases = loop_schedule.call_site_map.size();
    llvm::SmallVector<llvm::BasicBlock *, 8> outer_case_body_blocks;
    for (size_t i = 0; i < num_outer_cases; ++i) {
        outer_case_body_blocks.push_back(ctx.create_bb("outer_case_body"));
    }

    // Merge point after the cases of the outer switch. Jumps to `after_loop`.
    llvm::BasicBlock *after_outer_bb = ctx.create_bb("after_outer");

    // [=] KICK OFF THE LOOP

    // Go to loop entry at `loop_start`.
    auto start_bb = ctx->GetInsertBlock();
    ctx->CreateBr(loop_start_bb);

    // [=] LOOP START: Check iteration count.

    ctx->SetInsertPoint(loop_start_bb);
    llvm::PHINode *loop_var = ctx->CreatePHI(zero->getType(), 2, "i");
    loop_var->addIncoming(zero, start_bb);
    llvm::PHINode *hack_loop_var = ctx->CreatePHI(zero->getType(), 2, "hack_i");
    hack_loop_var->addIncoming(zero, start_bb);
    llvm::Value *cond = ctx->CreateICmpSGE(loop_var, loop_iterations, "for_cond");
    ctx->CreateCondBr(cond, after_loop_bb, outer_switch_bb);

    // HACK: Enable hack making mdl_read_* offsets depend on the hack loop variable
    m_sl_value_hack_loop_var = hack_loop_var;

    // [=] LOOP END: Increment loop counter.

    ctx->SetInsertPoint(loop_end_bb);
    // Increment loop counter and jump back to the loop start.
    llvm::Value *next_i_phi = ctx->CreateAdd(loop_var, ctx.get_constant(1));
    loop_var->addIncoming(next_i_phi, loop_end_bb);
    // Increment hack loop counter and jump back to the loop start.
    llvm::Value *next_hack_i_phi = ctx->CreateAdd(hack_loop_var, ctx.get_constant(37));
    hack_loop_var->addIncoming(next_hack_i_phi, loop_end_bb);
    ctx->CreateBr(loop_start_bb);

    // [=] OUTER SWITCH: One entry per called function.

    ctx->SetInsertPoint(outer_switch_bb);

    size_t call_index = 0;

    // special case: only 1 expensive call
    if (num_outer_cases == 1) {
        ctx->CreateBr(outer_case_body_blocks[0]);
    } else {
        // iterate until pre-last case and build if-cascade
        for (auto it = loop_schedule.call_site_map.begin(); call_index < num_outer_cases - 1;
                ++it, ++call_index) {
            auto const &ec = it->second;
            auto const &ecs_cases = ec.cases;

            llvm::BasicBlock *outer_case_bb = outer_case_body_blocks[call_index];

            llvm::Value *case_cond = nullptr;

            // build condition for this expensive call (loop_var == index1 | loop_var == idx2 | ...)
            // Explicitly use "or" to avoid shortcut evaluation messing up the control flow
            for (auto const &ecs_case : ecs_cases) {
                size_t v = ecs_case.index;

                llvm::Value *next_case_cond = ctx->CreateICmp(
                    llvm::ICmpInst::ICMP_EQ,
                    loop_var,
                    ctx.get_constant(int(v)));
                if (case_cond == nullptr) {
                    case_cond = next_case_cond;
                } else {
                    case_cond = ctx->CreateOr(case_cond, next_case_cond);
                }
            }

            llvm::BasicBlock *else_bb;
            // last if of if-cascade?
            if (call_index + 1 == num_outer_cases - 1) {
                else_bb = outer_case_body_blocks[call_index + 1];
            } else {
                else_bb = ctx.create_bb("outer_switch_cascade");
            }
            ctx->CreateCondBr(case_cond, outer_case_bb, else_bb);

            // continue cascade in else block
            ctx->SetInsertPoint(else_bb);
        }
    }

    call_index = 0;
    for (auto it = loop_schedule.call_site_map.begin(), end = loop_schedule.call_site_map.end();
            it != end; ++it, ++call_index) {
        auto const &ec = it->second;
        auto const &ecs_cases = ec.cases;

        llvm::BasicBlock *after_param_bb = ctx.create_bb("after_param_switch");

        ctx->SetInsertPoint(outer_case_body_blocks[call_index]);

        typedef vector<llvm::PHINode *>::Type Llvm_phi_vector;
        Llvm_phi_vector varying_param_phis(get_allocator());
        typedef vector<llvm::Value *>::Type Llvm_value_vector;
        typedef vector < std::pair < llvm::BasicBlock *, Llvm_value_vector >> ::Type Param_case_value_vector;
        Param_case_value_vector varying_param_case_values(get_allocator());

        llvm::BasicBlock* first_param_case_bb = ctx.create_bb("");

        llvm::SwitchInst *param_switch_instr = ctx->CreateSwitch(loop_var, first_param_case_bb,
            unsigned(ecs_cases.size()));
        DAG_call const *generic_call = nullptr;
        size_t argument_count = 0;

        for (auto const &ecs_case : ecs_cases) {

            clear_dag_node_map();
            m_manual_node_value_map.clear();

            llvm::BasicBlock *param_case_bb = first_param_case_bb ? first_param_case_bb : ctx.create_bb("");
            first_param_case_bb = nullptr;

            size_t v = ecs_case.index;

            char case_name[32];
            snprintf(case_name, sizeof(case_name), "param_case_body_%u_", unsigned(v));
            param_case_bb->setName(case_name);

            param_switch_instr->addCase(ctx.get_constant(int(v)), param_case_bb);
            ctx->SetInsertPoint(param_case_bb);

            size_t eval_idx = ecs_case.first;
            size_t eval_cnt = ecs_case.last;

            for (; eval_idx <= eval_cnt; ++eval_idx) {
                auto const &eval = loop_schedule.evaluations[eval_idx];
                if (eval.expensive) {
                    break;
                }
                generate_evaluation(eval, loop_var);
            }

            auto const &exp_eval = loop_schedule.evaluations[eval_idx];

            typedef vector<llvm::Value *>::Type Llvm_value_vector;
            Llvm_value_vector param_values(get_allocator());

            DAG_call const *ecs_call = cast<DAG_call>(exp_eval.node);
            if (generic_call == nullptr) {
                generic_call = ecs_call;
                argument_count = generic_call->get_argument_count();
            }
            for (size_t i = 0; i < argument_count; ++i) {
                if (!ec.equal_params.test_bit(i)) {
                    if (ec.material_param_params.test_bit(i)) {
                        DAG_parameter const *param = cast<DAG_parameter>(ecs_call->get_argument(i));
                        llvm::DataLayout const *dl = get_target_layout_data();
                        llvm::StructLayout const *sl = dl->getStructLayout(m_captured_args_type);
                        int param_offs = int(sl->getElementOffset(param->get_index()));

                        Expression_result res = Expression_result::value(ctx.get_constant(param_offs));
                        m_manual_node_value_map[exp_eval.node] = res;
                        param_values.push_back(res.as_value(ctx));
                    } else {
                        Expression_result res = translate_node(
                            ecs_call->get_argument(i),
                            m_cur_resolver);
                        m_manual_node_value_map[exp_eval.node] = res;
                        param_values.push_back(res.as_value(ctx));
                    }
                } else {
                    param_values.push_back(nullptr);
                }
            }

            llvm::BasicBlock *param_exit_bb = ctx->GetInsertBlock();
            varying_param_case_values.push_back({ param_exit_bb, std::move(param_values) });
            ctx->CreateBr(after_param_bb);
        }

        ctx->SetInsertPoint(after_param_bb);

        string pname(get_allocator());

        MDL_ASSERT(generic_call != nullptr);

        for (size_t i = 0; i < argument_count; ++i) {
            if (!ec.equal_params.test_bit(i)) {
                DAG_node const *arg = generic_call->get_argument(i);
                pname = "p";
                pname = pname + std::to_string(i).c_str();
                if (ec.material_param_params.test_bit(i)) {
                    llvm::PHINode *varying_param_phi = ctx->CreatePHI(zero->getType(), ecs_cases.size(), pname.c_str());
                    varying_param_phis.push_back(varying_param_phi);
                } else {
                    llvm::PHINode *varying_param_phi = ctx->CreatePHI(lookup_type(arg->get_type()), ecs_cases.size(), pname.c_str());
                    varying_param_phis.push_back(varying_param_phi);
                }
            } else {
                varying_param_phis.push_back(nullptr);
            }
        }

        // Merge all varying parameters from parameter switch into PHIs.

        for (size_t idx = 0; idx < ecs_cases.size(); ++idx) {
            llvm::BasicBlock *bb = varying_param_case_values[idx].first;
            auto &param_values = varying_param_case_values[idx].second;

            for (size_t i = 0; i < argument_count; ++i) {
                if (!ec.equal_params.test_bit(i)) {
                    llvm::Value *v = param_values[i];
                    llvm::PHINode *varying_param_phi = varying_param_phis[i];
                    varying_param_phi->addIncoming(v, bb);
                }
            }
        }

        clear_dag_node_map();
        m_manual_node_value_map.clear();

        vector<Expression_result>::Type actual_arguments(get_allocator());

        auto trans_param = [&](llvm::Value *offset, DAG_node const *arg) {
            MDL_ASSERT(target_is_structured_language());
            mi::mdl::IType const *param_type = arg->get_type();
            return Expression_result::offset(
                offset,
                Expression_result::OK_ARG_BLOCK,
                lookup_type(param_type, ctx.instantiate_type_size(param_type)),
                param_type);
            };
        for (size_t i = 0; i < argument_count; ++i) {
            DAG_node const *arg = generic_call->get_argument(i);
            if (!ec.equal_params.test_bit(i)) {
                if (ec.material_param_params.test_bit(i)) {
                    llvm::Value *ofs = varying_param_phis[i];
                    Expression_result r = trans_param(ofs, arg);
                    actual_arguments.push_back(r);

                } else {
                    llvm::Value *val = varying_param_phis[i];
                    Expression_result r = Expression_result::value(val);
                    actual_arguments.push_back(r);
                }
            } else {
                Expression_result res = translate_node(
                    arg,
                    m_cur_resolver);
                actual_arguments.push_back(res);
            }
        }

        set_argument_overrides(actual_arguments.data(), actual_arguments.size());
        Expression_result exp_res = translate_node(generic_call, m_cur_resolver);
        set_argument_overrides(nullptr, 0);

        llvm::BasicBlock *reduction_bb = ctx.create_bb("reduction_switch");
        ctx->CreateBr(reduction_bb);

        llvm::BasicBlock *after_reduction_bb = ctx.create_bb("after_reduction_switch");

        ctx->SetInsertPoint(reduction_bb);

        llvm::BasicBlock* first_reduction_case_bb = ctx.create_bb("");

        llvm::SwitchInst *reduction_switch_instr = ctx->CreateSwitch(loop_var, first_reduction_case_bb, unsigned(ecs_cases.size()));
        for (auto const &ecs_case : ecs_cases) {
            clear_dag_node_map();
            m_manual_node_value_map.clear();

            llvm::BasicBlock *reduction_case_bb = first_reduction_case_bb ? first_reduction_case_bb : ctx.create_bb("");
            first_reduction_case_bb = nullptr;

            size_t v = ecs_case.index;

            char case_name[32];
            snprintf(case_name, sizeof(case_name), "reduction_case_body_%u_", unsigned(v));
            reduction_case_bb->setName(case_name);

            reduction_switch_instr->addCase(ctx.get_constant(int(v)), reduction_case_bb);
            ctx->SetInsertPoint(reduction_case_bb);

            size_t eval_idx = ecs_case.first;
            size_t eval_cnt = ecs_case.last;

            for (; eval_idx <= eval_cnt; ++eval_idx) {
                auto const &eval = loop_schedule.evaluations[eval_idx];
                if (eval.expensive) {
                    break;
                }
            }

            Loop_schedule::Evaluation const &exp_eval = loop_schedule.evaluations[eval_idx];
            // For each reduction case, we map the expensive call node of this particular
            // case to the common expensive call result we created above.
            m_manual_node_value_map[exp_eval.node] = exp_res;

            if (exp_eval.kind == Loop_schedule::Evaluation::Kind::EK_WRITE_TEXTURE) {
                write_texture(exp_eval, exp_res);
            }
            if (exp_eval.is_geom_normal) {
                store_normal(exp_eval, exp_res);
            }

            ++eval_idx; // Skip expensive call.

            for (; eval_idx <= eval_cnt; ++eval_idx) {
                auto const &eval = loop_schedule.evaluations[eval_idx];
                generate_evaluation(eval, loop_var);
            }

            ctx->CreateBr(after_reduction_bb);
        }
        ctx->SetInsertPoint(after_reduction_bb);

        ctx->CreateBr(after_outer_bb);
    }

    // [=] AFTER OUTER SWITCH: Go to loop end.

    ctx->SetInsertPoint(after_outer_bb);
    ctx->CreateBr(loop_end_bb);

    // [=] AFTER LOOP: Finish init function.
    ctx->SetInsertPoint(after_loop_bb);

    // Clear manual mode map before returning.
    m_manual_node_value_map.clear();

    // HACK: Disable hack again
    m_sl_value_hack_loop_var = nullptr;
}

// Translate the init function of a distribution function to LLVM IR.
void LLVM_code_generator::translate_distribution_function_init(
    mi::mdl::vector<Schedule_entry>::Type const &schedule,
    Loop_schedule const &loop_schedule)
{
    Function_context &ctx = *m_ctx;

#if 0
    if (loop_schedule.schedule_loop) {
        translate_distribution_function_init_loop(loop_schedule);
        return;

    }
#endif
    llvm::Value *texture_results = NULL;
    if (schedule.size() != 0 &&
        // no texture results accessed, if the only scheduled node is geometry.normal
        !(schedule.size() == 1 &&
            schedule[0].has_special_kind(Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL)))
    {
        texture_results = get_texture_results();
    }

    // Non-loop-scheduled path.
    mi::mdl::vector<DAG_node const *>::Type cur_eval_state_results(get_allocator());

    size_t i;
    size_t n = schedule.size();
    size_t max_index_before_normal_update = ~0;

    // if displacement, cutout opacity or geometry.normal are part of the schedule,
    // they and all scheduled nodes before them need to be translated first.
    // find the maximum schedule index, which must be translated before updating the normal.
    // note: displacement and/or cutout opacity may depend on the result of the DAG node
    //       of geometry.normal
    for (i = 0; i < n; ++i) {
        if (schedule[i].has_special_kind(
                Distribution_function::SK_MATERIAL_GEOMETRY_DISPLACEMENT) ||
            schedule[i].has_special_kind(
                Distribution_function::SK_MATERIAL_GEOMETRY_CUTOUT_OPACITY) ||
            schedule[i].has_special_kind(
                Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL))
        {
            max_index_before_normal_update = i;
        }
    }

    // reset i to the start of the schedule again
    i = 0;

    llvm::Value *normal = nullptr;

    if (max_index_before_normal_update != ~0) {
        for (; i <= max_index_before_normal_update; ++i) {
            Schedule_entry const &entry = schedule[i];

            Expression_result res = translate_node(
                entry.node,
                m_cur_resolver);

            // use node cache independent of basic block. We know, this is sequential code.
            m_manual_node_value_map[entry.node] = res;

            // if the node is evaluation state dependent, we need to remove it from the manual node
            // value map later again.
            // Support for evaluation state dependency allows reusing independent node results after
            // geometry.normal updates state.normal.
            // Example: ::nvidia::vMaterials::AEC::Masonry::CMU_Running_Half_Bond_Splitface
            if (entry.is_eval_state_dependent) {
                cur_eval_state_results.push_back(entry.node);
            }

            // remember value of normal
            if (entry.has_special_kind(Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL)) {
                // type doesn't matter or fits already?
                if (res.get_value_type() != m_type_mapper.get_float3_type()) {
                    // convert to expected type
                    res = Expression_result::value(
                        ctx.load_and_convert(m_type_mapper.get_float3_type(), res.as_ptr(ctx)));
                }

                normal = res.as_value(ctx);
            }

            // store result in texture results?
            if (entry.texture_result_offset != ~0) {
                llvm::Value *res_value = res.as_value(ctx);

                if (target_is_structured_language()) {
                    store_to_float4_array(
                        res_value,
                        texture_results,
                        entry.texture_result_offset);
                } else {
                    llvm::Value *ptr = ctx.create_simple_gep_in_bounds(
                        texture_results, entry.texture_result_index);
                    ctx.convert_and_store(res_value, ptr);
                }
            }
        }
    }

    // set normal now, if necessary
    if (normal != nullptr) {
        // call state::adapt_normal(normal), if requested
        if (m_use_renderer_adapt_normal) {
            llvm::Function *adapt_normal = get_internal_function(m_int_func_state_adapt_normal);
            llvm::SmallVector<llvm::Value *, 3> args;
            args.push_back(ctx.get_state_parameter());
            if (target_uses_resource_data_parameter()) {
                args.push_back(ctx.get_resource_data_parameter());
            }
            args.push_back(normal);
            normal = call_rt_func(adapt_normal, args);
        }

        // call state::set_normal(normal)
        llvm::Function *set_func = get_internal_function(m_int_func_state_set_normal);
        llvm::Value *set_normal_args[] = {
            ctx.get_state_parameter(),
            normal
        };
        call_rt_func_void(set_func, set_normal_args);

        // clear DAG results, as they may depend on the evaluation state.
        // Reused nodes, for which we know, that they don't depend on the evaluation state
        // will still be available in the manual node value map.
        clear_dag_node_map();

        // the evaluation state changes now, so remove all results depending on it
        // from the manual node value map
        for (auto node : cur_eval_state_results) {
            m_manual_node_value_map.erase(node);
        }
        cur_eval_state_results.clear();
    }

    // translate the remaining scheduled nodes
    for (; i < n; ++i) {
        Schedule_entry const &entry = schedule[i];

        Expression_result res = translate_node(
            entry.node,
            m_cur_resolver);

        // use node cache independent of basic block. We know, this is sequential code.
        m_manual_node_value_map[entry.node] = res;

        // store result in texture results?
        if (entry.texture_result_offset != ~0) {
            llvm::Value *res_value = res.as_value(ctx);

            if (target_is_structured_language()) {
                store_to_float4_array(
                    res_value,
                    texture_results,
                    entry.texture_result_offset);
            } else {
                llvm::Value *ptr = ctx.create_simple_gep_in_bounds(
                    texture_results, entry.texture_result_index);
                ctx.convert_and_store(res_value, ptr);
            }
        }
    }

    // clear manual node value map again
    m_manual_node_value_map.clear();
}

} // mdl
} // mi

