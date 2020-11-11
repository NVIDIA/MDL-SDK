/***************************************************************************************************
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
 **************************************************************************************************/
/// \file

#include "pch.h"

#include <algorithm>

#include <llvm/ADT/SetVector.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Linker/Linker.h>

#include "mdl/compiler/compilercore/compilercore_errors.h"
#include "mdl/codegenerators/generator_dag/generator_dag_lambda_function.h"
#include "mdl/codegenerators/generator_dag/generator_dag_tools.h"
#include "mdl/codegenerators/generator_dag/generator_dag_walker.h"


#include "generator_jit.h"
#include "generator_jit_llvm.h"


namespace mi {
namespace mdl {

namespace {

/// Helper class representing a potential slot in the lambda or texture results buffer.
struct Lambda_result_slot
{
    /// The index of the corresponding lambda expression.
    size_t          expr_lambda_index;

    /// The LLVM return type for the lambda expression.
    llvm::Type      *llvm_ret_type;

    /// The calculated cost of evaluating the lambda expression.
    int             cost;

    /// The size in bytes of the lambda expression result.
    unsigned        size;

    /// The costs used for sorting.
    int             sort_cost;

    /// The constructor.
    Lambda_result_slot(
        size_t expr_lambda_index,
        llvm::Type *llvm_ret_type,
        int cost,
        unsigned size)
    : expr_lambda_index(expr_lambda_index)
    , llvm_ret_type(llvm_ret_type)
    , cost(cost)
    , size(size)
    {
        // when sorting, prefer high cost per byte
        sort_cost = cost * 16 / std::max(size, 1u);
    }
};

/// Helper class to compare lambda result slots.
struct Lambda_result_slot_compare
{
    /// Returns true if 'a' should be placed before 'b'.
    bool operator()(Lambda_result_slot const *a, Lambda_result_slot const *b) const
    {
        return a->sort_cost > b->sort_cost;
    }
};

/// Helper class to calculate the cost of a Lambda function.
class Cost_calculator : public IDAG_ir_visitor
{
public:
    enum Cost
    {
        COST_RESOURCE = 200,           ///< Cost of a reference to a texture, light profile or
                                       ///< BSDF measurement.
        COST_CALL_TEX_ACCESS = 100,    ///< Cost of calls to texture access functions.
        COST_CALL_UNKNOWN = 10,        ///< Cost of a call to a non-intrinsic function.
        COST_CALL = 1,                 ///< Cost of any other call (includes operators)

        MIN_STORE_RESULT_COST = 10     ///< Minimum cost needed for allowing to store the result
                                       ///< in the texture results or the local lambda results.
    };

    /// The constructor.
    Cost_calculator()
    : m_cost(0)
    {
    }

    /// Get the calculated costs.
    int get_cost() const { return m_cost; }

    /// Post-visit a Constant.
    ///
    /// \param cnst  the constant that is visited
    void visit(DAG_constant *cnst) MDL_FINAL
    {
        IValue const *v = cnst->get_value();
        IType const  *t = v->get_type();

        // note: this also collects invalid references ...
        switch (t->get_kind()) {
        case IType::TK_TEXTURE:
        case IType::TK_LIGHT_PROFILE:
        case IType::TK_BSDF_MEASUREMENT:
            m_cost += COST_RESOURCE;
            break;
        default:
            break;
        }
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
        switch (call->get_semantic())
        {
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
            // direct state access is for free for most fields
            break;

        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_COLOR:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT2:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT3:
        case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT4:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_COLOR:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT2:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT3:
        case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT4:
            m_cost += COST_CALL_TEX_ACCESS;
            break;

        case IDefinition::DS_UNKNOWN:
            m_cost += COST_CALL_UNKNOWN;
            break;

        default:
            m_cost += COST_CALL;
            break;
        }
    }

    /// Post-visit a Parameter.
    ///
    /// \param param  the parameter that is visited
    void visit(DAG_parameter *param) MDL_FINAL
    {
        // do nothing
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

    /// Calculate the cost of evaluating the given lambda function.
    ///
    /// \param alloc    the allocator
    /// \param lambda   the lambda function
    static int calc_lambda_cost(IAllocator *alloc, Lambda_function &lambda)
    {
        DAG_ir_walker walker(alloc, /*as_tree=*/false);
        Cost_calculator calculator;

        size_t num_exprs = lambda.get_root_expr_count();
        if (num_exprs > 0) {
            MDL_ASSERT(!"calculating costs for switch lambdas is not supported, yet");
            return 0;
        }

        walker.walk_node(const_cast<DAG_node *>(lambda.get_body()), &calculator);
        return calculator.get_cost();
    }

private:
    int m_cost;
};

/// Helper class to assign expression lambdas to slots in the texture and lambda results
/// and determine the order of evaluation.
class Expr_lambda_scheduler
{
public:
    /// Constructor.
    Expr_lambda_scheduler(
        IAllocator *alloc,
        llvm::LLVMContext &llvm_context,
        llvm::DataLayout const *data_layout,
        Type_mapper &type_mapper,
        llvm::StructType *float3_struct_type,
        unsigned num_texture_results,
        Distribution_function const &dist_func,
        mi::mdl::vector<int>::Type &lambda_result_indices,
        mi::mdl::vector<int>::Type &texture_result_indices,
        mi::mdl::vector<unsigned>::Type &texture_result_offsets,
        llvm::SmallVector<unsigned, 8> &lambda_result_exprs_init,
        llvm::SmallVector<unsigned, 8> &lambda_result_exprs_others,
        llvm::SmallVector<unsigned, 8> &texture_result_exprs)
      : m_alloc(alloc)
      , m_llvm_context(llvm_context)
      , m_data_layout(data_layout)
      , m_type_mapper(type_mapper)
      , m_float3_struct_type(float3_struct_type)
      , m_num_texture_results(num_texture_results)
      , m_dist_func(dist_func)
      , m_lambda_result_indices(lambda_result_indices)
      , m_texture_result_indices(texture_result_indices)
      , m_texture_result_offsets(texture_result_offsets)
      , m_lambda_infos(alloc)
      , m_lambda_slots(alloc)
      , m_lambda_result_exprs_init(lambda_result_exprs_init)
      , m_lambda_result_exprs_others(lambda_result_exprs_others)
      , m_texture_result_exprs(texture_result_exprs)
      , m_text_results_size(0)
      , m_text_results_cur_offs(0)
    {
        size_t expr_lambda_count = dist_func.get_expr_lambda_count();

        // Initialize all indices as not being set.
        m_lambda_result_indices.clear();
        m_lambda_result_indices.resize(expr_lambda_count, -1);
        m_texture_result_indices.clear();
        m_texture_result_indices.resize(expr_lambda_count, -1);
        m_texture_result_offsets.clear();

        m_lambda_infos.reserve(expr_lambda_count);
    }

    /// Schedule the lambda functions and assign them to texture result and lambda result slots
    /// if necessary.
    ///
    /// \param ignore_lambda_results  if true, no lambda results calculations will be scheduled
    void schedule_lambdas(bool ignore_lambda_results)
    {
        size_t expr_lambda_count = m_dist_func.get_expr_lambda_count();

        size_t geometry_normal_index = m_dist_func.get_special_lambda_function_index(
            IDistribution_function::SK_MATERIAL_GEOMETRY_NORMAL);

        // first collect information about all non-constant expression lambdas
        for (size_t i = 0; i < expr_lambda_count; ++i) {
            mi::base::Handle<mi::mdl::ILambda_function> expr_lambda(m_dist_func.get_expr_lambda(i));
            Lambda_function &lambda = *impl_cast<Lambda_function>(expr_lambda.get());

            // determine the cost of calculating the result of the expression lambda
            int cost = Cost_calculator::calc_lambda_cost(m_alloc, lambda);

            m_lambda_infos.push_back(Lambda_info(m_alloc));
            Lambda_info &cur = m_lambda_infos.back();

            // due to construction in backends_backends.cpp,
            // all required lambdas are already available
            cur.calc_dependencies(lambda.get_body(), m_lambda_infos);

            // the result of geometry.normal will be stored in state.normal
            if (i == geometry_normal_index)
                continue;

            // not worth storing the result?
            if (cost < Cost_calculator::MIN_STORE_RESULT_COST)
                continue;

            // constants are neither materialized as functions nor stored in the lambda results
            if (is<DAG_constant>(lambda.get_body()))
                continue;

            // don't store matrices in lambda results, they are far too expensive
            IType const *mdl_type = lambda.get_return_type();
            if (is<IType_matrix>(m_type_mapper.skip_deriv_type(mdl_type)))
                continue;

            // determine the size of the result
            llvm::Type *lambda_ret_type = m_type_mapper.lookup_type(m_llvm_context, mdl_type);

            // replace lambda float3 types by float3 struct type used in libbsdf
            if (lambda_ret_type == m_type_mapper.get_float3_type())
                lambda_ret_type = m_float3_struct_type;

            unsigned res_alloc_size = unsigned(m_data_layout->getTypeAllocSize(lambda_ret_type));

            // we want to materialize the result, so register a slot
            m_lambda_slots.push_back(Lambda_result_slot(
                i, lambda_ret_type, cost, res_alloc_size));
            cur.lambda_slot_index = m_lambda_slots.size() - 1;

            // set local costs and add to the total ones
            cur.local_cost = cost;
            cur.total_cost += cost;
            cur.local_size = res_alloc_size;
            cur.total_size += res_alloc_size;
        }

        // nothing to schedule? -> done
        if (m_lambda_slots.empty())
            return;

        // sort the expression lambdas by cost per byte.
        mi::mdl::vector<Lambda_result_slot *>::Type sorted_lambda_slots(m_alloc);
        sorted_lambda_slots.resize(m_lambda_slots.size());
        for (size_t i = 0, n = m_lambda_slots.size(); i < n; ++i)
            sorted_lambda_slots[i] = &m_lambda_slots[i];
        std::sort(
            sorted_lambda_slots.begin(), sorted_lambda_slots.end(),
            Lambda_result_slot_compare());

        m_text_results_size = size_t(m_num_texture_results * m_data_layout->getTypeAllocSize(
                m_type_mapper.get_float4_type()));

        // first try to add the results of the expression lambdas to texture results,
        // then to the lambda results
        for (size_t i = 0, n = sorted_lambda_slots.size(); i < n; ++i) {
            size_t expr_index = sorted_lambda_slots[i]->expr_lambda_index;

            // already calculated? -> skip
            if (m_texture_result_indices[expr_index] != -1)
                continue;

            Lambda_info &lambda_info = m_lambda_infos[expr_index];
            Lambda_info::Index_set &deps = lambda_info.dep_expr_indices;

            // calculate required size including dependencies which have not yet been added
            // (estimate as it ignores alignment)
            size_t required_size = lambda_info.local_size;
            bool needs_deps = false;
            for (Lambda_info::Index_set::const_iterator it = deps.begin(), end = deps.end();
                    it != end; ++it) {
                if (m_texture_result_indices[*it] == -1) {
                    // not calculated, yet
                    required_size += m_lambda_infos[*it].local_size;
                    needs_deps = true;
                }
            }

            // check whether we still have (roughly) enough space for the texture results
            if (m_text_results_cur_offs + required_size <= m_text_results_size) {
                bool calc_deps_failed = false;

                // not all dependencies available, yet?
                if (needs_deps) {
                    // Note: Index_set is already sorted and results can only depend on
                    //       results with smaller indices
                    for (Lambda_info::Index_set::const_iterator it = deps.begin(),
                            end = deps.end(); it != end; ++it) {
                        if (m_texture_result_indices[*it] == -1) {
                            if (!add_texture_result_entry(*it)) {
                                calc_deps_failed = true;
                                break;
                            }
                        }
                    }
                }

                // all dependencies available now?
                // -> try to add the lambda expression to the texture results
                if (!calc_deps_failed) {
                    add_texture_result_entry(unsigned(expr_index));
                }
            }
        }

        // if we should not schedule lambda results, we're done here
        if (ignore_lambda_results)
            return;

        // if geometry.normal has to be calculated, collect required lambda results
        // for use in the bsdf init function
        if (geometry_normal_index != ~0 && m_texture_result_indices[geometry_normal_index] == -1) {
            Lambda_info &lambda_info = m_lambda_infos[geometry_normal_index];
            add_lambda_result_dep_entries(lambda_info, /*for_init_func=*/ true);
        }

        // determine which lambda results are required by the other bsdf functions
        // after the init function (so especially without geometry.normal)
        // TODO: not only use lambda_result for init and all others, but also per main function
        Lambda_info df_info(m_alloc);
        for (size_t i = 0, n = m_dist_func.get_main_function_count(); i < n; ++i) {
            mi::base::Handle<ILambda_function> main_func(m_dist_func.get_main_function(i));
            df_info.calc_dependencies(main_func->get_body(), m_lambda_infos);

            add_lambda_result_dep_entries(df_info, /*for_init_func=*/ false);
        }
    }

    /// Create the texture results type.
    llvm::StructType *create_texture_results_type()
    {
        return llvm::StructType::create(
            m_llvm_context,
            m_texture_result_types,
            "struct.Texture_result_types",
            /*is_packed=*/ false);
    }

    /// Create the lambda results type.
    llvm::StructType *create_lambda_results_type()
    {
        return llvm::StructType::create(
            m_llvm_context,
            m_lambda_result_types,
            "struct.Lambda_result_types",
            /*is_packed=*/ false);
    }

private:
    /// Helper structure collecting information about expression lambdas and their dependencies.
    struct Lambda_info {
        typedef set<unsigned>::Type Index_set;

        /// Local cost of the lambda function without costs of dependencies.
        int local_cost;

        /// Total costs of the lambda function including costs of dependencies.
        int total_cost;

        /// Size of the lambda function result.
        unsigned local_size;

        /// Total size requirements for the result including dependencies (ignores alignment).
        unsigned total_size;

        /// Set of indices of expression lambdas required to calculate this expression lambda.
        Index_set dep_expr_indices;

        /// Corresponding index in the lambda slot list.
        unsigned lambda_slot_index;

        /// Constructor.
        Lambda_info(mi::mdl::IAllocator *alloc)
            : local_cost(0)
            , total_cost(0)
            , local_size(0)
            , total_size(0)
            , dep_expr_indices(Index_set::key_compare(), alloc)
            , lambda_slot_index(0)
        {}

        /// Add a dependency to another expression lambda.
        ///
        /// \param index  The index of the other expression lambda.
        /// \param infos  The lambda information list.
        void add_dependency(unsigned index, mi::mdl::vector<Lambda_info>::Type &infos) {
            Lambda_info const &o = infos[index];

            // add the lambda itself, if non-zero local cost (== gets materialized)
            if (o.local_cost > 0 && dep_expr_indices.insert(index).second == true) {
                // really added, update totals
                total_cost += o.local_cost;
                total_size += o.local_size;
            }

            // add all its dependencies
            for (Index_set::const_iterator it = o.dep_expr_indices.begin(),
                    end = o.dep_expr_indices.end(); it != end; ++it) {
                Lambda_info &cur = infos[*it];

                if (dep_expr_indices.insert(*it).second == true) {
                    // really added, update totals
                    total_cost += cur.local_cost;
                    total_size += cur.local_size;
                }
            }
        }

        /// Calculate the dependencies based on the given expression.
        ///
        /// \param expr   The expression to walk.
        /// \param infos  The lambda information list.
        void calc_dependencies(DAG_node const *expr, mi::mdl::vector<Lambda_info>::Type &infos) {
            switch (expr->get_kind()) {
            case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = mi::mdl::cast<DAG_temporary>(expr);
                expr = t->get_expr();
                calc_dependencies(expr, infos);
                return;
            }
            case DAG_node::EK_CONSTANT:
            case DAG_node::EK_PARAMETER:
                return;
            case DAG_node::EK_CALL:
            {
                DAG_call const *call = mi::mdl::cast<DAG_call>(expr);
                if (call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA) {
                    size_t lambda_index = strtoul(call->get_name(), NULL, 10);
                    add_dependency(unsigned(lambda_index), infos);
                    return;
                }

                int n_args = call->get_argument_count();
                for (int i = 0; i < n_args; ++i)
                    calc_dependencies(call->get_argument(i), infos);
                return;
            }
            }
            MDL_ASSERT(!"Unsupported DAG node kind");
            return;
        }
    };

    /// Ensure that all dependencies for the referenced lambda are available.
    ///
    /// \param lambda_info    The information structure of the lambda.
    /// \param for_init_func  True, if this lambda result entry will be used by the init function.
    void add_lambda_result_dep_entries(Lambda_info &lambda_info, bool for_init_func)
    {
        Lambda_info::Index_set &deps = lambda_info.dep_expr_indices;
        for (Lambda_info::Index_set::const_iterator it = deps.begin(),
                end = deps.end(); it != end; ++it) {
            // not available as texture result, yet?
            if (m_texture_result_indices[*it] == -1) {
                add_lambda_result_entry(*it, for_init_func);
            }
        }
    }

    /// Add a texture results entry, if possible.
    ///
    /// \param expr_index   The index of the expression lambda.
    ///
    /// \returns True, if it was possible to add it as a texture result.
    bool add_texture_result_entry(unsigned expr_index)
    {
        Lambda_result_slot &slot = m_lambda_slots[m_lambda_infos[expr_index].lambda_slot_index];
        llvm::Type *result_type = slot.llvm_ret_type;

        // check whether the result still fits and the alignment is compatible
        // (the beginning of the texture results is float4 = 16-byte aligned)
        size_t res_align = size_t(m_data_layout->getABITypeAlignment(result_type));
        size_t res_alloc_size = size_t(m_data_layout->getTypeAllocSize(result_type));
        size_t new_offs = (m_text_results_cur_offs + res_align - 1) & ~(res_align - 1);
        if (res_align <= 16 && new_offs + res_alloc_size <= m_text_results_size) {
            m_text_results_cur_offs = new_offs + res_alloc_size;

            // add to texture results
            m_texture_result_types.push_back(result_type);
            m_texture_result_exprs.push_back(unsigned(expr_index));
            m_texture_result_indices[expr_index] =
                int(m_texture_result_types.size() - 1);
            m_texture_result_offsets.push_back(unsigned(new_offs));
            return true;
        }
        return false;
    }

    /// Add a lambda results entry.
    ///
    /// \param expr_index     The index of the expression lambda.
    /// \param for_init_func  True, if this lambda result entry will be used by the init function.
    void add_lambda_result_entry(unsigned expr_index, bool for_init_func)
    {
        // the lambda results are all collected in one struct type (init and other bsdf functions)
        // because functions using lambda results may be used in both cases

        // only add to lambda result struct type, if slot is not chosen, yet
        if (m_lambda_result_indices[expr_index] == -1) {
            Lambda_result_slot &slot = m_lambda_slots[m_lambda_infos[expr_index].lambda_slot_index];
            llvm::Type *result_type = slot.llvm_ret_type;

            m_lambda_result_types.push_back(result_type);
            m_lambda_result_indices[expr_index] = int(m_lambda_result_types.size() - 1);
        }

        if (for_init_func)
            m_lambda_result_exprs_init.push_back(expr_index);
        else
            m_lambda_result_exprs_others.push_back(expr_index);
    }

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The LLVM context to use.
    llvm::LLVMContext &m_llvm_context;

    /// The LLVM data layout.
    llvm::DataLayout const *m_data_layout;

    /// The MDL to LLVM type mapper.
    Type_mapper &m_type_mapper;

    /// A float3 struct type used in libbsdf.
    llvm::StructType *m_float3_struct_type;

    /// The number of texture result entries.
    unsigned m_num_texture_results;

    /// The distribution function.
    Distribution_function const &m_dist_func;

    /// Array which maps expression lambda indices to lambda result indices.
    /// For expression lambdas without a lambda result entry the array contains -1.
    mi::mdl::vector<int>::Type &m_lambda_result_indices;

    /// Array which maps expression lambda indices to texture result indices.
    /// For expression lambdas without a texture result entry the array contains -1.
    mi::mdl::vector<int>::Type &m_texture_result_indices;

    /// Array which maps expression lambda indices to texture result offsets.
    mi::mdl::vector<unsigned>::Type &m_texture_result_offsets;

    /// The list of expression lambda infos.
    mi::mdl::vector<Lambda_info>::Type m_lambda_infos;

    /// The list of lambda result slots.
    mi::mdl::vector<Lambda_result_slot>::Type m_lambda_slots;

    /// The list of lambda result types.
    llvm::SmallVector<llvm::Type *, 8> m_lambda_result_types;

    /// The list of texture result types.
    llvm::SmallVector<llvm::Type *, 8> m_texture_result_types;

    /// The list of lambda result expression indices used by the init function.
    llvm::SmallVector<unsigned, 8> &m_lambda_result_exprs_init;

    /// The list of lambda result expression indices used by the other bsdf functions (not init).
    llvm::SmallVector<unsigned, 8> &m_lambda_result_exprs_others;

    /// The list of texture result expression indices.
    llvm::SmallVector<unsigned, 8> &m_texture_result_exprs;

    /// The maximum size of the texture results.
    size_t m_text_results_size;

    /// The current offset within the textures results data structure.
    size_t m_text_results_cur_offs;
};

} // anonymous namespace


/// Helper class for code generation of df::bsdf_component elements.
class Df_component_info
{
public:
    /// Constructor.
    ///
    /// \param code_gen  The code generator.
    Df_component_info(LLVM_code_generator &code_gen, IType::Kind kind)
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
    llvm::Function *get_df_function(LLVM_code_generator::Distribution_function_state state)
    {
        // no components registered -> black_bsdf()
        if (m_component_dfs.empty()) {
            std::string func_name;
            switch (m_kind)
            {
                case IType::TK_BSDF:
                case IType::TK_HAIR_BSDF:
                    func_name = "gen_black_bsdf";
                    break;

                case IType::TK_EDF:
                    func_name = "gen_black_edf";
                    break;

                default:
                    MDL_ASSERT(!"Invalid distribution kind for getting a DF function");
                    return NULL;
            }

            func_name += LLVM_code_generator::get_dist_func_state_suffix(state);
            llvm::Function *black_bsdf_func =
                m_code_gen.get_llvm_module()->getFunction(func_name);
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
        if (m_df_funcs[index] != NULL)
            return m_df_funcs[index];

        // no, temporarily set given state as current and instantiate the BSDFs
        LLVM_code_generator::Distribution_function_state old_state = m_code_gen.m_dist_func_state;
        m_code_gen.m_dist_func_state = state;

        llvm::SmallVector<llvm::Function *, 8> comp_funcs;
        for (DAG_node const *node : m_component_dfs) {
            comp_funcs.push_back(m_code_gen.instantiate_df(node));
        }

        m_code_gen.m_dist_func_state = old_state;

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

        llvm::Function *switch_func = llvm::Function::Create(
            switch_func_type,
            llvm::GlobalValue::InternalLinkage,
            "switch_func",
            m_code_gen.get_llvm_module());
        m_code_gen.set_llvm_function_attributes(switch_func);

        llvm::BasicBlock *start_block =
            llvm::BasicBlock::Create(llvm_context, "start", switch_func);
        llvm::BasicBlock *end_block = llvm::BasicBlock::Create(llvm_context, "end", switch_func);

        llvm::IRBuilder<> builder(start_block);
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


// Create the BSDF function types using the BSDF data types from the already linked libbsdf
// module.
void LLVM_code_generator::create_bsdf_function_types()
{
    // fetch the BSDF data types from the already linked libbsdf

    m_type_bsdf_sample_data = m_module->getTypeByName("struct.BSDF_sample_data");
    m_type_bsdf_evaluate_data = m_module->getTypeByName("struct.BSDF_evaluate_data");
    m_type_bsdf_pdf_data = m_module->getTypeByName("struct.BSDF_pdf_data");
    m_type_bsdf_auxiliary_data = m_module->getTypeByName("struct.BSDF_auxiliary_data");

    // create function types for the BSDF functions

    llvm::Type *ret_tp = m_type_mapper.get_void_type();
    llvm::Type *second_param_type;
    if (target_supports_lambda_results_parameter())
        second_param_type = m_type_mapper.get_exec_ctx_ptr_type();
    else
        second_param_type = m_type_mapper.get_state_ptr_type(m_state_mode);
    llvm::Type *float3_struct_ptr_type = Type_mapper::get_ptr(m_float3_struct_type);

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
    //     float3 *inherited_weight)

    llvm::Type *arg_types_eval[] = {
        Type_mapper::get_ptr(m_type_bsdf_evaluate_data),
        second_param_type,
        float3_struct_ptr_type,
        float3_struct_ptr_type
    };

    m_type_bsdf_evaluate_func = llvm::FunctionType::get(ret_tp, arg_types_eval, false);

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
    //     float3 *inherited_weight)

    llvm::Type *arg_types_auxiliary[] = {
        Type_mapper::get_ptr(m_type_bsdf_auxiliary_data),
        second_param_type,
        float3_struct_ptr_type,
        float3_struct_ptr_type
    };

    m_type_bsdf_auxiliary_func = llvm::FunctionType::get(ret_tp, arg_types_auxiliary, false);
}


// Create the EDF function types using the EDF data types from the already linked libbsdf
// module.
void LLVM_code_generator::create_edf_function_types()
{
    // fetch the EDF data types from the already linked libbsdf

    m_type_edf_sample_data = m_module->getTypeByName("struct.EDF_sample_data");
    m_type_edf_evaluate_data = m_module->getTypeByName("struct.EDF_evaluate_data");
    m_type_edf_pdf_data = m_module->getTypeByName("struct.EDF_pdf_data");
    m_type_edf_auxiliary_data = m_module->getTypeByName("struct.EDF_auxiliary_data");

    // create function types for the EDF functions

    llvm::Type *ret_tp = m_type_mapper.get_void_type();
    llvm::Type *second_param_type;
    if (target_supports_lambda_results_parameter())
        second_param_type = m_type_mapper.get_exec_ctx_ptr_type();
    else
        second_param_type = m_type_mapper.get_state_ptr_type(m_state_mode);
    llvm::Type *float3_struct_ptr_type = Type_mapper::get_ptr(m_float3_struct_type);

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
    //     float3 *inherited_weight)

    llvm::Type *arg_types_eval[] = {
        Type_mapper::get_ptr(m_type_edf_evaluate_data),
        second_param_type,
        float3_struct_ptr_type,
        float3_struct_ptr_type
    };

    m_type_edf_evaluate_func = llvm::FunctionType::get(ret_tp, arg_types_eval, false);

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
    //     float3 *inherited_weight)

    llvm::Type *arg_types_auxiliary[] = {
        Type_mapper::get_ptr(m_type_edf_auxiliary_data),
        second_param_type,
        float3_struct_ptr_type,
        float3_struct_ptr_type
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
    size_t                      *main_function_indices)
{
    m_dist_func = &dist_func;

#if 0
    static int dumpid = 0;
    std::string dumpname("df");
    dumpname += std::to_string(dumpid++);
    dumpname += ".gv";
    m_dist_func->dump(dumpname.c_str());
#endif

    IAllocator *alloc = m_arena.get_allocator();

    mi::base::Handle<ILambda_function> root_lambda_handle(m_dist_func->get_root_lambda());
    Lambda_function const *root_lambda = impl_cast<Lambda_function>(root_lambda_handle.get());

    create_captured_argument_struct(m_llvm_context, *root_lambda);

    // must be done before load_and_link_libbsdf() because of calls to texture runtime
    if (m_texruntime_with_derivs)
        m_deriv_infos = dist_func.get_derivative_infos();

    // create a module for the functions
    if (m_module == NULL) {
        create_module("lambda_mod", NULL);

        // initialize the module with user code
        if (!init_user_modules()) {
            // drop the module and give up
            drop_llvm_module(m_module);
            m_dist_func = NULL;
            return NULL;
        }

        if (m_target_lang == TL_HLSL) {
            init_hlsl_code_gen();
        }
    }

    // load libbsdf into the current module, if it was not initialized, yet
    if (m_type_bsdf_sample_data == NULL && !load_and_link_libbsdf(
            m_link_libbsdf_df_handle_slot_mode)) {
        // drop the module and give up
        drop_llvm_module(m_module);
        m_dist_func = NULL;
        return NULL;
    }

    create_resource_tables(*root_lambda);

    // determine which expression lambdas will be put into the init function (texture results)
    // and which will be calculated in the sample, eval and pdf functions (lambda results)
    llvm::SmallVector<unsigned, 8> texture_result_exprs;
    llvm::SmallVector<unsigned, 8> lambda_result_exprs_init;
    llvm::SmallVector<unsigned, 8> lambda_result_exprs_others;
    Expr_lambda_scheduler lambda_sched(
        get_allocator(),
        m_llvm_context,
        get_target_layout_data(),
        m_type_mapper,
        m_float3_struct_type,
        m_num_texture_results,
        *m_dist_func,
        m_lambda_result_indices,
        m_texture_result_indices,
        m_texture_result_offsets,
        lambda_result_exprs_init,
        lambda_result_exprs_others,
        texture_result_exprs);

    // for HLSL, we don't support lambda results
    lambda_sched.schedule_lambdas(/*ignore_lambda_results=*/ m_target_lang == TL_HLSL);

    m_texture_results_struct_type = lambda_sched.create_texture_results_type();
    m_lambda_results_struct_type = lambda_sched.create_lambda_results_type();

    size_t expr_lambda_count = m_dist_func->get_expr_lambda_count();

    // now generate LLVM functions for all non-constant expression lambdas
    for (size_t i = 0; i < expr_lambda_count; ++i) {
        mi::base::Handle<mi::mdl::ILambda_function> expr_lambda(m_dist_func->get_expr_lambda(i));
        Lambda_function &lambda = *impl_cast<Lambda_function>(expr_lambda.get());

        // constants are neither materialized as functions nor stored in the lambda results
        if (is<DAG_constant>(lambda.get_body()))
            continue;

        reset_lambda_state();

        // generic functions return the result by reference if supported
        m_lambda_force_sret         = target_supports_sret_for_lambda();

        // generic functions always includes a render state in its interface
        m_lambda_force_render_state = true;

        LLVM_context_data *ctx_data = get_or_create_context_data(&lambda);
        llvm::Function    *func     = ctx_data->get_function();
        unsigned          flags     = ctx_data->get_function_flags();

        char func_name[48];
        snprintf(func_name, sizeof(func_name), "expr_lambda_%u_", unsigned(i));
        func->setName(func_name);

        // force expression lambda function to be internal
        func->setLinkage(llvm::GlobalValue::InternalLinkage);

        if (is_always_inline_enabled())
            func->addFnAttr(llvm::Attribute::AlwaysInline);

        // if the result is not returned as an out parameter, mark the lambda function as read-only
        if ((flags & LLVM_context_data::FL_SRET) == 0) {
            func->setOnlyReadsMemory();
        }

        // ensure the function is finished by putting it into a block
        {
            Function_instance inst(alloc, &lambda);
            Function_context context(alloc, *this, inst, func, flags);

            // translate function body
            Expression_result res = translate_node(context, lambda.get_body(), resolver);
            context.create_return(res.as_value(context));
        }

        m_module_lambda_index_map[&lambda] = m_module_lambda_funcs.size();
        m_module_lambda_funcs.push_back(func);
    }

    reset_lambda_state();

    // let return type decide to allow init function without structure return parameter
    m_lambda_force_sret         = false;

    // distribution functions always includes a render state in its interface
    m_lambda_force_render_state = true;

    // the BSDF API functions create the lambda results they use, so no lambda results parameter
    m_lambda_force_no_lambda_results = true;

    // create init function

    {
        Function_instance inst(get_allocator(), root_lambda);

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

        if (is_always_inline_enabled())
            func->addFnAttr(llvm::Attribute::AlwaysInline);

        // remember function as an exported function
        IGenerated_code_executable::Function_kind func_kind =
            IGenerated_code_executable::FK_DF_INIT;

        IGenerated_code_executable::Distribution_kind dist_kind =
            IGenerated_code_executable::DK_NONE;

        // if there is only one main function, mark the init function with the corresponding
        // distribution kind for backward compatibility
        if (dist_func.get_main_function_count() == 1) {
            mi::base::Handle<mi::mdl::ILambda_function> main_func(m_dist_func->get_main_function(0));
            Lambda_function &lambda = *impl_cast<Lambda_function>(main_func.get());

            switch (lambda.get_body()->get_type()->get_kind()) {
            case IType::TK_BSDF:      dist_kind = IGenerated_code_executable::DK_BSDF;      break;
            case IType::TK_HAIR_BSDF: dist_kind = IGenerated_code_executable::DK_HAIR_BSDF; break;
            case IType::TK_EDF:       dist_kind = IGenerated_code_executable::DK_EDF;       break;
            default:
                break;
            }
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
        for (size_t i = 0, n = dist_func.get_df_handle_count(); i < n; ++i) {
            exp_func.add_df_handle(dist_func.get_df_handle(i));
        }

        Function_context context(alloc, *this, inst, func, flags);

        // translate the init function
        translate_distribution_function_init(
            context, texture_result_exprs, lambda_result_exprs_init);

        context.create_void_return();
    }

    // generic functions return the result by reference if supported
    m_lambda_force_sret = target_supports_sret_for_lambda();

    for (size_t i = 0, n = dist_func.get_main_function_count(); i < n; ++i) {
        m_cur_main_func_index = i;

        mi::base::Handle<mi::mdl::ILambda_function> main_func(m_dist_func->get_main_function(i));
        Lambda_function &lambda = *impl_cast<Lambda_function>(main_func.get());

        IGenerated_code_executable::Distribution_kind dist_kind =
            IGenerated_code_executable::DK_NONE;
        switch (lambda.get_body()->get_type()->get_kind()) {
        case IType::TK_BSDF:      dist_kind = IGenerated_code_executable::DK_BSDF;      break;
        case IType::TK_HAIR_BSDF: dist_kind = IGenerated_code_executable::DK_HAIR_BSDF; break;
        case IType::TK_EDF:       dist_kind = IGenerated_code_executable::DK_EDF;       break;
        default:
            break;
        }

        if (main_function_indices)
            main_function_indices[i] = m_exported_func_list.size();

        llvm::Twine base_name(lambda.get_name());
        Function_instance inst(get_allocator(), &lambda);

        // non-distribution function?
        if (dist_kind == IGenerated_code_executable::DK_NONE) {
            m_dist_func_state = Distribution_function_state(DFSTATE_NONE);

            LLVM_context_data *ctx_data = get_or_create_context_data(&lambda);
            llvm::Function    *func     = ctx_data->get_function();
            unsigned          flags     = ctx_data->get_function_flags();

            if (is_always_inline_enabled())
                func->addFnAttr(llvm::Attribute::AlwaysInline);

            m_exported_func_list.push_back(
                Exported_function(
                    get_allocator(),
                    func,
                    IGenerated_code_executable::DK_NONE,
                    lambda.get_execution_context() == ILambda_function::LEC_ENVIRONMENT
                        ? IGenerated_code_executable::FK_ENVIRONMENT
                        : IGenerated_code_executable::FK_LAMBDA,
                    m_captured_args_type != NULL ? next_arg_block_index : ~0));

            Function_context context(alloc, *this, inst, func, flags);

            // translate function body
            Expression_result res = translate_node(context, lambda.get_body(), resolver);
            context.create_return(res.as_value(context));

            continue;
        }

        // a distribution function

        llvm::GlobalVariable *mat_data_global = NULL;

        // create one LLVM function for each distribution function state
        for (int i = DFSTATE_SAMPLE; i < DFSTATE_END_STATE; ++i)
        {
            m_dist_func_state = Distribution_function_state(i);

            // we cannot use get_or_create_context_data here, because we need to force the creation
            // of a new function here, as the (const) root_lambda cannot be changed to reflect the
            // different states
            LLVM_context_data *ctx_data = declare_lambda(&lambda);
            m_context_data[inst] = ctx_data;

            llvm::Function *func = ctx_data->get_function();
            llvm_funcs.push_back(func);
            unsigned flags = ctx_data->get_function_flags();

            // set proper function name according to distribution function state
            func->setName(base_name + get_dist_func_state_suffix());

            if (is_always_inline_enabled())
                func->addFnAttr(llvm::Attribute::AlwaysInline);

            // remember function as an exported function
            IGenerated_code_executable::Function_kind func_kind =
                IGenerated_code_executable::FK_INVALID;
            switch (i) {
            case DFSTATE_SAMPLE:    func_kind = IGenerated_code_executable::FK_DF_SAMPLE;    break;
            case DFSTATE_EVALUATE:  func_kind = IGenerated_code_executable::FK_DF_EVALUATE;  break;
            case DFSTATE_PDF:       func_kind = IGenerated_code_executable::FK_DF_PDF;       break;
            case DFSTATE_AUXILIARY: func_kind = IGenerated_code_executable::FK_DF_AUXILIARY; break;
            default:
                MDL_ASSERT(!"Unexpected DF state");
                break;
            }

            // skip the auxiliary functions if deactivated
            if (!m_enable_auxiliary && i == DFSTATE_AUXILIARY)
                continue;

            m_exported_func_list.push_back(
                Exported_function(
                    get_allocator(),
                    func,
                    dist_kind,
                    func_kind,
                    m_captured_args_type != NULL ? next_arg_block_index : ~0));

            Exported_function &exp_func = m_exported_func_list.back();
            for (size_t i = 0, n = dist_func.get_main_func_df_handle_count(m_cur_main_func_index);
                    i < n; ++i) {
                exp_func.add_df_handle(dist_func.get_main_func_df_handle(m_cur_main_func_index, i));
            }

            Function_context context(alloc, *this, inst, func, flags);

            // translate the distribution function
            translate_distribution_function(
                context, lambda.get_body(), lambda_result_exprs_others, mat_data_global);
            context.create_void_return();
        }
    }

    // if we are compiling with derivatives, all waiting functions need to be compiled now,
    // to give them access to the derivative infos
    if (m_deriv_infos)
        compile_waiting_functions();

    // reset some fields
    m_deriv_infos = NULL;
    m_dist_func = NULL;
    m_cur_main_func_index = 0;
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
        case DFSTATE_INIT:      return "_init";
        case DFSTATE_SAMPLE:    return "_sample";
        case DFSTATE_EVALUATE:  return "_evaluate";
        case DFSTATE_PDF:       return "_pdf";
        case DFSTATE_AUXILIARY: return "_auxiliary";
        default:
            MDL_ASSERT(!"Invalid distribution function state");
            return "";
    }
}

// Returns the distribution function state requested by the given call.
LLVM_code_generator::Distribution_function_state
LLVM_code_generator::get_dist_func_state_from_call(llvm::CallInst *call)
{
    llvm::FunctionType *func_type = llvm::cast<llvm::FunctionType>(
        call->getCalledValue()->getType()->getPointerElementType());
    llvm::Type *df_data_type =
        func_type->getParamType(0)->getPointerElementType();

    if (df_data_type == m_type_bsdf_sample_data || df_data_type == m_type_edf_sample_data)
        return DFSTATE_SAMPLE;
    else if (df_data_type == m_type_bsdf_evaluate_data || df_data_type == m_type_edf_evaluate_data)
        return DFSTATE_EVALUATE;
    else if (df_data_type == m_type_bsdf_pdf_data || df_data_type == m_type_edf_pdf_data)
        return DFSTATE_PDF;
    else if (df_data_type == m_type_bsdf_auxiliary_data || df_data_type == m_type_edf_auxiliary_data)
        return DFSTATE_AUXILIARY;

    MDL_ASSERT(!"Invalid distribution function type called");
    return DFSTATE_NONE;
}

// Get the BSDF function for the given semantics and the current distribution function state
// from the BSDF library.
llvm::Function *LLVM_code_generator::get_libbsdf_function(DAG_call const *dag_call)
{
    IDefinition::Semantics sema = dag_call->get_semantic();
    IType::Kind kind = dag_call->get_type()->get_kind();

    std::string func_name;

    std::string suffix = "";

    // check for tint(color, color, bsdf) overload
    if (sema == IDefinition::DS_INTRINSIC_DF_TINT && dag_call->get_argument_count() == 3)
        suffix = "_rt";

    switch (kind)
    {
        case IType::Kind::TK_BSDF: suffix += "_bsdf"; break;
        case IType::Kind::TK_HAIR_BSDF: suffix += "_hair_bsdf"; break;
        case IType::Kind::TK_EDF:  suffix += "_edf"; break;
        default: break;
    }


    #define SEMA_CASE(val, name)  case IDefinition::val: func_name = name; break;

    switch (sema) {
        SEMA_CASE(DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF,
                  "diffuse_reflection_bsdf")
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
        SEMA_CASE(DS_INTRINSIC_DF_MEASURED_BSDF,
                  "measured_bsdf")

        SEMA_CASE(DS_INTRINSIC_DF_DIFFUSE_EDF,
                  "diffuse_edf")
        SEMA_CASE(DS_INTRINSIC_DF_MEASURED_EDF,
                  "measured_edf")
        SEMA_CASE(DS_INTRINSIC_DF_SPOT_EDF,
                  "spot_edf")

        // Unsupported: DS_INTRINSIC_DF_ANISOTROPIC_VDF

        SEMA_CASE(DS_INTRINSIC_DF_NORMALIZED_MIX,
                  "normalized_mix" + suffix)
        SEMA_CASE(DS_INTRINSIC_DF_CLAMPED_MIX,
                  "clamped_mix" + suffix)
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
                  "directional_factor")
        SEMA_CASE(DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR,
                  "measured_curve_factor")
        SEMA_CASE(DS_INTRINSIC_DF_MEASURED_FACTOR,
                  "measured_factor")

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

    return m_module->getFunction("gen_" + func_name + get_dist_func_state_suffix());
}

// Determines the semantics for a libbsdf df function name.
IDefinition::Semantics LLVM_code_generator::get_libbsdf_function_semantics(llvm::StringRef name)
{
    llvm::StringRef basename;
    if (name.endswith("_sample"))
        basename = name.drop_back(7);
    else if (name.endswith("_evaluate"))
        basename = name.drop_back(9);
    else if (name.endswith("_pdf"))
        basename = name.drop_back(4);
    else if (name.endswith("_auxiliary"))
        basename = name.drop_back(10);
    else
        return IDefinition::DS_UNKNOWN;

    if (basename.endswith("_mix_bsdf"))
        basename = basename.drop_back(5);
    if (basename.endswith("_mix_edf"))
        basename = basename.drop_back(4);

    if (basename == "black_bsdf")
        return IDefinition::DS_INVALID_REF_CONSTRUCTOR;
    if (basename == "black_edf")
        return IDefinition::DS_INVALID_REF_CONSTRUCTOR;

    // df::tint(color, color, bsdf) overload?
    if (basename == "tint_rt_bsdf")
        return IDefinition::DS_INTRINSIC_DF_TINT;

    // df::tint(color, edf) overload?
    if (basename == "tint_edf")
        return IDefinition::DS_INTRINSIC_DF_TINT;

    // df::tint(color, bsdf) overload?
    if (basename == "tint_bsdf")
        return IDefinition::DS_INTRINSIC_DF_TINT;

    // df::tint(color, hair_bsdf) overload?
    if (basename == "tint_hair_bsdf")
        return IDefinition::DS_INTRINSIC_DF_TINT;

    string builtin_name("::df::", get_allocator());
    builtin_name.append(basename.data(), basename.size());

    return m_compiler->get_builtin_semantic(builtin_name.c_str());
}

// Check whether the given parameter of the given df function is an array parameter.
bool LLVM_code_generator::is_libbsdf_array_parameter(IDefinition::Semantics sema, int df_param_idx)
{
    switch (sema)
    {
        case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR:
        case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER:
            return df_param_idx == 0;

        default:
            return false;
    }
}

// Translates a potential runtime call in a libbsdf function to a call to the according
// intrinsic, converting the arguments as necessary.
bool LLVM_code_generator::translate_libbsdf_runtime_call(
    llvm::CallInst *call,
    llvm::BasicBlock::iterator &ii,
    Function_context &ctx)
{
    unsigned num_params_eaten = 0;

    llvm::Function *called_func = call->getCalledFunction();
    if (called_func == NULL)
        return true;   // ignore indirect function invocation

    llvm::StringRef func_name = called_func->getName();
    if (!func_name.startswith("_Z") || !called_func->isDeclaration())
        return true;   // ignore non-mangled functions and functions with definitions

    // try to resolve the function name to the LLVM function of an intrinsic

    string demangled_name(get_allocator());
    MDL_name_mangler mangler(get_allocator(), demangled_name);
    if (!mangler.demangle(func_name.data(), func_name.size()))
        demangled_name.assign(func_name.data(), func_name.size());

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
    bool handled = false;

    if (demangled_name.compare(0, 9, "::state::") == 0)
    {
        // special case of an internal function not available in MDL?
        if (demangled_name == "::state::set_normal(float3)")
        {
            func = get_internal_function(m_int_func_state_set_normal);

            Function_instance inst(
                get_allocator(), reinterpret_cast<size_t>(m_int_func_state_set_normal));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::get_texture_results()")
        {
            func = get_internal_function(m_int_func_state_get_texture_results);

            Function_instance inst(
                get_allocator(), reinterpret_cast<size_t>(m_int_func_state_get_texture_results));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::get_arg_block()")
        {
            func = get_internal_function(m_int_func_state_get_arg_block);

            Function_instance inst(
                get_allocator(), reinterpret_cast<size_t>(m_int_func_state_get_arg_block));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::call_lambda_float(int)")
        {
            func = get_internal_function(m_int_func_state_call_lambda_float);

            Function_instance inst(
                get_allocator(), reinterpret_cast<size_t>(m_int_func_state_call_lambda_float));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::call_lambda_float3(int)")
        {
            func = get_internal_function(m_int_func_state_call_lambda_float3);

            Function_instance inst(
                get_allocator(), reinterpret_cast<size_t>(m_int_func_state_call_lambda_float3));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::call_lambda_uint(int)")
        {
            func = get_internal_function(m_int_func_state_call_lambda_uint);

            Function_instance inst(
                get_allocator(), reinterpret_cast<size_t>(m_int_func_state_call_lambda_uint));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::get_arg_block_float(int)")
        {
            func = get_internal_function(m_int_func_state_get_arg_block_float);

            Function_instance inst(
                get_allocator(), reinterpret_cast<size_t>(m_int_func_state_get_arg_block_float));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::get_arg_block_float3(int)")
        {
            func = get_internal_function(m_int_func_state_get_arg_block_float3);

            Function_instance inst(
                get_allocator(), reinterpret_cast<size_t>(m_int_func_state_get_arg_block_float3));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::get_arg_block_uint(int)")
        {
            func = get_internal_function(m_int_func_state_get_arg_block_uint);

            Function_instance inst(
                get_allocator(), reinterpret_cast<size_t>(m_int_func_state_get_arg_block_uint));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::get_arg_block_bool(int)")
        {
            func = get_internal_function(m_int_func_state_get_arg_block_bool);

            Function_instance inst(
                get_allocator(), reinterpret_cast<size_t>(m_int_func_state_get_arg_block_bool));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::get_measured_curve_value(int,int)")
        {
            func = get_internal_function(m_int_func_state_get_measured_curve_value);

            Function_instance inst(get_allocator(),
                reinterpret_cast<size_t>(m_int_func_state_get_measured_curve_value));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::adapt_microfacet_roughness(float2)")
        {
            func = get_internal_function(m_int_func_state_adapt_microfacet_roughness);

            Function_instance inst(get_allocator(),
                reinterpret_cast<size_t>(m_int_func_state_adapt_microfacet_roughness));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::tex_resolution_2d(int)")
        {
            demangled_name = "::tex::resolution(texture_2d)";
        }
        else if (demangled_name == "::state::tex_is_valid_2d(int)")
        {
            demangled_name = "::tex::is_valid(texture_2d)";
        }
        else if (demangled_name == 
            "::state::tex_lookup_float3_2d(int,float2,int,int,float2,float2)")
        {
            demangled_name = "::tex::lookup_float3(texture_2d,"
                "float2,::tex::wrap_mode,::tex::wrap_mode,float2,float2)";
        }
        else if (demangled_name == 
            "::state::tex_lookup_float_3d(int,float3,int,int,int,float2,float2,float2)")
        {
            demangled_name = "::tex::lookup_float(texture_3d,"
                "float3,::tex::wrap_mode,::tex::wrap_mode,::tex::wrap_mode,float2,float2,float2)";
        }
        else if (demangled_name == 
            "::state::tex_lookup_float3_3d(int,float3,int,int,int,float2,float2,float2)")
        {
            demangled_name = "::tex::lookup_float3(texture_3d,"
                "float3,::tex::wrap_mode,::tex::wrap_mode,::tex::wrap_mode,float2,float2,float2)";
        }
        else if (demangled_name == "::state::bsdf_measurement_resolution(int,int)")
        {
            func = get_internal_function(m_int_func_df_bsdf_measurement_resolution);

            Function_instance inst(get_allocator(),
                reinterpret_cast<size_t>(m_int_func_df_bsdf_measurement_resolution));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::bsdf_measurement_evaluate(int,float2,float2,int)")
        {
            func = get_internal_function(m_int_func_df_bsdf_measurement_evaluate);

            Function_instance inst(get_allocator(),
                reinterpret_cast<size_t>(m_int_func_df_bsdf_measurement_evaluate));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::bsdf_measurement_sample(int,float2,float3,int)")
        {
            func = get_internal_function(m_int_func_df_bsdf_measurement_sample);

            Function_instance inst(get_allocator(),
                reinterpret_cast<size_t>(m_int_func_df_bsdf_measurement_sample));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::bsdf_measurement_pdf(int,float2,float2,int)")
        {
            func = get_internal_function(m_int_func_df_bsdf_measurement_pdf);

            Function_instance inst(get_allocator(),
                reinterpret_cast<size_t>(m_int_func_df_bsdf_measurement_pdf));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::bsdf_measurement_albedos(int,float2)")
        {
            func = get_internal_function(m_int_func_df_bsdf_measurement_albedos);

            Function_instance inst(get_allocator(),
                reinterpret_cast<size_t>(m_int_func_df_bsdf_measurement_albedos));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::light_profile_evaluate(int,float2)")
        {
            func = get_internal_function(m_int_func_df_light_profile_evaluate);

            Function_instance inst(get_allocator(),
                reinterpret_cast<size_t>(m_int_func_df_light_profile_evaluate));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::light_profile_sample(int,float3)")
        {
            func = get_internal_function(m_int_func_df_light_profile_sample);

            Function_instance inst(get_allocator(),
                reinterpret_cast<size_t>(m_int_func_df_light_profile_sample));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::light_profile_pdf(int,float2)")
        {
            func = get_internal_function(m_int_func_df_light_profile_pdf);

            Function_instance inst(get_allocator(),
                reinterpret_cast<size_t>(m_int_func_df_light_profile_pdf));
            p_data = get_context_data(inst);
            handled = true;
        }
        else if (demangled_name == "::state::get_bsdf_data_texture_id(Bsdf_data_kind)")
        {
            // will be handled by finalize_module() when all resources of the link unit are known
            return true;
        }
    }

    if (!handled)
    {
        // find last "::" before the parameters
        size_t parenpos = demangled_name.find('(');
        size_t colonpos = demangled_name.rfind("::", parenpos);
        if (colonpos == string::npos || colonpos == 0)
            return true;  // not in a module, maybe a builtin function

        string module_name = demangled_name.substr(0, colonpos);
        string signature = demangled_name.substr(colonpos + 2);
        IDefinition const *def = m_compiler->find_stdlib_signature(
            module_name.c_str(), signature.c_str());
        if (def == NULL)
            return true;  // not one of our modules, maybe a builtin function

        if (m_target_lang == TL_HLSL)
            func = get_hlsl_intrinsic_function(def, /*return_derivs=*/ false);
 
        if(func == NULL)
            func = get_intrinsic_function(def, /*return_derivs=*/ false);

        // check for MDL function with array return and retrieve array size
        MDL_ASSERT(def->get_type()->get_kind() == IType::TK_FUNCTION);
        IType_function const *mdl_func_type = static_cast<IType_function const *>(def->get_type());
        IType const *mdl_ret_type = mdl_func_type->get_return_type();
        if (mdl_ret_type->get_kind() == IType::TK_ARRAY) {
            IType_array const *mdl_array_type = static_cast<IType_array const *>(mdl_ret_type);
            MDL_ASSERT(mdl_array_type->is_immediate_sized());
            ret_array_size = unsigned(mdl_array_type->get_size());
        }

        Function_instance inst(get_allocator(), def, /*return_derivs=*/ false);
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
            // pass exc_state_param parameter
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
        llvm::Value *arg = call->getArgOperand(i);
        llvm::Type *arg_type = arg->getType();
        llvm::Type *param_type = func_type->getParamType(llvm_args.size());

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
        if (llvm::isa<llvm::PointerType>(param_type))
            param_elem_type = param_type->getPointerElementType();

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

    llvm::Value *res = ctx->CreateCall(func, llvm_args);

    // Runtime call case: f_r(&res,a,b)?
    if (runtime_res_ptr != NULL) {
        if (ret_array_size != 0)
            res = ctx->CreateLoad(runtime_res_ptr);
        else
            res = ctx.load_and_convert(orig_res_type, runtime_res_ptr);
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
    int df_param_idx,
    IType::Kind kind)
{
    llvm::SmallPtrSet<llvm::Value *, 16> visited;
    llvm::SmallVector<llvm::Value *, 16> worklist;

    llvm::Type *int_type = m_type_mapper.get_int_type();

    worklist.push_back(arg);
    while (!worklist.empty()) {
        llvm::Value *cur = worklist.pop_back_val();
        if (visited.count(cur))
            continue;
        visited.insert(cur);

        int num_stores = 0;
        for (auto user : cur->users())
        {
            if (llvm::StoreInst *inst = llvm::dyn_cast<llvm::StoreInst>(user)) {
                // for stores, also follow the variable which is written
                worklist.push_back(inst->getPointerOperand());
                ++num_stores;
            } else if (llvm::CallInst *inst = llvm::dyn_cast<llvm::CallInst>(user)) {
                // found a call, store the parameter index as metadata
                llvm::Metadata *param_idx = llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(int_type, df_param_idx));
                llvm::MDNode *md = llvm::MDNode::get(m_llvm_context, param_idx);
                switch (kind)
                {
                    case IType::TK_BSDF:
                    case IType::TK_HAIR_BSDF:
                        inst->setMetadata(m_bsdf_param_metadata_id, md);
                        break;                
                    case IType::TK_EDF:
                        inst->setMetadata(m_edf_param_metadata_id, md);
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

    llvm::Type *ret_tp = func->getReturnType();
    auto it = func->arg_begin();
    if (ret_tp->isVoidTy() && it != func->arg_end() && it->getType()->isPointerTy())
        flags |= LLVM_context_data::FL_SRET; // set SRET if the function returns void
    //if (target_supports_sret_for_lambda())
    //    flags |= LLVM_context_data::FL_SRET;  // will be mapped to inout
    if (target_uses_resource_data_parameter())
        flags |= LLVM_context_data::FL_HAS_RES;
    if (target_uses_exception_state_parameter())
        flags |= LLVM_context_data::FL_HAS_EXC;
    if (target_supports_captured_argument_parameter())
        flags |= LLVM_context_data::FL_HAS_CAP_ARGS;
    if (target_supports_lambda_results_parameter())
        flags |= LLVM_context_data::FL_HAS_EXEC_CTX | LLVM_context_data::FL_HAS_LMBD_RES;
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

    // collect all functions available before linking
    // note: we cannot use the function pointers, as linking removes some function declarations and
    //       may reuse the old pointers
    hash_set<string, string_hash<string> >::Type old_func_names(get_allocator());
    for (llvm::Function &f : m_module->functions()) {
        if (!f.isDeclaration())
            old_func_names.insert(string(f.getName().begin(), f.getName().end(), get_allocator()));
    }

    if (llvm::Linker::linkModules(*m_module, std::move(libbsdf))) {
        // true means linking has failed
        error(LINKING_LIBBSDF_FAILED, "unknown linker error");
        MDL_ASSERT(!"Linking libbsdf failed");
        return false;
    }

    m_float3_struct_type = m_module->getTypeByName("struct.float3");
    if (m_float3_struct_type == NULL) {
        // name was lost during linking? get it from
        //    void @black_bsdf_sample(
        //        %struct.BSDF_sample_data* nocapture %data,
        //        %class.State* nocapture readnone %state,
        //        %struct.float3* nocapture readnone %inherited_normal)

        llvm::Function *func = m_module->getFunction("black_bsdf_sample");
        MDL_ASSERT(func);
        llvm::FunctionType *func_type = func->getFunctionType();
        m_float3_struct_type = llvm::cast<llvm::StructType>(
            func_type->getParamType(2)->getPointerElementType());
        MDL_ASSERT(m_float3_struct_type);
    }


    create_bsdf_function_types();
    create_edf_function_types();

    m_bsdf_param_metadata_id = m_llvm_context.getMDKindID("libbsdf.bsdf_param");
    m_edf_param_metadata_id = m_llvm_context.getMDKindID("libbsdf.edf_param");

    llvm::Type *int_type = m_type_mapper.get_int_type();
    unsigned alloca_addr_space = m_module->getDataLayout().getAllocaAddrSpace();

    // find all functions which were added by linking the libbsdf module,
    // collect in vector as module functions will be modified, later
    vector<llvm::Function *>::Type libbsdf_funcs(get_allocator());
    for (llvm::Function &f : m_module->functions()) {
        // just a declaration or did already exist before linking? -> skip
        if (f.isDeclaration() || old_func_names.count(
                string(f.getName().begin(), f.getName().end(), get_allocator())) != 0)
            continue;

        // Found a libbsdf function
        libbsdf_funcs.push_back(&f);
    }

    // iterate over all functions added from the libbsdf module
    for (llvm::Function *func : libbsdf_funcs) {
        // make all functions from libbsdf internal to allow global dead code elimination
        func->setLinkage(llvm::GlobalValue::InternalLinkage);

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
                        if (!translate_libbsdf_runtime_call(call, II, ctx))
                            return false;
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
                }
                else if (df_data_type == m_type_bsdf_evaluate_data) {
                    new_func_type = m_type_bsdf_evaluate_func;
                    df_kind = IType::TK_BSDF; // or TK_HAIR_BSDF
                    has_inherited_weight = true;
                }
                else if (df_data_type == m_type_bsdf_pdf_data) {
                    new_func_type = m_type_bsdf_pdf_func;
                    df_kind = IType::TK_BSDF; // or TK_HAIR_BSDF
                }
                else if (df_data_type == m_type_bsdf_auxiliary_data) {
                    new_func_type = m_type_bsdf_auxiliary_func;
                    df_kind = IType::TK_BSDF; // or TK_HAIR_BSDF
                    has_inherited_weight = true;
                }
                // edf
                else if (df_data_type == m_type_edf_sample_data) {
                    new_func_type = m_type_edf_sample_func;
                    df_kind = IType::TK_EDF;
                }
                else if (df_data_type == m_type_edf_evaluate_data) {
                    new_func_type = m_type_edf_evaluate_func;
                    df_kind = IType::TK_EDF;
                    has_inherited_weight = true;
                }
                else if (df_data_type == m_type_edf_pdf_data) {
                    new_func_type = m_type_edf_pdf_func;
                    df_kind = IType::TK_EDF;
                }
                else if (df_data_type == m_type_edf_auxiliary_data) {
                    new_func_type = m_type_edf_auxiliary_func;
                    df_kind = IType::TK_EDF;
                    has_inherited_weight = true;
                }
                else
                    new_func_type = NULL;

                std::string df_arg = "";
                std::string df_arg_var = "";
                std::string df_struct_name = "";
                switch (df_kind)
                {
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

                // for HLSL, check for interpreter functions
                if (m_target_lang == TL_HLSL && func->getName().startswith("mdl_bsdf_")) {
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
                    set_llvm_function_attributes(new_func);
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

                    // replace all uses of parameters
                    old_df_data_param->replaceAllUsesWith(new_df_data_param);
                    old_state_param->replaceAllUsesWith(new llvm::BitCastInst(
                        new_state_param,
                        old_state_param->getType(),
                        "state_cast",
                        &*param_init_insert_point));
                    old_scratch_space_param->replaceAllUsesWith(new_scratch_space_param);
                    old_material_param->replaceAllUsesWith(new_material_param);

                    func->eraseFromParent();
                    continue;
                }

                IDefinition::Semantics sema = get_libbsdf_function_semantics(func->getName());

                if (new_func_type != NULL && (is_df_semantics(sema) ||
                        sema == IDefinition::DS_INVALID_REF_CONSTRUCTOR)) {
                    // this is a BSDF API function

                    // For DF instantiation, any DF parameters (like tint or roughness) are
                    // replaced by local variable placeholders, which will be replaced by the real
                    // values or lambda function calls during instantiation.

                    llvm::Function *old_func = func;

                    llvm::Function::arg_iterator old_arg_it  = old_func->arg_begin();
                    llvm::Function::arg_iterator old_arg_end = old_func->arg_end();
                    llvm::Value *df_data          = old_arg_it++;
                    llvm::Value *exec_ctx         = old_arg_it++;
                    llvm::Value *inherited_normal = old_arg_it++;
                    llvm::Value *inherited_weight = NULL;
                    if (has_inherited_weight)
                        inherited_weight = old_arg_it++;

                    llvm::Function *new_func = llvm::Function::Create(
                        new_func_type,
                        llvm::GlobalValue::InternalLinkage,
                        "",
                        m_module);
                    set_llvm_function_attributes(new_func);
                    new_func->setName("gen_" + func->getName());
                    new_func->getBasicBlockList().splice(
                        new_func->begin(), old_func->getBasicBlockList());

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
                    if (has_inherited_weight)
                        inherited_weight_param = arg_it++;

                    // replace all uses of parameters which will not be removed
                    df_data->replaceAllUsesWith(data_param);
                    exec_ctx->replaceAllUsesWith(new llvm::BitCastInst(
                        exec_ctx_param,
                        exec_ctx->getType(),
                        "exec_ctx_cast",
                        &*param_init_insert_point));
                    inherited_normal->replaceAllUsesWith(inherited_normal_param);
                    if (has_inherited_weight)
                        inherited_weight->replaceAllUsesWith(inherited_weight_param);

                    // introduce local variables for all used DF parameters
                    bool skipped_df_idx_inc = false;
                    for (int i = 0, df_idx = 0; old_arg_it != old_arg_end; ++i, ++old_arg_it) {
                        int cur_df_idx = df_idx;

                        // Determine parameter index for next iteration
                        if (skipped_df_idx_inc) {
                            skipped_df_idx_inc = false;
                            ++df_idx;
                        }
                        else if (is_libbsdf_array_parameter(sema, cur_df_idx)) {
                            // array parameters consist of a pointer and a length in libbsdf
                            // and both get the same associated df parameter index
                            skipped_df_idx_inc = true;
                        }
                        else
                            ++df_idx;

                        if (old_arg_it->use_empty())
                            continue;

                        llvm::AllocaInst *arg_var;
                        llvm::Value *arg_val;
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
                                elem_type->getStructName() == df_struct_name.c_str())
                            {
                                // for DF parameters, we mark the calls to the DF methods
                                // with metadata instead of the local variables.
                                // The argument value is not necessary, but we keep it, in case
                                // the uses are not optimized away.
                                // Note: we don't do this for the DFs inside xDF_component!
                                mark_df_calls(old_arg_it, cur_df_idx, df_kind);

                                arg_var = NULL;
                            }
                        } else {
                            // for non-pointer types we also need to load the value
                            // and replace the argument by the load, not the alloca
                            arg_var = new llvm::AllocaInst(
                                old_arg_it->getType(),
                                alloca_addr_space,
                                df_arg_var,
                                &*new_func->getEntryBlock().begin());
                            arg_val = new llvm::LoadInst(
                                arg_var, df_arg, &*param_init_insert_point);
                        }

                        // do we need to set metadata?
                        if (arg_var != NULL) {
                            llvm::ConstantAsMetadata *param_idx = llvm::ConstantAsMetadata::get(
                                llvm::ConstantInt::get(int_type, cur_df_idx));
                            llvm::MDNode *md = llvm::MDNode::get(m_llvm_context, param_idx);

                            switch (df_kind)
                            {
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

// Generate a call to an expression lambda function.
Expression_result LLVM_code_generator::generate_expr_lambda_call(
    Function_context                &ctx,
    mi::mdl::ILambda_function const *expr_lambda,
    llvm::Value                     *opt_results_buffer,
    size_t                          opt_result_index)
{
    Lambda_function const *expr_lambda_impl = impl_cast<Lambda_function>(expr_lambda);

    // expression and special lambdas always return by reference via first parameter,
    // if supported by the target

    llvm::Function *func = m_module_lambda_funcs[m_module_lambda_index_map[expr_lambda]];
    llvm::Type *lambda_retptr_type = func->getFunctionType()->getParamType(0);
    llvm::Type *lambda_res_type = lambda_retptr_type->getPointerElementType();
    llvm::Type *dest_type = NULL;
    llvm::Value *res_pointer;

    llvm::Value *opt_dest_ptr = NULL;
    if (opt_results_buffer != NULL && m_target_lang != TL_HLSL) {
        opt_dest_ptr = ctx.create_simple_gep_in_bounds(
            opt_results_buffer, ctx.get_constant(int(opt_result_index)));
    }

    if (!target_supports_sret_for_lambda())
        res_pointer = NULL;
    else if (opt_dest_ptr != NULL &&
            (dest_type = opt_dest_ptr->getType()->getPointerElementType()) == lambda_res_type)
        res_pointer = opt_dest_ptr;
    else
        res_pointer = ctx.create_local(lambda_res_type, "res_buf");

    llvm::SmallVector<llvm::Value *, 6> lambda_args;
    if (res_pointer)
        lambda_args.push_back(res_pointer);
    lambda_args.push_back(ctx.get_state_parameter());
    if (target_uses_resource_data_parameter())
        lambda_args.push_back(ctx.get_resource_data_parameter());
    if (target_uses_exception_state_parameter())
        lambda_args.push_back(ctx.get_exc_state_parameter());
    if (target_supports_captured_argument_parameter())
        lambda_args.push_back(ctx.get_cap_args_parameter());
    if (target_supports_lambda_results_parameter() && expr_lambda_impl->uses_lambda_results())
        lambda_args.push_back(ctx.get_lambda_results_parameter());

    llvm::CallInst *call = ctx->CreateCall(func, lambda_args);
    if (res_pointer == NULL) {
        if (opt_results_buffer != NULL && m_target_lang == TL_HLSL) {
            store_to_float4_array(
                ctx,
                call,
                opt_results_buffer,
                m_texture_result_offsets[opt_result_index]);
        }
        return Expression_result::value(call);
    }

    if (opt_dest_ptr != NULL && dest_type != lambda_res_type) {
        llvm::Value *res = ctx->CreateLoad(res_pointer);
        ctx.convert_and_store(res, opt_dest_ptr);
        return Expression_result::ptr(opt_dest_ptr);
    }

    return Expression_result::ptr(res_pointer);
}

// Generate a call to an expression lambda function.
Expression_result LLVM_code_generator::generate_expr_lambda_call(
    Function_context &ctx,
    size_t           lambda_index,
    llvm::Value      *opt_results_buffer,
    size_t           opt_result_index)
{
    mi::base::Handle<mi::mdl::ILambda_function> expr_lambda(
        m_dist_func->get_expr_lambda(lambda_index));
    return generate_expr_lambda_call(ctx, expr_lambda.get(), opt_results_buffer, opt_result_index);
}

// Store a value inside a float4 array at the given byte offset, updating the offset.
void LLVM_code_generator::store_to_float4_array_impl(
    Function_context &ctx,
    llvm::Value *val,
    llvm::Value *dest,
    unsigned &dest_offs)
{
    llvm::Type *val_type = val->getType();

    if (llvm::IntegerType *it = llvm::dyn_cast<llvm::IntegerType>(val_type)) {
        if (it->getBitWidth() > 8)
            dest_offs = (dest_offs + 3) & ~3;

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
            if ((dest_offs & 3) != 0)
                val = ctx->CreateShl(val, (dest_offs & 3) * 8);
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

    if (llvm::isa<llvm::CompositeType>(val_type)) {
        size_t size = size_t(
            ctx.get_code_gen().get_target_layout_data()->getTypeAllocSize(val_type));
        unsigned compound_start_offs = dest_offs;

        uint64_t n;
        if (llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(val_type)) {
            n = st->getNumElements();
        } else {
            n = llvm::cast<llvm::SequentialType>(val_type)->getNumElements();
        }

        for (uint64_t i = 0; i < n; ++i) {
            llvm::Value *elem = ctx.create_extract(val, unsigned(i));
            store_to_float4_array_impl(ctx, elem, dest, dest_offs);
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
    Function_context &ctx,
    llvm::Value *val,
    llvm::Value *dest,
    unsigned dest_offs)
{
    // call wrapped function with a copy of the offset, so only the copy is changed
    store_to_float4_array_impl(ctx, val, dest, dest_offs);
}

// Load a value inside a float4 array at the given byte offset, updating the offset.
llvm::Value *LLVM_code_generator::load_from_float4_array_impl(
    Function_context &ctx,
    llvm::Type       *val_type,
    llvm::Value      *src,
    unsigned         &src_offs)
{
    if (llvm::IntegerType *it = llvm::dyn_cast<llvm::IntegerType>(val_type)) {
        if (it->getBitWidth() > 8)
            src_offs = (src_offs + 3) & ~3;

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

            if ((src_offs & 3) != 0)
                val = ctx->CreateLShr(val, (src_offs & 3) * 8);
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

    if (llvm::CompositeType *ct = llvm::dyn_cast<llvm::CompositeType>(val_type)) {
        size_t size = size_t(
            ctx.get_code_gen().get_target_layout_data()->getTypeAllocSize(val_type));
        unsigned compound_start_offs = src_offs;

        uint64_t n;
        if (llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(val_type)) {
            n = st->getNumElements();
        } else {
            n = llvm::cast<llvm::SequentialType>(val_type)->getNumElements();
        }

        llvm::Value *res = llvm::UndefValue::get(val_type);
        for (uint64_t i = 0; i < n; ++i) {
            llvm::Value *elem = load_from_float4_array_impl(
                ctx, ct->getTypeAtIndex(unsigned(i)), src, src_offs);
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
    Function_context &ctx,
    llvm::Type       *val_type,
    llvm::Value      *src,
    unsigned         src_offs)
{
    // call wrapped function with a copy of the offset, so only the copy is changed
    return load_from_float4_array_impl(ctx, val_type, src, src_offs);
}

// Translate a precalculated lambda function to LLVM IR.
Expression_result LLVM_code_generator::translate_precalculated_lambda(
    Function_context &ctx,
    size_t           lambda_index,
    llvm::Type       *expected_type)
{
    mi::base::Handle<mi::mdl::ILambda_function> expr_lambda(
        m_dist_func->get_expr_lambda(lambda_index));

    Expression_result res;

    // translate constants directly
    if (DAG_constant const *c = as<DAG_constant>(expr_lambda->get_body())) {
        res = translate_value(ctx, c->get_value());
    }

    // is the result available in the texture results from the MDL SDK state
    else if (m_texture_result_indices[lambda_index] != -1) {
        if (m_target_lang == TL_HLSL) {
            res = Expression_result::value(load_from_float4_array(
                ctx,
                m_texture_results_struct_type->getElementType(
                    m_texture_result_indices[lambda_index]),
                get_texture_results(ctx),
                m_texture_result_offsets[m_texture_result_indices[lambda_index]]));
        } else {
            res = Expression_result::ptr(ctx->CreateConstGEP2_32(
                nullptr,
                get_texture_results(ctx),
                0,
                unsigned(m_texture_result_indices[lambda_index])));
        }
    }

    // was the result locally precalculated?
    else if (m_lambda_result_indices[lambda_index] != -1) {
        res = Expression_result::ptr(ctx->CreateConstGEP2_32(
            nullptr,
            ctx->CreateBitCast(
                ctx.get_lambda_results_parameter(),
                m_type_mapper.get_ptr(m_lambda_results_struct_type)),
            0,
            unsigned(m_lambda_result_indices[lambda_index])));
    }

    // calculate on demand, should be cheap if we get here
    else {
        res = generate_expr_lambda_call(ctx, lambda_index);
    }

    // type doesn't matter or fits already?
    if (expected_type == NULL || res.get_value_type() == expected_type) return res;

    // convert to expected type
    return Expression_result::value(ctx.load_and_convert(expected_type, res.as_ptr(ctx)));
}

// Translate a DAG call argument which may be a precalculated lambda function to LLVM IR.
Expression_result LLVM_code_generator::translate_call_arg(
    Function_context &ctx,
    DAG_node const   *arg,
    llvm::Type       *expected_type)
{
    // translate constants directly
    if (DAG_constant const *c = as<DAG_constant>(arg)) {
        Expression_result res = translate_value(ctx, c->get_value());
        if (res.get_value_type() == expected_type)
            return res;

        return Expression_result::value(ctx.load_and_convert(expected_type, res.as_ptr(ctx)));
    }

    // determine expression lambda index
    MDL_ASSERT(arg->get_kind() == DAG_node::EK_CALL);
    DAG_call const *arg_call = mi::mdl::cast<DAG_call>(arg);
    MDL_ASSERT(arg_call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA);
    size_t lambda_index = strtoul(arg_call->get_name(), NULL, 10);

    return translate_precalculated_lambda(ctx, lambda_index, expected_type);
}

// Get the BSDF parameter ID metadata for an instruction.
int LLVM_code_generator::get_metadata_df_param_id(
    llvm::Instruction *inst, 
    IType::Kind kind)
{

    llvm::MDNode *md = NULL;
    switch (kind)
    {
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

    if (md == NULL) return -1;

    llvm::ConstantInt *param_idx_val =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(md->getOperand(0));
    if (param_idx_val == NULL) {
        MDL_ASSERT(!"Invalid BSDF alloca parameter metadata");
        return -1;
    }
    return int(param_idx_val->getValue().getZExtValue());
}

bool LLVM_code_generator::rewrite_weight_memcpy_addr(
    Function_context &ctx,
    llvm::Value *weight_array,
    llvm::BitCastInst *addr_bitcast,
    llvm::Value *index,
    llvm::SmallVector<llvm::Instruction *, 16> &delete_list)
{
    // check for
    //   <C> = bitcast %struct.color_xDF_component* <X> to i8*
    //   call void @llvm.memcpy.p0i8.p0i8.i64(i8* <Y>, i8* <C>, i64 12, i32 4, i1 false)

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
    llvm::Value *new_cast = llvm::BitCastInst::Create(
        llvm::Instruction::BitCast, weight_ptr, addr_bitcast->getType(), "", weight_ptr);
    addr_bitcast->replaceAllUsesWith(new_cast);
    delete_list.push_back(addr_bitcast);

    return true;
}

// Rewrite all usages of a BSDF component variable using the given weight array and the
// BSDF function, which can either be a switch function depending on the array index
// or the same function for all indices.
void LLVM_code_generator::rewrite_df_component_usages(
    Function_context                           &ctx,
    llvm::AllocaInst                           *inst,
    llvm::Value                                *weight_array,
    Df_component_info                          &comp_info,
    llvm::SmallVector<llvm::Instruction *, 16> &delete_list)
{
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
            rewrite_weight_memcpy_addr(ctx, weight_array, cast, null_val, delete_list);
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
                rewrite_weight_memcpy_addr(ctx, weight_array, cast, component_idx_val, delete_list);
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
            llvm::Value *new_gep;

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
            } else {
                // replace by access on same index of weight array (can be float or color)
                llvm::Value *idxs[] = {
                    llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type()),
                    component_idx_val
                };
                new_gep = llvm::GetElementPtrInst::Create(nullptr, weight_array, idxs, "", gep);
            }
            gep->replaceAllUsesWith(new_gep);
            continue;
        }

        // access to component?
        if (struct_idx == 1) {
            // We have to rewrite all accesses.
            // The code we search for should look like this:
            //  - %elemptr = getelementptr %components, %i, 1, 0
            //  - %funcptr = load %elemptr
            //  - call %funcptr
            // So iterate over all usages of the gep and the loads
            for (auto gep_user : gep->users()) {
                llvm::LoadInst *load = llvm::dyn_cast<llvm::LoadInst>(gep_user);
                MDL_ASSERT(load);

                for (auto load_user : load->users()) {
                    llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(load_user);
                    MDL_ASSERT(call);
                    MDL_ASSERT(call->getType() != llvm::IntegerType::get(m_llvm_context, 1) &&
                        "bsdfs in bsdf_component currently don't support is_black()");

                    Distribution_function_state call_state = get_dist_func_state_from_call(call);
                    llvm::Function *df_func = comp_info.get_df_function(call_state);

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
                    llvm_args.push_back(call->getArgOperand(0));     // res_pointer
                    llvm_args.push_back(ctx.has_exec_ctx_parameter() ?
                        ctx.get_exec_ctx_parameter() : ctx.get_state_parameter());
                    llvm_args.push_back(call->getArgOperand(2));     // inherited_normal param
                    if (call_state == Distribution_function_state::DFSTATE_EVALUATE ||
                        call_state == Distribution_function_state::DFSTATE_AUXILIARY)
                            llvm_args.push_back(call->getArgOperand(3));  // inherited_weight param
                    if (comp_info.is_switch_function())
                        llvm_args.push_back(idx_val);                // BSDF function index
                    llvm::CallInst::Create(df_func, llvm_args, "", call);
                    delete_list.push_back(call);
                }
                delete_list.push_back(load);
            }
            continue;
        }

        MDL_ASSERT(!"Invalid access to BSDF_component structure");
    }
}

// Handle BSDF array parameter during BSDF instantiation.
void LLVM_code_generator::handle_df_array_parameter(
    Function_context                           &ctx,
    llvm::AllocaInst                           *inst,
    DAG_node const                             *arg,
    llvm::SmallVector<llvm::Instruction *, 16> &delete_list)
{
    llvm::Type  *elem_type      = inst->getAllocatedType();

    // special handling for constant array parameters
    if (arg->get_kind() == DAG_node::EK_CONSTANT) {
        DAG_constant const *arg_const = mi::mdl::cast<DAG_constant>(arg);
        mi::mdl::IValue const *arg_val = arg_const->get_value();
        MDL_ASSERT(arg_val->get_kind() == mi::mdl::IValue::VK_ARRAY);
        mi::mdl::IValue_array const *arg_array = mi::mdl::cast<mi::mdl::IValue_array>(arg_val);
        int elem_count = arg_array->get_component_count();

        // is it an array size parameter? -> replace by the number of elements
        if (elem_type == m_type_mapper.get_int_type()) {
            Expression_result res = Expression_result::value(
                llvm::ConstantInt::get(m_type_mapper.get_int_type(), elem_count));
            inst->replaceAllUsesWith(res.as_ptr(ctx));
            return;
        }

        // is it a float3 array? (this should be a RGB color array)
        if (elem_type == m_float3_struct_type) {
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

        // is it a BSDF_component array?
        bool color_df_component = false;
        if (elem_type->isStructTy() && !llvm::cast<llvm::StructType>(elem_type)->isLiteral() && (
                elem_type->getStructName() == "struct.BSDF_component" ||
                elem_type->getStructName() == "struct.EDF_component" ||
                (color_df_component = (
                    elem_type->getStructName() == "struct.color_BSDF_component" ||
                    elem_type->getStructName() == "struct.color_EDF_component")))) {

            llvm::Type *weight_type = color_df_component ?
                m_float3_struct_type : m_type_mapper.get_float_type(); 

            // create a global constant weight array
            llvm::ArrayType *weight_array_type =
                llvm::ArrayType::get(weight_type, elem_count);
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

            llvm::Constant *array = llvm::ConstantArray::get(weight_array_type, elems);
            llvm::Value *weight_array_global = new llvm::GlobalVariable(
                *m_module,
                weight_array_type,
                /*isConstant=*/ true,
                llvm::GlobalValue::InternalLinkage,
                array,
                "_global_libbsdf_const");


            const IType_array* array_type = as<IType_array>(arg->get_type());
            const IType_struct* element_type = as<IType_struct>(array_type->get_element_type());
            IType::Kind df_kind = element_type->get_compound_type(1)->get_kind();

            // only "xdf()" can be part of a constant, so use an empty component info
            Df_component_info comp_info(*this, df_kind);

            // rewrite all usages of the components variable
            rewrite_df_component_usages(
                ctx,
                inst,
                weight_array_global,
                comp_info,
                delete_list);
            return;
        }

        MDL_ASSERT(!"Unsupported constant array parameter type");
        return;
    }

    MDL_ASSERT(arg->get_kind() == DAG_node::EK_CALL);
    DAG_call const *arg_call = mi::mdl::cast<DAG_call>(arg);
    IType_array const *arg_type = mi::mdl::cast<IType_array>(arg_call->get_type());
    MDL_ASSERT(arg_type->is_immediate_sized() && "array type must be instantiated");
    int elem_count = arg_type->get_size();

    // is it an array size parameter? -> replace by the number of elements
    if (elem_type == m_type_mapper.get_int_type()) {
        Expression_result res = Expression_result::value(
            llvm::ConstantInt::get(m_type_mapper.get_int_type(), elem_count));
        inst->replaceAllUsesWith(res.as_ptr(ctx));
        return;
    }

    // is it a float3 array? (this should be a color array)
    if (elem_type == m_float3_struct_type) {
        // make sure the color array is initialized at the beginning of the body
        ctx.move_to_body_start();

        llvm::ArrayType *color_array_type = llvm::ArrayType::get(
            m_float3_struct_type, elem_count);
        llvm::Value *color_array;

        if (arg_call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA) {
            // the whole array is calculated by a lambda function

            // read precalculated lambda result
            size_t lambda_index = strtoul(arg_call->get_name(), NULL, 10);
            Expression_result array_res = translate_precalculated_lambda(
                ctx, lambda_index, color_array_type);
            color_array = array_res.as_ptr(ctx);
        } else {
            // only single elements of the array a lambda functions

            // create local color array
            color_array = ctx.create_local(color_array_type, "colors");

            for (int i = 0; i < elem_count; ++i) {
                DAG_node const *color_node = arg_call->get_argument(i);
                Expression_result color_res = translate_call_arg(
                    ctx, color_node, m_float3_struct_type);

                // store result in colors array
                llvm::Value *idxs[] = {
                    llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type()),
                    llvm::ConstantInt::get(m_type_mapper.get_int_type(), i)
                };
                ctx->CreateStore(color_res.as_value(ctx),
                    ctx->CreateGEP(color_array, idxs, "colors_elem"));
            }
        }

        llvm::Value *casted_array = ctx->CreateBitCast(
            color_array, m_float3_struct_type->getPointerTo());
        inst->replaceAllUsesWith(casted_array);
        return;
    }

    // is it a BSDF_component array?
    bool color_df_component = false;
    if (elem_type->isStructTy() && !llvm::cast<llvm::StructType>(elem_type)->isLiteral() && (
            elem_type->getStructName() == "struct.BSDF_component" ||
            elem_type->getStructName() == "struct.EDF_component" ||
            (color_df_component = (
                elem_type->getStructName() == "struct.color_BSDF_component" ||
                elem_type->getStructName() == "struct.color_EDF_component")))) {

        // make sure the weight array is initialized at the beginning of the body
        ctx.move_to_body_start();

        llvm::Type *weight_type = color_df_component ?
            m_float3_struct_type : m_type_mapper.get_float_type(); 

        // create local weight array and instantiate all BSDF components
        llvm::ArrayType *weight_array_type = llvm::ArrayType::get(weight_type, elem_count);
        llvm::Value *weight_array = ctx.create_local(weight_array_type, "weights");


        // get df kind
        const IType_array* array_type = as<IType_array>(arg->get_type());
        const IType_struct* element_type = as<IType_struct>(array_type->get_element_type());
        IType::Kind df_kind = element_type->get_compound_type(1)->get_kind();

        Df_component_info comp_info(*this, df_kind);

        for (int i = 0; i < elem_count; ++i) {
            DAG_node const *elem_node = arg_call->get_argument(i);

            Expression_result weight_res;

            // is the i-th element a BSDF_component constant?
            if (elem_node->get_kind() == DAG_node::EK_CONSTANT) {
                DAG_constant const *constant = mi::mdl::cast<DAG_constant>(elem_node);
                mi::mdl::IValue_struct const *value =
                    mi::mdl::cast<IValue_struct>(constant->get_value());
                mi::mdl::IValue const *weight_val = value->get_field("weight");
                weight_res = translate_value(ctx, weight_val);

                // only "bsdf()" can be part of a constant
                MDL_ASSERT(value->get_field("component")->get_kind() ==
                    mi::mdl::IValue::VK_INVALID_REF);
                comp_info.add_component_df(elem_node);
            } else {
                // should be a BSDF_component constructor call
                MDL_ASSERT(elem_node->get_kind() == DAG_node::EK_CALL);
                DAG_call const *elem_call = mi::mdl::cast<DAG_call>(elem_node);
                DAG_node const *weight_node = elem_call->get_argument("weight");
                weight_res = translate_call_arg(ctx, weight_node, weight_type);

                // instantiate BSDF for component parameter of the constructor
                DAG_node const *component_node = elem_call->get_argument("component");
                comp_info.add_component_df(component_node);
            }

            // store result in weights array
            llvm::Value *idxs[] = {
                llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type()),
                llvm::ConstantInt::get(m_type_mapper.get_int_type(), i)
            };
            ctx->CreateStore(weight_res.as_value(ctx),
                ctx->CreateGEP(weight_array, idxs, "weights_elem"));
        }

        // rewrite all usages of the components variable
        rewrite_df_component_usages(
            ctx,
            inst,
            weight_array,
            comp_info,
            delete_list);
        return;
    }

    MDL_ASSERT(!"Unsupported array parameter type");
}

/// Recursively instantiate a ternary operator of type BSDF.
llvm::Function *LLVM_code_generator::instantiate_ternary_df(
    DAG_call const *dag_call)
{
    // Create a new function with the type for current distribution function state
    IType::Kind kind = dag_call->get_type()->get_kind();
    llvm::FunctionType *func_type;
    std::string operator_name;
    switch (kind)
    {
        case IType::Kind::TK_BSDF:
        case IType::Kind::TK_HAIR_BSDF:
        {
            switch (m_dist_func_state)
            {
                case DFSTATE_SAMPLE:    func_type = m_type_bsdf_sample_func; break;
                case DFSTATE_EVALUATE:  func_type = m_type_bsdf_evaluate_func; break;
                case DFSTATE_PDF:       func_type = m_type_bsdf_pdf_func; break;
                case DFSTATE_AUXILIARY: func_type = m_type_bsdf_auxiliary_func; break;
                default:
                    MDL_ASSERT(!"Invalid bsdf distribution function state");
                    return NULL;
            }
            if(kind == IType::Kind::TK_HAIR_BSDF)
                operator_name = "ternary_hair_bsdf";
            else
                operator_name = "ternary_bsdf";
            break;
        }

        case IType::Kind::TK_EDF:
        {
            switch (m_dist_func_state)
            {
                case DFSTATE_SAMPLE:    func_type = m_type_edf_sample_func; break;
                case DFSTATE_EVALUATE:  func_type = m_type_edf_evaluate_func; break;
                case DFSTATE_PDF:       func_type = m_type_edf_pdf_func; break;
                case DFSTATE_AUXILIARY: func_type = m_type_edf_auxiliary_func; break;
                default:
                    MDL_ASSERT(!"Invalid edf distribution function state");
                    return NULL;
            }
            operator_name = "ternary_edf";
            break;
        }

        default:
            MDL_ASSERT(!"Invalid distribution kind");
            return NULL;
    }

    llvm::Function *func = llvm::Function::Create(
        func_type,
        llvm::GlobalValue::InternalLinkage,
        operator_name,
        m_module);
    set_llvm_function_attributes(func);

    {
        // Context needs a non-empty start block, so create a jump to a second block
        llvm::BasicBlock *start_bb = llvm::BasicBlock::Create(m_llvm_context, "start", func);
        llvm::BasicBlock *body_bb = llvm::BasicBlock::Create(m_llvm_context, "body", func);
        start_bb->getInstList().push_back(llvm::BranchInst::Create(body_bb));

        Function_context ctx(
            get_allocator(),
            *this,
            func,
            get_df_function_flags(func),
            true);

        ctx->SetInsertPoint(body_bb);

        // Find lambda expression for condition and generate code
        DAG_node const *cond = dag_call->get_argument(0);
        Expression_result res = translate_call_arg(ctx, cond, m_type_mapper.get_bool_type());

        // Generate code for "if (cond) call[1](args); else call[2](args);"

        llvm::BasicBlock *cond_true_bb = ctx.create_bb("cond_true");
        llvm::BasicBlock *cond_false_bb = ctx.create_bb("cond_false");
        llvm::BasicBlock *end_bb = ctx.create_bb("end");

        llvm::Value *cond_res = res.as_value(ctx);
        llvm::Value *cond_bool = ctx->CreateICmpNE(
            cond_res,
            llvm::Constant::getNullValue(cond_res->getType()));
        ctx->CreateCondBr(cond_bool, cond_true_bb, cond_false_bb);

        llvm::Value *df_true_func = instantiate_df(dag_call->get_argument(1));
        llvm::Value *df_false_func = instantiate_df(dag_call->get_argument(2));
        llvm::Function::arg_iterator arg_it = ctx.get_first_parameter();
        llvm::Value *inherited_normal = arg_it;
        llvm::Value *inherited_weight = NULL;
        if (m_dist_func_state == DFSTATE_EVALUATE || m_dist_func_state == DFSTATE_AUXILIARY)
            inherited_weight = ++arg_it;

        llvm::SmallVector<llvm::Value *, 4> llvm_args;
        llvm_args.push_back(func->arg_begin());            // res_pointer
        llvm_args.push_back(ctx.has_exec_ctx_parameter()
            ? ctx.get_exec_ctx_parameter() : ctx.get_state_parameter());
        llvm_args.push_back(inherited_normal);
        if (m_dist_func_state == DFSTATE_EVALUATE || m_dist_func_state == DFSTATE_AUXILIARY)
            llvm_args.push_back(inherited_weight);

        // True case
        ctx->SetInsertPoint(cond_true_bb);
        ctx->CreateCall(df_true_func, llvm_args, "");
        ctx->CreateBr(end_bb);

        // False case
        ctx->SetInsertPoint(cond_false_bb);
        ctx->CreateCall(df_false_func, llvm_args, "");
        ctx->CreateBr(end_bb);

        ctx->SetInsertPoint(end_bb);
        ctx->CreateRetVoid();
    }

    // Return the now finalized function
    return func;
}

// Recursively instantiate a BSDF specified by the given DAG node from code in the BSDF library
// according to the current distribution function state.
llvm::Function *LLVM_code_generator::instantiate_df(
    DAG_node const *node)
{
    // get DF function according to semantics and current state
    // and clone it into the current module.

    llvm::Function *df_lib_func;

    if (DAG_constant const *c = as<DAG_constant>(node)) {
        IValue const *value = c->get_value();

        // check for "bsdf()" or "df::bsdf_component(weight, bsdf())" constant
        if ( (
                // "bsdf()"
                is<IValue_invalid_ref>(value) && 
                (is<IType_bsdf>(value->get_type()) || is<IType_hair_bsdf>(value->get_type()))
            ) || (
                // "df::bsdf_component(weight, bsdf())"
                is<IValue_struct>(value) &&
                strcmp(cast<IValue_struct>(value)->get_type()->get_symbol()->get_name(),
                    "::df::bsdf_component") == 0
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

        // check for "edf()" or "df::edf_component(weight, edf())" constant
        if ( (
                // "edf()"
                is<IValue_invalid_ref>(value) && is<IType_edf>(value->get_type())
            ) || (
                // "df::edf_component(weight, edf())"
                is<IValue_struct>(value) &&
                strcmp(cast<IValue_struct>(value)->get_type()->get_symbol()->get_name(),
                "::df::edf_component") == 0
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

    DAG_call const *dag_call = cast<DAG_call>(node);

    Instantiated_dfs::const_iterator it = m_instantiated_dfs[m_dist_func_state].find(dag_call);
    if (it != m_instantiated_dfs[m_dist_func_state].end())
        return it->second;
    
    IDefinition::Semantics sema = dag_call->get_semantic();
    if (sema == operator_to_semantic(IExpression::OK_TERNARY)) {
        llvm::Function *res_func = instantiate_ternary_df(dag_call);
        m_instantiated_dfs[m_dist_func_state][dag_call] = res_func;
        return res_func;
    }

    bool is_elemental = is_elemental_df_semantics(sema);
    IType::Kind kind = dag_call->get_type()->get_kind();

    df_lib_func = get_libbsdf_function(dag_call);
    if (df_lib_func == NULL) {
        char const *suffix;
        switch (dag_call->get_type()->get_kind())
        {
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

    llvm::ValueToValueMapTy ValueMap;
    llvm::Function *bsdf_func = llvm::CloneFunction(df_lib_func, ValueMap);
    if (is_always_inline_enabled())
        bsdf_func->addFnAttr(llvm::Attribute::AlwaysInline);

    Function_context ctx(
        get_allocator(),
        *this,
        bsdf_func,
        get_df_function_flags(bsdf_func),
        true);

    llvm::SmallVector<llvm::Instruction *, 16> delete_list;

    // process all calls to BSDF parameter accessors
    for (llvm::Function::iterator BI = bsdf_func->begin(), BE = bsdf_func->end(); BI != BE; ++BI) {
        for (llvm::BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
            if (llvm::AllocaInst *inst = llvm::dyn_cast<llvm::AllocaInst>(II)) {
                llvm::Type *elem_type = inst->getAllocatedType();

                // check for lambda call BSDF parameters
                int param_idx = get_metadata_df_param_id(inst, kind);
                if (param_idx < 0) continue;
                DAG_node const *arg = dag_call->get_argument(param_idx);

                // special handling for array parameters
                if (is_libbsdf_array_parameter(sema, param_idx)) {
                    handle_df_array_parameter(ctx, inst, arg, delete_list);
                    continue;
                }

                ctx.move_to_body_start();

                // special handling for handle parameters
                if (is_elemental && param_idx == dag_call->get_argument_count() - 1 &&
                        strcmp(dag_call->get_parameter_name(param_idx), "handle") == 0) {
                    MDL_ASSERT(is<DAG_constant>(arg) && "DF handle must be a constant");
                    if (DAG_constant const *handle_const = as<DAG_constant>(arg)) {
                        IValue const *handle_val = handle_const->get_value();
                        IValue_string const *handle_str = as<IValue_string>(handle_val);
                        MDL_ASSERT(handle_str && "DF handle must be string");
                        if (handle_str) {
                            char const *handle_name = handle_str->get_value();

                            int handle_id = -1;
                            for (size_t i = 0, n = m_dist_func->get_main_func_df_handle_count(
                                    m_cur_main_func_index); i < n; ++i) {
                                if (strcmp(handle_name, m_dist_func->get_main_func_df_handle(
                                        m_cur_main_func_index, i)) == 0) {
                                    handle_id = i;
                                    break;
                                }
                            }

                            MDL_ASSERT(handle_id != -1);
                            Expression_result res = Expression_result::value(
                                ctx.get_constant(handle_id));
                            inst->replaceAllUsesWith(res.as_ptr(ctx));
                            continue;
                        }
                    }
                }

                Expression_result res = translate_call_arg(ctx, arg, elem_type);
                inst->replaceAllUsesWith(res.as_ptr(ctx));
                continue;
            }

            if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(II)) {
                // check for calls to BSDFs
                int param_idx = get_metadata_df_param_id(call, kind);
                if (param_idx >= 0) {
                    // instantiate the BSDF function according to the DAG call argument
                    DAG_node const *arg = dag_call->get_argument(param_idx);

                    // identify access to is_black() by checking for bool return type
                    llvm::Type *bool_type = llvm::IntegerType::get(m_llvm_context, 1);
                    if (call->getType() == bool_type) {
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
                    } else {
                        Distribution_function_state old_state = m_dist_func_state;
                        m_dist_func_state = get_dist_func_state_from_call(call);

                        llvm::Value *param_bsdf_func = instantiate_df(arg);

                        // call it with state parameters added
                        llvm::SmallVector<llvm::Value *, 4> llvm_args;
                        llvm_args.push_back(call->getArgOperand(0));  // res_pointer
                        llvm_args.push_back(ctx.has_exec_ctx_parameter()
                            ? ctx.get_exec_ctx_parameter() : ctx.get_state_parameter());
                        llvm_args.push_back(call->getArgOperand(2));  // inherited_normal
                        if (m_dist_func_state == DFSTATE_EVALUATE ||
                            m_dist_func_state == DFSTATE_AUXILIARY)
                                llvm_args.push_back(call->getArgOperand(3));  // inherited_weight
                        llvm::CallInst::Create(param_bsdf_func, llvm_args, "", call);

                        m_dist_func_state = old_state;
                    }

                    // mark call instruction for deletion
                    delete_list.push_back(call);
                    continue;
                }

                llvm::Function *called_func = call->getCalledFunction();
                if (called_func == NULL) continue;   // ignore indirect function invocation

                // check for calls to special functions
                llvm::StringRef func_name = called_func->getName();
                if (!func_name.startswith("get_"))
                    continue;

                IDistribution_function::Special_kind special_kind;
                if (func_name == "get_material_ior")
                    special_kind = IDistribution_function::SK_MATERIAL_IOR;
                else if (func_name == "get_material_thin_walled")
                    special_kind = IDistribution_function::SK_MATERIAL_THIN_WALLED;
                else if (func_name == "get_material_volume_absorption_coefficient")
                    special_kind = IDistribution_function::SK_MATERIAL_VOLUME_ABSORPTION;
                else
                    continue;

                size_t index = m_dist_func->get_special_lambda_function_index(special_kind);
                MDL_ASSERT(index != ~0 && "Invalid special lambda function");

                ctx->SetInsertPoint(call);

                // determine expected return type (either type of call or from first argument)
                llvm::Type *expected_type = call->getType();
                if (expected_type == llvm::Type::getVoidTy(m_llvm_context))
                    expected_type = call->getArgOperand(0)->getType()->getPointerElementType();

                Expression_result res = translate_precalculated_lambda(ctx, index, expected_type);
                if (call->getType() != expected_type)
                    ctx->CreateStore(res.as_value(ctx), call->getArgOperand(0));
                else
                    call->replaceAllUsesWith(res.as_value(ctx));

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

    m_instantiated_dfs[m_dist_func_state][dag_call] = bsdf_func;

    return bsdf_func;
}


// Translate a DAG node pointing to a DF to LLVM IR.
Expression_result LLVM_code_generator::translate_distribution_function(
    Function_context                     &ctx,
    DAG_node const                       *df_node,
    llvm::SmallVector<unsigned, 8> const &lambda_result_exprs,
    llvm::GlobalVariable                 *mat_data_global)
{
    MDL_ASSERT(
        is<IType_df>(df_node->get_type()->skip_type_alias())
        && (
        (
            is<DAG_call>(df_node) &&
            (
                is_df_semantics(cast<DAG_call>(df_node)->get_semantic())
                ||
                cast<DAG_call>(df_node)->get_semantic() == operator_to_semantic(
                    IExpression::OK_TERNARY)
                )
            ) || (
                df_node->get_kind() == DAG_node::EK_CONSTANT &&
                cast<DAG_constant>(df_node)->get_value()->get_kind() == IValue::VK_INVALID_REF
                )
            )
    );

    // allocate the lambda results struct and make it available in the context
    llvm::Value *lambda_results = NULL;
    if (target_supports_lambda_results_parameter()) {
        lambda_results = ctx.create_local(m_lambda_results_struct_type, "lambda_results");
        ctx.override_lambda_results(
            ctx->CreateBitCast(lambda_results, m_type_mapper.get_void_ptr_type()));
    }

    // calculate all required non-constant expression lambdas
    for (size_t i = 0, n = lambda_result_exprs.size(); i < n; ++i) {
        size_t expr_index = lambda_result_exprs[i];

        generate_expr_lambda_call(ctx, expr_index, lambda_results, i);
    }

    // get the current normal
    IDefinition const *def = m_compiler->find_stdlib_signature("::state", "normal()");
    llvm::Function *func = get_intrinsic_function(def, /*return_derivs=*/ false);
    llvm::Value *args[] = { ctx.get_state_parameter() };
    llvm::Value *normal = call_rt_func(ctx, func, args);

    // convert to type used in libbsdf
    llvm::Value *normal_buf = ctx.create_local(m_float3_struct_type, "normal_buf");
    ctx.convert_and_store(normal, normal_buf);

    // initialize evaluate and auxiliary data
    mi::mdl::IType::Kind df_kind = df_node->get_type()->get_kind();
    llvm::Constant *zero = ctx.get_constant(0.0f);
    if (m_dist_func_state == DFSTATE_EVALUATE || m_dist_func_state == DFSTATE_AUXILIARY) {

        llvm::Constant *elems[] = {zero, zero, zero};

        // no handles
        if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_NONE) {
            llvm::Value *value_ptr = NULL;
            if (df_kind == mi::mdl::IType::TK_BSDF || df_kind == mi::mdl::IType::TK_HAIR_BSDF) {
                switch (m_dist_func_state)
                {
                    case DFSTATE_EVALUATE: {

                        value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(4)) }); // bsdf_diffuse
                        ctx->CreateStore(
                            llvm::ConstantStruct::get(m_float3_struct_type, elems), value_ptr);

                        value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(5)) }); // bsdf_glossy
                        ctx->CreateStore(
                            llvm::ConstantStruct::get(m_float3_struct_type, elems), value_ptr);
                        break;
                    }
                    case DFSTATE_AUXILIARY: {
                        value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(3)) }); // albedo
                        ctx->CreateStore(
                            llvm::ConstantStruct::get(m_float3_struct_type, elems), value_ptr);

                        value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(4)) }); // normal
                        ctx->CreateStore(
                            llvm::ConstantStruct::get(m_float3_struct_type, elems), value_ptr);
                        break;
                    }
                    default:
                        break;
                }
            } else if (df_kind == mi::mdl::IType::TK_EDF && m_dist_func_state == DFSTATE_EVALUATE) {
                value_ptr = ctx->CreateGEP(
                    ctx.get_function()->arg_begin(),
                    { ctx.get_constant(int(0)), ctx.get_constant(int(2)) });        // edf
                ctx->CreateStore(
                    llvm::ConstantStruct::get(m_float3_struct_type, elems), value_ptr);
            }
        }
        // fixed size array or user data
        else {
            // number of elements in the buffer/array
            llvm::Value *handle_count = NULL;
            if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_POINTER) { // DF_HSM_POINTER
                int handle_count_idx = -1;
                if (df_kind == mi::mdl::IType::TK_BSDF || df_kind == mi::mdl::IType::TK_HAIR_BSDF)
                    handle_count_idx = m_dist_func_state == DFSTATE_EVALUATE ? 5 : 4;
                else if (df_kind == mi::mdl::IType::TK_EDF)
                    handle_count_idx = m_dist_func_state == DFSTATE_EVALUATE ? 2 : -1;

                if(handle_count_idx >=0)
                    handle_count = ctx->CreateLoad(
                        ctx.create_simple_gep_in_bounds(
                            ctx.get_function()->arg_begin(), handle_count_idx));
            } else {                                                            // DF_HSM_FIXED_X
                handle_count = ctx.get_constant(
                    static_cast<int>(m_link_libbsdf_df_handle_slot_mode));
            }
            
            if (handle_count)
            {
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
                if (df_kind == mi::mdl::IType::TK_BSDF || df_kind == mi::mdl::IType::TK_HAIR_BSDF) {
                    value_0_idx = 
                        m_dist_func_state == DFSTATE_EVALUATE ? 5 : 4; // bsdf_diffuse/albedo
                    value_1_idx = 
                        m_dist_func_state == DFSTATE_EVALUATE ? 6 : 5; // bsdf_specular/normal
                }
                else if (df_kind == mi::mdl::IType::TK_EDF &&
                         m_dist_func_state == DFSTATE_EVALUATE) {
                    value_0_idx = 3;                                  // edf
                }

                // get pointer and write zeros
                if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_POINTER)
                {
                    // for user buffers there is an additional 'handle_count' -> +1
                    if (value_0_idx >= 0)
                    {
                        llvm::Value *result_value_ptr_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(value_0_idx + 1)) });
                        llvm::Value *result_value_ptr = ctx->CreateLoad(result_value_ptr_ptr);
                        result_value_ptr = ctx->CreateGEP(result_value_ptr, cur_index);
                        ctx->CreateStore(llvm::ConstantStruct::get(
                            m_float3_struct_type, elems), result_value_ptr);
                    }

                    if (value_1_idx >= 0)
                    {
                        llvm::Value *result_value_ptr_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(value_1_idx + 1)) });
                        llvm::Value *result_value_ptr = ctx->CreateLoad(result_value_ptr_ptr);
                        result_value_ptr = ctx->CreateGEP(result_value_ptr, cur_index);
                        ctx->CreateStore(llvm::ConstantStruct::get(
                            m_float3_struct_type, elems), result_value_ptr);
                    }
                }
                else
                { // m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_FIXED_X
                    if (value_0_idx >= 0)
                    {
                        llvm::Value *result_value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), 
                              ctx.get_constant(int(value_0_idx)), 
                              cur_index });
                        ctx->CreateStore(llvm::ConstantStruct::get(
                            m_float3_struct_type, elems), result_value_ptr);
                    }

                    if (value_1_idx >= 0)
                    {
                        llvm::Value *result_value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), 
                              ctx.get_constant(int(value_1_idx)), 
                              cur_index });
                        ctx->CreateStore(llvm::ConstantStruct::get(
                            m_float3_struct_type, elems), result_value_ptr);
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
            ctx.get_exc_state_parameter(),
            ctx.create_simple_gep_in_bounds(exec_ctx, 2u));
        ctx->CreateStore(
            ctx.get_cap_args_parameter(),
            ctx.create_simple_gep_in_bounds(exec_ctx, 3u));
        ctx->CreateStore(
            ctx.get_lambda_results_parameter(),  // actually our overridden local struct
            ctx.create_simple_gep_in_bounds(exec_ctx, 4u));
    }
    // recursively instantiate the DF
    llvm::Function *df_func = instantiate_df(df_node);
    if (df_func == NULL) {
        MDL_ASSERT(!"BSDF instantiation failed");
        return Expression_result::undef(lookup_type(df_node->get_type()));
    }

    // call the instantiated distribution function
    llvm::SmallVector<llvm::Value *, 4> df_args;
    df_args.push_back(ctx.get_function()->arg_begin());  // result pointer
    df_args.push_back(exec_ctx ? exec_ctx : ctx.get_state_parameter());
    df_args.push_back(normal_buf);
    if (m_dist_func_state == DFSTATE_EVALUATE || m_dist_func_state == DFSTATE_AUXILIARY) {
        llvm::Value *weight_buf = ctx.create_local(m_float3_struct_type, "weight"); 
        llvm::Constant *one = ctx.get_constant(1.0f);                   //inherited_weight
        llvm::Constant *elems[] = { one, one, one };
        ctx->CreateStore(llvm::ConstantStruct::get(m_float3_struct_type, elems), weight_buf);
        df_args.push_back(weight_buf);
    }
    llvm::CallInst *callinst = ctx->CreateCall(df_func, df_args);

    if ((df_kind == mi::mdl::IType::TK_BSDF || df_kind == mi::mdl::IType::TK_HAIR_BSDF) && 
        m_dist_func_state == DFSTATE_AUXILIARY)
    {
        // normalize function
        IDefinition const *norm_def = m_compiler->find_stdlib_signature(
            "::math", "normalize(float3)");
        llvm::Function *norm_func = get_intrinsic_function(norm_def, /*return_derivs=*/ false);

        // no handles
        if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_NONE)
        {
            // find normal in the data structure (element at index 4)
            llvm::Value *result_normal_ptr = ctx->CreateGEP(
                ctx.get_function()->arg_begin(),  // result pointer
                { ctx.get_constant(int(0)), ctx.get_constant(int(4)) });

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
            llvm::Value *result_normalized = call_rt_func(ctx, norm_func, {result_normal});
            ctx.convert_and_store(result_normalized, result_normal_ptr);
            ctx->CreateBr(if_non_zero_block_end);

            ctx->SetInsertPoint(if_non_zero_block_end);

            return Expression_result::value(callinst);
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
        llvm::BasicBlock *loop_block = ctx.create_bb("normal_loop");
        llvm::BasicBlock *loop_block_end = ctx.create_bb("normal_loop_end");

        llvm::Value *index_ptr = ctx.create_local(m_type_mapper.get_int_type(), "normal_index");
        ctx->CreateStore(ctx.get_constant(int(0)), index_ptr);

        // start loop
        ctx->CreateBr(loop_block);
        ctx->SetInsertPoint(loop_block);
        llvm::Value *cur_index = ctx->CreateLoad(index_ptr);

        // get a pointer to the normal at the current index
        llvm::Value *result_normal_ptr = NULL;
        if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_POINTER)
        {
            llvm::Value *result_normal_ptr_ptr = ctx->CreateGEP(
                ctx.get_function()->arg_begin(),
                {ctx.get_constant(int(0)), ctx.get_constant(int(6))});
            result_normal_ptr = ctx->CreateLoad(result_normal_ptr_ptr);
            result_normal_ptr = ctx->CreateGEP(result_normal_ptr, cur_index);
        } else { // m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_FIXED_X
            result_normal_ptr = ctx->CreateGEP(
                ctx.get_function()->arg_begin(),
                {ctx.get_constant(int(0)), ctx.get_constant(int(5)), cur_index});
        }

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
        llvm::Value *result_normalized = call_rt_func(ctx, norm_func, {result_normal});
        ctx.convert_and_store(result_normalized, result_normal_ptr);
        ctx->CreateBr(if_non_zero_block_end);

        ctx->SetInsertPoint(if_non_zero_block_end);

        // increment index, next iteration or end of loop 
        llvm::Value *new_index = ctx->CreateAdd(cur_index, ctx.get_constant(1));
        ctx->CreateStore(new_index, index_ptr);
        llvm::Value *cond = ctx->CreateICmpSLT(new_index, handle_count);
        ctx->CreateCondBr(cond, loop_block, loop_block_end);
        ctx->SetInsertPoint(loop_block_end);
    }
    return Expression_result::value(callinst);
}

// Translate the init function of a distribution function to LLVM IR.
void LLVM_code_generator::translate_distribution_function_init(
    Function_context &ctx,
    llvm::SmallVector<unsigned, 8> const &texture_result_exprs,
    llvm::SmallVector<unsigned, 8> const &lambda_result_exprs)
{
    // allocate the lambda results struct and make it available in the context
    llvm::Value *lambda_results = NULL;
    if (target_supports_lambda_results_parameter()) {
        lambda_results = ctx.create_local(m_lambda_results_struct_type, "lambda_results");
        ctx.override_lambda_results(
            ctx->CreateBitCast(lambda_results, m_type_mapper.get_void_ptr_type()));
    }

    // call state::get_texture_results()
    llvm::Value *texture_results = NULL;
    if (texture_result_exprs.size() != 0)
        texture_results = get_texture_results(ctx);

    // calculate the normal from geometry.normal
    llvm::Value *normal = NULL;

    size_t geometry_normal_index = m_dist_func->get_special_lambda_function_index(
        IDistribution_function::SK_MATERIAL_GEOMETRY_NORMAL);
    if (geometry_normal_index != ~0) {
        // SK_MATERIAL_GEOMETRY_NORMAL is only set, if it is not state::normal().
        // we need to update the state in this case

        // calculate all texture results required to calculate geometry.normal
        for (size_t i = 0, n = texture_result_exprs.size(); i < n; ++i) {
            size_t expr_index = texture_result_exprs[i];
            if (expr_index > geometry_normal_index) continue;

            generate_expr_lambda_call(ctx, expr_index, texture_results, i);
        }

        // calculate all non-constant expression lambdas required to calculate geometry.normal
        // (this may include geometry.normal itself, if it is reused).
        // The Expr_lambda_scheduler ensures the correct order of the expression lambdas.
        for (size_t i = 0, n = lambda_result_exprs.size(); i < n; ++i) {
            size_t expr_index = lambda_result_exprs[i];
            if (expr_index > geometry_normal_index) continue;

            generate_expr_lambda_call(ctx, expr_index, lambda_results, i);
        }

        normal = translate_precalculated_lambda(
            ctx,
            geometry_normal_index,
            m_type_mapper.get_float3_type()).as_value(ctx);

        // call state::set_normal(normal)
        llvm::Function *set_func = get_internal_function(m_int_func_state_set_normal);
        llvm::Value *set_normal_args[] = {
            ctx.get_state_parameter(),
            normal
        };
        call_rt_func_void(ctx, set_func, set_normal_args);
    }

    // calculate remaining texture results depending on evaluated geometry.normal
    for (size_t i = 0, n = texture_result_exprs.size(); i < n; ++i) {
        size_t expr_index = texture_result_exprs[i];

        // already processed during geometry.normal evaluation?
        if (geometry_normal_index != ~0 && expr_index <= geometry_normal_index) continue;

        generate_expr_lambda_call(ctx, expr_index, texture_results, i);
    }
}

} // mdl
} // mi

