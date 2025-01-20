/***************************************************************************************************
 * Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Linker/Linker.h>

#include "mdl/compiler/compilercore/compilercore_errors.h"
#include "mdl/codegenerators/generator_dag/generator_dag_lambda_function.h"
#include "mdl/codegenerators/generator_dag/generator_dag_tools.h"
#include "mdl/codegenerators/generator_dag/generator_dag_walker.h"

#include "generator_jit.h"
#include "generator_jit_llvm.h"


#define DEBUG_TYPE "df_instantiation"

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
        size_t     expr_lambda_index,
        llvm::Type *llvm_ret_type,
        int        cost,
        unsigned   size)
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
        switch (call->get_semantic()) {
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
        DAG_ir_walker walker(alloc);
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
        IAllocator                      *alloc,
        llvm::LLVMContext               &llvm_context,
        llvm::DataLayout const          *data_layout,
        Type_mapper                     &type_mapper,
        llvm::StructType                *float3_struct_type,
        unsigned                        num_texture_results,
        Distribution_function const     &dist_func,
        mi::mdl::vector<int>::Type      &lambda_result_indices,
        mi::mdl::vector<int>::Type      &texture_result_indices,
        mi::mdl::vector<unsigned>::Type &texture_result_offsets,
        llvm::SmallVector<unsigned, 8>  &lambda_result_exprs_init,
        llvm::SmallVector<unsigned, 8>  &lambda_result_exprs_others,
        llvm::SmallVector<unsigned, 8>  &texture_result_exprs)
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
            if (i == geometry_normal_index) {
                continue;
            }

            // not worth storing the result?
            if (cost < Cost_calculator::MIN_STORE_RESULT_COST) {
                continue;
            }

            // constants are neither materialized as functions nor stored in the lambda results
            if (is<DAG_constant>(lambda.get_body())) {
                continue;
            }

            // don't store matrices in lambda results, they are far too expensive
            IType const *mdl_type = lambda.get_return_type();
            if (is<IType_matrix>(m_type_mapper.skip_deriv_type(mdl_type))) {
                continue;
            }

            // determine the size of the result
            llvm::Type *lambda_ret_type = m_type_mapper.lookup_type(m_llvm_context, mdl_type);

            // replace lambda float3 types by float3 struct type used in libbsdf
            if (lambda_ret_type == m_type_mapper.get_float3_type()) {
                lambda_ret_type = m_float3_struct_type;
            }

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
        if (m_lambda_slots.empty()) {
            return;
        }

        // sort the expression lambdas by cost per byte.
        mi::mdl::vector<Lambda_result_slot *>::Type sorted_lambda_slots(m_alloc);
        sorted_lambda_slots.resize(m_lambda_slots.size());
        for (size_t i = 0, n = m_lambda_slots.size(); i < n; ++i) {
            sorted_lambda_slots[i] = &m_lambda_slots[i];
        }
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
            if (m_texture_result_indices[expr_index] != -1) {
                continue;
            }

            Lambda_info &lambda_info = m_lambda_infos[expr_index];
            Lambda_info::Index_set &deps = lambda_info.dep_expr_indices;

            // calculate required size including dependencies which have not yet been added
            // (estimate as it ignores alignment)
            size_t required_size = lambda_info.local_size;
            bool needs_deps = false;
            for (Lambda_info::Index_set::const_iterator it(deps.begin()), end(deps.end());
                it != end;
                ++it)
            {
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
                    for (Lambda_info::Index_set::const_iterator it(deps.begin()), end(deps.end());
                        it != end;
                        ++it)
                    {
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
        if (ignore_lambda_results) {
            return;
        }

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
        typedef ptr_hash_set<DAG_node const>::Type Node_set;
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
        void calc_dependencies(DAG_node const* expr, mi::mdl::vector<Lambda_info>::Type& infos) {
            Node_set visited_nodes(
                0, Node_set::hasher(), Node_set::key_equal(), infos.get_allocator());
            do_calc_dependencies(expr, infos, visited_nodes);
        }

    private:
        /// Calculate the dependencies based on the given expression.
        void do_calc_dependencies(
                DAG_node const *expr,
                mi::mdl::vector<Lambda_info>::Type &infos,
                Node_set &visited_nodes)
        {
            if (visited_nodes.count(expr)) {
                return;
            }
            visited_nodes.insert(expr);

            switch (expr->get_kind()) {
            case DAG_node::EK_TEMPORARY:
                {
                    // should not happen, but we can handle it
                    DAG_temporary const *t = mi::mdl::cast<DAG_temporary>(expr);
                    expr = t->get_expr();
                    do_calc_dependencies(expr, infos, visited_nodes);
                    return;
                }
            case DAG_node::EK_CONSTANT:
            case DAG_node::EK_PARAMETER:
                return;
            case DAG_node::EK_CALL:
                {
                    DAG_call const *call = mi::mdl::cast<DAG_call>(expr);
                    if (call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA) {
                        size_t lambda_index = LLVM_code_generator::get_lambda_index_from_call(call);
                        add_dependency(unsigned(lambda_index), infos);
                        return;
                    }

                    int n_args = call->get_argument_count();
                    for (int i = 0; i < n_args; ++i) {
                        do_calc_dependencies(call->get_argument(i), infos, visited_nodes);
                    }
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
        for (Lambda_info::Index_set::const_iterator it(deps.begin()), end(deps.end());
            it != end;
            ++it)
        {
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

        if (for_init_func) {
            m_lambda_result_exprs_init.push_back(expr_index);
        } else {
            m_lambda_result_exprs_others.push_back(expr_index);
        }
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
        LLVM_code_generator::Distribution_function_state old_state = m_code_gen.m_dist_func_state;
        m_code_gen.m_dist_func_state = state;

        llvm::SmallVector<llvm::Function *, 8> comp_funcs;
        for (DAG_node const *node : m_component_dfs) {
            comp_funcs.push_back(m_code_gen.instantiate_df(caller_ctx, node));
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

    // BSDF_API float3 thin_film_bsdf_get_factor(
    //     BSDF_evaluate_data *data, Execution_context *ctx, float3 *inherited_normal)

    llvm::Type *arg_types_get_factor[] = {
        Type_mapper::get_ptr(m_type_bsdf_evaluate_data),
        second_param_type,
        float3_struct_ptr_type
    };

    m_type_bsdf_get_factor_func = llvm::FunctionType::get(
        m_float3_struct_type, arg_types_get_factor, false);

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

    // BSDF_API float3 tint_edf_get_factor(
    //     EDF_evaluate_data *data, Execution_context *ctx, float3 *inherited_normal)

    llvm::Type *arg_types_get_factor[] = {
        Type_mapper::get_ptr(m_type_edf_evaluate_data),
        second_param_type,
        float3_struct_ptr_type
    };

    m_type_edf_get_factor_func = llvm::FunctionType::get(
        m_float3_struct_type, arg_types_get_factor, false);

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
    if (m_texruntime_with_derivs) {
        m_deriv_infos = dist_func.get_derivative_infos();
    }

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

    // for HLSL and GLSL we don't support lambda results
    lambda_sched.schedule_lambdas(/*ignore_lambda_results=*/ target_is_structured_language());

    m_texture_results_struct_type = lambda_sched.create_texture_results_type();
    m_lambda_results_struct_type  = lambda_sched.create_lambda_results_type();

    size_t expr_lambda_count = m_dist_func->get_expr_lambda_count();

    // now generate LLVM functions for all non-constant expression lambdas
    for (size_t i = 0; i < expr_lambda_count; ++i) {
        mi::base::Handle<mi::mdl::ILambda_function> expr_lambda(m_dist_func->get_expr_lambda(i));
        Lambda_function &lambda = *impl_cast<Lambda_function>(expr_lambda.get());

        // constants are neither materialized as functions nor stored in the lambda results
        if (is<DAG_constant>(lambda.get_body())) {
            continue;
        }

        reset_lambda_state();

        // generic functions return the result by reference if supported
        m_lambda_force_sret         = m_lambda_return_mode == Return_mode::RETMODE_SRET;

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

        add_generated_attributes(func);

        // if the result is not returned as an out parameter, mark the lambda function as read-only
        if ((flags & LLVM_context_data::FL_SRET) == 0) {
            func->setOnlyReadsMemory();
        }

        // ensure the function is finished by putting it into a block
        {
            Function_instance inst(alloc, &lambda, target_supports_storage_spaces());
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

        if (main_function_indices != NULL) {
            main_function_indices[0] = m_exported_func_list.size();
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

    for (size_t idx = 0, n_main = dist_func.get_main_function_count(); idx < n_main; ++idx) {
        m_cur_main_func_index = idx;

        mi::base::Handle<mi::mdl::ILambda_function> main_func(m_dist_func->get_main_function(idx));
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

        if (main_function_indices != NULL) {
            main_function_indices[idx + 1] = m_exported_func_list.size();
        }

        llvm::Twine base_name(lambda.get_name());
        Function_instance inst(get_allocator(), &lambda, target_supports_storage_spaces());

        // Don't allow returning structs at ABI level, even in value mode
        m_lambda_force_sret = m_lambda_return_mode == Return_mode::RETMODE_SRET
            || (m_lambda_return_mode == Return_mode::RETMODE_VALUE &&
                is<mi::mdl::IType_struct>(lambda.get_return_type()->skip_type_alias()));

        // only force, when actually supported by backend
        m_lambda_force_sret &= target_supports_sret_for_lambda();

        // non-distribution function?
        if (dist_kind == IGenerated_code_executable::DK_NONE) {
            m_dist_func_state = Distribution_function_state(DFSTATE_NONE);

            LLVM_context_data *ctx_data = get_or_create_context_data(&lambda);
            llvm::Function    *func     = ctx_data->get_function();
            unsigned          flags     = ctx_data->get_function_flags();

            add_generated_attributes(func);

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

            // allocate the lambda results struct and make it available in the context
            llvm::Value* lambda_results = NULL;
            if (target_supports_lambda_results_parameter()) {
                lambda_results = context.create_local(
                    m_lambda_results_struct_type, "lambda_results");
                context.override_lambda_results(
                    context->CreateBitCast(lambda_results, m_type_mapper.get_void_ptr_type()));
            }

            // calculate all required non-constant expression lambdas
            for (size_t j = 0, n_others = lambda_result_exprs_others.size(); j < n_others; ++j) {
                size_t expr_index = lambda_result_exprs_others[j];
                size_t result_index = m_lambda_result_indices[expr_index];

                generate_expr_lambda_call(context, expr_index, lambda_results, result_index);
            }

            // translate function body
            Expression_result res = translate_node(context, lambda.get_body(), resolver);
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
            LLVM_context_data *ctx_data = declare_lambda(&lambda);
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
            for (size_t i = 0, n = dist_func.get_main_func_df_handle_count(m_cur_main_func_index);
                i < n;
                ++i)
            {
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
    if (m_deriv_infos != NULL) {
        compile_waiting_functions();
    }

    // reset some fields
    m_scatter_components_map.clear();
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
    llvm::BasicBlock::iterator &ii,
    Function_context           &ctx)
{
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

    add_promoted_arguments(ctx, promote, llvm_args);

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
                        if (!translate_libbsdf_runtime_call(call, II, ctx)) {
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

                    // introduce local variables for all used DF parameters
                    bool skipped_df_idx_inc = false;
                    for (int i = 0, df_idx = 0; old_arg_it != old_arg_end; ++i, ++old_arg_it) {
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
                                // for DF parameters, we mark the calls to the DF methods
                                // with metadata additionally to the local variables.
                                // We still need the meta data on the local variables,
                                // as the select methods have two DF parameters.
                                // The argument value is not necessary, but we keep it, in case
                                // the uses are not optimized away.
                                // Note: we don't do this for the DFs inside xDF_component!
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

// Generate a call to an expression lambda function.
Expression_result LLVM_code_generator::generate_expr_lambda_call(
    Function_context                &ctx,
    mi::mdl::ILambda_function const *expr_lambda,
    llvm::Value                     *opt_results_buffer,
    size_t                          opt_result_index)
{
    Lambda_function const *expr_lambda_impl = impl_cast<Lambda_function>(expr_lambda);

    // expression and special lambdas always return by reference via first parameter,
    // if supported by the target, unless "value" lambda return mode is used

    llvm::Function *func = m_module_lambda_funcs[m_module_lambda_index_map[expr_lambda]];
    llvm::Type *lambda_retptr_type = func->getFunctionType()->getParamType(0);
    llvm::Type *lambda_res_type = lambda_retptr_type->getPointerElementType();
    llvm::Type *dest_type = NULL;
    llvm::Value *res_pointer;

    llvm::Value *opt_dest_ptr = NULL;
    if (opt_results_buffer != NULL && !target_is_structured_language()) {
        opt_dest_ptr = ctx.create_simple_gep_in_bounds(
            opt_results_buffer, ctx.get_constant(int(opt_result_index)));
    }

    if (!func->getReturnType()->isVoidTy()) {
        // lambda function directly returns the result
        res_pointer = NULL;
    } else if (opt_dest_ptr != NULL &&
            (dest_type = opt_dest_ptr->getType()->getPointerElementType()) == lambda_res_type) {
        res_pointer = opt_dest_ptr;
    } else {
        res_pointer = ctx.create_local(lambda_res_type, "res_buf");
    }

    llvm::SmallVector<llvm::Value *, 6> lambda_args;
    if (res_pointer != NULL) {
        lambda_args.push_back(res_pointer);
    }
    lambda_args.push_back(ctx.get_state_parameter());
    if (target_uses_resource_data_parameter()) {
        lambda_args.push_back(ctx.get_resource_data_parameter());
    }
    if (target_uses_exception_state_parameter()) {
        lambda_args.push_back(ctx.get_exc_state_parameter());
    }
    if (target_supports_captured_argument_parameter()) {
        lambda_args.push_back(ctx.get_cap_args_parameter());
    }
    if (target_supports_lambda_results_parameter() && expr_lambda_impl->uses_lambda_results()) {
        lambda_args.push_back(ctx.get_lambda_results_parameter());
    }

    m_state_usage_analysis.add_call(ctx.get_function(), func);
    llvm::CallInst *call = ctx->CreateCall(func, lambda_args);
    if (res_pointer == NULL) {
        if (opt_results_buffer != NULL) {
            if (target_is_structured_language()) {
                store_to_float4_array(
                    ctx,
                    call,
                    opt_results_buffer,
                    m_texture_result_offsets[opt_result_index]);
            } else {
                ctx.convert_and_store(call, opt_dest_ptr);
                return Expression_result::ptr(opt_dest_ptr);
            }
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
    llvm::Value      *val,
    llvm::Value      *dest,
    unsigned         &dest_offs)
{
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
                ctx, llvm::GetElementPtrInst::getTypeAtIndex(val_type, unsigned(i)), src, src_offs);
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

    if (DAG_constant const *c = as<DAG_constant>(expr_lambda->get_body())) {
        // translate constants directly
        res = translate_value(ctx, c->get_value());
    } else if (m_texture_result_indices[lambda_index] != -1) {
        // is the result available in the texture results from the MDL SDK state
        if (target_is_structured_language()) {
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
    } else if (m_lambda_result_indices[lambda_index] != -1) {
        // was the result locally precalculated?
        res = Expression_result::ptr(ctx->CreateConstGEP2_32(
            nullptr,
            ctx->CreateBitCast(
                ctx.get_lambda_results_parameter(),
                m_type_mapper.get_ptr(m_lambda_results_struct_type)),
            0,
            unsigned(m_lambda_result_indices[lambda_index])));
    } else {
        // calculate on demand, should be cheap if we get here
        res = generate_expr_lambda_call(ctx, lambda_index);
    }

    // type doesn't matter or fits already?
    if (expected_type == NULL || res.get_value_type() == expected_type) {
        return res;
    }

    // convert to expected type
    return Expression_result::value(ctx.load_and_convert(expected_type, res.as_ptr(ctx)));
}

// Get the lambda index from a lambda DAG call.
size_t LLVM_code_generator::get_lambda_index_from_call(DAG_call const *call)
{
    MDL_ASSERT(call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA);
    size_t lambda_index = strtoul(call->get_name(), NULL, 10);
    return lambda_index;
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
        if (res.get_value_type() == expected_type) {
            return res;
        }

        return Expression_result::value(ctx.load_and_convert(expected_type, res.as_ptr(ctx)));
    }

    // determine expression lambda index
    MDL_ASSERT(arg->get_kind() == DAG_node::EK_CALL);
    DAG_call const *arg_call = mi::mdl::cast<DAG_call>(arg);
    size_t lambda_index = get_lambda_index_from_call(arg_call);

    return translate_precalculated_lambda(ctx, lambda_index, expected_type);
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
    Function_context                           &ctx,
    llvm::AllocaInst                           *inst,
    llvm::Value                                *weight_array,
    llvm::Value                                *df_flags_array,
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
                            ctx->SetInsertPoint(call);

                            llvm::Value *comp_val = ctx->CreateLoad(
                                ctx.create_simple_gep_in_bounds(df_flags_array, component_idx_val),
                                "df_flags");
                            llvm::Value *allowed_val = call->getArgOperand(0);
                            llvm::Value *union_val = ctx->CreateAnd(comp_val, allowed_val);
                            llvm::Value *comp = ctx->CreateICmpNE(union_val, ctx.get_constant(0));
                            call->replaceAllUsesWith(comp);
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

        // is it a constant BSDF_component array?
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
                ctx,
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
            size_t lambda_index = get_lambda_index_from_call(arg_call);
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

    // is it a non-constant BSDF_component array?
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
                weight_res = translate_value(ctx, weight_val);

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
                weight_res = translate_call_arg(ctx, weight_node, weight_type);

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
            ctx,
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
    // Optimize thin_film special case:
    //      cond ? thin_film(ior, thickness, base) : base
    //   -> thin_film(ior, cond ? thickness : 0, base)
    //
    //      cond ? base : thin_film(ior, thickness, base)
    //   -> thin_film(ior, cond ? 0 : thickness, base)
    if (dag_call->get_semantic() == operator_to_semantic(IExpression::OK_TERNARY) &&
        dag_call->get_type()->get_kind() == IType::TK_BSDF) {
        DAG_call const *true_node  = as<DAG_call>(dag_call->get_argument(1));
        DAG_call const *false_node = as<DAG_call>(dag_call->get_argument(2));
        if (true_node && false_node) {
            if (true_node->get_semantic() == IDefinition::DS_INTRINSIC_DF_THIN_FILM) {
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
            if (false_node->get_semantic() == IDefinition::DS_INTRINSIC_DF_THIN_FILM) {
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

    // Create a new function with the type for current distribution function state
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
        // Context needs a non-empty start block, so create a jump to a second block
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

        // Find lambda expression for condition and generate code
        DAG_node const *cond = dag_call->get_argument(0);
        Expression_result res = translate_call_arg(ctx, cond, m_type_mapper.get_bool_type());

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

        // Generate "if(cond)-then-else; return"

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
                true_inherited_weight_val = llvm::UndefValue::get(inherited_weight_val->getType());
                for (int i = 0; i < 3; ++i) {
                    true_inherited_weight_val = ctx.create_insert(
                        true_inherited_weight_val,
                        ctx->CreateFMul(
                            ctx.create_extract(factor_val, i),
                            ctx.create_extract(inherited_weight_val, i)),
                        i);
                }
            }

            // handle false_bsdf, if it is a factor BSDF
            //   cond ? common_node : factor(common_node)
            //   cond ? factor(common_node) : factor_2(common_node)
            llvm::Value *false_inherited_weight_val = inherited_weight_val;
            if (false_bsdf != common_node) {
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
                false_inherited_weight_val = llvm::UndefValue::get(inherited_weight_val->getType());
                for (int i = 0; i < 3; ++i) {
                    false_inherited_weight_val = ctx.create_insert(
                        false_inherited_weight_val,
                        ctx->CreateFMul(
                            ctx.create_extract(inherited_weight_val, i),
                            ctx.create_extract(factor, i)),
                        i);
                }
            }

            // execute common code in end block
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
            instantiate_and_call_df(
                ctx,
                true_bsdf,
                m_dist_func_state,
                res_pointer,
                inherited_normal,
                inherited_weight,
                true_term);

            // False case
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

    // Return the now finalized function
    return func;
}

// Returns true, if the given DAG node is a call to diffuse_reflection_bsdf(color(1), 0, color(0)).
bool LLVM_code_generator::is_default_diffuse_reflection(DAG_node const *node)
{
    // match diffuse_reflection_bsdf(*)
    DAG_call const *dag_call = as<DAG_call>(node);
    if (dag_call == NULL ||
        dag_call->get_semantic() != IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF)
    {
        return false;
    }

    // match tint argument as color(1.0f)
    DAG_constant const *tint_arg = as<DAG_constant>(dag_call->get_argument(0));
    if (tint_arg == NULL) {
        return false;
    }

    IValue_rgb_color const *tint_value = as<IValue_rgb_color>(tint_arg->get_value());
    if (tint_value == NULL || !tint_value->is_one()) {
        return false;
    }

    // match roughness argument as 0.0f
    DAG_constant const *roughness_arg = as<DAG_constant>(dag_call->get_argument(1));
    if (roughness_arg == NULL) {
        return false;
    }

    IValue_float const *roughness_value = as<IValue_float>(roughness_arg->get_value());
    if (roughness_value == NULL || !roughness_value->is_zero()) {
        return false;
    }

    // match multiscatter_tint argument as color(0.0f)
    DAG_constant const *multiscatter_arg = as<DAG_constant>(dag_call->get_argument(2));
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
    int scatter_arg_index = -1;

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
        scatter_arg_index = 1;
        break;

    case IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF:
        scatter_arg_index = 2;
        break;

    case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
        scatter_arg_index = 5;
        break;

    case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFLAKE_SHEEN_BSDF:
        {
            Df_flags multiscatter_comps = get_bsdf_scatter_components(dag_call->get_argument(3));
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
        // last argument is the base bsdf
        res = get_bsdf_scatter_components(
            dag_call->get_argument(dag_call->get_argument_count() - 1));
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
            // the last three parameters are layer, base and normal
            Df_flags layer_comps = get_bsdf_scatter_components(
                dag_call->get_argument(dag_call->get_argument_count() - 3));
            Df_flags base_comps = get_bsdf_scatter_components(
                dag_call->get_argument(dag_call->get_argument_count() - 2));
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
    if (scatter_arg_index != -1) {
        DAG_node const *scatter_arg = dag_call->get_argument(scatter_arg_index);
        if (DAG_constant const *scatter_const = as<DAG_constant>(scatter_arg)) {
            int scatter_const_int = cast<IValue_enum>(scatter_const->get_value())->get_value();
            res = Df_flags(scatter_const_int + 1);
        } else {
            // for now, if we get a lambda expression call here (can be material parameter or code),
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
    Distribution_function_state old_state = m_dist_func_state;
    m_dist_func_state = df_state;

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

    m_dist_func_state = old_state;

    return call;
}

// Recursively instantiate a BSDF specified by the given DAG node from code in the BSDF library
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

    while (inner !=  NULL && inner->get_semantic() == IDefinition::DS_INTRINSIC_DF_THIN_FILM) {
        thin_film_node = inner;
        arg            = thin_film_node->get_argument(2);
        inner          = cast<DAG_call>(arg);
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

    // get DF function according to semantics and current state
    // and clone it into the current module.

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

    llvm::ValueToValueMapTy ValueMap;
    llvm::Function *bsdf_func = llvm::CloneFunction(df_lib_func, ValueMap);
    add_generated_attributes(bsdf_func);
    if (m_enable_noinline && !is_always_inline_enabled()) {
        bsdf_func->addFnAttr(llvm::Attribute::NoInline);
    }
    m_state_usage_analysis.register_cloned_function(bsdf_func, df_lib_func);

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

    // process all calls to BSDF parameter accessors
    size_t n_args = dag_call->get_argument_count();
    for (llvm::Function::iterator BI = bsdf_func->begin(), BE = bsdf_func->end(); BI != BE; ++BI) {
        for (llvm::BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
            if (llvm::AllocaInst *inst = llvm::dyn_cast<llvm::AllocaInst>(II)) {
                llvm::Type *elem_type = inst->getAllocatedType();

                // ignore BSDF and EDF struct allocas
                if (elem_type->isStructTy() &&
                        !llvm::cast<llvm::StructType>(elem_type)->isLiteral() &&
                        (elem_type->getStructName() == "struct.BSDF"
                        || elem_type->getStructName() == "struct.EDF") ) {
                    continue;
                }

                // check for lambda call BSDF parameters
                int param_idx = get_metadata_df_param_id(inst, kind);
                if (param_idx < 0) {
                    continue;
                }

                DAG_node const *arg = NULL;
                if (param_idx < n_args) {
                    // get the parameter from the BSDF call
                    arg = dag_call->get_argument(param_idx);
                } else {
                    // get extra parameter from the modifier
                    MDL_ASSERT(thin_film_node != NULL);
                    arg = thin_film_node->get_argument(param_idx - n_args);
                }

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
                        if (handle_str != nullptr) {
                            char const *handle_name = handle_str->get_value();

                            // translate the handle name into a handle ID
                            int handle_id = -1;
                            for (size_t i = 0, n = m_dist_func->get_main_func_df_handle_count(
                                    m_cur_main_func_index); i < n; ++i) {
                                if (strcmp(handle_name, m_dist_func->get_main_func_df_handle(
                                        m_cur_main_func_index, i)) == 0) {
                                    handle_id = i;
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

                Expression_result res = translate_call_arg(ctx, arg, elem_type);

                // in "ternary operator with thin_film" optimization mode and current parameter
                // is the coating_thickness?
                if (opt_ctx.m_ternary_cond != NULL && param_idx == n_args) {
                    // set thickness to 0, if condition says thin_film should be skipped
                    Expression_result cond_res = translate_call_arg(
                        ctx, opt_ctx.m_ternary_cond, m_type_mapper.get_bool_type());
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

                inst->replaceAllUsesWith(res.as_ptr(ctx));
                continue;
            }

            if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(II)) {
                // check for calls to BSDFs
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
                                    instantiate_and_call_df(
                                        ctx,
                                        true_arg,
                                        new_state,
                                        param_res_pointer,
                                        param_true_inherited_normal,
                                        nullptr,
                                        then_term);

                                    // handle false case
                                    instantiate_and_call_df(
                                        ctx,
                                        false_arg,
                                        new_state,
                                        param_res_pointer,
                                        param_false_inherited_normal,
                                        nullptr,
                                        else_term);

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

                IDistribution_function::Special_kind special_kind;
                if (func_name == "get_material_ior") {
                    special_kind = IDistribution_function::SK_MATERIAL_IOR;
                } else if (func_name == "get_material_thin_walled") {
                    special_kind = IDistribution_function::SK_MATERIAL_THIN_WALLED;
                } else if (func_name == "get_material_volume_absorption_coefficient") {
                    special_kind = IDistribution_function::SK_MATERIAL_VOLUME_ABSORPTION;
                } else {
                    continue;
                }

                size_t index = m_dist_func->get_special_lambda_function_index(special_kind);
                MDL_ASSERT(index != ~0 && "Invalid special lambda function");

                ctx->SetInsertPoint(call);

                // determine expected return type (either type of call or from first argument)
                llvm::Type *expected_type = call->getType();
                if (expected_type == llvm::Type::getVoidTy(m_llvm_context)) {
                    expected_type = call->getArgOperand(0)->getType()->getPointerElementType();
                }

                Expression_result res = translate_precalculated_lambda(ctx, index, expected_type);
                if (call->getType() != expected_type) {
                    ctx->CreateStore(res.as_value(ctx), call->getArgOperand(0));
                } else {
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
        size_t expr_index   = lambda_result_exprs[i];
        size_t result_index = m_lambda_result_indices[expr_index];

        generate_expr_lambda_call(ctx, expr_index, lambda_results, result_index);
    }

    // get the current normal
    llvm::Value *normal_buf;
    {
        IDefinition const *def = m_compiler->find_stdlib_signature("::state", "normal()");
        llvm::Function *func = get_intrinsic_function(def, /*return_derivs=*/ false);
        llvm::Value *args[] = { ctx.get_state_parameter() };
        llvm::Value *normal = call_rt_func(ctx, func, args);

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

        llvm::Constant *elems[] = {zero, zero, zero};

        if (m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_NONE) {
            // no handles
            llvm::Value *value_ptr = NULL;
            if (df_kind == mi::mdl::IType::TK_BSDF || df_kind == mi::mdl::IType::TK_HAIR_BSDF) {
                switch (m_dist_func_state) {
                case DFSTATE_EVALUATE:
                    {
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
                    }
                    break;
                case DFSTATE_AUXILIARY:
                    {
                        value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(3)) }); // albedo diff
                        ctx->CreateStore(
                            llvm::ConstantStruct::get(m_float3_struct_type, elems), value_ptr);

                        value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(4)) }); // albedo glos
                        ctx->CreateStore(
                            llvm::ConstantStruct::get(m_float3_struct_type, elems), value_ptr);

                        value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(5)) }); // normal
                        ctx->CreateStore(
                            llvm::ConstantStruct::get(m_float3_struct_type, elems), value_ptr);

                        value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(6)) }); // roughness
                        ctx->CreateStore(
                            llvm::ConstantStruct::get(m_float3_struct_type, elems), value_ptr);
                    }
                    break;
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
                        ctx.create_simple_gep_in_bounds(
                            ctx.get_function()->arg_begin(), handle_count_idx));
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
                    if (value_0_idx >= 0) {
                        llvm::Value *result_value_ptr_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(value_0_idx + 1)) });
                        llvm::Value *result_value_ptr = ctx->CreateLoad(result_value_ptr_ptr);
                        result_value_ptr = ctx->CreateGEP(result_value_ptr, cur_index);
                        ctx->CreateStore(llvm::ConstantStruct::get(
                            m_float3_struct_type, elems), result_value_ptr);
                    }

                    if (value_1_idx >= 0) {
                        llvm::Value *result_value_ptr_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(value_1_idx + 1)) });
                        llvm::Value *result_value_ptr = ctx->CreateLoad(result_value_ptr_ptr);
                        result_value_ptr = ctx->CreateGEP(result_value_ptr, cur_index);
                        ctx->CreateStore(llvm::ConstantStruct::get(
                            m_float3_struct_type, elems), result_value_ptr);
                    }

                    if (value_2_idx >= 0) {
                        llvm::Value* result_value_ptr_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(value_2_idx + 1)) });
                        llvm::Value* result_value_ptr = ctx->CreateLoad(result_value_ptr_ptr);
                        result_value_ptr = ctx->CreateGEP(result_value_ptr, cur_index);
                        ctx->CreateStore(llvm::ConstantStruct::get(
                            m_float3_struct_type, elems), result_value_ptr);
                    }
                    if (value_3_idx >= 0) {
                        llvm::Value* result_value_ptr_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), ctx.get_constant(int(value_3_idx + 1)) });
                        llvm::Value* result_value_ptr = ctx->CreateLoad(result_value_ptr_ptr);
                        result_value_ptr = ctx->CreateGEP(result_value_ptr, cur_index);
                        ctx->CreateStore(llvm::ConstantStruct::get(
                            m_float3_struct_type, elems), result_value_ptr);
                    }
                } else {
                    // m_link_libbsdf_df_handle_slot_mode == mi::mdl::DF_HSM_FIXED_X
                    if (value_0_idx >= 0) {
                        llvm::Value *result_value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), 
                              ctx.get_constant(int(value_0_idx)), 
                              cur_index });
                        ctx->CreateStore(llvm::ConstantStruct::get(
                            m_float3_struct_type, elems), result_value_ptr);
                    }

                    if (value_1_idx >= 0) {
                        llvm::Value *result_value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)), 
                              ctx.get_constant(int(value_1_idx)), 
                              cur_index });
                        ctx->CreateStore(llvm::ConstantStruct::get(
                            m_float3_struct_type, elems), result_value_ptr);
                    }

                    if (value_2_idx >= 0) {
                        llvm::Value* result_value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)),
                              ctx.get_constant(int(value_2_idx)),
                              cur_index });
                        ctx->CreateStore(llvm::ConstantStruct::get(
                            m_float3_struct_type, elems), result_value_ptr);
                    }

                    if (value_3_idx >= 0) {
                        llvm::Value* result_value_ptr = ctx->CreateGEP(
                            ctx.get_function()->arg_begin(),
                            { ctx.get_constant(int(0)),
                              ctx.get_constant(int(value_3_idx)),
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
        llvm::Value *weight_buf = ctx.create_local(m_float3_struct_type, "weight"); 
        llvm::Constant *one = ctx.get_constant(1.0f);                   //inherited_weight
        llvm::Constant *elems[] = { one, one, one };
        ctx->CreateStore(llvm::ConstantStruct::get(m_float3_struct_type, elems), weight_buf);
        df_args.push_back(weight_buf);
    }
    m_state_usage_analysis.add_call(ctx.get_function(), df_func);
    ctx->CreateCall(df_func, df_args);

    // at the end of the sample function, call the pdf function to calculate the pdf result
    if (m_dist_func_state == DFSTATE_SAMPLE) {
        bool is_edf = df_kind == mi::mdl::IType::TK_EDF;
        llvm::Value *pdf_data = ctx.create_local(
            is_edf ? m_type_edf_pdf_data : m_type_bsdf_pdf_data, "pdf_data");
        llvm::Value *sample_data = result_pointer;

        // copy over the values from the sample data to a pdf data struct
        if (is_edf) {
            // only k1 needs to be copied
            llvm::Value *k1_val =
                ctx->CreateLoad(ctx->CreateStructGEP(sample_data, 1));
            ctx->CreateStore(k1_val, ctx->CreateStructGEP(pdf_data, 0));
        } else {
            // copy first 4 struct fields
            for (unsigned i = 0; i < 4; ++i) {
                llvm::Value *data =
                    ctx->CreateLoad(ctx->CreateStructGEP(sample_data, i));
                ctx->CreateStore(data, ctx->CreateStructGEP(pdf_data, i));
            }

            // copy libbsdf flags if used
            if (m_libbsdf_flags_in_bsdf_data) {
                llvm::Value *flags =
                    ctx->CreateLoad(ctx->CreateStructGEP(sample_data, 9));
                ctx->CreateStore(flags, ctx->CreateStructGEP(pdf_data, 5));
            }
        }

        // instantiate pdf function
        Distribution_function_state old_state = m_dist_func_state;
        m_dist_func_state = DFSTATE_PDF;
        llvm::Function *param_bsdf_func = instantiate_df(ctx, df_node);
        m_dist_func_state = old_state;

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
            pdf_val, ctx->CreateStructGEP(sample_data, is_edf ? 2 : 5));
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
                llvm::Value *result_normalized = call_rt_func(ctx, norm_func, {result_normal});
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
            llvm::Value *result_normalized = call_rt_func(ctx, norm_func, { result_normal });
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
    if (texture_result_exprs.size() != 0) {
        texture_results = get_texture_results(ctx);
    }

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
            if (expr_index > geometry_normal_index) {
                continue;
            }

            generate_expr_lambda_call(ctx, expr_index, texture_results, i);
        }

        // calculate all non-constant expression lambdas required to calculate geometry.normal
        // (this may include geometry.normal itself, if it is reused).
        // The Expr_lambda_scheduler ensures the correct order of the expression lambdas.
        for (size_t i = 0, n = lambda_result_exprs.size(); i < n; ++i) {
            size_t expr_index = lambda_result_exprs[i];
            if (expr_index > geometry_normal_index) {
                continue;
            }

            size_t result_index = m_lambda_result_indices[expr_index];
            generate_expr_lambda_call(ctx, expr_index, lambda_results, result_index);
        }

        normal = translate_precalculated_lambda(
            ctx,
            geometry_normal_index,
            m_type_mapper.get_float3_type()).as_value(ctx);

        // call state::adapt_normal(normal), if requested
        if (m_use_renderer_adapt_normal) {
            llvm::Function *adapt_normal = get_internal_function(m_int_func_state_adapt_normal);
            llvm::SmallVector<llvm::Value *, 3> args;
            args.push_back(ctx.get_state_parameter());
            if (target_uses_resource_data_parameter()) {
                args.push_back(ctx.get_resource_data_parameter());
            }
            args.push_back(normal);
            normal = call_rt_func(ctx, adapt_normal, args);
        }

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
        if (geometry_normal_index != ~0 && expr_index <= geometry_normal_index) {
            continue;
        }

        generate_expr_lambda_call(ctx, expr_index, texture_results, i);
    }
}

} // mdl
} // mi

