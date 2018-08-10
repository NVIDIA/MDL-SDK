/***************************************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Linker.h>

#include "mdl/compiler/compilercore/compilercore_errors.h"
#include "mdl/codegenerators/generator_dag/generator_dag_lambda_function.h"
#include "mdl/codegenerators/generator_dag/generator_dag_tools.h"
#include "mdl/codegenerators/generator_dag/generator_dag_walker.h"

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
        sort_cost = cost * 16 / MISTD::max(size, 1u);
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
        IAllocator *alloc,
        llvm::LLVMContext &llvm_context,
        llvm::DataLayout const *data_layout,
        Type_mapper &type_mapper,
        llvm::StructType *float3_struct_type,
        unsigned num_texture_results,
        Distribution_function const &dist_func,
        mi::mdl::vector<int>::Type &lambda_result_indices,
        mi::mdl::vector<int>::Type &texture_result_indices,
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

        m_lambda_infos.reserve(expr_lambda_count);
    }

    /// Schedule the lambda functions and assign them to texture result and lambda result slots
    /// if necessary.
    void schedule_lambdas()
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

            // determine the size of the result
            IType const *mdl_type = lambda.get_return_type();
            llvm::Type *lambda_ret_type = m_type_mapper.lookup_type(m_llvm_context, mdl_type);

            // replace lambda float3 types by float3 struct type used in libbsdf, except for
            // geometry normal
            if (i != geometry_normal_index && lambda_ret_type == m_type_mapper.get_float3_type())
                lambda_ret_type = m_float3_struct_type;

            unsigned res_alloc_size = unsigned(m_data_layout->getTypeAllocSize(lambda_ret_type));

            m_lambda_infos.push_back(Lambda_info(m_alloc));
            Lambda_info &cur = m_lambda_infos.back();

            // due to construction in backends_backends.cpp,
            // all required lambdas are already available
            cur.calc_dependencies(lambda.get_body(), m_lambda_infos);

            // constants are neither materialized as functions nor stored in the lambda results
            if (is<DAG_constant>(lambda.get_body()))
                continue;

            // not worth storing the result?
            if (cost < Cost_calculator::MIN_STORE_RESULT_COST)
                continue;

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
        MISTD::sort(
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

        // if geometry.normal has to be calculated, collect required lambda results
        // for use in the bsdf init function
        if (geometry_normal_index != ~0 && m_texture_result_indices[geometry_normal_index] == -1) {
            Lambda_info &lambda_info = m_lambda_infos[geometry_normal_index];
            add_lambda_result_dep_entries(lambda_info, /*for_init_func=*/ true);
        }

        // determine which lambda results are required by the other bsdf functions
        // after the init function (so especially without geometry.normal)
        Lambda_info df_info(m_alloc);
        mi::base::Handle<ILambda_function> main_df(m_dist_func.get_main_df());
        df_info.calc_dependencies(main_df->get_body(), m_lambda_infos);

        add_lambda_result_dep_entries(df_info, /*for_init_func=*/ false);
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

// Prepare types and prototypes in the current LLVM module used by libbsdf.
void LLVM_code_generator::prepare_libbsdf_prototypes(llvm::Module *libbsdf)
{
    // prepare float3 type
    llvm::Type *members_float3_struct[] = {
        m_type_mapper.get_float_type(),        // x
        m_type_mapper.get_float_type(),        // y
        m_type_mapper.get_float_type(),        // z
    };
    m_float3_struct_type = llvm::StructType::create(
        m_llvm_context, members_float3_struct, "struct.float3", /*is_packed=*/false);

    // declare a dummy function in both the current and the libbsdf module with all non-basic
    // types (except the special BSDF type) used by BSDF function parameters to ensure proper
    // type mapping when linking libbsdf. Currently this is only:
    //  - float3
    //  - color -> float3
    llvm::Type *ret_tp = m_type_mapper.get_void_type();
    llvm::Type *arg_types[] = {
        m_float3_struct_type,
    };

    llvm::Function::Create(
        llvm::FunctionType::get(ret_tp, arg_types, false),
        llvm::GlobalValue::ExternalLinkage,
        "dummy_function_for_linking",
        m_module);

    // find the libbsdf float3 type by using the prototype of black_bsdf_sample:
    //   define void @black_bsdf_sample(%struct.BSDF_sample_data* %a, %struct.float3* %b)
    //
    // we cannot use libbsdf->getTypeByName() because it will just ask the context which
    // has the names of all modules
    llvm::Function *func = libbsdf->getFunction("black_bsdf_sample");
    MDL_ASSERT(func && "libbsdf must contain a black_bsdf_sample function!");
    llvm::Type *libbsdf_float3 = func->getFunctionType()->getParamType(1)->getPointerElementType();

    llvm::Type *arg_types_libbsdf[] = {
        libbsdf_float3
    };

    llvm::Function::Create(
        llvm::FunctionType::get(ret_tp, arg_types_libbsdf, false),
        llvm::GlobalValue::ExternalLinkage,
        "dummy_function_for_linking",
        libbsdf);
}

// Create the BSDF function types using the BSDF data types from the already linked libbsdf
// module.
void LLVM_code_generator::create_bsdf_function_types()
{
    // fetch the BSDF data types from the already linked libbsdf

    m_type_bsdf_sample_data = m_module->getTypeByName("struct.BSDF_sample_data");
    m_type_bsdf_evaluate_data = m_module->getTypeByName("struct.BSDF_evaluate_data");
    m_type_bsdf_pdf_data = m_module->getTypeByName("struct.BSDF_pdf_data");

    // create function types for the BSDF functions

    llvm::Type *ret_tp = m_type_mapper.get_void_type();
    llvm::Type *state_ptr_type = m_type_mapper.get_state_ptr_type(m_state_mode);
    llvm::Type *res_data_pair_ptr_type = m_type_mapper.get_res_data_pair_ptr_type();
    llvm::Type *exc_state_ptr_type = m_type_mapper.get_exc_state_ptr_type();
    llvm::Type *void_ptr_type = m_type_mapper.get_void_ptr_type();
    llvm::Type *float3_struct_ptr_type = Type_mapper::get_ptr(m_float3_struct_type);

    // BSDF_API void diffuse_reflection_bsdf_sample(
    //     BSDF_sample_data *data, MDL_SDK_State *state,
    //     MDL_SDK_Res_data_pair *, MDL_SDK_Exception_state *,
    //     void *captured_args, void *lambda_results, float3 *inherited_normal)

    llvm::Type *arg_types_sample[] = {
        Type_mapper::get_ptr(m_type_bsdf_sample_data),
        state_ptr_type,
        res_data_pair_ptr_type,
        exc_state_ptr_type,
        void_ptr_type,
        void_ptr_type,
        float3_struct_ptr_type
    };

    m_type_bsdf_sample_func = llvm::FunctionType::get(ret_tp, arg_types_sample, false);

    // BSDF_API void diffuse_reflection_bsdf_evaluate(
    //     BSDF_evaluate_data *data, MDL_SDK_State *state,
    //     MDL_SDK_Res_data_pair *, MDL_SDK_Exception_state *,
    //     void *captured_args, void *lambda_results, float3 *inherited_normal)

    llvm::Type *arg_types_eval[] = {
        Type_mapper::get_ptr(m_type_bsdf_evaluate_data),
        state_ptr_type,
        res_data_pair_ptr_type,
        exc_state_ptr_type,
        void_ptr_type,
        void_ptr_type,
        float3_struct_ptr_type
    };

    m_type_bsdf_evaluate_func = llvm::FunctionType::get(ret_tp, arg_types_eval, false);

    // BSDF_API void diffuse_reflection_bsdf_pdf(
    //     BSDF_pdf_data *data, MDL_SDK_State *state,
    //     MDL_SDK_Res_data_pair *, MDL_SDK_Exception_state *,
    //     void *captured_args, void *lambda_results, float3 *inherited_normal)

    llvm::Type *arg_types_pdf[] = {
        Type_mapper::get_ptr(m_type_bsdf_pdf_data),
        state_ptr_type,
        res_data_pair_ptr_type,
        exc_state_ptr_type,
        void_ptr_type,
        void_ptr_type,
        float3_struct_ptr_type
    };

    m_type_bsdf_pdf_func = llvm::FunctionType::get(ret_tp, arg_types_pdf, false);
}

// Compile a distribution function into an LLVM Module and return the LLVM module.
llvm::Module *LLVM_code_generator::compile_distribution_function(
    bool                        incremental,
    Distribution_function const &dist_func,
    ICall_name_resolver const   *resolver,
    Function_vector             &llvm_funcs)
{
    m_dist_func = &dist_func;

    IAllocator *alloc = m_arena.get_allocator();

    mi::base::Handle<ILambda_function> root_lambda_handle(m_dist_func->get_main_df());
    Lambda_function const *root_lambda = impl_cast<Lambda_function>(root_lambda_handle.get());

    create_captured_argument_struct(m_llvm_context, *root_lambda);

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
    }

    // load libbsdf into the current module, if it was not initialized, yet
    if (m_type_bsdf_sample_data == NULL && !load_and_link_libbsdf()) {
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
        lambda_result_exprs_init,
        lambda_result_exprs_others,
        texture_result_exprs);

    lambda_sched.schedule_lambdas();

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

        // generic functions return the result by reference
        m_lambda_force_sret         = true;

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

        // ensure the function is finished by putting it into a block
        {
            Function_instance inst(alloc, &lambda);
            Function_context context(alloc, *this, inst, func, flags);

            llvm::Function::arg_iterator arg_it = get_first_parameter(func, ctx_data);

            for (size_t i = 0, n = lambda.get_parameter_count(); i < n; ++i, ++arg_it) {
                mi::mdl::IType const       *p_type = lambda.get_parameter_type(i);
                bool                       by_ref  = is_passed_by_reference(p_type);
                LLVM_context_data          *p_data;

                // lambda parameters will never be written
                p_data = context.create_context_data(i, arg_it, by_ref);
            }

            // translate function body
            Expression_result res = translate_node(context, lambda.get_body(), resolver);
            context.create_return(res.as_value(context));
        }

        m_dist_func_lambda_map[&lambda] = func;
    }

    reset_lambda_state();

    // let return type decide to allow init function without structure return parameter
    m_lambda_force_sret         = false;

    // distribution functions always includes a render state in its interface
    m_lambda_force_render_state = true;

    // the BSDF API functions create the lambda results they use, so no lambda results parameter
    m_lambda_force_no_lambda_results = true;

    llvm::Twine base_name(root_lambda->get_name());
    Function_instance inst(get_allocator(), root_lambda);

    // create one LLVM function for each distribution function state
    for (int i = DFSTATE_INIT; i < DFSTATE_END_STATE; ++i)
    {
        m_dist_func_state = Distribution_function_state(i);

        // we cannot use get_or_create_context_data here, because we need to force the creation of
        // a new function here, as the (const) root_lambda cannot be changed to reflect the
        // different states
        LLVM_context_data *ctx_data = declare_lambda(root_lambda);
        m_context_data[inst] = ctx_data;

        llvm::Function *func = ctx_data->get_function();
        llvm_funcs.push_back(func);
        unsigned flags = ctx_data->get_function_flags();

        // set proper function name according to distribution function state
        func->setName(base_name + get_dist_func_state_suffix());

        Function_context context(alloc, *this, inst, func, flags);

        llvm::Function::arg_iterator arg_it = get_first_parameter(func, ctx_data);

        for (size_t j = 0, n = root_lambda->get_parameter_count(); j < n; ++j, ++arg_it) {
            mi::mdl::IType const       *p_type = root_lambda->get_parameter_type(j);
            bool                       by_ref = is_passed_by_reference(p_type);
            LLVM_context_data          *p_data;

            // lambda parameters will never be written
            p_data = context.create_context_data(j, arg_it, by_ref);
        }

        if (i == DFSTATE_INIT) {
            // translate the init function
            translate_distribution_function_init(
                context, texture_result_exprs, lambda_result_exprs_init);
        } else {
            // translate the distribution function
            translate_distribution_function(context, lambda_result_exprs_others);
        }
        context.create_void_return();
    }

    // reset some fields
    m_dist_func_lambda_map.clear();
    m_dist_func = NULL;

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
char const *LLVM_code_generator::get_dist_func_state_suffix()
{
    switch (m_dist_func_state) {
        case DFSTATE_INIT:     return "_init";
        case DFSTATE_SAMPLE:   return "_sample";
        case DFSTATE_EVALUATE: return "_evaluate";
        case DFSTATE_PDF:      return "_pdf";
        default:
            MDL_ASSERT(!"Invalid distribution function state");
            return "";
    }
}

// Get the BSDF function for the given semantics and the current distribution function state
// from the BSDF library.
llvm::Function *LLVM_code_generator::get_libbsdf_function(IDefinition::Semantics sema)
{
    MISTD::string func_name;

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

        // Unsupported: DS_INTRINSIC_DF_MEASURED_BSDF
        // Unsupported: DS_INTRINSIC_DF_DIFFUSE_EDF
        // Unsupported: DS_INTRINSIC_DF_MEASURED_EDF
        // Unsupported: DS_INTRINSIC_DF_SPOT_EDF
        // Unsupported: DS_INTRINSIC_DF_ANISOTROPIC_VDF

        SEMA_CASE(DS_INTRINSIC_DF_NORMALIZED_MIX,
                  "normalized_mix")
        SEMA_CASE(DS_INTRINSIC_DF_CLAMPED_MIX,
                  "clamped_mix")
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
                  "tint")
        SEMA_CASE(DS_INTRINSIC_DF_DIRECTIONAL_FACTOR,
                  "directional_factor")
        SEMA_CASE(DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR,
                  "measured_curve_factor")

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
                  "color_normalized_mix")
        SEMA_CASE(DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
                  "color_clamped_mix")
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

        default:
            MDL_ASSERT(!is_df_semantics(sema) && "unsupported DF function found");
            return NULL;
    }

    #undef SEMA_CASE

    return m_module->getFunction(func_name + get_dist_func_state_suffix());
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
    else
        return IDefinition::DS_UNKNOWN;

    if (basename == "black_bsdf")
        return IDefinition::DS_INVALID_REF_CONSTRUCTOR;

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

    llvm::Function *func;
    LLVM_context_data *p_data;
    unsigned ret_array_size = 0;

    // special case of an internal function not available in MDL?
    if (demangled_name == "state::set_normal(float3)") {
        func = get_internal_function(m_int_func_state_set_normal);

        Function_instance inst(
            get_allocator(), reinterpret_cast<size_t>(m_int_func_state_set_normal));
        p_data = get_context_data(inst);
    } else {
        size_t colonpos = demangled_name.rfind("::");
        if (colonpos == string::npos)
            return true;  // not in a module, maybe a builtin function

        string module_name = "::" + demangled_name.substr(0, colonpos);
        string signature = demangled_name.substr(colonpos + 2);
        IDefinition const *def = m_compiler->find_stdlib_signature(
            module_name.c_str(), signature.c_str());
        if (def == NULL)
            return true;  // not one of our modules, maybe a builtin function

        func = get_intrinsic_function(def);

        // check for MDL function with array return and retrieve array size
        MDL_ASSERT(def->get_type()->get_kind() == IType::TK_FUNCTION);
        IType_function const *mdl_func_type = static_cast<IType_function const *>(def->get_type());
        IType const *mdl_ret_type = mdl_func_type->get_return_type();
        if (mdl_ret_type->get_kind() == IType::TK_ARRAY) {
            IType_array const *mdl_array_type = static_cast<IType_array const *>(mdl_ret_type);
            MDL_ASSERT(mdl_array_type->is_immediate_sized());
            ret_array_size = unsigned(mdl_array_type->get_size());
        }

        Function_instance inst(get_allocator(), def);
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

    bool first_param_eaten = false;
    llvm::Type *orig_res_type = called_func->getReturnType();
    llvm::Value *orig_res_ptr = NULL;
    llvm::Value *runtime_res_ptr = NULL;

    // Original call case: f(&res,a,b)?
    if (ret_array_size == 0 && orig_res_type == m_type_mapper.get_void_type()) {
        orig_res_ptr = call->getArgOperand(0);
        orig_res_type = llvm::cast<llvm::PointerType>(orig_res_ptr->getType())->getElementType();
        first_param_eaten = true;
    }

    // Runtime call case: f_r(&res,a,b)?
    if (p_data->is_sret_return()) {
        runtime_res_ptr = ctx.create_local(p_data->get_return_type(), "runtime_call_result");
        llvm_args.push_back(runtime_res_ptr);
    }

    if (p_data->has_state_param()) {
        // pass state parameter
        MDL_ASSERT(ctx.get_state_parameter() != NULL);
        llvm_args.push_back(ctx.get_state_parameter());
    }

    if (p_data->has_resource_data_param()) {
        // pass resource_data parameter
        MDL_ASSERT(ctx.get_resource_data_parameter() != NULL);
        llvm_args.push_back(ctx.get_resource_data_parameter());
    }

    if (p_data->has_exc_state_param()) {
        // pass exc_state_param parameter
        MDL_ASSERT(ctx.get_exc_state_parameter() != NULL);
        llvm_args.push_back(ctx.get_exc_state_parameter());
    }

    if (p_data->has_object_id_param()) {
        // should not happen, as we always require the render state
        MDL_ASSERT(!"Object ID parameter not supported, yet");
        return false;
    }

    if (p_data->has_transform_params()) {
        // should not happen, as we always require the render state
        MDL_ASSERT(!"Transform parameters not supported, yet\n");
        return false;
    }

    // insert new code before the old call
    ctx->SetInsertPoint(call);
    llvm::FunctionType *func_type = func->getFunctionType();

    // handle all remaining arguments (except for array return arguments)
    unsigned n_args = call->getNumArgOperands();
    for (unsigned i = first_param_eaten ? 1 : 0; i < n_args - ret_array_size; ++i) {
        llvm::Value *arg = call->getArgOperand(i);
        llvm::Type *arg_type = arg->getType();
        llvm::Type *param_type = func_type->getParamType(llvm_args.size());

        // are argument and parameter types identical?
        if (arg_type == param_type)
            llvm_args.push_back(arg);
        else {
            // no, a conversion is required
            if (llvm::isa<llvm::PointerType>(arg_type) && llvm::isa<llvm::PointerType>(param_type))
            {
                // convert from one memory representation to another
                llvm::PointerType *param_ptr_type = llvm::cast<llvm::PointerType>(param_type);
                llvm::Value *val = ctx.load_and_convert(param_ptr_type->getElementType(), arg);

                llvm::Value *convert_tmp_ptr = ctx.create_local(
                    param_ptr_type->getElementType(), "convert_tmp");
                ctx->CreateStore(val, convert_tmp_ptr);

                llvm_args.push_back(convert_tmp_ptr);
            } else if (llvm::isa<llvm::PointerType>(arg_type) &&
                llvm::isa<llvm::VectorType>(param_type))
            {
                // load memory representation into a vector
                llvm::Value *val = ctx.load_and_convert(param_type, arg);
                llvm_args.push_back(val);
            } else {
                MDL_ASSERT(!"Unsupported parameter conversion");
                return false;
            }
        }
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
        llvm::Value *convert_tmp_ptr = ctx.create_local(res->getType(), "convert_tmp");
        ctx->CreateStore(res, convert_tmp_ptr);
        res = ctx.load_and_convert(orig_res_type, convert_tmp_ptr);
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
    ii = ii->getParent()->getInstList().erase(call)->getPrevNode();
    return true;
}

// Transitively walk over the uses of the given argument and mark any calls as BSDF calls,
// storing the provided parameter index as "libbsdf.bsdf_param" metadata.
void LLVM_code_generator::mark_bsdf_calls(llvm::Argument *arg, int bsdf_param_idx)
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
        for (llvm::Value::use_iterator ui = cur->use_begin(), ue = cur->use_end(); ui != ue; ++ui)
        {
            if (llvm::StoreInst *inst = llvm::dyn_cast<llvm::StoreInst>(*ui)) {
                // for stores, also follow the variable which is written
                worklist.push_back(inst->getPointerOperand());
                ++num_stores;
            } else if (llvm::CallInst *inst = llvm::dyn_cast<llvm::CallInst>(*ui)) {
                // found a call, store the parameter index as metadata
                llvm::Value *param_idx = llvm::ConstantInt::get(int_type, bsdf_param_idx);
                llvm::MDNode *md = llvm::MDNode::get(m_llvm_context, param_idx);
                inst->setMetadata(m_bsdf_param_metadata_id, md);
            } else {
                // for all other uses, just follow the use
                worklist.push_back(*ui);
            }
        }

        // if we have more than one store to the same variable, the code is probably not supported
        MDL_ASSERT(num_stores <= 1);
    }
}

// Load and link libbsdf into the current LLVM module.
bool LLVM_code_generator::load_and_link_libbsdf()
{
    llvm::Module *libbsdf = load_libbsdf(m_llvm_context);
    MDL_ASSERT(libbsdf != NULL);

    // clear target triple to avoid LLVM warning on console about mixing different targets
    // when linking libbsdf ("x86_x64-pc-win32") with libdevice ("nvptx-unknown-unknown").
    // Using an nvptx target for libbsdf would cause struct parameters to be split, which we
    // try to avoid.
    libbsdf->setTargetTriple("");

    prepare_libbsdf_prototypes(libbsdf);

    // collect all functions available before linking
    ptr_hash_set<llvm::Function>::Type old_funcs(get_allocator());
    for (llvm::Module::iterator FI = m_module->begin(), FE = m_module->end(); FI != FE; ++FI) {
        old_funcs.insert(&*FI);
    }

    MISTD::string errorInfo;
    if (llvm::Linker::LinkModules(m_module, libbsdf, llvm::Linker::DestroySource, &errorInfo)) {
        // true means linking has failed
        error(LINKING_LIBBSDF_FAILED, errorInfo);
        MDL_ASSERT(!"Linking libbsdf failed");
        return false;
    }

    // find all functions which were added by linking the libbsdf module
    ptr_hash_set<llvm::Function>::Type libbsdf_funcs(get_allocator());
    for (llvm::Module::iterator FI = m_module->begin(), FE = m_module->end(); FI != FE; ++FI) {
        libbsdf_funcs.insert(&*FI);
    }
    for (ptr_hash_set<llvm::Function>::Type::const_iterator FI = old_funcs.begin(),
            FE = old_funcs.end(); FI != FE; ++FI) {
        libbsdf_funcs.erase(*FI);
    }

    create_bsdf_function_types();

    m_bsdf_param_metadata_id = m_llvm_context.getMDKindID("libbsdf.bsdf_param");

    llvm::Type *int_type = m_type_mapper.get_int_type();

    // iterate over all functions added from the libbsdf module
    for (ptr_hash_set<llvm::Function>::Type::const_iterator FI = libbsdf_funcs.begin(),
            FE = libbsdf_funcs.end(); FI != FE; ++FI) {
        llvm::Function *func = *FI;

        // disable all alignment settings on functions, as PTX does not support them
        // (this is fixed in LLVM release 3.7 (svn commit 232004))
        func->setAlignment(0);

        // skip function declarations
        if (func->isDeclaration())
            continue;

        // make all functions from libbsdf internal to allow global dead code elimination
        func->setLinkage(llvm::GlobalValue::InternalLinkage);

        // check whether this is a BSDF API function, for which we need to update the prototype
        if (func->getArgumentList().size() >= 1) {
            llvm::Function::arg_iterator old_arg_it  = *func->arg_begin();
            llvm::Function::arg_iterator old_arg_end = *func->arg_end();
            llvm::Value *bsdf_data        = old_arg_it++;
            llvm::Value *inherited_normal = old_arg_it++;

            // is the type of the first parameter one of the BSDF data types?
            if (llvm::PointerType *bsdf_data_ptr_type =
                    llvm::dyn_cast<llvm::PointerType>(bsdf_data->getType()))
            {
                llvm::Type *bsdf_data_type = bsdf_data_ptr_type->getElementType();

                llvm::FunctionType *new_func_type;
                if (bsdf_data_type == m_type_bsdf_sample_data)
                    new_func_type = m_type_bsdf_sample_func;
                else if (bsdf_data_type == m_type_bsdf_evaluate_data)
                    new_func_type = m_type_bsdf_evaluate_func;
                else if (bsdf_data_type == m_type_bsdf_pdf_data)
                    new_func_type = m_type_bsdf_pdf_func;
                else
                    new_func_type = NULL;

                IDefinition::Semantics sema = get_libbsdf_function_semantics(func->getName());

                if (new_func_type != NULL && (is_df_semantics(sema) ||
                        sema == IDefinition::DS_INVALID_REF_CONSTRUCTOR)) {
                    // this is a BSDF API function, so change function type to add parameters:
                    //  - MDL_SDK_State           *state,
                    //  - MDL_SDK_Res_data_pair   *res_data_pair,
                    //  - MDL_SDK_Exception_state *exc_state
                    //  - void                    *captured_args
                    //  - void                    *lambda_results

                    // change function type by creating new function with blocks of old function
                    llvm::Function *new_func = llvm::Function::Create(
                        new_func_type,
                        llvm::GlobalValue::InternalLinkage,
                        "",
                        m_module);
                    new_func->takeName(func);
                    new_func->getBasicBlockList().splice(
                        new_func->begin(), func->getBasicBlockList());

                    // tell context where to find the state parameters
                    llvm::Function::arg_iterator arg_it = new_func->arg_begin();
                    llvm::Value *data_param             = arg_it++;
                    arg_it++;                                        // skip state_param
                    arg_it++;                                        // skip res_data_param
                    arg_it++;                                        // skip exc_state_param
                    arg_it++;                                        // skip captured_args_param
                    arg_it++;                                        // skip lambda_results_param
                    llvm::Value *inherited_normal_param = arg_it++;

                    bsdf_data->replaceAllUsesWith(data_param);
                    inherited_normal->replaceAllUsesWith(inherited_normal_param);

                    // make sure we don't introduce initialization code before alloca instructions
                    llvm::BasicBlock::iterator param_init_insert_point = new_func->front().begin();
                    while (llvm::isa<llvm::AllocaInst>(param_init_insert_point))
                        ++param_init_insert_point;

                    // introduce local variables for all used BSDF parameters
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
                                "bsdf_arg",
                                new_func->getEntryBlock().begin());

                            if (elem_type->isStructTy() &&
                                elem_type->getStructName() == "struct.BSDF")
                            {
                                // for BSDF parameters, we mark the calls to the BSDF methods
                                // with metadata instead of the local variables.
                                // The argument value is not necessary, but we keep it, in case
                                // the uses are not optimized away.
                                // Note: we don't do this for the BSDFs inside BSDF_component!
                                mark_bsdf_calls(old_arg_it, cur_df_idx);

                                arg_var = NULL;
                            }
                        } else {
                            // for non-pointer types we also need to load the value
                            // and replace the argument by the load, not the alloca
                            arg_var = new llvm::AllocaInst(
                                old_arg_it->getType(),
                                "bsdf_arg_var",
                                new_func->getEntryBlock().begin());
                            arg_val = new llvm::LoadInst(
                                arg_var, "bsdf_arg", param_init_insert_point);
                        }

                        // do we need to set metadata?
                        if (arg_var != NULL) {
                            llvm::Value *param_idx = llvm::ConstantInt::get(int_type, cur_df_idx);
                            llvm::MDNode *md = llvm::MDNode::get(m_llvm_context, param_idx);
                            arg_var->setMetadata(m_bsdf_param_metadata_id, md);
                        }

                        old_arg_it->replaceAllUsesWith(arg_val);
                    }

                    func->eraseFromParent();
                    func = new_func;
                }
            }
        }

        Function_context ctx(
            get_allocator(),
            *this,
            func,
            LLVM_context_data::FL_SRET | LLVM_context_data::FL_HAS_STATE |
            LLVM_context_data::FL_HAS_RES | LLVM_context_data::FL_HAS_EXC |
            LLVM_context_data::FL_HAS_CAP_ARGS,
            false);  // don't optimize, because of parameter handling via uninitialized allocas

        // search for all CallInst instructions and link runtime function calls to the
        // corresponding intrinsics
        for (llvm::Function::iterator BI = func->begin(), BE = func->end(); BI != BE; ++BI) {
            for (llvm::BasicBlock::iterator II = *BI->begin(); II != *BI->end(); ++II) {
                if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(II)) {
                    if (!translate_libbsdf_runtime_call(call, II, ctx))
                        return false;
                }
            }
        }
    }

    return true;
}

/// Generate a call to an expression lambda function.
Expression_result LLVM_code_generator::generate_expr_lambda_call(
    Function_context &ctx,
    size_t           lambda_index,
    llvm::Value *opt_dest_ptr)
{
    mi::base::Handle<mi::mdl::ILambda_function> expr_lambda(
        m_dist_func->get_expr_lambda(lambda_index));
    Lambda_function const *expr_lambda_impl = impl_cast<Lambda_function>(expr_lambda.get());

    // expression and special lambdas always return by reference via first parameter

    llvm::Function *func = m_dist_func_lambda_map[expr_lambda.get()];
    llvm::Type *lambda_retptr_type = func->getFunctionType()->getParamType(0);
    llvm::Type *lambda_res_type = lambda_retptr_type->getPointerElementType();
    llvm::Type *dest_type = NULL;
    llvm::Value *res_pointer;
    if (opt_dest_ptr != NULL &&
            (dest_type = opt_dest_ptr->getType()->getPointerElementType()) == lambda_res_type)
        res_pointer = opt_dest_ptr;
    else
        res_pointer = ctx.create_local(lambda_res_type, "res_buf");

    llvm::SmallVector<llvm::Value *, 6> lambda_args;
    lambda_args.push_back(res_pointer);
    lambda_args.push_back(ctx.get_state_parameter());
    lambda_args.push_back(ctx.get_resource_data_parameter());
    lambda_args.push_back(ctx.get_exc_state_parameter());
    lambda_args.push_back(ctx.get_cap_args_parameter());
    if (expr_lambda_impl->uses_lambda_results())
        lambda_args.push_back(ctx.get_lambda_results_parameter());

    ctx->CreateCall(func, lambda_args);

    if (opt_dest_ptr != NULL && dest_type != lambda_res_type) {
        llvm::Value *res = ctx->CreateLoad(res_pointer);
        ctx.convert_and_store(res, opt_dest_ptr);
        return Expression_result::ptr(opt_dest_ptr);
    }

    return Expression_result::ptr(res_pointer);
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
        res = Expression_result::ptr(ctx->CreateConstGEP2_32(
            get_texture_results(ctx),
            0,
            unsigned(m_texture_result_indices[lambda_index])));
    }

    // was the result locally precalculated?
    else if (m_lambda_result_indices[lambda_index] != -1) {
        res = Expression_result::ptr(ctx->CreateConstGEP2_32(
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
    if (expected_type == NULL || res.as_value(ctx)->getType() == expected_type) return res;

    // convert to expected type
    return Expression_result::value(ctx.load_and_convert(expected_type, res.as_ptr(ctx)));
}

// Generates a switch function calling the BSDF function identified by the last parameter
// with the provided arguments.
llvm::Function *LLVM_code_generator::generate_bsdf_switch_func(
    llvm::ArrayRef<llvm::Function *> const &funcs)
{
    size_t num_funcs = funcs.size();
    llvm::Type *int_type = m_type_mapper.get_int_type();

    llvm::FunctionType *bsdf_func_type = funcs[0]->getFunctionType();

    llvm::SmallVector<llvm::Type *, 8> arg_types;
    arg_types.append(bsdf_func_type->param_begin(), bsdf_func_type->param_end());
    arg_types.push_back(int_type);

    llvm::FunctionType *switch_func_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(m_llvm_context), arg_types, false);

    llvm::Function *switch_func = llvm::Function::Create(
        switch_func_type,
        llvm::GlobalValue::InternalLinkage,
        "",
        m_module);

    llvm::BasicBlock *start_block = llvm::BasicBlock::Create(m_llvm_context, "start", switch_func);
    llvm::BasicBlock *end_block = llvm::BasicBlock::Create(m_llvm_context, "end", switch_func);

    llvm::IRBuilder<> builder(start_block);
    llvm::SwitchInst *switch_inst =
        builder.CreateSwitch(--switch_func->arg_end(), end_block, num_funcs);

    // collect the arguments for the BSDF functions to be called (without the index argument)
    llvm::SmallVector<llvm::Value *, 8> arg_values;
    for (llvm::Function::arg_iterator ai = switch_func->arg_begin(), ae = --switch_func->arg_end();
            ai != ae; ++ai)
    {
        arg_values.push_back(ai);
    }

    // generate the switch cases with the calls to the corresponding BSDF function
    for (size_t i = 0; i < num_funcs; ++i) {
        llvm::BasicBlock *case_block =
            llvm::BasicBlock::Create(m_llvm_context, "case", switch_func);
        switch_inst->addCase(
            llvm::ConstantInt::get(m_llvm_context, llvm::APInt(32, uint64_t(i))),
            case_block);
        builder.SetInsertPoint(case_block);
        builder.CreateCall(funcs[i], arg_values);
        builder.CreateBr(end_block);
    }

    builder.SetInsertPoint(end_block);
    builder.CreateRetVoid();

    // optimize function to improve inlining
    m_func_pass_manager->run(*switch_func);

    return switch_func;
}

// Get the BSDF parameter ID metadata for an instruction.
int LLVM_code_generator::get_metadata_bsdf_param_id(llvm::Instruction *inst)
{
    llvm::MDNode *md = inst->getMetadata(m_bsdf_param_metadata_id);
    if (md == NULL) return -1;

    llvm::ConstantInt *param_idx_val = llvm::dyn_cast<llvm::ConstantInt>(md->getOperand(0));
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
    //   <C> = bitcast %struct.color_BSDF_component* <X> to i8*
    //   call void @llvm.memcpy.p0i8.p0i8.i64(i8* <Y>, i8* <C>, i64 12, i32 4, i1 false)

    // ensure, that all usages of this cast are memcpys of a weight
    for (llvm::Value::use_iterator cast_ui = addr_bitcast->use_begin(),
            cast_ue = addr_bitcast->use_end(); cast_ui != cast_ue; ++cast_ui) {
        llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(*cast_ui);
        if (call == NULL) {
            MDL_ASSERT(
                !"Unsupported usage of color_BSDF_component parameter with bitcast");
            return false;
        }
        llvm::Function *called_func = call->getCalledFunction();
        if (!called_func->getName().startswith("llvm.memcpy.")) {
            MDL_ASSERT(
                !"Unsupported usage of color_BSDF_component parameter with bitcast/call");
            return false;
        }
        if (call->getNumArgOperands() != 5 ||
                call->getArgOperand(1) != addr_bitcast ||             // source is cast
                !ctx.is_constant_value(call->getArgOperand(2), 12)) { // size of float3
            MDL_ASSERT(
                !"Unsupported usage of color_BSDF_component parameter with memcpy");
            return false;
        }
    }

    // rewrite cast to use pointer to index'th weight in weight array
    llvm::Value *null_val = llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type());
    llvm::Value *idxs[] = { null_val, index };
    llvm::GetElementPtrInst *weight_ptr = llvm::GetElementPtrInst::Create(
        weight_array, idxs, "", addr_bitcast);
    llvm::Value *new_cast = llvm::BitCastInst::Create(
        llvm::Instruction::BitCast, weight_ptr, addr_bitcast->getType(), "", weight_ptr);
    addr_bitcast->replaceAllUsesWith(new_cast);
    delete_list.push_back(addr_bitcast);

    return true;
}

// Rewrite all usages of a BSDF component variable using the given weight array and the
// BSDF function, which can either be a switch function depending on the array index
// or the same function for all indices.
void LLVM_code_generator::rewrite_bsdf_component_usages(
    Function_context                           &ctx,
    llvm::AllocaInst                           *inst,
    llvm::Value                                *weight_array,
    llvm::Function                             *bsdf_func,
    bool                                       is_switch_func,
    llvm::SmallVector<llvm::Instruction *, 16> &delete_list)
{
    // These rewrites are performed:
    //  - bsdf_component[i].weight -> weights[i]
    //  - bsdf_component[i].component.sample() -> bsdf_func(...) or bsdf_func(..., i)
    for (llvm::Value::use_iterator ui = inst->use_begin(),
            ue = inst->use_end(); ui != ue; ++ui) {
        llvm::GetElementPtrInst *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(*ui);
        if (gep == NULL) {
            // check for
            //   <C> = bitcast %struct.color_BSDF_component* <X> to i8*
            //   call void @llvm.memcpy.p0i8.p0i8.i64(i8* <Y>, i8* <C>, i64 12, i32 4, i1 false)
            llvm::BitCastInst *cast = llvm::dyn_cast<llvm::BitCastInst>(*ui);
            if (cast == NULL) {
                MDL_ASSERT(!"Unsupported usage of color_BSDF_component parameter");
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

            for (llvm::Value::use_iterator gep_ui = gep->use_begin(),
                    gep_ue = gep->use_end(); gep_ui != gep_ue; ++gep_ui) {
                llvm::BitCastInst *cast = llvm::dyn_cast<llvm::BitCastInst>(*gep_ui);
                if (cast == NULL) {
                    MDL_ASSERT(!"Unsupported gep usage of color_BSDF_component parameter");
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
            //   color_bsdf_component[i].weight.x/y/z -> color_weights[i].x/y/z
            if (gep->getNumOperands() == 4) {
                // replace by access to same color component on same index of color array
                llvm::Value *col_comp_idx_val = gep->getOperand(3);
                llvm::Value *idxs[] = {
                    llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type()),
                    component_idx_val,
                    col_comp_idx_val
                };
                new_gep = llvm::GetElementPtrInst::Create(weight_array, idxs, "", gep);
            } else {
                // replace by access on same index of weight array (can be float or color)
                llvm::Value *idxs[] = {
                    llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type()),
                    component_idx_val
                };
                new_gep = llvm::GetElementPtrInst::Create(weight_array, idxs, "", gep);
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
            for (llvm::Value::use_iterator gep_ui = gep->use_begin(),
                    gep_ue = gep->use_end(); gep_ui != gep_ue; ++gep_ui) {
                llvm::LoadInst *load = llvm::dyn_cast<llvm::LoadInst>(*gep_ui);
                MDL_ASSERT(load);

                for (llvm::Value::use_iterator load_ui = load->use_begin(),
                        load_ue = load->use_end(); load_ui != load_ue; ++load_ui) {
                    llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(*load_ui);
                    MDL_ASSERT(call);
                    MDL_ASSERT(call->getType() != llvm::IntegerType::get(m_llvm_context, 1) &&
                        "bsdfs in bsdf_component currently don't support is_black()");

                    // convert 64-bit index to 32-bit index
                    llvm::Value *idx_val = new llvm::TruncInst(
                        component_idx_val,
                        m_type_mapper.get_int_type(),
                        "",
                        call);

                    // call it with state parameters added
                    llvm::SmallVector<llvm::Value *, 7> llvm_args;
                    llvm_args.push_back(call->getArgOperand(0));     // res_pointer
                    llvm_args.push_back(ctx.get_state_parameter());
                    llvm_args.push_back(ctx.get_resource_data_parameter());
                    llvm_args.push_back(ctx.get_exc_state_parameter());
                    llvm_args.push_back(ctx.get_cap_args_parameter());
                    llvm_args.push_back(ctx.get_lambda_results_parameter());
                    llvm_args.push_back(call->getArgOperand(1));     // inherited_normal parameter
                    if (is_switch_func)
                        llvm_args.push_back(idx_val);                // BSDF function index
                    llvm::CallInst::Create(bsdf_func, llvm_args, "", call);
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
void LLVM_code_generator::handle_bsdf_array_parameter(
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
        bool color_bsdf_component = false;
        if (elem_type->isStructTy() && (
                elem_type->getStructName() == "struct.BSDF_component" ||
                (color_bsdf_component = (elem_type->getStructName() ==
                                         "struct.color_BSDF_component")))) {

            llvm::Type *weight_type = color_bsdf_component ?
                m_float3_struct_type : m_type_mapper.get_float_type(); 

            // create a global constant weight array
            llvm::ArrayType *weight_array_type =
                llvm::ArrayType::get(weight_type, elem_count);
            llvm::SmallVector<llvm::Constant *, 8> elems(elem_count);
            for (int i = 0; i < elem_count; ++i) {
                MDL_ASSERT(arg_array->get_value(i)->get_kind() == mi::mdl::IValue::VK_STRUCT);
                mi::mdl::IValue_struct const *comp_val =
                    mi::mdl::cast<mi::mdl::IValue_struct>(arg_array->get_value(i));
                if (color_bsdf_component) {
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

            // only "bsdf()" can be part of a constant
            mi::mdl::string func_name("black_bsdf", get_allocator());
            func_name += get_dist_func_state_suffix();
            llvm::Function *black_bsdf_func = m_module->getFunction(func_name.c_str());
            MDL_ASSERT(black_bsdf_func &&
                "libbsdf is missing an implementation of bsdf(): black_bsdf_*");

            // rewrite all usages of the components variable
            rewrite_bsdf_component_usages(
                ctx,
                inst,
                weight_array_global,
                black_bsdf_func,
                /*is_switch_func=*/ false,
                delete_list);
            return;
        }

        MDL_ASSERT(!"Unsupported constant array parameter type");
        return;
    }

    MDL_ASSERT(arg->get_kind() == DAG_node::EK_CALL);
    DAG_call const *arg_call = mi::mdl::cast<DAG_call>(arg);
    int elem_count = arg_call->get_argument_count();

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

        // create local color array
        llvm::ArrayType *color_array_type = llvm::ArrayType::get(m_float3_struct_type, elem_count);
        llvm::Value *color_array = ctx.create_local(color_array_type, "colors");

        for (int i = 0; i < elem_count; ++i) {
            // the i-th element should have been rewritten to a lambda call
            DAG_node const *color_node = arg_call->get_argument(i);
            MDL_ASSERT(color_node->get_kind() == DAG_node::EK_CALL);
            DAG_call const *color_call = mi::mdl::cast<DAG_call>(color_node);
            MDL_ASSERT(color_call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA);

            // read precalculated lambda result
            size_t lambda_index = strtoul(color_call->get_name(), NULL, 10);
            Expression_result color_res = translate_precalculated_lambda(
                ctx, lambda_index, m_float3_struct_type);

            // store result in colors array
            llvm::Value *idxs[] = {
                llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type()),
                llvm::ConstantInt::get(m_type_mapper.get_int_type(), i)
            };
            ctx->CreateStore(color_res.as_value(ctx),
                ctx->CreateGEP(color_array, idxs, "colors_elem"));
        }
        llvm::Value *casted_array = ctx->CreateBitCast(
            color_array, m_float3_struct_type->getPointerTo());
        inst->replaceAllUsesWith(casted_array);
        return;
    }

    // is it a BSDF_component array?
    bool color_bsdf_component = false;
    if (elem_type->isStructTy() && (
          elem_type->getStructName() == "struct.BSDF_component" ||
          (color_bsdf_component = (elem_type->getStructName() == "struct.color_BSDF_component")))) {
        // make sure the weight array is initialized at the beginning of the body
        ctx.move_to_body_start();

        llvm::Type *weight_type = color_bsdf_component ?
            m_float3_struct_type : m_type_mapper.get_float_type(); 

        // create local weight array and instantiate all BSDF components
        llvm::ArrayType *weight_array_type = llvm::ArrayType::get(weight_type, elem_count);
        llvm::Value *weight_array = ctx.create_local(weight_array_type, "weights");
        llvm::SmallVector<llvm::Function *, 8> bsdf_funcs(elem_count);

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
                mi::mdl::string func_name("black_bsdf", get_allocator());
                func_name += get_dist_func_state_suffix();
                bsdf_funcs[i] = m_module->getFunction(func_name.c_str());
                MDL_ASSERT(bsdf_funcs[i] &&
                    "libbsdf is missing an implementation of bsdf(): black_bsdf_*");
            } else {
                // should be a BSDF_component constructor call
                MDL_ASSERT(elem_node->get_kind() == DAG_node::EK_CALL);
                DAG_call const *elem_call = mi::mdl::cast<DAG_call>(elem_node);

                // the weight argument of the constructor should have been rewritten
                // to a lambda call
                DAG_node const *weight_node = elem_call->get_argument("weight");
                MDL_ASSERT(weight_node->get_kind() == DAG_node::EK_CALL);
                DAG_call const *weight_call = mi::mdl::cast<DAG_call>(weight_node);
                MDL_ASSERT(weight_call->get_semantic() ==
                    IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA);

                // read precalculated lambda result
                size_t lambda_index = strtoul(weight_call->get_name(), NULL, 10);
                weight_res = translate_precalculated_lambda(
                    ctx, lambda_index, weight_type);

                // instantiate BSDF for component parameter of the constructor
                DAG_node const *component_node = elem_call->get_argument("component");
                bsdf_funcs[i] = instantiate_bsdf(component_node);
            }

            // store result in weights array
            llvm::Value *idxs[] = {
                llvm::ConstantInt::getNullValue(m_type_mapper.get_int_type()),
                llvm::ConstantInt::get(m_type_mapper.get_int_type(), i)
            };
            ctx->CreateStore(weight_res.as_value(ctx),
                ctx->CreateGEP(weight_array, idxs, "weights_elem"));
        }

        // generate the switch function for the instantiated BSDF components
        // (we don't use function pointers to be compatible with OptiX)
        llvm::Function *bsdf_switch_func = generate_bsdf_switch_func(bsdf_funcs);

        // rewrite all usages of the components variable
        rewrite_bsdf_component_usages(
            ctx,
            inst,
            weight_array,
            bsdf_switch_func,
            /*is_switch_func=*/ true,
            delete_list);
        return;
    }

    MDL_ASSERT(!"Unsupported array parameter type");
}

/// Recursively instantiate a ternary operator of type BSDF.
llvm::Function *LLVM_code_generator::instantiate_ternary_bsdf(
    DAG_call const *dag_call)
{
    // Create a new function with the type for current distribution function state

    llvm::FunctionType *func_type;
    switch (m_dist_func_state) {
        case DFSTATE_SAMPLE:   func_type = m_type_bsdf_sample_func; break;
        case DFSTATE_EVALUATE: func_type = m_type_bsdf_evaluate_func; break;
        case DFSTATE_PDF:      func_type = m_type_bsdf_pdf_func; break;
        default:
            MDL_ASSERT(!"Invalid distribution function state");
            return NULL;
    }

    llvm::Function *func = llvm::Function::Create(
        func_type,
        llvm::GlobalValue::InternalLinkage,
        "ternary_bsdf",
        m_module);

    {
        // Context needs a non-empty start block, so create a jump to a second block
        llvm::BasicBlock *start_bb = llvm::BasicBlock::Create(m_llvm_context, "start", func);
        llvm::BasicBlock *body_bb = llvm::BasicBlock::Create(m_llvm_context, "body", func);
        start_bb->getInstList().push_back(llvm::BranchInst::Create(body_bb));

        Function_context ctx(
            get_allocator(),
            *this,
            func,
            LLVM_context_data::FL_SRET | LLVM_context_data::FL_HAS_STATE |
            LLVM_context_data::FL_HAS_RES | LLVM_context_data::FL_HAS_EXC |
            LLVM_context_data::FL_HAS_CAP_ARGS | LLVM_context_data::FL_HAS_LMBD_RES,
            true);

        ctx->SetInsertPoint(body_bb);

        // Find lambda expression for condition and generate code
        DAG_node const *cond = dag_call->get_argument(0);
        MDL_ASSERT(cond->get_kind() == DAG_node::EK_CALL);
        DAG_call const *cond_call = mi::mdl::cast<DAG_call>(cond);
        MDL_ASSERT(cond_call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA);
        size_t lambda_index = strtoul(cond_call->get_name(), NULL, 10);

        Expression_result res = translate_precalculated_lambda(
            ctx, lambda_index, m_type_mapper.get_bool_type());

        // Generate code for "if (cond) call[1](args); else call[2](args);"

        llvm::BasicBlock *cond_true_bb = ctx.create_bb("cond_true");
        llvm::BasicBlock *cond_false_bb = ctx.create_bb("cond_false");
        llvm::BasicBlock *end_bb = ctx.create_bb("end");

        llvm::Value *cond_res = res.as_value(ctx);
        llvm::Value *cond_bool = ctx->CreateICmpNE(
            cond_res,
            llvm::Constant::getNullValue(cond_res->getType()));
        ctx->CreateCondBr(cond_bool, cond_true_bb, cond_false_bb);

        llvm::Value *bsdf_true_func = instantiate_bsdf(dag_call->get_argument(1));
        llvm::Value *bsdf_false_func = instantiate_bsdf(dag_call->get_argument(2));

        llvm::Value *llvm_args[] = {
            func->arg_begin(),            // res_pointer
            ctx.get_state_parameter(),
            ctx.get_resource_data_parameter(),
            ctx.get_exc_state_parameter(),
            ctx.get_cap_args_parameter(),
            ctx.get_lambda_results_parameter(),
            ctx.get_first_parameter()   // inherited_normal parameter
        };

        // True case
        ctx->SetInsertPoint(cond_true_bb);
        ctx->CreateCall(bsdf_true_func, llvm_args, "");
        ctx->CreateBr(end_bb);

        // False case
        ctx->SetInsertPoint(cond_false_bb);
        ctx->CreateCall(bsdf_false_func, llvm_args, "");
        ctx->CreateBr(end_bb);

        ctx->SetInsertPoint(end_bb);
        ctx->CreateRetVoid();
    }

    // Return the now finalized function
    return func;
}

// Recursively instantiate a BSDF specified by the given DAG node from code in the BSDF library
// according to the current distribution function state.
llvm::Function *LLVM_code_generator::instantiate_bsdf(
    DAG_node const *node)
{
    // get BSDF function according to semantics and current state
    // and clone it into the current module.

    llvm::Function *bsdf_lib_func;

    // check for "bsdf()" constant
    if (node->get_kind() == DAG_node::EK_CONSTANT &&
        cast<DAG_constant>(node)->get_value()->get_kind() == IValue::VK_INVALID_REF &&
        cast<DAG_constant>(node)->get_value()->get_type()->get_kind() == IType::TK_BSDF)
    {
        mi::mdl::string func_name("black_bsdf", get_allocator());
        func_name += get_dist_func_state_suffix();
        bsdf_lib_func = m_module->getFunction(func_name.c_str());
        if (bsdf_lib_func == NULL) {
            MDL_ASSERT(!"libbsdf is missing an implementation of bsdf(): black_bsdf_*");
            return NULL;
        }
        return bsdf_lib_func;   // the black_bsdf needs no instantiation, return it directly
    }

    if (node->get_kind() != DAG_node::EK_CALL) {
        MDL_ASSERT(!"Unsupported DAG node");
        return NULL;
    }

    DAG_call const *dag_call = cast<DAG_call>(node);
    IDefinition::Semantics sema = dag_call->get_semantic();
    if (sema == operator_to_semantic(IExpression::OK_TERNARY))
        return instantiate_ternary_bsdf(dag_call);

    bsdf_lib_func = get_libbsdf_function(sema);
    if (bsdf_lib_func == NULL) {
        MDL_ASSERT(!"BSDF function not supported by libbsdf, yet");
        return NULL;
    }

    llvm::ValueToValueMapTy ValueMap;
    llvm::Function *bsdf_func = llvm::CloneFunction(
        bsdf_lib_func,
        ValueMap,
        false);          // ModuleLevelChanges

    m_module->getFunctionList().push_back(bsdf_func);

    Function_context ctx(
        get_allocator(),
        *this,
        bsdf_func,
        LLVM_context_data::FL_SRET | LLVM_context_data::FL_HAS_STATE |
        LLVM_context_data::FL_HAS_RES | LLVM_context_data::FL_HAS_EXC |
        LLVM_context_data::FL_HAS_CAP_ARGS | LLVM_context_data::FL_HAS_LMBD_RES,
        true);

    llvm::SmallVector<llvm::Instruction *, 16> delete_list;

    // process all calls to BSDF parameter accessors
    for (llvm::Function::iterator BI = bsdf_func->begin(), BE = bsdf_func->end(); BI != BE; ++BI) {
        for (llvm::BasicBlock::iterator II = *BI->begin(); II != *BI->end(); ++II) {
            if (llvm::AllocaInst *inst = llvm::dyn_cast<llvm::AllocaInst>(II)) {
                llvm::Type *elem_type = inst->getAllocatedType();

                // check for lambda call BSDF parameters
                int param_idx = get_metadata_bsdf_param_id(inst);
                if (param_idx < 0) continue;
                DAG_node const *arg = dag_call->get_argument(param_idx);

                // special handling for array parameters
                if (is_libbsdf_array_parameter(sema, param_idx)) {
                    handle_bsdf_array_parameter(ctx, inst, arg, delete_list);
                    continue;
                }

                // determine expression lambda index
                MDL_ASSERT(arg->get_kind() == DAG_node::EK_CALL);
                DAG_call const *arg_call = mi::mdl::cast<DAG_call>(arg);
                MDL_ASSERT(arg_call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA);
                size_t lambda_index = strtoul(arg_call->get_name(), NULL, 10);

                ctx.move_to_body_start();

                Expression_result res = translate_precalculated_lambda(
                    ctx, lambda_index, elem_type);
                inst->replaceAllUsesWith(res.as_ptr(ctx));
                continue;
            }

            if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(II)) {
                // check for calls to BSDFs
                int param_idx = get_metadata_bsdf_param_id(call);
                if (param_idx >= 0) {
                    // instantiate the BSDF function according to the DAG call argument
                    DAG_node const *arg = dag_call->get_argument(param_idx);

                    // identify access to is_black() by checking for bool return type
                    llvm::Type *bool_type = llvm::IntegerType::get(m_llvm_context, 1);
                    if (call->getType() == bool_type) {
                        // replace is_black() by true, if the DAG call argument is "bsdf()"
                        // constant, otherwise replace it by false
                        bool is_black = (arg->get_kind() == DAG_node::EK_CONSTANT &&
                                cast<DAG_constant>(arg)->get_value()->get_kind() ==
                                    IValue::VK_INVALID_REF &&
                                cast<DAG_constant>(arg)->get_value()->get_type()->get_kind() ==
                                    IType::TK_BSDF);

                        call->replaceAllUsesWith(
                            llvm::ConstantInt::get(bool_type, is_black ? 1 : 0));
                    } else {
                        llvm::Value *param_bsdf_func = instantiate_bsdf(arg);

                        // call it with state parameters added
                        llvm::Value *llvm_args[] = {
                            call->getArgOperand(0),  // res_pointer
                            ctx.get_state_parameter(),
                            ctx.get_resource_data_parameter(),
                            ctx.get_exc_state_parameter(),
                            ctx.get_cap_args_parameter(),
                            ctx.get_lambda_results_parameter(),
                            call->getArgOperand(1)   // inherited_normal parameter
                        };
                        llvm::CallInst::Create(param_bsdf_func, llvm_args, "", call);
                    }

                    // mark call instruction for deletion
                    delete_list.push_back(call);
                    continue;
                }

                llvm::Function *called_func = call->getCalledFunction();
                if (called_func == NULL) continue;   // ignore indirect function invocation

                // check for calls to special functions
                llvm::StringRef func_name = called_func->getName();
                if (!called_func->isDeclaration() || !func_name.startswith("get_"))
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

    return bsdf_func;
}

// Translate a DAG node pointing to a DF to LLVM IR.
Expression_result LLVM_code_generator::translate_distribution_function(
    Function_context &ctx,
    llvm::SmallVector<unsigned, 8> const &lambda_result_exprs)
{
    mi::base::Handle<ILambda_function> root_lambda(m_dist_func->get_main_df());
    DAG_node const *body = root_lambda->get_body();
    MDL_ASSERT(
        body->get_type()->get_kind() == IType::TK_BSDF &&
        (
            body->get_kind() == DAG_node::EK_CALL &&
            (
                is_df_semantics(cast<DAG_call>(body)->get_semantic())
            ||
                cast<DAG_call>(body)->get_semantic() == operator_to_semantic(
                    IExpression::OK_TERNARY)
            )
        )
        ||
        (
            body->get_kind() == DAG_node::EK_CONSTANT &&
            cast<DAG_constant>(body)->get_value()->get_kind() == IValue::VK_INVALID_REF
        )
    );

    // allocate the lambda results struct and make it available in the context
    llvm::Value *lambda_results = ctx.create_local(m_lambda_results_struct_type, "lambda_results");
    ctx.override_lambda_results(
        ctx->CreateBitCast(lambda_results, m_type_mapper.get_void_ptr_type()));

    // calculate all required non-constant expression lambdas
    for (size_t i = 0, n = lambda_result_exprs.size(); i < n; ++i) {
        size_t expr_index = lambda_result_exprs[i];

        llvm::Value *dest_ptr = ctx->CreateConstGEP2_32(
            lambda_results,
            0,
            unsigned(m_lambda_result_indices[expr_index]));
        generate_expr_lambda_call(ctx, expr_index, dest_ptr);
    }

    // get the current normal
    IDefinition const *def = m_compiler->find_stdlib_signature("::state", "normal()");
    llvm::Function *func = get_intrinsic_function(def);
    llvm::Value *args[] = { ctx.get_state_parameter() };
    llvm::Value *normal = call_rt_func(ctx, func, args);

    // convert to type used in libbsdf
    llvm::Value *normal_buf = ctx.create_local(m_float3_struct_type, "normal_buf");
    ctx.convert_and_store(normal, normal_buf);

    // recursively instantiate the BSDF and call it
    llvm::Function *bsdf_func = instantiate_bsdf(body);
    if (bsdf_func == NULL) {
        MDL_ASSERT(!"BSDF instantiation failed");
        return Expression_result::undef(lookup_type(body->get_type()));
    }

    llvm::Value *bsdf_args[] = {
        ctx.get_function()->arg_begin(),  // result pointer
        ctx.get_state_parameter(),
        ctx.get_resource_data_parameter(),
        ctx.get_exc_state_parameter(),
        ctx.get_cap_args_parameter(),
        ctx.get_lambda_results_parameter(),  // actually our overridden local struct
        normal_buf
    };
    llvm::CallInst *callinst = ctx->CreateCall(bsdf_func, bsdf_args);

    return Expression_result::value(callinst);
}

// Translate the init function of a distribution function to LLVM IR.
void LLVM_code_generator::translate_distribution_function_init(
    Function_context &ctx,
    llvm::SmallVector<unsigned, 8> const &texture_result_exprs,
    llvm::SmallVector<unsigned, 8> const &lambda_result_exprs)
{
    // allocate the lambda results struct and make it available in the context
    llvm::Value *lambda_results = ctx.create_local(m_lambda_results_struct_type, "lambda_results");
    ctx.override_lambda_results(
        ctx->CreateBitCast(lambda_results, m_type_mapper.get_void_ptr_type()));

    // call state::get_texture_results()
    llvm::Value *texture_results = get_texture_results(ctx);

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

            llvm::Value *dest_ptr = ctx.create_simple_gep_in_bounds(
                texture_results, ctx.get_constant(int(i)));
            generate_expr_lambda_call(ctx, expr_index, dest_ptr);
        }

        // calculate all non-constant expression lambdas required to calculate geometry.normal
        // (this may include geometry.normal itself, if it is reused).
        // The Expr_lambda_scheduler ensures the correct order of the expression lambdas.
        for (size_t i = 0, n = lambda_result_exprs.size(); i < n; ++i) {
            size_t expr_index = lambda_result_exprs[i];
            if (expr_index > geometry_normal_index) continue;

            llvm::Value *dest_ptr = ctx->CreateConstGEP2_32(lambda_results, 0, unsigned(i));
            generate_expr_lambda_call(ctx, expr_index, dest_ptr);
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

        llvm::Value *dest_ptr = ctx.create_simple_gep_in_bounds(
            texture_results, ctx.get_constant(int(i)));
        generate_expr_lambda_call(ctx, expr_index, dest_ptr);
    }
}

} // mdl
} // mi
