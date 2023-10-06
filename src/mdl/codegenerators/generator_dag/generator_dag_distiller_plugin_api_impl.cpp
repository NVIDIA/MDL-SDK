/******************************************************************************
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "generator_dag_distiller_plugin_api_impl.h"

#include <mi/base/handle.h>

#include "mdl/compiler/compilercore/compilercore_modules.h"
#include "mdl/compiler/compilercore/compilercore_def_table.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"

#include "generator_dag_tools.h"
#include "generator_dag_ir_checker.h"
#include "generator_dag_builder.h"

#include <mi/mdl/mdl_distiller_node_types.h>

#include <string>
#include <iostream>

namespace mi {
namespace mdl {

extern Node_types s_node_types;

#define MDL_DIST_PLUG_DEBUG 0

#if MDL_DIST_PLUG_DEBUG
// Utility for debugging. Prints a value to stderr.
void print_value(mi::mdl::IValue const *v) {
    switch (v->get_kind()) {
    case mi::mdl::IValue::Kind::VK_BAD:
        std::cerr << "<bad>";
        break;

    case mi::mdl::IValue::Kind::VK_BOOL:
        std::cerr << cast<IValue_bool>(v)->get_value();
        break;

    case mi::mdl::IValue::Kind::VK_INT:
        std::cerr << cast<IValue_int>(v)->get_value();
        break;

    case mi::mdl::IValue::Kind::VK_ENUM:
    {
        IType_enum const *t = cast<IType_enum>(v->get_type());
        int value_idx = cast<IValue_enum>(v)->get_index();
        int code;
        ISymbol const *name;
        t->get_value(value_idx, name, code);
        std::cerr << t->get_symbol()->get_name() << "::";
        std::cerr << name->get_name();
        break;
    }

    case mi::mdl::IValue::Kind::VK_FLOAT:
        std::cerr << cast<IValue_float>(v)->get_value();
        break;

    case mi::mdl::IValue::Kind::VK_DOUBLE:
        std::cerr << cast<IValue_double>(v)->get_value();
        break;

    case mi::mdl::IValue::Kind::VK_STRING:
        std::cerr << "\"" << cast<IValue_string>(v)->get_value() << "\"";
        break;

    case mi::mdl::IValue::Kind::VK_VECTOR:
    {
        IValue_vector const *vv = cast<IValue_vector>(v);
        std::cerr << "float"
                  << vv->get_component_count()
                  << "(";
        for (int i = 0; i < vv->get_component_count(); i++) {
            if (IValue_float const *f = as<IValue_float>(vv->get_value(i))) {
                if (i > 0)
                    std::cerr << ",";
                std::cerr << f->get_value();
            } else if (IValue_int const *f = as<IValue_int>(vv->get_value(i))) {
                if (i > 0)
                    std::cerr << ",";
                std::cerr << f->get_value();
            } else {
                MDL_ASSERT(!"unreachable");
            }
        }
        std::cerr << ")";
        break;
    }

    case mi::mdl::IValue::Kind::VK_MATRIX:
        std::cerr << "<matrix>";
        break;

    case mi::mdl::IValue::Kind::VK_ARRAY:
        std::cerr << "<array>";
        break;

    case mi::mdl::IValue::Kind::VK_RGB_COLOR:
    {
        std::cerr << "color(";
        for (int i = 0; i < 3; i++) {
            IValue_float const *f = cast<IValue_rgb_color>(v)->get_value(i);
            if (i > 0)
                std::cerr << ",";
            std::cerr << f->get_value();
        }
        std::cerr << ")";
        break;
    }

    case mi::mdl::IValue::Kind::VK_STRUCT:
        std::cerr << "<struct>";
        break;

    case mi::mdl::IValue::Kind::VK_INVALID_REF:
    {
        IType const *t = v->get_type();
        switch (t->get_kind()) {
        case IType::Kind::TK_BSDF:
            std::cerr << "bsdf()";
            break;
        case IType::Kind::TK_EDF:
            std::cerr << "edf()";
            break;
        case IType::Kind::TK_VDF:
            std::cerr << "vdf()";
            break;
        case IType::Kind::TK_HAIR_BSDF:
            std::cerr << "hair_bsdf()";
            break;
        default:
            std::cerr << "<invalid_ref>";
            break;
        }
        break;
    }

    case mi::mdl::IValue::Kind::VK_TEXTURE:
        std::cerr << "<texture>";
        break;

    case mi::mdl::IValue::Kind::VK_LIGHT_PROFILE:
        std::cerr << "<light_profile>";
        break;

    case mi::mdl::IValue::Kind::VK_BSDF_MEASUREMENT:
        std::cerr << "<bsdf_measurement>";
        break;

    default:
        std::cerr << "<unknown>";
        break;
    }
}

// Utility for debugging. Prints i*2 spaces.
void indent(int i) {
    for (int x = 0; x < i; x++) {
        std::cerr << "  ";
    }
}

// Utility for debugging. Prints attribubtes of the given node, if
// there are any.
void Distiller_plugin_api_impl::mb_print_attrs(IGenerated_code_dag::IMaterial_instance const *inst, DAG_node const *node, int level) {
    auto inner = m_attribute_map.find(node);
    if (inner != m_attribute_map.end()) {
        std::cerr << "\x1b[32m[[ ";
        int i = 0;
        for (auto iit : inner->second) {
            if (i > 0) {
                std::cerr << ",";
            }
            if (level > 10) {
                std::cerr << "\n";
                indent(level);
            }
            if (iit.second) {
                std::cerr << iit.first << "=";
                pprint_node(inst, iit.second, level + 1);
            } else {
                std::cerr << iit.first << "=null";
            }
            i++;
        }
        std::cerr << " ]]\x1b[0m";
    }
}

// Utility for debugging. Prints a node with indentation and some
// colors. Only works properly on terminals with ANSI color code
// emulation.
void Distiller_plugin_api_impl::pprint_node(IGenerated_code_dag::IMaterial_instance const *inst,
                                            DAG_node const *node, int level) {
  restart:
    switch (node->get_kind()) {
    case DAG_node::EK_CONSTANT:
    {
        DAG_constant const *c = cast<DAG_constant>(node);
        IValue const *value = c->get_value();
        print_value(value);
        mb_print_attrs(inst, node, level);
        break;
    }
    case DAG_node::EK_PARAMETER:
    {
        DAG_parameter const *p    = cast<DAG_parameter>(node);
        int                 index = p->get_index();
        std::cerr << "p" << index;
        mb_print_attrs(inst, node, level);
        break;
    }
    case DAG_node::EK_TEMPORARY:
    {
        DAG_temporary const *t = cast<DAG_temporary>(node);
        int index = t->get_index();
        DAG_node const *val = t->get_expr();
        switch (val->get_kind()) {
        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            node = val;
            goto restart;
        default:
            std::cerr << "t" << index;
            mb_print_attrs(inst, node, level);
            break;
        }
        break;
    }
    case DAG_node::EK_CALL:
    {
        DAG_call const         *call    = cast<DAG_call>(node);
        int                    n_params = call->get_argument_count();
//      IDefinition::Semantics sema     = call->get_semantic();
        char const             *name    = call->get_name();
        int np = std::string(name).find('(');
        if (np == std::string::npos) {
            np = strlen(name);
        }
        std::string n(name, 0, np);

        std::cerr << n.c_str() << "(";
        for (int i = 0; i < n_params; ++i) {
            if (i > 0) {
                std::cerr << ",";
            }
            if (level > 4 || i > 3) {
                std::cerr << "\n";
                indent(level);
            }
            pprint_node(inst, call->get_argument(i), level + 1);
        }
        std::cerr << ")";
        mb_print_attrs(inst, node, level);
        break;
    }
    default:
        MDL_ASSERT(!"unexpected case");
        break;
    }
}
#endif

/// Make mixer canonical by sorting its arguments in ascending order of its selector value
class Normalize_mixers_rules : public mi::mdl::IRule_matcher {
public:
    // Return the strategy to be used with this rule set.
    mi::mdl::Rule_eval_strategy get_strategy() const MDL_FINAL;

    /// Return the number of imported MDL modules in this rule set.
    size_t get_target_material_name_count() const MDL_FINAL {
        return 0;
    }

    /// Return the name of the imported MDL module with the given
    /// index for this rule set.
    char const *get_target_material_name(size_t i) const {
        MDL_ASSERT(!"unreachable");
        return "";
    }

    DAG_node const *matcher(
        IRule_matcher_event        *event_handler,
        IDistiller_plugin_api      &plugin_api,
        DAG_node const             *node,
        const mi::mdl::Distiller_options    *options,
        mi::mdl::Rule_result_code  &result_code) const MDL_FINAL;

    bool postcond(
        IRule_matcher_event        *event_handler,
        IDistiller_plugin_api      &plugin_api,
        DAG_node const             *root,
        mi::mdl::Distiller_options const    *options) const MDL_FINAL;

    // Return the name of the rule set (for diagnostic messages)
    char const * get_rule_set_name() const MDL_FINAL;


    /// Set the Node_types object to use.
    void set_node_types(Node_types *node_types) {
        m_node_types = node_types;
    };

private:
    Node_types *m_node_types = nullptr;
};

// Return the strategy to be used with this rule set.
mi::mdl::Rule_eval_strategy Normalize_mixers_rules::get_strategy() const
{
    return mi::mdl::RULE_EVAL_BOTTOM_UP;
}

// Return the name of the rule set (for diagnostic messages)
char const * Normalize_mixers_rules::get_rule_set_name() const
{
    return "Normalize_mixers_rules";
}

// Run the matcher.
mi::mdl::DAG_node const* Normalize_mixers_rules::matcher(
    IRule_matcher_event       *event_handler,
    IDistiller_plugin_api     &api,
    DAG_node const            *node,
    const mi::mdl::Distiller_options  *,
    mi::mdl::Rule_result_code &result_code) const
{
    switch (api.get_selector(node)) {
    case DS_DIST_BSDF_MIX_2:  // match for bsdf_mix_2, edf_mix_2, vdf_mix_2
    case DS_DIST_EDF_MIX_2:
    case DS_DIST_VDF_MIX_2:
// RUID 1
    if (true) {
        DAG_node const *v_w0 = api.get_remapped_argument(node, 0);
        DAG_node const *v_b0 = api.get_remapped_argument(node, 1);
        DAG_node const *v_w1 = api.get_remapped_argument(node, 2);
        DAG_node const *v_b1 = api.get_remapped_argument(node, 3);
        if (api.get_selector(v_b0) <= api.get_selector(v_b1))
            break;

        if (event_handler != NULL)
            event_handler->rule_match_event("Normalize_mixers_rules",
                                            /* RUID */ 1, "..._mix_2", "<internal>", 0);

        DAG_call::Call_argument args[4];
        args[0].arg = v_w0;
        args[1].arg = v_b0;
        args[2].arg = v_w1;
        args[3].arg = v_b1;
        return api.create_mixer_call( args, 4);
    }
    break;
    case DS_DIST_BSDF_MIX_3:  // match for bsdf_mix_3, edf_mix_3, vdf_mix_3
    case DS_DIST_EDF_MIX_3:
    case DS_DIST_VDF_MIX_3:
// deadrule RUID 2
    if (true) {
        DAG_node const *v_w0 = api.get_remapped_argument(node, 0);
        DAG_node const *v_b0 = api.get_remapped_argument(node, 1);
        DAG_node const *v_w1 = api.get_remapped_argument(node, 2);
        DAG_node const *v_b1 = api.get_remapped_argument(node, 3);
        DAG_node const *v_w2 = api.get_remapped_argument(node, 4);
        DAG_node const *v_b2 = api.get_remapped_argument(node, 5);

        if ((api.get_selector(v_b0) <= api.get_selector(v_b1))
            && (api.get_selector(v_b1) <= api.get_selector(v_b2)))
            break;

        if (event_handler != NULL)
            event_handler->rule_match_event("Normalize_mixers_rules",
                                            /* RUID */ 2, "..._mix_3", "<internal>", 0);

        DAG_call::Call_argument args[6];
        args[0].arg = v_w0;
        args[1].arg = v_b0;
        args[2].arg = v_w1;
        args[3].arg = v_b1;
        args[4].arg = v_w2;
        args[5].arg = v_b2;
        return api.create_mixer_call( args, 6);
    }
    break;
    case DS_DIST_BSDF_COLOR_MIX_2:  // match for ..._color_mix_2
    case DS_DIST_EDF_COLOR_MIX_2:
// deadrule RUID 3
    if (true) {
        DAG_node const *v_w0 = api.get_remapped_argument(node, 0);
        DAG_node const *v_b0 = api.get_remapped_argument(node, 1);
        DAG_node const *v_w1 = api.get_remapped_argument(node, 2);
        DAG_node const *v_b1 = api.get_remapped_argument(node, 3);
        if (api.get_selector(v_b0) <= api.get_selector(v_b1))
            break;

        if (event_handler != NULL)
            event_handler->rule_match_event("Normalize_mixers_rules",
                                            /* RUID */ 3, "..._color_mix_2", "<internal>", 0);

        DAG_call::Call_argument args[4];
        args[0].arg = v_w0;
        args[1].arg = v_b0;
        args[2].arg = v_w1;
        args[3].arg = v_b1;
        return api.create_color_mixer_call( args, 4);
    }
    break;
    case DS_DIST_BSDF_COLOR_MIX_3:  // match for ..._color_mix_3
    case DS_DIST_EDF_COLOR_MIX_3:
// deadrule RUID 4
    if (true) {
        DAG_node const *v_w0 = api.get_remapped_argument(node, 0);
        DAG_node const *v_b0 = api.get_remapped_argument(node, 1);
        DAG_node const *v_w1 = api.get_remapped_argument(node, 2);
        DAG_node const *v_b1 = api.get_remapped_argument(node, 3);
        DAG_node const *v_w2 = api.get_remapped_argument(node, 4);
        DAG_node const *v_b2 = api.get_remapped_argument(node, 5);

        if ((api.get_selector(v_b0) <= api.get_selector(v_b1))
            && (api.get_selector(v_b1) <= api.get_selector(v_b2)))
            break;

        if (event_handler != NULL)
            event_handler->rule_match_event("Normalize_mixers_rules",
                                            /* RUID */ 4, "..._color_mix_3", "<internal>", 0);

        DAG_call::Call_argument args[6];
        args[0].arg = v_w0;
        args[1].arg = v_b0;
        args[2].arg = v_w1;
        args[3].arg = v_b1;
        args[4].arg = v_w2;
        args[5].arg = v_b2;
        return api.create_color_mixer_call( args, 6);
    }
    break;
    default:
        break;
    }

    return node;
}

bool Normalize_mixers_rules::postcond(
    IRule_matcher_event *,
    IDistiller_plugin_api &,
    DAG_node const *,
    const mi::mdl::Distiller_options *) const
{
    return true;
}

// Constructor.
Distiller_plugin_api_impl::Distiller_plugin_api_impl(
    IGenerated_code_dag::IMaterial_instance const *instance,
    ICall_name_resolver                           *call_resolver)
: m_alloc(impl_cast<Generated_code_dag::Material_instance>(instance)->get_allocator())
, m_compiler(impl_cast<Generated_code_dag::Material_instance>(instance)->get_mdl())
, m_printer(m_alloc)
, m_type_factory(NULL)
, m_value_factory(NULL)
, m_node_factory(NULL)
//, m_marker_map(0, Visited_node_map::hasher(), Visited_node_map::key_equal(), m_alloc)
//, m_attribute_marker_map(0, Visited_node_map::hasher(), Visited_node_map::key_equal(), m_alloc)
, m_strategy(RULE_EVAL_BOTTOM_UP)
, m_matcher(NULL)
, m_event_handler(NULL)
, m_options(NULL)
, m_call_resolver(call_resolver)
, m_checker(m_alloc, m_call_resolver)
, m_normalize_mixers(false)
, m_attribute_map(m_alloc)
{
    m_global_ior[0] = 1.4f;
    m_global_ior[1] = 1.4f;
    m_global_ior[2] = 1.4f;
}

/// Factory to create a newly allocated distiller plugin API. 
IDistiller_plugin_api* IDistiller_plugin_api::get_new_distiller_plugin_api(
    IGenerated_code_dag::IMaterial_instance const *instance,
    ICall_name_resolver                           *call_resolver) {
    return new Distiller_plugin_api_impl( instance, call_resolver);
}

/// Immediately deletes this distiller plugin API
void Distiller_plugin_api_impl::release() const {
    delete this;
}

// Create a constant.
DAG_constant const *Distiller_plugin_api_impl::create_constant(IValue const *value)
{
    DAG_constant const *res = m_node_factory->create_constant(m_value_factory->import(value));
    m_checker.check_const(res);
    return res;
}

// Create a temporary reference.
DAG_temporary const *Distiller_plugin_api_impl::create_temporary(DAG_node const *node, int index)
{
    DAG_temporary const *res = m_node_factory->create_temporary(node, index);
    m_checker.check_temp(res);
    return res;
}

/// Create a call.
DAG_node const *Distiller_plugin_api_impl::create_call(
    char const                    *name,
    IDefinition::Semantics        sema,
    DAG_call::Call_argument const call_args[],
    int                           num_call_args,
    IType const                   *ret_type)
{
    DAG_node const *res = m_node_factory->create_call(
        name, sema, call_args, num_call_args, m_type_factory->import(ret_type));
    m_checker.check_node(res);
    return res;
}

/// Create a function call for a non-overloaded function. All parameter
/// and return types are deduced from the function definition.
DAG_node const *Distiller_plugin_api_impl::create_function_call(
    char const             *name,
    DAG_node const * const call_args[],
    size_t                 num_call_args)
{
    for (size_t i = 0; i < num_call_args; ++i)
        m_checker.check_node(call_args[i]);

    IDefinition::Semantics sema = IDefinition::DS_UNKNOWN;
    IType const            *ret_type = 0;

    // build argument list for overload resolution lookup
    vector<string>::Type parameter_types(get_allocator());
    string parameter_types_str(get_allocator());
    parameter_types_str += '(';
    for (size_t i = 0; i < num_call_args; ++i) {
        if (i > 0)
            parameter_types_str += ',';
        IType const *t = call_args[i]->get_type();
        m_printer.print(t->skip_type_alias());
        parameter_types.push_back(m_printer.get_line());
        parameter_types_str += parameter_types.back();
    }
    parameter_types_str += ')';

    // find the module name if the name has one
    Size last_scope_pos = 0;
    for ( Size i = 1; name[i] != '\0'; ++i) {
        if ( name[i] == ':' && name[i+1] == ':')
            last_scope_pos = i;
    }

    mi::base::Handle<IModule const> owner;
    if ( last_scope_pos != 0) {
        // direct lookup of module name
        string module_name( name, last_scope_pos, get_allocator());
        owner = m_call_resolver->get_owner_module(module_name.c_str());
    } else { 
        // find the module through the function definition
        string signature = name + parameter_types_str;
        owner = m_call_resolver->get_owner_module(signature.c_str());
    }

    if (!owner.is_valid_interface()) {
        MDL_ASSERT(!"module or function call name does not exist");
        return NULL;
    }

    vector<const char*>::Type parameter_types_cstr(get_allocator());
    for (const auto& s: parameter_types)
        parameter_types_cstr.push_back(s.c_str());
    size_t n = parameter_types_cstr.size();
    mi::base::Handle<IOverload_result_set const> res(
        owner->find_overload_by_signature(name, n > 0 ? &parameter_types_cstr[0] : 0, n));

    if (!res.is_valid_interface()) {
        MDL_ASSERT(!"function overload not found");
        return NULL;
    }

    char const *sig = res->first_signature();
    if (sig == NULL) {
        MDL_ASSERT(!"function overload not found");
        return NULL;
    }

    IDefinition const *def = res->first();

    if (res->next() != NULL) {
        MDL_ASSERT(!"ambiguous overload");
        return NULL;
    }

    IType_function const *f_ty = cast<IType_function>(def->get_type());

    if (num_call_args != f_ty->get_parameter_count()) {
        // std::cerr << "ERR: " << name << '(' << params.c_str() << ')' << std::endl;
        MDL_ASSERT(!"wrong number of arguments (cannot handle defaults)");
        return NULL;
    }

    ret_type = f_ty->get_return_type();

    // FIXME: do type binding, uniform propagation
    sema = def->get_semantics();

    VLA<DAG_call::Call_argument> args(m_alloc, num_call_args);
    for (size_t i = 0; i < num_call_args; ++i) {
        IType const   *p_tp;
        ISymbol const *p_sym;

        f_ty->get_parameter(int(i), p_tp, p_sym);

        args[i].param_name = p_sym->get_name();
        args[i].arg        = call_args[i];
    }

    // enable optimizations for function calls
    Option_store<DAG_node_factory_impl, bool> optimizations(
        *m_node_factory, &DAG_node_factory_impl::enable_opt, true);

    return create_call(sig, sema, args.data(), args.size(), ret_type);
}

// Create a 1-, 2-, or 3-mixer call, with 2, 4, or 6 parameters respectively.
DAG_node const *Distiller_plugin_api_impl::create_mixer_call(
    DAG_call::Call_argument const call_args[],
    int                           num_call_args)
{
    MDL_ASSERT( num_call_args == 2 || num_call_args == 4 || num_call_args == 6);
    int n = num_call_args / 2;

    // permute order for canonical order
    size_t permutation[6] = {0,1,2,3,4,5};
    if ( n == 2) {
        if ( get_selector(call_args[1].arg) > get_selector(call_args[3].arg)) {
            std::swap( permutation[0], permutation[2]);
            std::swap( permutation[1], permutation[3]);
        }
    } else if ( n == 3) {
        if ( get_selector(call_args[1].arg) > get_selector(call_args[3].arg)) {
            std::swap( permutation[0], permutation[2]);
            std::swap( permutation[1], permutation[3]);
        }
        if ( get_selector(call_args[permutation[3]].arg) > get_selector(call_args[5].arg)) {
            std::swap( permutation[2], permutation[4]);
            std::swap( permutation[3], permutation[5]);
        }
        if ( get_selector(call_args[permutation[1]].arg)
             > get_selector(call_args[permutation[3]].arg)) {
            std::swap( permutation[0], permutation[2]);
            std::swap( permutation[1], permutation[3]);
        }
    }

    // Determine type of mixer: BSDF, EDF, VDF
    IType const* itp1 = get_bsdf_component_type();
    IType const* itp2 = get_bsdf_component_array_type(n);
    const char* tp1 = "::df::bsdf_component(float,bsdf)";
    const char* tp2 = "::df::normalized_mix(::df::bsdf_component[N])";
    IType const* arg_type = call_args[1].arg->get_type();
    if ( arg_type->get_kind() == mi::mdl::IType::TK_EDF) {
        itp1 = get_edf_component_type();
        itp2 = get_edf_component_array_type(n);
        tp1 = "::df::edf_component(float,edf)";
        tp2 = "::df::normalized_mix(::df::edf_component[N])";
    } else if ( arg_type->get_kind() == mi::mdl::IType::TK_VDF) {
        itp1 = get_vdf_component_type();
        itp2 = get_vdf_component_array_type(n);
        tp1 = "::df::vdf_component(float,vdf)";
        tp2 = "::df::normalized_mix(::df::vdf_component[N])";
    }

    // room for up to three DF_component structs
    DAG_call::Call_argument comp_args[3];

    for ( int i = 0; i < n; ++ i) {
        DAG_call::Call_argument args[2];

        args[0].param_name = "weight";
        args[0].arg        = call_args[permutation[2*i]].arg;
        args[1].param_name = "component";
        args[1].arg        = call_args[permutation[2*i+1]].arg;

        static char const * const idx[3] = {"value0", "value1", "value2"};
        comp_args[i].param_name = idx[i];
        comp_args[i].arg = create_call(
            tp1,
            mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR,
            args, 2,
            itp1);
    }

    // create bsdf_component's array and mixer
    DAG_node const * array = create_call(
        get_array_constructor_signature(),
        mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR,
        comp_args, n,
        itp2);
    DAG_call::Call_argument mix_arg[1] = {{ array, "components"}};
    return create_call(
        tp2,
        mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX,
        mix_arg, 1,
        arg_type);
}

// Create a 1-, 2-, or 3-color-mixer call, with 2, 4, or 6 parameters respectively.
// Color-mixer exists for BSDF, EDF, and VDF.
DAG_node const *Distiller_plugin_api_impl::create_color_mixer_call(
    DAG_call::Call_argument const call_args[],
    int                           num_call_args)
{
    MDL_ASSERT( num_call_args == 2 || num_call_args == 4 || num_call_args == 6);
    int n = num_call_args / 2;

    // permute order for canonical order
    size_t permutation[6] = {0,1,2,3,4,5};
    if ( n == 2) {
        if ( get_selector(call_args[1].arg) > get_selector(call_args[3].arg)) {
            std::swap( permutation[0], permutation[2]);
            std::swap( permutation[1], permutation[3]);
        }
    } else if ( n == 3) {
        if ( get_selector(call_args[1].arg) > get_selector(call_args[3].arg)) {
            std::swap( permutation[0], permutation[2]);
            std::swap( permutation[1], permutation[3]);
        }
        if ( get_selector(call_args[permutation[3]].arg) > get_selector(call_args[5].arg)) {
            std::swap( permutation[2], permutation[4]);
            std::swap( permutation[3], permutation[5]);
        }
        if ( get_selector(call_args[permutation[1]].arg)
             > get_selector(call_args[permutation[3]].arg)) {
            std::swap( permutation[0], permutation[2]);
            std::swap( permutation[1], permutation[3]);
        }
    }

    // Determine type of mixer: BSDF, EDF, or VDF
    IType const* itp1 = get_color_bsdf_component_type();
    IType const* itp2 = get_color_bsdf_component_array_type(n);
    const char* tp1 = "::df::color_bsdf_component(color,bsdf)";
    const char* tp2 = "::df::color_normalized_mix(::df::color_bsdf_component[N])";
    IType const* arg_type = call_args[1].arg->get_type();
    if ( arg_type->get_kind() == mi::mdl::IType::TK_EDF) {
        itp1 = get_color_edf_component_type();
        itp2 = get_color_edf_component_array_type(n);
        tp1 = "::df::color_edf_component(color,edf)";
        tp2 = "::df::color_normalized_mix(::df::color_edf_component[N])";
    } else if ( arg_type->get_kind() == mi::mdl::IType::TK_VDF) {
        itp1 = get_color_vdf_component_type();
        itp2 = get_color_vdf_component_array_type(n);
        tp1 = "::df::color_vdf_component(color,edf)";
        tp2 = "::df::color_normalized_mix(::df::color_vdf_component[N])";
    }
    // room for up to three DF_component structs
    DAG_call::Call_argument comp_args[3];

    for ( int i = 0; i < n; ++ i) {
        DAG_call::Call_argument args[2];

        args[0].param_name = "weight";
        args[0].arg        = call_args[permutation[2*i]].arg;
        args[1].param_name = "component";
        args[1].arg        = call_args[permutation[2*i+1]].arg;

        static char const * const idx[3] = {"value0", "value1", "value2"};
        comp_args[i].param_name = idx[i];
        comp_args[i].arg = create_call(
            tp1,
            mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR,
            args, 2,
            itp1);
    }

    // create color_..df_component's array and mixer
    DAG_node const * array = create_call(
        get_array_constructor_signature(),
        mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR,
        comp_args, n,
        itp2);
    DAG_call::Call_argument mix_arg[1] = {{ array, "components"}};
    DAG_node const * res = create_call(
        tp2,
        mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
        mix_arg, 1,
        arg_type);
    MDL_ASSERT(
        ( arg_type->get_kind() == mi::mdl::IType::TK_BSDF &&
          get_selector(res) == mi::mdl::DS_DIST_BSDF_COLOR_MIX_1 + num_call_args/2 - 1)
        || ( arg_type->get_kind() == mi::mdl::IType::TK_EDF &&
             get_selector(res) == mi::mdl::DS_DIST_EDF_COLOR_MIX_1 + num_call_args/2 - 1)
    );
    return res;
}

// Create a parameter reference.
DAG_parameter const *Distiller_plugin_api_impl::create_parameter(IType const *type, int index)
{
    DAG_parameter const *res =
        m_node_factory->create_parameter(m_type_factory->import(type), index);
    m_checker.check_parameter(res);
    return res;
}

// Enable common subexpression elimination.
bool Distiller_plugin_api_impl::enable_cse(bool flag)
{
    return m_node_factory->enable_cse(flag);
}

// Enable optimization.
bool Distiller_plugin_api_impl::enable_opt(bool flag)
{
    return m_node_factory->enable_opt(flag);
}

// Enable unsafe math optimizations.
bool Distiller_plugin_api_impl::enable_unsafe_math_opt(bool flag)
{
    return m_node_factory->enable_unsafe_math_opt(flag);
}

// Get the type factory associated with this expression factory.
IType_factory *Distiller_plugin_api_impl::get_type_factory()
{
    return m_value_factory->get_type_factory();
}

// Get the value factory associated with this expression factory.
IValue_factory *Distiller_plugin_api_impl::get_value_factory()
{
    return m_value_factory;
}

// Return the type for ::df::bsdf_component
IType const *Distiller_plugin_api_impl::get_bsdf_component_type()
{
    // TBD: cache this type access in c'tor
    mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
    Definition const *def = find_exported_def(df_mod.get(), "bsdf_component");
    return def->get_type();
}

// Return the type for ::df::edf_component
IType const *Distiller_plugin_api_impl::get_edf_component_type()
{
    // TBD: cache this type access in c'tor
    mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
    Definition const *def = find_exported_def(df_mod.get(), "edf_component");
    return def->get_type();
}

// Return the type for ::df::vdf_component
IType const *Distiller_plugin_api_impl::get_vdf_component_type()
{
    // TBD: cache this type access in c'tor
    mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
    Definition const *def = find_exported_def(df_mod.get(), "vdf_component");
    return def->get_type();
}

// Return the type for ::df::bsdf_component
IType const *Distiller_plugin_api_impl::get_bsdf_component_array_type( int n_values)
{
    // TBD: cache this type access in c'tor, n_values between 1 and 3
    IType const* elem_type = get_bsdf_component_type();
    return m_type_factory->create_array(elem_type, n_values);
}

// Return the type for ::df::edf_component
IType const *Distiller_plugin_api_impl::get_edf_component_array_type( int n_values)
{
    // TBD: cache this type access in c'tor, n_values between 1 and 3
    IType const* elem_type = get_edf_component_type();
    return m_type_factory->create_array(elem_type, n_values);
}

// Return the type for ::df::vdf_component
IType const *Distiller_plugin_api_impl::get_vdf_component_array_type( int n_values)
{
    // TBD: cache this type access in c'tor, n_values between 1 and 3
    IType const* elem_type = get_vdf_component_type();
    return m_type_factory->create_array(elem_type, n_values);
}

// Return the type for ::df::color_bsdf_component
IType const *Distiller_plugin_api_impl::get_color_bsdf_component_type()
{
    // TBD: cache this type access in c'tor
    mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
    Definition const *def = find_exported_def(df_mod.get(), "color_bsdf_component");
    return def->get_type();
}

// Return the type for ::df::color_edf_component
IType const *Distiller_plugin_api_impl::get_color_edf_component_type()
{
    // TBD: cache this type access in c'tor
    mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
    Definition const *def = find_exported_def(df_mod.get(), "color_edf_component");
    return def->get_type();
}

// Return the type for ::df::color_vdf_component
IType const *Distiller_plugin_api_impl::get_color_vdf_component_type()
{
    // TBD: cache this type access in c'tor
    mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
    Definition const *def = find_exported_def(df_mod.get(), "color_vdf_component");
    return def->get_type();
}

// Return the type for ::df::color_bsdf_component
IType const *Distiller_plugin_api_impl::get_color_bsdf_component_array_type( int n_values)
{
    // TBD: cache this type access in c'tor, n_values between 1 and 3
    IType const* elem_type = get_color_bsdf_component_type();
    return m_type_factory->create_array(elem_type, n_values);
}

// Return the type for ::df::color_edf_component
IType const *Distiller_plugin_api_impl::get_color_edf_component_array_type( int n_values)
{
    // TBD: cache this type access in c'tor, n_values between 1 and 3
    IType const* elem_type = get_color_edf_component_type();
    return m_type_factory->create_array(elem_type, n_values);
}

// Return the type for ::df::color_vdf_component
IType const *Distiller_plugin_api_impl::get_color_vdf_component_array_type( int n_values)
{
    // TBD: cache this type access in c'tor, n_values between 1 and 3
    IType const* elem_type = get_color_vdf_component_type();
    return m_type_factory->create_array(elem_type, n_values);
}

// Return the type for bool
IType const *Distiller_plugin_api_impl::get_bool_type()
{
    // TBD: cache this type access in c'tor
    return m_type_factory->create_bool();
}


// Creates an operator, handles types.
DAG_node const *Distiller_plugin_api_impl::create_unary(
    Unary_operator op,
    DAG_node const *o)
{
    m_checker.check_node(o);

    IExpression_unary::Operator un_op = IExpression_unary::Operator(op);

    IType const *ret_type = o->get_type();

    DAG_call::Call_argument args[1];

    IType const *ot = o->get_type()->skip_type_alias();

    // we do not have extra operators for enums, so use int here
    if (is<IType_enum>(ot)) {
        o = convert_enum_to_int(o);
        ot = o->get_type()->skip_type_alias();
    }

    args[0].param_name = "x";
    args[0].arg        = o;

    IDefinition::Semantics sema  = operator_to_semantic(un_op);

    string name(DAG_builder::unary_op_to_name(un_op), get_allocator());
    name += '(';
    m_printer.print(ot->skip_type_alias());
    name += m_printer.get_line();
    name += ')';

    // enable optimizations for function calls
    Option_store<DAG_node_factory_impl, bool> optimizations(
        *m_node_factory, &DAG_node_factory_impl::enable_opt, true);

    return create_call(name.c_str(), sema, args, 1, ret_type);
}

// Creates an operator, handles types.
DAG_node const *Distiller_plugin_api_impl::create_binary(
    Binary_operator op,
    DAG_node const  *l,
    DAG_node const  *r)
{
    m_checker.check_node(l);
    m_checker.check_node(r);

    IExpression_binary::Operator bin_op = IExpression_binary::Operator(op);

    DAG_call::Call_argument args[2];

    if (bin_op == IExpression_binary::OK_ARRAY_INDEX) {
        // the index operator is mapped to DS_INTRINSIC_DAG_INDEX_ACCESS
        IType const *lt       = l->get_type()->skip_type_alias();
        IType const *ret_type = NULL;

        if (IType_compound const *v_tp = as<IType_compound>(lt)) {
            ret_type = v_tp->get_compound_type(0);
        } else {
            MDL_ASSERT(!"index not on compound type");
        }

        string name(DAG_builder::binary_op_to_name(bin_op), get_allocator());
        name += "(<0>[],int)";

        args[0].param_name = "a";
        args[0].arg = l;
        args[1].param_name = "i";
        args[1].arg = r;

        // enable optimizations for function calls
        Option_store<DAG_node_factory_impl, bool> optimizations(
            *m_node_factory, &DAG_node_factory_impl::enable_opt, true);

        IDefinition::Semantics sema = operator_to_semantic(bin_op);
        return create_call(
            name.c_str(), sema, args, 2, ret_type);
    }

    DAG_node_factory_impl::normalize(bin_op, l, r);

    // for mixed operations, choose the type of higher kind as return type, which chooses
    // color over float3 over float over int.
    IType const *ret_type = l->get_type();
    if ( ret_type->get_kind() < r->get_type()->get_kind())
        ret_type = r->get_type();
    // relational ops have bool return type
    if ((bin_op >= IExpression_binary::OK_LESS) && (bin_op <= IExpression_binary::OK_NOT_EQUAL))
        ret_type = get_bool_type();

    IType const *lt = l->get_type()->skip_type_alias();
    IType const *rt = r->get_type()->skip_type_alias();

    // we do not have extra operators for enums, so use int here
    if (is<IType_enum>(lt)) {
        l = convert_enum_to_int(l);
        lt = l->get_type()->skip_type_alias();
    }
    if (is<IType_enum>(rt)) {
        r = convert_enum_to_int(r);
        rt = r->get_type()->skip_type_alias();
    }

    args[0].param_name = "x";
    args[0].arg        = l;
    args[1].param_name = "y";
    args[1].arg        = r;

    IDefinition::Semantics sema  = operator_to_semantic(bin_op);

    string name(DAG_builder::binary_op_to_name(bin_op), get_allocator());
    name += '(';
    m_printer.print(lt);
    name += m_printer.get_line();
    name += ',';
    m_printer.print(rt);
    name += m_printer.get_line();
    name += ')';

    // enable optimizations for function calls
    Option_store<DAG_node_factory_impl, bool> optimizations(
        *m_node_factory, &DAG_node_factory_impl::enable_opt, true);

    return create_call(name.c_str(), sema, args, 2, ret_type);
}

// Creates a ternary operator.
DAG_node const *Distiller_plugin_api_impl::create_ternary(
    DAG_node const *cond,
    DAG_node const *t_expr,
    DAG_node const *f_expr)
{
    m_checker.check_node(cond);
    m_checker.check_node(t_expr);
    m_checker.check_node(f_expr);

    IType const *t_type = t_expr->get_type();
    IType const *f_type = f_expr->get_type();

    IType const *arg_type = t_type->skip_type_alias();

    // compute return type modifiers
    IType::Modifiers modifiers = t_type->get_type_modifiers() & IType::MK_UNIFORM;
    modifiers &= f_type->get_type_modifiers();

    // note: we normally enforce that the cond type is uniform, but it will not hurt
    // to compute is here the right way
    modifiers &= cond->get_type()->get_type_modifiers();

    IType const *ret_type = m_type_factory->create_alias(arg_type, /*name=*/NULL, modifiers);

    DAG_call::Call_argument args[3];

    args[0].arg        = cond;
    args[0].param_name = "cond";
    args[1].arg        = t_expr;
    args[1].param_name = "true_exp";
    args[2].arg        = f_expr;
    args[2].param_name = "false_exp";

    return create_call(
        get_ternary_operator_signature(),
        operator_to_semantic(IExpression::OK_TERNARY),
        args,
        dimension_of(args),
        ret_type);
}

// Creates a SELECT operator on a struct or vector.
DAG_node const *Distiller_plugin_api_impl::create_select(
    DAG_node const *s,
    char const     *member)
{
    m_checker.check_node(s);

    IType const *l_type = s->get_type();
    if (IType_vector const *v_type = as<IType_vector>(l_type)) {
        // convert vector selects into index operations

        string name(
            DAG_builder::binary_op_to_name(IExpression_binary::OK_ARRAY_INDEX), get_allocator());
        name += "(<0>[],int)";

        int index = -1;
        switch (member[0]) {
        case 'x': index = 0; break;
        case 'y': index = 1; break;
        case 'z': index = 2; break;
        case 'w': index = 3; break;
        default:
            break;
        }
        MDL_ASSERT(member[1] == '\0' && 0 <= index && index < v_type->get_size());

        // create an index access
        DAG_call::Call_argument call_args[2];

        call_args[0].arg        = s;
        call_args[0].param_name = "a";

        call_args[1].arg        = create_int_constant(index);
        call_args[1].param_name = "i";

        // enable optimizations for function calls
        Option_store<DAG_node_factory_impl, bool> optimizations(
            *m_node_factory, &DAG_node_factory_impl::enable_opt, true);

        return create_call(
            name.c_str(),
            operator_to_semantic(IExpression_binary::OK_ARRAY_INDEX),
            call_args, 2,
            v_type->get_element_type());
    }

    IType_struct const *s_type = as<IType_struct>(l_type);
    MDL_ASSERT(s_type != NULL && "select not on struct type");
    if (s_type == NULL)
        return NULL;

    IType const *ret_type = NULL;
    for (int i = 0, n = s_type->get_field_count(); i < n; ++i) {
        IType const *p_type;
        ISymbol const *p_sym;

        s_type->get_field(i, p_type, p_sym);
        if (strcmp(p_sym->get_name(), member) == 0) {
            // found the member
            ret_type = p_type;
            break;
        }
    }

    MDL_ASSERT(ret_type && "member does not exist");
    if (ret_type == NULL)
        return NULL;

    m_printer.print(s_type);
    m_printer.print('.');
    m_printer.print(member);
    m_printer.print('(');
    m_printer.print(s_type);
    m_printer.print(')');

    DAG_call::Call_argument arg;

    arg.arg        = s;
    arg.param_name = "s";

    string op_name(m_printer.get_line());

    // enable optimizations for function calls
    Option_store<DAG_node_factory_impl, bool> optimizations(
        *m_node_factory, &DAG_node_factory_impl::enable_opt, true);

    return create_call(
        op_name.c_str(), IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS, &arg, 1, ret_type);
}

// Creates an array constructor.
DAG_node const *Distiller_plugin_api_impl::create_array(
    IType const            *elem_type,
    DAG_node const * const values[],
    size_t                 n_values)
{
    for (size_t i = 0; i < n_values; ++i)
        m_checker.check_node(values[i]);

    if (elem_type == NULL) {
        if (n_values == 0) {
            MDL_ASSERT(!"element type must be given");
            return NULL;
        }
        elem_type = values[0]->get_type();
    }

    Memory_arena arena(m_alloc);
    VLA<DAG_call::Call_argument> args(m_alloc, n_values);

    for (size_t i = 0; i < n_values; ++i) {
        m_printer.print("value");
        m_printer.print(i);

        args[i].param_name = Arena_strdup(arena, m_printer.get_line().c_str());
        args[i].arg        = values[i];
    }

    IType const *ret_type = m_type_factory->create_array(elem_type, n_values);

    // enable optimizations for function calls
    Option_store<DAG_node_factory_impl, bool> optimizations(
        *m_node_factory, &DAG_node_factory_impl::enable_opt, true);

    return create_call(
        get_array_constructor_signature(),
        IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR,
        args.data(),
        args.size(),
        ret_type);
}

// Creates a boolean constant.
DAG_constant const *Distiller_plugin_api_impl::create_bool_constant(bool f)
{
    IValue_bool const *val_f = m_value_factory->create_bool(f);
    return create_constant(val_f);
}

// Creates an integer constant.
DAG_constant const *Distiller_plugin_api_impl::create_int_constant(int i)
{
    IValue_int const *val_i = m_value_factory->create_int(i);
    return create_constant(val_i);
}

// Creates a constant of the predefined intensity_mode enum
DAG_constant const *Distiller_plugin_api_impl::create_emission_enum_constant(int i)
{
    IType_enum *enum_type = m_type_factory->get_predefined_enum(
        IType_enum::EID_INTENSITY_MODE);
    IValue_enum const *val = m_value_factory->create_enum(enum_type, i);
    return create_constant(val);
}

// Creates a constant of the df::scatter_mode enum.
DAG_constant const *Distiller_plugin_api_impl::create_scatter_enum_constant(int i)
{
    mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
    Definition const *def = find_exported_def(df_mod.get(), "scatter_mode");

    IType_enum const *enum_type = cast<IType_enum>(m_type_factory->import(def->get_type()));
    IValue_enum const *val = m_value_factory->create_enum(enum_type, i);
    return create_constant(val);
}

// Creates a constant of the tex::wrap_mode enum.
DAG_constant const *Distiller_plugin_api_impl::create_wrap_mode_enum_constant(int i)
{
    mi::base::Handle<Module const> tex_mod(find_builtin_module("::tex"));
    Definition const *def = find_exported_def(tex_mod.get(), "wrap_mode");

    IType_enum const *enum_type = cast<IType_enum>(m_type_factory->import(def->get_type()));
    IValue_enum const *val = m_value_factory->create_enum(enum_type, i);
    return create_constant(val);
}

// Creates a floating point constant.
DAG_constant const *Distiller_plugin_api_impl::create_float_constant(float f)
{
    IValue_float const *val_f = m_value_factory->create_float(f);
    return create_constant(val_f);
}

// Creates a float3 constant.
DAG_constant const *Distiller_plugin_api_impl::create_float3_constant(float x, float y, float z)
{
    IType_float  const *float_type  = m_type_factory->create_float();
    IType_vector const *float3_type = m_type_factory->create_vector(float_type, 3);
    IValue const *values[] = {
        m_value_factory->create_float(x),
        m_value_factory->create_float(y),
        m_value_factory->create_float(z)
    };
    IValue_vector const *v =
        m_value_factory->create_vector(float3_type, values, dimension_of(values));
    return create_constant(v);
}

// Creates a RGB color constant.
DAG_constant const *Distiller_plugin_api_impl::create_color_constant(float r, float g, float b)
{
    IValue_float const *val_r = m_value_factory->create_float(r);
    IValue_float const *val_g = m_value_factory->create_float(g);
    IValue_float const *val_b = m_value_factory->create_float(b);
    IValue_rgb_color const *color  = m_value_factory->create_rgb_color(val_r, val_g, val_b);
    return create_constant(color);
}

/// Creates a RGB color constant of the global material IOR value.
DAG_constant const *Distiller_plugin_api_impl::create_global_ior() 
{ 
    return create_color_constant( m_global_ior[0], m_global_ior[1], m_global_ior[2]);
}

/// Creates a float constant of the global material IOR green value.
DAG_constant const *Distiller_plugin_api_impl::create_global_float_ior()
{ 
    return create_float_constant( m_global_ior[1]);
}


// Creates a string constant.
DAG_constant const *Distiller_plugin_api_impl::create_string_constant(char const *s)
{
    IValue_string const *val_s = m_value_factory->create_string(s);
    return create_constant(val_s);
}

// Creates an invalid bsdf.
DAG_constant const *Distiller_plugin_api_impl::create_bsdf_constant()
{
    return create_constant(
        m_value_factory->create_invalid_ref(
            m_type_factory->create_bsdf()));
}

// Creates an invalid edf.
DAG_constant const *Distiller_plugin_api_impl::create_edf_constant()
{
    return create_constant(
        m_value_factory->create_invalid_ref(
            m_type_factory->create_edf()));
}

// Creates an invalid vdf.
DAG_constant const *Distiller_plugin_api_impl::create_vdf_constant()
{
    return create_constant(
        m_value_factory->create_invalid_ref(
            m_type_factory->create_vdf()));
}

// Creates an invalid hair_bsdf.
DAG_constant const *Distiller_plugin_api_impl::create_hair_bsdf_constant()
{
    return create_constant(
        m_value_factory->create_invalid_ref(
            m_type_factory->create_hair_bsdf()));
}

// Create a bsdf_component for a mixer; can be a call or a constant.
DAG_node const *Distiller_plugin_api_impl::create_bsdf_component(
    DAG_node const *weight_arg,
    DAG_node const *bsdf_arg)
{
    if ( is<DAG_constant>(weight_arg) && is<DAG_constant>(bsdf_arg)) {
        mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
        Definition const *def = find_exported_def(df_mod.get(), "bsdf_component");

        IType_struct const *struct_type = cast<IType_struct>(
            m_type_factory->import(def->get_type()));
        IValue const * values[2] = {
            cast<DAG_constant>(weight_arg)->get_value(),
            cast<DAG_constant>(bsdf_arg)->get_value()
        };
        IValue_struct const *val = m_value_factory->create_struct(struct_type, values, 2);
        return create_constant(val);
    }
    DAG_call::Call_argument args[2];
    args[0].param_name = "weight";
    args[0].arg        = weight_arg;
    args[1].param_name = "component";
    args[1].arg        = bsdf_arg;
    return create_call(
        "::df::bsdf_component(float,bsdf)",
        IDefinition::DS_ELEM_CONSTRUCTOR,
        args, 2,
        get_bsdf_component_type());
}

// Create a edf_component for a mixer; can be a call or a constant.
DAG_node const *Distiller_plugin_api_impl::create_edf_component(
    DAG_node const *weight_arg,
    DAG_node const *edf_arg)
{
    if ( is<DAG_constant>(weight_arg) && is<DAG_constant>(edf_arg)) {
        mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
        Definition const *def = find_exported_def(df_mod.get(), "edf_component");

        IType_struct const *struct_type = cast<IType_struct>(
            m_type_factory->import(def->get_type()));
        IValue const * values[2] = {
            cast<DAG_constant>(weight_arg)->get_value(),
            cast<DAG_constant>(edf_arg)->get_value()
        };
        IValue_struct const *val = m_value_factory->create_struct(struct_type, values, 2);
        return create_constant(val);
    }
    DAG_call::Call_argument args[2];
    args[0].param_name = "weight";
    args[0].arg        = weight_arg;
    args[1].param_name = "component";
    args[1].arg        = edf_arg;
    return create_call(
        "::df::edf_component(float,edf)",
        IDefinition::DS_ELEM_CONSTRUCTOR,
        args, 2,
        get_edf_component_type());
}

// Create a vdf_component for a mixer; can be a call or a constant.
DAG_node const *Distiller_plugin_api_impl::create_vdf_component(
    DAG_node const *weight_arg,
    DAG_node const *vdf_arg)
{
    if ( is<DAG_constant>(weight_arg) && is<DAG_constant>(vdf_arg)) {
        mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
        Definition const *def = find_exported_def(df_mod.get(), "vdf_component");

        IType_struct const *struct_type = cast<IType_struct>(
            m_type_factory->import(def->get_type()));
        IValue const * values[2] = {
            cast<DAG_constant>(weight_arg)->get_value(),
            cast<DAG_constant>(vdf_arg)->get_value()
        };
        IValue_struct const *val = m_value_factory->create_struct(struct_type, values, 2);
        return create_constant(val);
    }
    DAG_call::Call_argument args[2];
    args[0].param_name = "weight";
    args[0].arg        = weight_arg;
    args[1].param_name = "component";
    args[1].arg        = vdf_arg;
    return create_call(
        "::df::vdf_component(float,vdf)",
        IDefinition::DS_ELEM_CONSTRUCTOR,
        args, 2,
        get_vdf_component_type());
}

// Create a color_bsdf_component for a mixer; can be a call or a constant.
DAG_node const *Distiller_plugin_api_impl::create_color_bsdf_component(
    DAG_node const *weight_arg,
    DAG_node const *bsdf_arg)
{
    if ( is<DAG_constant>(weight_arg) && is<DAG_constant>(bsdf_arg)) {
        mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
        Definition const *def = find_exported_def(df_mod.get(), "color_bsdf_component");

        IType_struct const *struct_type = cast<IType_struct>(
            m_type_factory->import(def->get_type()));
        IValue const * values[2] = {
            cast<DAG_constant>(weight_arg)->get_value(),
            cast<DAG_constant>(bsdf_arg)->get_value()
        };
        IValue_struct const *val = m_value_factory->create_struct(struct_type, values, 2);
        return create_constant(val);
    }
    DAG_call::Call_argument args[2];
    args[0].param_name = "weight";
    args[0].arg        = weight_arg;
    args[1].param_name = "component";
    args[1].arg        = bsdf_arg;
    return create_call(
        "::df::color_bsdf_component(color,bsdf)",
        IDefinition::DS_ELEM_CONSTRUCTOR,
        args, 2,
        get_color_bsdf_component_type());
}

// Create a color_edf_component for a mixer; can be a call or a constant.
DAG_node const *Distiller_plugin_api_impl::create_color_edf_component(
    DAG_node const *weight_arg,
    DAG_node const *edf_arg)
{
    if ( is<DAG_constant>(weight_arg) && is<DAG_constant>(edf_arg)) {
        mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
        Definition const *def = find_exported_def(df_mod.get(), "color_edf_component");

        IType_struct const *struct_type = cast<IType_struct>(
            m_type_factory->import(def->get_type()));
        IValue const * values[2] = {
            cast<DAG_constant>(weight_arg)->get_value(),
            cast<DAG_constant>(edf_arg)->get_value()
        };
        IValue_struct const *val = m_value_factory->create_struct(struct_type, values, 2);
        return create_constant(val);
    }
    DAG_call::Call_argument args[2];
    args[0].param_name = "weight";
    args[0].arg        = weight_arg;
    args[1].param_name = "component";
    args[1].arg        = edf_arg;
    return create_call(
        "::df::color_edf_component(color,edf)",
        IDefinition::DS_ELEM_CONSTRUCTOR,
        args, 2,
        get_color_edf_component_type());
}

// Create a color_vdf_component for a mixer; can be a call or a constant.
DAG_node const *Distiller_plugin_api_impl::create_color_vdf_component(
    DAG_node const *weight_arg,
    DAG_node const *vdf_arg)
{
    if ( is<DAG_constant>(weight_arg) && is<DAG_constant>(vdf_arg)) {
        mi::base::Handle<Module const> df_mod(find_builtin_module("::df"));
        Definition const *def = find_exported_def(df_mod.get(), "color_vdf_component");

        IType_struct const *struct_type = cast<IType_struct>(
            m_type_factory->import(def->get_type()));
        IValue const * values[2] = {
            cast<DAG_constant>(weight_arg)->get_value(),
            cast<DAG_constant>(vdf_arg)->get_value()
        };
        IValue_struct const *val = m_value_factory->create_struct(struct_type, values, 2);
        return create_constant(val);
    }
    DAG_call::Call_argument args[2];
    args[0].param_name = "weight";
    args[0].arg        = weight_arg;
    args[1].param_name = "component";
    args[1].arg        = vdf_arg;
    return create_call(
        "::df::color_vdf_component(color,vdf)",
        IDefinition::DS_ELEM_CONSTRUCTOR,
        args, 2,
        get_color_vdf_component_type());
}

// Create a constant node for a given type and value.
DAG_constant const *Distiller_plugin_api_impl::mk_constant(char const *const_type, char const *value)
{
    if ( 0 == strcmp( const_type, "float")) {
        return create_float_constant( float( atof(value)));
    } else if ( 0 == strcmp( const_type, "color")) {
        if ( 0 == strcmp( value, "color()")) {
            return create_color_constant( 0.0, 0.0, 0.0);
        } else if ( 0 == strcmp( value, "color(1.0)")) {
            return create_color_constant( 1.0, 1.0, 1.0);
        }
    } else if ( 0 == strcmp( const_type, "string")) {
        return create_string_constant( value);
    } else if ( 0 == strcmp( const_type, "bool")) {
        if ( 0 == strcmp( value, "false")) {
            return create_bool_constant( false);
        }
        return create_bool_constant( true);
    } else if ( 0 == strcmp( const_type, "bsdf")) {
        return create_bsdf_constant();
    } else if ( 0 == strcmp( const_type, "edf")) {
        return create_edf_constant();
    } else if ( 0 == strcmp( const_type, "vdf")) {
        return create_vdf_constant();
    } else if ( 0 == strcmp( const_type, "hair_bsdf")) {
        return create_hair_bsdf_constant();
    } else if ( 0 == strcmp( const_type, "intensity_mode")) {
        if ( 0 == strcmp( value, "intensity_radiant_exitance"))
            return create_emission_enum_constant(0);
        return create_emission_enum_constant(1);
    } else if ( 0 == strcmp( const_type, "::df::scatter_mode")) {
        if ( 0 == strcmp( value, "::df::scatter_reflect"))
            return create_scatter_enum_constant(0);
        if ( 0 == strcmp( value, "::df::scatter_transmit"))
            return create_scatter_enum_constant(1);
        return create_scatter_enum_constant(2);
    } else if ( 0 == strcmp( const_type, "material_emission")) {
        IValue const * values[3];
        values[0] = m_value_factory->create_invalid_ref( m_type_factory->create_edf());
        IValue_float const *val = m_value_factory->create_float(0.0);
        values[1] = m_value_factory->create_rgb_color(val, val, val);
        values[2] = m_value_factory->create_enum(m_type_factory->get_predefined_enum(
                                                     IType_enum::EID_INTENSITY_MODE), 0);
        return create_constant(
            m_value_factory->create_struct(
                m_type_factory->get_predefined_struct( IType_struct::SID_MATERIAL_EMISSION),
                values, 3));
    }
    MDL_ASSERT( ! "Unknown type or value to build constant");
    return 0;
}

// Create DAG_node's for possible default values of Node_types parameter.
DAG_node const *Distiller_plugin_api_impl::mk_default(char const *param_type, char const *param_default)
{
    if ( 0 == strcmp( param_type, "float3")) {
        if ( 0 == strcmp( param_default, "float3(0.0)"))
            return create_float3_constant( 0.0, 0.0, 0.0);
        if ( 0 == strcmp( param_default, "::state::normal()"))
            return create_function_call( "::state::normal", 0, 0);
        if ( 0 == strcmp( param_default, "::state::texture_tangent_u(0)")) {
            DAG_node const *args[1] = {
                create_int_constant(0)
            };
            return create_function_call( "::state::texture_tangent_u", args, dimension_of(args));
        }
    }
    return mk_constant( param_type, param_default);
}

// Returns the argument count if node is non-null and of the call kind or a compound constant,
// and 0 otherwise.
size_t Distiller_plugin_api_impl::get_compound_argument_size(DAG_node const *node)
{
    if (node == NULL)
        return 0;
    if (DAG_call const *c = as<DAG_call>(node))
        return size_t( c->get_argument_count());
    if (DAG_constant const *l = as<DAG_constant>(node)) {
        if (IValue_compound const *s = as<IValue_compound>(l->get_value())) {
            return size_t( s->get_component_count());
        }
    }
    return 0;
}

// Return the i-th argument if node is non-null and of the call kind, or a compound constant,
// and NULL otherwise.
DAG_node const *Distiller_plugin_api_impl::get_compound_argument(DAG_node const *node, size_t i)
{
    if (node == NULL)
        return NULL;

    if (DAG_call const *c = as<DAG_call>(node)) {
        if ( i < c->get_argument_count())
            return c->get_argument(i);
        return NULL;
    }
    if (DAG_constant const *l = as<DAG_constant>(node)) {
        // give access to (literal) arguments of compound literal types
        if (IValue_compound const *s = as<IValue_compound>(l->get_value())) {
            MDL_ASSERT(i < s->get_component_count());
            IValue const *arg = s->get_value(i);
            if ( i < s->get_component_count())
                return create_constant(arg);
            return NULL;
        }
    }
    return NULL;
}

// Return the i-th argument if node is non-null and of the call kind, or a compound constant,
// and NULL otherwise; remaps index for special case handling of mixers and parameter
// order of glossy BSDFs.
DAG_node const *Distiller_plugin_api_impl::get_remapped_argument(DAG_node const* node, size_t i)
{
    if (node == NULL)
        return NULL;

    if (DAG_call const *c = as<DAG_call>(node)) {
        IDefinition::Semantics sema = c->get_semantic();

        // remap parameters for mixer
        switch (sema) {
        case IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
        case IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX:
            // remap parameter access: [0]-->[0][0],  [1]-->[0][1],  [2]-->[1][0], ...
            {
                DAG_node const *components_array = c->get_argument(0);
                // can be call or constant
                return get_compound_argument(
                    get_compound_argument(components_array, i / 2), i % 2);
            }


        default:
            break;
        }

        return c->get_argument(i);
    }
    return get_compound_argument(node, i);
}

// Returns the name of the i-th parameter of node, or NULL if there is none or node is NULL.
char const *Distiller_plugin_api_impl::get_compound_parameter_name(
    DAG_node const *node,
    size_t i) const
{
    if (node == NULL)
        return NULL;
    if (DAG_call const *c = as<DAG_call>(node))
        return c->get_parameter_name(i);
    if (DAG_constant const *c = as<DAG_constant>(node)) {
        if (IValue_struct const *s = as<IValue_struct>(c->get_value())) {
             IType const *type;
             ISymbol const *name;
            s->get_type()->get_field( i, type, name);
            return name->get_name();
        }
        if (is<IValue_array>(c->get_value())) {
            if (i < 16) {
                static char const * const idx[] = {
                    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "10", "11", "12", "13", "14", "15"
                };
                return idx[i];
            }
            return "N";
        }
    }
    return NULL;
}

// Returns true if node evaluates to true
bool Distiller_plugin_api_impl::eval_if(DAG_node const *node)
{
    if (DAG_constant const *c = as<DAG_constant>(node)) {
        if (IValue_bool const *b = as<IValue_bool>(c->get_value())) {
            return b->get_value();
        }
    }
    return false;
}

// Returns true if node is not evaluating to false, i.e., it either evaluates
// to true or cannot be evaluated.
bool Distiller_plugin_api_impl::eval_maybe_if(DAG_node const *node)
{
    if (DAG_constant const *c = as<DAG_constant>(node)) {
        if (IValue_bool const *b = as<IValue_bool>(c->get_value())) {
            return b->get_value();
        }
    }
    return true;
}

void Distiller_plugin_api_impl::dump_attributes(IGenerated_code_dag::IMaterial_instance const *inst) {
#if MDL_DIST_PLUG_DEBUG
    for (auto oit : m_attribute_map) {
        std::cerr << "[>>] node: " << oit.first << "\n";
        auto &inner = oit.second;
        for (auto iit : inner) {
            if (iit.second) {
                std::cerr << "  [=] " << iit.first << " (" << ((void*) iit.first) << ") "
                          << ": attr: " << iit.second << "\n";
                std::cerr << "    node:\n";
                pprint_node(inst,iit.second, 3);
            } else {
                std::cerr << "  [=] " << iit.first << ": attr: null\n";
            }
        }
    }
#endif
}

void Distiller_plugin_api_impl::dump_attributes(IGenerated_code_dag::IMaterial_instance const *inst,
                                                DAG_node const *node) {
#if MDL_DIST_PLUG_DEBUG
    auto inner = m_attribute_map.find(node);
    if (inner != m_attribute_map.end()) {
        for (auto iit : inner->second) {
            if (iit.second) {
                std::cerr << "  [=] " << iit.first << " (" << ((void*) iit.first) << ") "
                          << ": attr: " << iit.second << "\n";
                std::cerr << "    node:\n";
                pprint_node(inst, iit.second, 3);
            } else {
                std::cerr << "  [=] " << iit.first << ": attr: null\n";
            }
        }
    }
#endif
}

void Distiller_plugin_api_impl::set_attribute(DAG_node const * node,
                                              char const *name,
                                              DAG_node const *value) {
    Attr_map::iterator inner = m_attribute_map.find(node);
    if (inner == m_attribute_map.end()) {
        Attr_map::mapped_type new_map(m_alloc);
        inner = m_attribute_map.insert({node, new_map}).first;
    }
    inner->second.insert({name, value});
}

void Distiller_plugin_api_impl::set_attribute(IGenerated_code_dag::IMaterial_instance const *i_inst,
                                              DAG_node const * node,
                                              char const *name,
                                              mi::Sint32 value) {
    Generated_code_dag::Material_instance *inst =
        const_cast<Generated_code_dag::Material_instance *>(impl_cast<Generated_code_dag::Material_instance>(i_inst));
    Store<IType_factory *>         s_type_factory(m_type_factory,   inst->get_type_factory());
    Store<IValue_factory *>        s_value_factory(m_value_factory, inst->get_value_factory());
    Store<DAG_node_factory_impl *> s_node_factory(m_node_factory,   inst->get_node_factory());

    m_checker.set_owner(inst->get_node_factory());

    DAG_node const *v = create_int_constant(value);
    set_attribute(node, name, v);
}

void Distiller_plugin_api_impl::set_attribute(IGenerated_code_dag::IMaterial_instance const *i_inst,
                                              DAG_node const * node,
                                              char const *name,
                                              mi::Float32 value) {
    Generated_code_dag::Material_instance *inst =
        const_cast<Generated_code_dag::Material_instance *>(impl_cast<Generated_code_dag::Material_instance>(i_inst));
    Store<IType_factory *>         s_type_factory(m_type_factory,   inst->get_type_factory());
    Store<IValue_factory *>        s_value_factory(m_value_factory, inst->get_value_factory());
    Store<DAG_node_factory_impl *> s_node_factory(m_node_factory,   inst->get_node_factory());

    m_checker.set_owner(inst->get_node_factory());

    DAG_node const *v = create_float_constant(value);
    set_attribute(node, name, v);
}

void Distiller_plugin_api_impl::remove_attributes(DAG_node const * node) {
    auto inner = m_attribute_map.find(node);
    if (inner != m_attribute_map.end()) {
        m_attribute_map.erase(inner);
    }
}

DAG_node const * Distiller_plugin_api_impl::get_attribute(DAG_node const * node,
                                                          char const *name) {
    Attr_map::iterator inner = m_attribute_map.find(node);
    if (inner != m_attribute_map.end()) {
        Node_attr_map::iterator val = inner->second.find(name);
        if (val != inner->second.end())
            return val->second;
    }
    return nullptr;
}

bool Distiller_plugin_api_impl::attribute_exists(DAG_node const * node,
                                                 char const *name) {
    Attr_map::iterator inner = m_attribute_map.find(node);
    if (inner != m_attribute_map.end()) {
        Node_attr_map::iterator val = inner->second.find(name);
        if (val != inner->second.end())
            return true;
    }
    return false;
}

void Distiller_plugin_api_impl::import_attributes(IGenerated_code_dag::IMaterial_instance const *inst) {
    Visited_node_map marker_map(0, Visited_node_map::hasher(), Visited_node_map::key_equal(),
                                m_alloc);
    for (auto &outer : m_attribute_map) {
        for (auto &inner : outer.second) {
            DAG_node const *copy = copy_dag(inner.second, marker_map);
            inner.second = copy;
        }
    }
}

void Distiller_plugin_api_impl::move_attributes(DAG_node const *to_node,
                                                DAG_node const *from_node) {
    if (to_node == from_node) {
        return;
    }

    // Visited_node_map marker_map(0, Visited_node_map::hasher(), Visited_node_map::key_equal(),
    //                             m_alloc);

    auto inner = m_attribute_map.find(from_node);
    if (inner != m_attribute_map.end()) {
        for (Node_attr_map::iterator it = inner->second.begin();
             it != inner->second.end();
             ++it) {
//            marker_map.clear();
            // DAG_node const *copy = copy_dag(it->second, marker_map);
            // set_attribute(to_node, it->first, copy);
            set_attribute(to_node, it->first, it->second);
        }
        remove_attributes(from_node);
    }
}

IGenerated_code_dag::IMaterial_instance *Distiller_plugin_api_impl::apply_rules(
    IGenerated_code_dag::IMaterial_instance const *i_inst,
    IRule_matcher                                 &matcher,
    IRule_matcher_event                           *event_handler,
    const mi::mdl::Distiller_options              *options,
    mi::Sint32&                                   error)
{
    Store<mi::mdl::Distiller_options const*> opt_store(m_options, options);

    Generated_code_dag::Material_instance const *inst =
        impl_cast<Generated_code_dag::Material_instance>(i_inst);

#if MDL_DIST_PLUG_DEBUG
    std::cerr << "\n\x1b[35m[+++] Applying rules for rule set " << matcher.get_rule_set_name() <<
        " (root: " << inst->get_constructor() << ")...\x1b[0m\n";
    for (size_t i = 0; i < inst->get_temporary_count(); i++) {
        DAG_node const *temp = inst->get_temporary_value(i);
        std::cerr << "\x1b[33mt" << i << "\x1b[0m = ";
        pprint_node(inst, temp, 1);
        std::cerr << "\n";
    }
    std::cerr << "\x1b[33minput\x1b[0m = ";
    pprint_node(inst, inst->get_constructor(), 1);
    std::cerr << "\n";
#endif

    m_checker.enable_temporaries(false);
    m_checker.enable_parameters(inst->get_parameter_count() > 0);

    DAG_node const *original_root = inst->get_constructor();

    // this creates a clone but removed ALL temporaries
    Generated_code_dag::Material_instance *curr = inst->clone(
        m_alloc, Generated_code_dag::Material_instance::CF_DEFAULT, /*unsafe_math_opt=*/false);

    size_t import_count = matcher.get_target_material_name_count();

    for (size_t i = 0; i < import_count; i++) {
        char const *module_name = matcher.get_target_material_name(i);
        mi::base::Handle<IModule const> owner;

        owner = m_call_resolver->get_owner_module(module_name);

        if (!owner.is_valid_interface()) {
            error = -3;
            return curr;
        }
    }

    // mark that we modify the material by distilling
    curr->set_property(Generated_code_dag::Material_instance::IP_DISTILLED, true);

    // [TMM note 1] If target material model mode is requested in the
    // distiller options, we mark the result material accordingly.
    // Note that this is not the only way the material will be marked
    // as such (see below at [TMM note 2]).
    if (options->target_material_model_mode) {
        curr->set_property(Generated_code_dag::Material_instance::IP_TARGET_MATERIAL_MODEL, true);
    }

    Store<IType_factory *>         s_type_factory(m_type_factory,   curr->get_type_factory());
    Store<IValue_factory *>        s_value_factory(m_value_factory, curr->get_value_factory());
    Store<DAG_node_factory_impl *> s_node_factory(m_node_factory,   curr->get_node_factory());
    Store<Rule_eval_strategy>      s_strategy(m_strategy, matcher.get_strategy());
    Store<IRule_matcher *>         s_matcher(m_matcher, &matcher);
    Store<IRule_matcher_event *>   s_event_handler(m_event_handler, event_handler);

    m_matcher->set_node_types(&s_node_types);

    DAG_node const *root = curr->get_constructor();

    // disable optimizations, keep CSE
    Option_store<DAG_node_factory_impl, bool> optimizations(
        *m_node_factory, &DAG_node_factory_impl::enable_opt, false);

    m_checker.set_owner(curr->get_node_factory());

    // set global_ior in options if it is a constant
    m_global_ior[0] = 1.4f;
    m_global_ior[1] = 1.4f;
    m_global_ior[2] = 1.4f;
    DAG_node const *ior = get_compound_argument(root,3);
    if (ior != NULL && is<DAG_constant>(ior) && is<IType_color>(ior->get_type())) {
        IValue_rgb_color const *rgb = as<IValue_rgb_color>(as<DAG_constant>(ior)->get_value());
        m_global_ior[0] = rgb->get_value(0)->get_value();
        m_global_ior[1] = rgb->get_value(1)->get_value();
        m_global_ior[2] = rgb->get_value(2)->get_value();
    }

    m_checker.enable_temporaries(true);
    m_checker.set_owner(NULL);
    m_checker.check_instance(inst);

    m_checker.enable_temporaries(false);
    m_checker.set_owner(curr->get_node_factory());

#if 0
    static unsigned idx = 0;
    {
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "_%04u_%s_a2", idx, m_matcher->get_rule_set_name());

        curr->dump_instance_dag(buffer);
    }
#endif

    import_attributes(curr);

    Visited_node_map attr_marker_map(0, Visited_node_map::hasher(), Visited_node_map::key_equal(), m_alloc);
    move_attributes_deep(root, original_root, attr_marker_map, 0, false);

#if 0
    std::cerr << "{{=}} root attributes on entry (" << root << "):\n";
    dump_attributes(root);
#endif

    string root_name("material", m_alloc);

    Visited_node_map replace_marker_map(0, Visited_node_map::hasher(), Visited_node_map::key_equal(), m_alloc);
    DAG_node const* new_root = replace(root, root_name, replace_marker_map);

    bool postcond_result = m_matcher->postcond(event_handler, *this, new_root, m_options);
    MDL_ASSERT( postcond_result);
    if (!postcond_result) {
       error = -3;
    } else {
        // [TMM note 2] We check the distilling result. If the call at
        // the root is not a BSDF call, the result must be a target
        // material model mode, so we have to set the corresponding
        // property on the output compiled material, or it will break
        // other phases (like hash calculation).

        DAG_call const *call = cast<DAG_call>(new_root);
        IDefinition::Semantics sema = call->get_semantic();
        if (sema == IDefinition::DS_UNKNOWN &&
            (curr->get_properties() & Generated_code_dag::Material_instance::IP_TARGET_MATERIAL_MODEL) == 0) {
            curr->set_property(Generated_code_dag::Material_instance::IP_TARGET_MATERIAL_MODEL, true);
        }

        curr->set_constructor(cast<DAG_call>(new_root));

        // rebuild the temporaries
        curr->build_temporaries();
        curr->calc_hashes();

        m_checker.enable_temporaries(true);
        m_checker.set_owner(NULL);
        m_checker.check_instance(curr);
    }
#if 0
    {
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "_%04u_%s_z2", idx++, m_matcher->get_rule_set_name());

        curr->dump_instance_dag(buffer);
    }
#endif

#if 0
    std::cerr << "{{=}} root attributes on exit (" << new_root << "):\n";
    dump_attributes(new_root);
#endif
#if MDL_DIST_PLUG_DEBUG
    std::cerr << "\x1b[35m[+++] Applying rules for rule set " << matcher.get_rule_set_name() << "... done:\x1b[0m\n";
    for (size_t i = 0; i < curr->get_temporary_count(); i++) {
        DAG_node const *temp = curr->get_temporary_value(i);
        std::cerr << "\x1b[33mt" << i << "\x1b[0m = ";
        pprint_node(curr, temp, 1);
        std::cerr << "\n";
    }
    std::cerr << "\x1b[33mresult\x1b[0m = ";
    pprint_node(curr, curr->get_constructor(), 1);
    std::cerr << "\n";
#endif
    return curr;
}

/// Skip a temporary node.
static DAG_node const *skip_temporary(DAG_node const *node)
{
    if (DAG_temporary const *t = as<DAG_temporary>(node))
        node = t->get_expr();
    return node;
}

// Returns a new material instance as a merge of two material instances based
// on a material field selection mask choosing the top-level material fields
// between the two materials.
IGenerated_code_dag::IMaterial_instance *Distiller_plugin_api_impl::merge_materials(
    IGenerated_code_dag::IMaterial_instance const *i_mat0,
    IGenerated_code_dag::IMaterial_instance const *i_mat1,
    IDistiller_plugin_api::Field_selector  field_selector)
{
    Generated_code_dag::Material_instance const *inst0 =
        impl_cast<Generated_code_dag::Material_instance>(i_mat0);
    Generated_code_dag::Material_instance const *inst1 =
        impl_cast<Generated_code_dag::Material_instance>(i_mat1);

    DAG_node const *original_root = inst0->get_constructor();

    // this creates a clone but removed ALL temporaries
    Generated_code_dag::Material_instance *curr = inst0->clone(
        m_alloc, Generated_code_dag::Material_instance::CF_DEFAULT, /*unsafe_math_opt=*/false);

    // mark that we modify the material by distilling
    curr->set_property(Generated_code_dag::Material_instance::IP_DISTILLED, true);

    m_checker.enable_temporaries(false);
    m_checker.enable_parameters(inst0->get_parameter_count() > 0);

    Store<IType_factory *>         s_type_factory(m_type_factory,   curr->get_type_factory());
    Store<IValue_factory *>        s_value_factory(m_value_factory, curr->get_value_factory());
    Store<DAG_node_factory_impl *> s_node_factory(m_node_factory,   curr->get_node_factory());

    DAG_call const *root0 = inst0->get_constructor();

    // disable optimizations, keep CSE
    Option_store<DAG_node_factory_impl, bool> optimizations(
        *m_node_factory, &DAG_node_factory_impl::enable_opt, false);

    import_attributes(curr);

    // make a deep copy, this also removes ANY temporaries
    Visited_node_map merge_marker_map(0, Visited_node_map::hasher(), Visited_node_map::key_equal(), m_alloc);

    root0 = cast<DAG_call>(copy_dag(root0, merge_marker_map));
    merge_marker_map.clear();

    m_checker.set_owner(curr->get_node_factory());

    m_checker.enable_temporaries(true);
    m_checker.set_owner(NULL);
    m_checker.check_instance(inst0);

    m_checker.enable_temporaries(false);
    m_checker.set_owner(curr->get_node_factory());

    merge_marker_map.clear();

    DAG_call const *root1 = inst1->get_constructor();

#if MDL_DIST_PLUG_DEBUG
    std::cerr << "\x1b[35m{{-^-}} ------ merge_materials:\x1b[0m ------\n";
    for (size_t i = 0; i < inst0->get_temporary_count(); i++) {
        DAG_node const* temp = inst0->get_temporary_value(i);
        std::cerr << "\x1b[33mt" << i << "\x1b[0m = ";
        pprint_node(inst0, temp, 1);
        std::cerr << "\n";
    }
    std::cerr << "\x1b[31m>>> root0:\x1b[0m\n";
    pprint_node(inst0, root0, 1);
    std::cerr << "\n";
    std::cerr << "---------------------------------------\n";
    for (size_t i = 0; i < inst1->get_temporary_count(); i++) {
        DAG_node const* temp = inst1->get_temporary_value(i);
        std::cerr << "\x1b[33mt" << i << "\x1b[0m = ";
        pprint_node(inst1, temp, 1);
        std::cerr << "\n";
    }
    std::cerr << "\x1b[31m>>> root1:\x1b[0m\n";
    pprint_node(inst1, root1, 1);
    std::cerr << "\n";
    std::cerr << "---------------------------------------\n";
#endif
    // get material.material_geometry
    DAG_call const *geom0 = as<DAG_call>(skip_temporary(root0->get_argument(5)));
    DAG_call const *geom1 = as<DAG_call>(skip_temporary(root1->get_argument(5)));

    // check for constant surface_geometry entries
    bool skip_geom_copy = false;
    if (geom0 == NULL) {
        DAG_constant const *geom0_const =
            cast<DAG_constant>(skip_temporary(root0->get_argument(5)));
        geom0 = cast<DAG_call>(replace_constant_by_call(geom0_const));
    }
    if (geom1 == NULL) {
        DAG_constant const *geom1_const =
            cast<DAG_constant>(skip_temporary(root1->get_argument(5)));
        geom1 = cast<DAG_call>(replace_constant_by_call(geom1_const));
        skip_geom_copy = true;
    }

    // get material.backface
    DAG_call const *backface0 = as<DAG_call>(skip_temporary(root0->get_argument(2)));
    DAG_call const *backface1 = as<DAG_call>(skip_temporary(root1->get_argument(2)));

    // check for constant backface entries
    bool skip_backface_copy = false;
    if (backface0 == NULL) {
        DAG_constant const *backface0_const =
            cast<DAG_constant>(skip_temporary(root0->get_argument(2)));
        backface0 = cast<DAG_call>(replace_constant_by_call(backface0_const));
    }
    if (backface1 == NULL) {
        DAG_constant const *backface1_const =
            cast<DAG_constant>(skip_temporary(root1->get_argument(2)));
        backface1 = cast<DAG_call>(replace_constant_by_call(backface1_const));
        skip_backface_copy = true;
    }

    // create new material_geometry
    DAG_call::Call_argument g_args[3];
    g_args[0].param_name = geom0->get_parameter_name(0);
    g_args[0].arg = ((field_selector & Base::FS_MATERIAL_GEOMETRY_DISPLACEMENT)
                     ? ( skip_geom_copy
                         ? geom1->get_argument(0)
                         : copy_dag(geom1->get_argument(0), merge_marker_map))
                     : geom0->get_argument(0));

    g_args[1].param_name = geom0->get_parameter_name(1);
    g_args[1].arg = ((field_selector & Base::FS_MATERIAL_GEOMETRY_CUTOUT_OPACITY)
                     ? (skip_geom_copy
                        ? geom1->get_argument(1)
                        : copy_dag(geom1->get_argument(1), merge_marker_map))
                     : geom0->get_argument(1));
    g_args[2].param_name = geom0->get_parameter_name(2);
    g_args[2].arg = ((field_selector & Base::FS_MATERIAL_GEOMETRY_NORMAL)
                     ? (skip_geom_copy
                        ? geom1->get_argument(2)
                        : copy_dag(geom1->get_argument(2), merge_marker_map))
                     : geom0->get_argument(2));

    DAG_node const *new_geom = create_call(
        geom0->get_name(),
        geom0->get_semantic(),
        g_args, dimension_of(g_args),
        geom0->get_type());

    // create new backface
    DAG_call::Call_argument b_args[2];
    b_args[0].param_name = backface0->get_parameter_name(0);
    b_args[0].arg = ((field_selector & Base::FS_MATERIAL_BACKFACE_SCATTERING)
        ? (skip_backface_copy
            ? backface1->get_argument(0)
            : copy_dag(backface1->get_argument(0), merge_marker_map))
        : backface0->get_argument(0));

    b_args[1].param_name = backface0->get_parameter_name(1);
    b_args[1].arg = ((field_selector & Base::FS_MATERIAL_BACKFACE_EMISSION)
        ? (skip_backface_copy
            ? backface1->get_argument(1)
            : copy_dag(backface1->get_argument(1), merge_marker_map))
        : backface0->get_argument(1));

    DAG_node const *new_backface = create_call(
        backface0->get_name(),
        backface0->get_semantic(),
        b_args, dimension_of(b_args),
        backface0->get_type());

    // create new material root
    DAG_call::Call_argument args[7];
    args[0].param_name = root0->get_parameter_name(0);
    args[0].arg        = root0->get_argument(0);
    args[1].param_name = root0->get_parameter_name(1);
    args[1].arg        = root0->get_argument(1);
    args[2].param_name = root0->get_parameter_name(2);
    args[2].arg        = new_backface;
    args[3].param_name = root0->get_parameter_name(3);
    args[3].arg        = root0->get_argument(3);
    args[4].param_name = root0->get_parameter_name(4);
    args[4].arg        = root0->get_argument(4);
    args[5].param_name = root0->get_parameter_name(5);
    args[5].arg        = new_geom;
    args[6].param_name = root0->get_parameter_name(6);
    args[6].arg        = root0->get_argument(6);

    DAG_node const *new_root = create_call(
        root0->get_name(),
        root0->get_semantic(),
        args, dimension_of(args),
        root0->get_type());

    Visited_node_map attr_marker_map(0, Visited_node_map::hasher(), Visited_node_map::key_equal(), m_alloc);
    move_attributes_deep(new_root, original_root, attr_marker_map, 0, true);

    curr->set_constructor(cast<DAG_call>(new_root));

    // rebuild the temporaries
    curr->build_temporaries();

    m_checker.enable_temporaries(true);
    m_checker.set_owner(NULL);
    m_checker.check_instance(curr);

    return curr;
}

// Creates a (deep) copy of a node.
DAG_node const *Distiller_plugin_api_impl::copy_dag(
    DAG_node const *node,
    Visited_node_map &marker_map)
{
restart:
    Visited_node_map::const_iterator it = marker_map.find(node);
    if (it != marker_map.end()) {
        // already processed
        return it->second;
    }

    DAG_node const *original_node = node;

    switch (node->get_kind()) {
    case DAG_node::EK_CONSTANT:
        {
            DAG_constant const *c     = cast<DAG_constant>(node);
            IValue const       *value = c->get_value();
            node = m_node_factory->create_constant(m_value_factory->import(value));
        }
        break;
    case DAG_node::EK_PARAMETER:
        {
            DAG_parameter const *p    = cast<DAG_parameter>(node);
            IType const         *type = p->get_type();
            int                 index = p->get_index();
            node = m_node_factory->create_parameter(m_type_factory->import(type), index);
        }
        break;
    case DAG_node::EK_TEMPORARY:
        {
            DAG_temporary const *t = cast<DAG_temporary>(node);
            node = t->get_expr();
            goto restart;
        }
    case DAG_node::EK_CALL:
        {
            DAG_call const         *call    = cast<DAG_call>(node);
            int                    n_params = call->get_argument_count();
            IDefinition::Semantics sema     = call->get_semantic();
            char const             *name    = call->get_name();
            IType const            *type    = m_type_factory->import(call->get_type());

            VLA<DAG_call::Call_argument> args(m_alloc, n_params);

            for (int i = 0; i < n_params; ++i) {
                args[i].param_name = call->get_parameter_name(i);
                args[i].arg        = copy_dag(call->get_argument(i), marker_map);
            }

            node = m_node_factory->create_call(
                name, sema, args.data(), args.size(), type);
        }
        break;
    }
//    move_attributes(node, original_node);
    marker_map.insert(Visited_node_map::value_type(original_node, node));
    return node;
}

// Copy the attributes from `from_node` to `to_node` for the whole graph.
void Distiller_plugin_api_impl::move_attributes_deep(
    DAG_node const *to_node, DAG_node const *from_node,
    Visited_node_map &marker_map, int level,
    bool ignore_mismatched_nodes)
{
  restart:
    MDL_ASSERT(from_node);

    Visited_node_map::const_iterator it = marker_map.find(from_node);
    if (it != marker_map.end()) {
        // already processed
        return;
    }

  restart_inner:
    MDL_ASSERT(to_node);

    switch (to_node->get_kind()) {
    case DAG_node::EK_CONSTANT:
    case DAG_node::EK_PARAMETER:
    case DAG_node::EK_CALL:
        break;

    case DAG_node::EK_TEMPORARY:
    {
        DAG_temporary const *to_t = cast<DAG_temporary>(to_node);
        to_node = to_t->get_expr();
        goto restart_inner;
    }
    }

    switch (from_node->get_kind()) {
    case DAG_node::EK_CONSTANT:
    case DAG_node::EK_PARAMETER:
        if (ignore_mismatched_nodes && (from_node->get_kind() != to_node->get_kind())) {
            marker_map.insert(Visited_node_map::value_type(from_node, from_node));
            return;
        }
        MDL_ASSERT(from_node->get_kind() == to_node->get_kind());
        break;

    case DAG_node::EK_TEMPORARY:
    {
        DAG_temporary const *from_t = cast<DAG_temporary>(from_node);
        from_node = from_t->get_expr();
        goto restart;
    }

    case DAG_node::EK_CALL:
    {
        if (ignore_mismatched_nodes && (from_node->get_kind() != to_node->get_kind())) {
            marker_map.insert(Visited_node_map::value_type(from_node, from_node));
            return;
        }

        MDL_ASSERT(from_node->get_kind() == to_node->get_kind());

        DAG_call const *from_call = cast<DAG_call>(from_node);
        DAG_call const *to_call   = cast<DAG_call>(to_node);

        int n_params_from = from_call->get_argument_count();
        int n_params_to   = to_call->get_argument_count();

        if (ignore_mismatched_nodes &&
            (strcmp(from_call->get_name(), to_call->get_name()) ||
             n_params_from != n_params_to)) {
            marker_map.insert(Visited_node_map::value_type(from_node, from_node));
            return;
        }

        MDL_ASSERT(n_params_from == n_params_to);

        for (int i = 0; i < n_params_from; ++i) {
            move_attributes_deep(to_call->get_argument(i),
                                 from_call->get_argument(i),
                                 marker_map,
                                 level + 1, ignore_mismatched_nodes);
        }
        break;
    }
    }
    move_attributes(to_node, from_node);
    marker_map.insert(Visited_node_map::value_type(from_node, from_node));
}

// Get a standard library module.
Module const *Distiller_plugin_api_impl::find_builtin_module(
    char const *mod_name)
{
    MDL const *compiler = impl_cast<MDL>(m_compiler.get());
    Module const *mod = compiler->find_builtin_module(string(mod_name, get_allocator()));
    if (mod != NULL)
        mod->retain();
    return mod;
}

// Find an exported definition of a module.
Definition *Distiller_plugin_api_impl::find_exported_def(Module const *mod, char const *name)
{
    Symbol_table const &symtab = mod->get_symbol_table();
    ISymbol const      *sym    = symtab.lookup_symbol(name);

    if (sym == NULL)
        return NULL;

    Definition_table const &def_tag = mod->get_definition_table();
    Scope const            *global = def_tag.get_global_scope();

    Definition *def = global->find_definition_in_scope(sym);
    if (def == NULL) {
        return NULL;
    }
    if (!def->has_flag(Definition::DEF_IS_EXPORTED))
        return NULL;
    return def;
}

// Compute the node selector for the matcher, either the semantic for a DAG_call
// node, or one of the Distiller_extended_node_semantics covering DAG_constant
// of type bsdf, edf or vdf respectively, or for DAG_constant's and DAG_call's of
// one of the material structs, and selectors for mix_1, mix_2, mix_3,
// clamped_mix_1, ..., as well as a special selector for local_normal.
// All other nodes return 0.
int Distiller_plugin_api_impl::get_selector( DAG_node const* node) const {
    int selector = 0;
    switch ( node->get_kind()) {
    case DAG_node::EK_CONSTANT:
    {
        DAG_constant const *c = cast<DAG_constant>(node);
        switch ( c->get_type()->get_kind()) {
        case IType::TK_BSDF:
            selector = DS_DIST_DEFAULT_BSDF;
            break;
        case IType::TK_HAIR_BSDF:
            selector = DS_DIST_DEFAULT_HAIR_BSDF;
            break;
        case IType::TK_EDF:
            selector = DS_DIST_DEFAULT_EDF;
            break;
        case IType::TK_VDF:
            selector = DS_DIST_DEFAULT_VDF;
            break;
        case IType::TK_STRUCT:
        {
            int id = as<IType_struct>(c->get_type())->get_predefined_id();
            switch ( id) {
            case IType_struct::SID_MATERIAL_EMISSION:
                selector = DS_DIST_STRUCT_MATERIAL_EMISSION;
                break;
            case IType_struct::SID_MATERIAL_SURFACE:
                selector = DS_DIST_STRUCT_MATERIAL_SURFACE;
                break;
            case IType_struct::SID_MATERIAL_VOLUME:
                selector = DS_DIST_STRUCT_MATERIAL_VOLUME;
                break;
            case IType_struct::SID_MATERIAL_GEOMETRY:
                selector = DS_DIST_STRUCT_MATERIAL_GEOMETRY;
                break;
            case IType_struct::SID_MATERIAL:
                selector = DS_DIST_STRUCT_MATERIAL;
                break;
            default:
                break;
            }
            break;
        }
        default:
            break;
        }
    }
    break;
    case DAG_node::EK_CALL:
    {
        DAG_call const *c = cast<DAG_call>(node);
        selector = c->get_semantic();
        if ( selector == IDefinition::DS_ELEM_CONSTRUCTOR) {
            if (is<IType_struct>(node->get_type())) {
                int id = cast<IType_struct>(node->get_type())->get_predefined_id();
                switch ( id) {
                case IType_struct::SID_MATERIAL_EMISSION:
                    selector = DS_DIST_STRUCT_MATERIAL_EMISSION;
                    break;
                case IType_struct::SID_MATERIAL_SURFACE:
                    selector = DS_DIST_STRUCT_MATERIAL_SURFACE;
                    break;
                case IType_struct::SID_MATERIAL_VOLUME:
                    selector = DS_DIST_STRUCT_MATERIAL_VOLUME;
                    break;
                case IType_struct::SID_MATERIAL_GEOMETRY:
                    selector = DS_DIST_STRUCT_MATERIAL_GEOMETRY;
                    break;
                case IType_struct::SID_MATERIAL:
                    selector = DS_DIST_STRUCT_MATERIAL;
                    break;
                default:
                    break;
                }
            }
        } else if ((selector == IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX)
                   || (selector == IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX)
                   || (selector == IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX)
                   || (selector == IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX)
                   || (selector == IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX)
                   || (selector == IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX))
        {
            if (selector == IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX)
                selector = DS_DIST_BSDF_MIX_1;
            else if (selector == IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX)
                selector = DS_DIST_BSDF_CLAMPED_MIX_1;
            else if (selector == IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX)
                selector = DS_DIST_BSDF_COLOR_MIX_1;
            else if (selector == IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX)
                selector = DS_DIST_BSDF_COLOR_CLAMPED_MIX_1;
            else if (selector == IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX)
                selector = DS_DIST_BSDF_UNBOUNDED_MIX_1;
            else if (selector == IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX)
                selector = DS_DIST_BSDF_COLOR_UNBOUNDED_MIX_1;
            // Invariant: the array argument cannot be a constant here
            DAG_call const* array = as<DAG_call>(c->get_argument(0));
            MDL_ASSERT( array);
            if ( array->get_argument_count() == 2)
                selector += 1;
            if ( array->get_argument_count() >= 3)
                selector += 2;
            // check for EDF and VDF
            IType_array const* array_type = as<IType_array>(array->get_type());
            MDL_ASSERT( array_type);
            IType_struct const* comp_type = as<IType_struct>(array_type->get_element_type());
            MDL_ASSERT( comp_type);
            IType const* field_type = comp_type->get_compound_type(1);
            switch ( field_type->get_kind()) {
            case mi::mdl::IType::TK_BSDF:
                break;
            case mi::mdl::IType::TK_EDF:
                selector += DS_DIST_EDF_MIX_1 - DS_DIST_BSDF_MIX_1;
                break;
            case mi::mdl::IType::TK_VDF:
                selector += DS_DIST_VDF_MIX_1 - DS_DIST_BSDF_MIX_1;
                break;
            default:
                MDL_ASSERT(!"Malformed AST with a mixer node whose array is none of the DFs.");
            }
        } else if (selector == IDefinition::DS_INTRINSIC_DF_TINT) {
            selector = DS_DIST_BSDF_TINT;
            if (c->get_argument_count() == 2) {
                switch (c->get_argument(1)->get_type()->get_kind()) {
                case mi::mdl::IType::TK_BSDF:
                    selector = DS_DIST_BSDF_TINT; break;
                case mi::mdl::IType::TK_COLOR:
                    // special case for nvidia::distilling_support::local_normal
                    selector = DS_DIST_BSDF_TINT; break;
                case mi::mdl::IType::TK_EDF:
                    selector = DS_DIST_EDF_TINT; break;
                case mi::mdl::IType::TK_VDF:
                    selector = DS_DIST_VDF_TINT; break;
                case mi::mdl::IType::TK_HAIR_BSDF:
                    selector = DS_DIST_HAIR_BSDF_TINT; break;
                default:
                    MDL_ASSERT(!"Unsupported tint modifier");
                }
            } else {
                MDL_ASSERT(c->get_argument_count() == 3 && "Unsupported tint overload");
                selector = DS_DIST_BSDF_TINT2;
            }
        } else if (selector == IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR) {
            selector = DS_DIST_BSDF_DIRECTIONAL_FACTOR;
            if (c->get_argument_count() == 4) {
                const IType::Kind k = c->get_argument(3)->get_type()->get_kind();
                switch (k) {
                case mi::mdl::IType::TK_BSDF:
                    selector = DS_DIST_BSDF_DIRECTIONAL_FACTOR; 
                    break;

                case mi::mdl::IType::TK_EDF:
                    selector = DS_DIST_EDF_DIRECTIONAL_FACTOR; 
                    break;

                case mi::mdl::IType::TK_COLOR:
                    // FIXME: This happens when local normals are
                    // calculated, where BSDFs can be replaced by
                    // color values.

                    selector = DS_DIST_BSDF_DIRECTIONAL_FACTOR; 
                    break;

                default:
                    MDL_ASSERT(!"Unsupported directional_factor modifier");
                }
            } else {
                MDL_ASSERT(c->get_argument_count() == 4 && "Unsupported directional_factor overload");
            }
        } else if (selector == (IDefinition::DS_OP_BASE + IExpression::OK_TERNARY)) {
            DAG_call const *c1 = as<DAG_call>(c->get_argument(1));
            if ( c1 && c1->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR
                 && is<IType_struct>(c1->get_type())
                 && cast<IType_struct>(c1->get_type())->get_predefined_id()
                     == IType_struct::SID_MATERIAL)
            {
                selector = DS_DIST_MATERIAL_CONDITIONAL_OPERATOR;
            } else if ( c->get_argument(1)->get_type()->get_kind() == mi::mdl::IType::TK_BSDF) {
                selector = DS_DIST_BSDF_CONDITIONAL_OPERATOR;
            } else if ( c->get_argument(1)->get_type()->get_kind() == mi::mdl::IType::TK_EDF) {
                selector = DS_DIST_EDF_CONDITIONAL_OPERATOR;
            } else if ( c->get_argument(1)->get_type()->get_kind() == mi::mdl::IType::TK_VDF) {
                selector = DS_DIST_VDF_CONDITIONAL_OPERATOR;
            }
        } else if (selector == IDefinition::DS_UNKNOWN) {
            if ( strcmp ( c->get_name(),
                          "::nvidia::distilling_support::local_normal(float,float3)") == 0)
                selector = DS_DIST_LOCAL_NORMAL;
        }
    }
    break;
    case DAG_node::EK_TEMPORARY:
    case DAG_node::EK_PARAMETER:
        break;
    }
    return selector;
}

// Convert a enum typed value to int.
DAG_node const *Distiller_plugin_api_impl::convert_enum_to_int(DAG_node const *n)
{
    IType_enum const *e_tp = as<IType_enum>(n->get_type());
    string name(e_tp->get_symbol()->get_name(), get_allocator());

    string prefix(get_allocator());

    size_t pos = name.rfind("::");
    if (pos != string::npos) {
        prefix = name.substr(0, pos + 2);
    }

    // the name of a conversion operator is <prefix>int(<enum_type>)
    m_printer.print(prefix.c_str());
    m_printer.print("int(");
    m_printer.print(e_tp);
    m_printer.print(')');

    name = m_printer.get_line();

    DAG_call::Call_argument args[1];

    args[0].arg        = n;
    args[0].param_name = "x";

    // enable optimizations for function calls
    Option_store<DAG_node_factory_impl, bool> optimizations(
        *m_node_factory, &DAG_node_factory_impl::enable_opt, true);

    return create_call(
        name.c_str(),
        IDefinition::DS_CONV_OPERATOR,
        args,
        1,
        m_type_factory->create_int());
}

// Convert a material emission value into a constructor as a call.
DAG_node const *Distiller_plugin_api_impl::conv_material_emission_value(
    IValue_struct const *emission_value)
{
    DAG_call::Call_argument e_args[3];
    e_args[0].param_name = "emission";
    e_args[0].arg        = create_constant(emission_value->get_value(0));
    e_args[1].param_name = "intensity";
    e_args[1].arg        = create_constant(emission_value->get_value(1));
    e_args[2].param_name = "mode";
    e_args[2].arg        = create_constant(emission_value->get_value(2));
    return create_call(
        "material_emission(edf,color,intensity_mode)",
        IDefinition::DS_ELEM_CONSTRUCTOR,
        e_args, 3,
        get_type_factory()->get_predefined_struct(IType_struct::SID_MATERIAL_EMISSION));
}

// Convert a material surface value into a constructor as a call.
DAG_node const *Distiller_plugin_api_impl::conv_material_surface_value(
    IValue_struct const *surface_value)
{
    DAG_call::Call_argument args[2];
    args[0].param_name = "scattering";
    args[0].arg        = create_constant(surface_value->get_value(0));
    args[1].param_name = "emission";
    args[1].arg        = conv_material_emission_value(
        cast<IValue_struct>(surface_value->get_value(1)));
    return create_call(
        "material_surface(bsdf,material_emission)",
        IDefinition::DS_ELEM_CONSTRUCTOR,
        args, 2,
        get_type_factory()->get_predefined_struct(IType_struct::SID_MATERIAL_SURFACE));
}

// Convert a material volume value into a constructor as a call.
DAG_node const *Distiller_plugin_api_impl::conv_material_volume_value(
    IValue_struct const *volume_value)
{
    DAG_call::Call_argument args[4];
    args[0].param_name = "scattering";
    args[0].arg        = create_constant(volume_value->get_value(0));
    args[1].param_name = "absorption_coefficient";
    args[1].arg        = create_constant(volume_value->get_value(1));
    args[2].param_name = "scattering_coefficient";
    args[2].arg        = create_constant(volume_value->get_value(2));
    args[3].param_name = "emission_intensity";
    args[3].arg        = create_constant(volume_value->get_value(3));
    return create_call(
        "material_volume(vdf,color,color,color)",
        IDefinition::DS_ELEM_CONSTRUCTOR,
        args, 4,
        get_type_factory()->get_predefined_struct(IType_struct::SID_MATERIAL_VOLUME));
}

// Convert a material geometry value into a constructor as a call.
DAG_node const *Distiller_plugin_api_impl::conv_material_geometry_value(
    IValue_struct const *geom_value)
{
    DAG_call::Call_argument args[3];
    args[0].param_name = "displacement";
    args[0].arg        = create_constant(geom_value->get_value(0));
    args[1].param_name = "cutout_opacity";
    args[1].arg        = create_constant(geom_value->get_value(1));
    args[2].param_name = "normal";
    args[2].arg        = create_constant(geom_value->get_value(2));
    return create_call(
        "material_geometry(float3,float,float3)",
        IDefinition::DS_ELEM_CONSTRUCTOR,
        args, 3,
        get_type_factory()->get_predefined_struct(IType_struct::SID_MATERIAL_GEOMETRY));
}

// Convert a material value into a constructor as a call.
DAG_node const *Distiller_plugin_api_impl::conv_material_value(
    IValue_struct const *mat_value)
{
    DAG_call::Call_argument args[7];
    args[0].param_name = "thin_walled";
    args[0].arg        = create_constant(mat_value->get_value(0));
    args[1].param_name = "surface";
    args[1].arg        = conv_material_surface_value(cast<IValue_struct>(mat_value->get_value(1)));
    args[2].param_name = "backface";
    args[2].arg        = conv_material_surface_value(cast<IValue_struct>(mat_value->get_value(2)));
    args[3].param_name = "ior";
    args[3].arg        = create_constant(mat_value->get_value(3));
    args[4].param_name = "volume";
    args[4].arg        = conv_material_volume_value(cast<IValue_struct>(mat_value->get_value(4)));
    args[5].param_name = "geometry";
    args[5].arg        = conv_material_geometry_value(cast<IValue_struct>(mat_value->get_value(5)));
    args[6].param_name = "hair";
    args[6].arg        = create_constant(mat_value->get_value(6));
    return create_call(
        "material(bool,material_surface,material_surface,color,material_volume,material_geometry,"
        "hair_bsdf)",
        IDefinition::DS_ELEM_CONSTRUCTOR,
        args, 7,
        get_type_factory()->get_predefined_struct(IType_struct::SID_MATERIAL));
}

// Replace standard material structures with a DAG_call if they happen to be constants.
DAG_node const *Distiller_plugin_api_impl::replace_constant_by_call(DAG_node const *node)
{
    if (DAG_constant const *c = as<DAG_constant>(node)) {
        if (IValue_struct const *s = as<IValue_struct>(c->get_value())) {
            IType_struct::Predefined_id id = s->get_type()->get_predefined_id();
            switch (id) {
            case IType_struct::SID_MATERIAL_SURFACE:
                return conv_material_surface_value(s);
            case IType_struct::SID_MATERIAL_EMISSION:
                return conv_material_emission_value(s);
            case IType_struct::SID_MATERIAL_VOLUME:
                return conv_material_volume_value(s);
            case IType_struct::SID_MATERIAL_GEOMETRY:
                return conv_material_geometry_value(s);
            case IType_struct::SID_MATERIAL:
                return conv_material_value(s);
            case IType_struct::SID_USER:
                if (0 == strcmp("::df::bsdf_component", s->get_type()->get_symbol()->get_name())) {
                    DAG_call::Call_argument args[2];
                    args[0].param_name = "weight";
                    args[0].arg        = create_constant(s->get_value(0));
                    args[1].param_name = "component";
                    args[1].arg        = create_constant(s->get_value(1));
                    return create_call(
                        "::df::bsdf_component(float,bsdf)",
                        IDefinition::DS_ELEM_CONSTRUCTOR,
                        args, 2,
                        get_bsdf_component_type());
                }
                break;
            }
        }
    } else if (DAG_call const *call = as<DAG_call>(node)) {
        // check if we find a constant one level deeper in a mixer BSDF, in which
        // case it is equivalent to a black BSDF, e.g., bsdf().
        int selector = call->get_semantic();
        if ((selector == IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX)
            || (selector == IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX)
            || (selector == IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX)
            || (selector == IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX)
            || (selector == IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX)
            || (selector == IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX))
        {
            if (DAG_constant const *array = as<DAG_constant>(call->get_argument(0))) {
                // determine BSDF, EDF, or VDF components array and create apropos literal
                IValue_array const *array_val = cast<IValue_array>(array->get_value());
                IType_array const *array_type = array_val->get_type();
                IType_struct const *comp_type = cast<IType_struct>(array_type->get_element_type());
                IType const *field_type = comp_type->get_compound_type(1);
                switch (field_type->get_kind()) {
                case mi::mdl::IType::TK_BSDF:
                    node = create_bsdf_constant();
                    break;
                case mi::mdl::IType::TK_EDF:
                    node = create_edf_constant();
                    break;
                case mi::mdl::IType::TK_VDF:
                    node = create_vdf_constant();
                    break;
                default:
                    MDL_ASSERT(!"Malformed AST with a mixer node whose array is none of the DFs.");
                }
            }
        }
    }
    return node;
}

// Do replacement using a strategy.
DAG_node const *Distiller_plugin_api_impl::replace(
    DAG_node const *node,
    string const   &path,
    Visited_node_map &marker_map)
{
    Visited_node_map::const_iterator it = marker_map.find(node);
    if (it != marker_map.end()) {
        // already processed
        return it->second;
    }
    MDL_ASSERT(!is<DAG_temporary>(node) && "temporaries should be skipped at that point");
    DAG_node const *original_node = node;

    DAG_node const *old_node = node;
    node = replace_constant_by_call(node);
    move_attributes(node, old_node);

    if ( is<DAG_call>(node) && m_strategy == RULE_EVAL_BOTTOM_UP) {
        DAG_call const *c = cast<DAG_call>(node);
        int n = c->get_argument_count();

        VLA<DAG_call::Call_argument> n_args(m_alloc, n);

        for (int i = 0; i < n; ++i) {
            DAG_node const *arg = c->get_argument(i);
            string path2( m_alloc);
            if (m_options->trace)
                path2 = path + "." + c->get_parameter_name(i);

            n_args[i].arg        = replace(arg, path2, marker_map);
            n_args[i].param_name = c->get_parameter_name(i);
        }

        DAG_node const *old_node = node;
        node = create_call(
            c->get_name(),
            c->get_semantic(),
            n_args.data(), n,
            c->get_type());
        move_attributes(node, old_node);

        // mixer nodes might need a renormalization here
        if ( m_normalize_mixers &&
             ( (c->get_semantic() == IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX)
               || (c->get_semantic() == IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX)
               || (c->get_semantic() == IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX)
               || (c->get_semantic() == IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX)
               || (c->get_semantic() == IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX)
               || (c->get_semantic() == IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX)))
        {
            Rule_result_code dummy_result_code = RULE_RECURSE;
            Normalize_mixers_rules normalize_mixers;

            DAG_node const *old_node = node;
            node = normalize_mixers.matcher(
                m_event_handler, *this, node, m_options, dummy_result_code);
            move_attributes(node, old_node);
        }
    }

    Rule_result_code result_code = RULE_RECURSE;
    do {
        //if ( m_options->trace)
        //    std::cerr << "Check path '" << path.c_str() << "':" << std::endl;
        if ( m_event_handler != NULL)
            m_event_handler->path_check_event( m_matcher->get_rule_set_name(), path.c_str());

        result_code = RULE_RECURSE;

        // node can be of kind: EK_CONSTANT:, EK_CALL, EK_PARAMETER

        // After this call, the returned node will have the relevant
        // attributes. Either new ones, or if there was no match, the
        // original ones.
        node = m_matcher->matcher(
            m_event_handler, *this, node, m_options, result_code);
    } while (result_code == RULE_REPEAT_RULES);

    if (m_strategy == RULE_EVAL_TOP_DOWN && is<DAG_call>(node)) {
        // node is still a call (might otherwise be reduced to a constant)
        DAG_call const *n_c = cast<DAG_call>(node);

        if (result_code != RULE_SKIP_RECURSION) {
            int n = n_c->get_argument_count();

            VLA<DAG_call::Call_argument> n_args(m_alloc, n);

            for (int i = 0; i < n; ++i) {
                DAG_node const *arg = n_c->get_argument(i);
                string path2( m_alloc);
                if (m_options->trace)
                    path2 = path + "." + n_c->get_parameter_name(i);

                n_args[i].arg        = replace(arg, path2, marker_map);
                n_args[i].param_name = n_c->get_parameter_name(i);
            }

            DAG_node const *old_node = node;
            node = create_call(
                n_c->get_name(),
                n_c->get_semantic(),
                n_args.data(), n,
                n_c->get_type());
            move_attributes(node, old_node);
        }

        // mixer nodes might need a renormalization here
        if (m_normalize_mixers) {
            IDefinition::Semantics sema = n_c->get_semantic();

            if ((sema == IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX)
                || (sema == IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX)
                || (sema == IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX)
                || (sema == IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX)
                || (sema == IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX)
                || (sema == IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX))
            {
                Rule_result_code dummy_result_code = RULE_RECURSE;
                Normalize_mixers_rules normalize_mixers;

                DAG_node const *old_node = node;
                node = normalize_mixers.matcher(
                    m_event_handler, *this, node, m_options, dummy_result_code);
                move_attributes(node, old_node);
            }
        }
    }

    marker_map.insert(Visited_node_map::value_type(original_node, node));
    return node;
}

// Checks recursively for all call nodes if the property test_fct returns true.
// The recursive body for all_nodes().
bool Distiller_plugin_api_impl::all_nodes_rec(
    IRule_matcher::Checker_function test_fct,
    DAG_node const                 *node,
    char const                     *path,
    Visited_node_map &marker_map)
{
    MDL_ASSERT(!is<DAG_temporary>(node) && "temporaries should be skipped at that point");

    Visited_node_map::const_iterator it = marker_map.find(node);
    if (it != marker_map.end()) {
        // already processed
        return true;
    }

    bool result = test_fct(*this, node);

    if (!result && m_event_handler != NULL)
        m_event_handler->postcondition_failed_path(path);

    // recurse only in DAG_call nodes here
    if ( DAG_call const * c = as<DAG_call>(node)) {

        for (int i = 0, n = c->get_argument_count(); result && i < n; ++i) {
            DAG_node const *arg = c->get_argument(i);

            string path2( m_alloc);
            if (m_options->trace)
                path2 = path2 + path + "." + c->get_parameter_name(i);

            result = result && all_nodes_rec( test_fct, arg, path2.c_str(), marker_map);
        }
    }
    marker_map.insert(Visited_node_map::value_type(node, node));

    return result;
}

// Checks recursively for all call nodes if the property test_fct returns true.
bool Distiller_plugin_api_impl::all_nodes(
    IRule_matcher::Checker_function test_fct,
    DAG_node const *node)
{
    Visited_node_map all_marker_map(0, Visited_node_map::hasher(), Visited_node_map::key_equal(), m_alloc);
    return all_nodes_rec( test_fct, node, "material", all_marker_map);
}

/// Set the normalization of mixer node flag and return its previous value.
bool Distiller_plugin_api_impl::set_normalize_mixers( bool new_value)
{
    bool tmp = m_normalize_mixers;
    m_normalize_mixers = new_value;
    return tmp;
}

/// Normalize mixer nodes and set respective flag to keep them normalized
IGenerated_code_dag::IMaterial_instance *Distiller_plugin_api_impl::normalize_mixers(
    IGenerated_code_dag::IMaterial_instance const *inst,
    IRule_matcher_event                           *event_handler,
    const mi::mdl::Distiller_options              *options,
    mi::Sint32                                    &error)
{
    set_normalize_mixers(true);
    Normalize_mixers_rules normalize_mixers;
    return apply_rules( inst, normalize_mixers, event_handler, options, error);
}

} // mdl
} // mi
