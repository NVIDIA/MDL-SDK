/******************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/mdl/mdl_values.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_modules.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"

#include "generator_dag_ir_checker.h"

namespace mi {
namespace mdl {

#if !defined(NDEBUG) || defined(DEBUG)

// Constructor.
DAG_ir_checker::DAG_ir_checker(
    IAllocator                  *alloc,
    ICall_name_resolver         *call_resolver)
: m_alloc(alloc)
, m_temporaries(alloc)
, m_call_resolver(call_resolver)
, m_node_fact(NULL)
, m_tf(NULL)
, m_vf(NULL)
, m_errors(0)
, m_allow_temporary(false)
, m_allow_parameters(false)
, m_collect_temporary(false)
{
}

// Check the given instance.
bool DAG_ir_checker::check_instance(Generated_code_dag::Material_instance const *inst)
{
    Store<DAG_node_factory_impl const *> node_fact(m_node_fact, &inst->get_node_factory());
    Store<Type_factory const *>          type_fact(m_tf,        &inst->get_type_factory());
    Store<Value_factory const *>         value_fact(m_vf,       &inst->get_value_factory());

    DAG_ir_walker walker(m_alloc, /*as_tree=*/false);

    m_errors = 0;

    Store<bool>           collect_temporary(m_collect_temporary, true);
    Clear_scope<Node_vec> clear_tmps(m_temporaries);

    walker.walk_instance(const_cast<Generated_code_dag::Material_instance *>(inst), this);

    // check default arguments
    for (size_t i = 0, n = inst->get_parameter_count(); i < n; ++i) {
        IValue const *v = inst->get_parameter_default(i);

        if (!m_vf->is_owner(v)) {
            error(NULL, EC_VALUE_NOT_OWNED);
        }
    }

    if (m_collect_temporary) {
        size_t n_tmps = inst->get_temporary_count();

        for (size_t i = 0, n = m_temporaries.size(); i < n; ++i) {
            DAG_node const *expr = m_temporaries[i];

            if (i > n_tmps) {
                error(NULL, EC_TEMP_INDEX_TOO_HIGH);
                break;
            }
            if (expr != inst->get_temporary_value(i)) {
                error(NULL, EC_WRONG_TEMP);
            }
        }
    }

    return m_errors == 0;
}

// Check a DAG node.
size_t DAG_ir_checker::check_node(DAG_node const *node)
{
    if (node == NULL) {
        Store<size_t> stored_errors(m_errors, 0);

        error(node, EC_NULL_DAG);
        return m_errors;
    }

    switch (node->get_kind()) {
    case DAG_node::EK_CONSTANT:
        return check_const(cast<DAG_constant>(node));
    case DAG_node::EK_TEMPORARY:
        return check_temp(cast<DAG_temporary>(node));
    case DAG_node::EK_CALL:
        return check_call(cast<DAG_call>(node));
    case DAG_node::EK_PARAMETER:
        return check_parameter(cast<DAG_parameter>(node));
    }
    MDL_ASSERT(!"unsupported node kind");
    return 0;
}

// Check a constant.
size_t DAG_ir_checker::check_const(DAG_constant const *cnst)
{
    Store<size_t> stored_errors(m_errors, 0);

    if (cnst == NULL) {
        error(cnst, EC_NULL_DAG);
        return m_errors;
    }

    IValue const *v = cnst->get_value();
    if (v == NULL)
        error(cnst, EC_NULL_VALUE);
    else if (!m_vf->is_owner(v))
        error(cnst, EC_VALUE_NOT_OWNED);
    if (!m_node_fact->is_owner(cnst))
        error(cnst, EC_DAG_NOT_OWNED);
    return m_errors;
}

// Check a call.
size_t DAG_ir_checker::check_call(DAG_call const *call)
{
    Store<size_t> stored_errors(m_errors, 0);

    if (call == NULL) {
        error(call, EC_NULL_DAG);
        return m_errors;
    }

    IType const *type = call->get_type();
    if (type == NULL)
        error(call, EC_NULL_TYPE);
    else if (!m_tf->is_owner(type))
        error(call, EC_TYPE_NOT_OWNED);
    if (!m_node_fact->is_owner(call))
        error(call, EC_DAG_NOT_OWNED);

    char const *signature = call->get_name();

    mi::base::Handle<IModule const> owner(m_call_resolver->get_owner_module(signature));

    if (!owner.is_valid_interface()) {
        error(call, EC_UNRESOLVED_MODULE_NAME);
        return m_errors;
    }

    IDefinition::Semantics sema = call->get_semantic();

    // handle DAG-intrinsics
    IType_function const *ftype = NULL;

    int n_params        = 0;
    bool has_def        = true;
    bool def_is_uniform = false;

    switch (sema) {
    case IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
        has_def        = false;
        def_is_uniform = true;
        n_params       = 1;
//      names          = { "s" };
        break;
    case IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR:
        has_def        = false;
        def_is_uniform = true;
        n_params       = call->get_argument_count();
        break;
    case IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
        has_def        = false;
        def_is_uniform = true;
        n_params       = 1;
//      names          = { "a" };
        break;
    case IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID:
        has_def        = false;
        def_is_uniform = true;
        n_params       = 2;
//      names          = { "object_id", "expr" };
        break;
    case IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS:
        has_def        = false;
        def_is_uniform = true;
        n_params       = 3;
//      names          = { "world_to_object", "object_to_world", "expr" };
        break;
    default:
        if (sema == operator_to_semantic(IExpression::OK_TERNARY)) {
            // ternary operator has no def
            has_def        = false;
            def_is_uniform = true;
            n_params       = 3;
//          names          = { "cond", "true_exp", "false_exp" };
        } else if (sema == operator_to_semantic(IExpression::OK_ARRAY_INDEX)) {
            // array index operator has no def
            has_def = false;
            def_is_uniform = true;
            n_params = 2;
//          names          = { "a", "i" };
        } else {
            MDL_ASSERT(!is_DAG_semantics(sema) && "DAG semantic not handled");
        }
        break;
    }

    Module const      *mod = impl_cast<Module>(owner.get());
    IDefinition const *def = mod->find_signature(signature, /*only_exported=*/false);

    if (has_def && def == NULL) {
        error(call, EC_UNDECLARED_NAME);
        return m_errors;
    }

    if (has_def) {
        if (def->get_semantics() != sema) {
            error(call, EC_SEMANTIC_MISMATCH);
        }

        ftype    = cast<IType_function>(def->get_type());
        n_params = ftype->get_parameter_count();
    }

    // check parameters
    bool all_auto_are_uniform = true;

    if (n_params != call->get_argument_count()) {
        error(call, EC_PARAM_COUNT);
    } else {
        for (int i = 0; i < n_params; ++i) {
            IType const   *p_type = NULL;
            ISymbol const *p_sym  = NULL;

            if (has_def) {
                ftype->get_parameter(i, p_type, p_sym);

                if (strcmp(p_sym->get_name(), call->get_parameter_name(i)) != 0) {
                    error(call, EC_PARAM_NAME_MISMATCH);
                }
            }

            DAG_node const *arg = call->get_argument(i);
            if (arg == NULL) {
                error(call, EC_NULL_ARG);
            } else {
                IType const *arg_tp = arg->get_type();

                if (!has_def) {
                    // so far parameter types are not checked
                    p_type = arg_tp->skip_type_alias();
                }

                bool arg_is_uniform =
                    arg->get_kind() == DAG_node::EK_CONSTANT ||
                    (arg_tp->get_type_modifiers() & IType::MK_UNIFORM) != 0;

                IType_array const *p_a_tp = as<IType_array>(p_type);
                IType_array const *a_a_tp = as<IType_array>(arg_tp);

                IType const *c_p_tp = p_type;
                IType const *c_a_tp = arg_tp;

                if (p_a_tp != NULL && a_a_tp != NULL &&
                    !p_a_tp->is_immediate_sized() && a_a_tp->is_immediate_sized())
                {
                    // assigning an immediate size to a deferred size is ok
                    c_p_tp = p_a_tp->get_element_type();
                    c_a_tp = a_a_tp->get_element_type();
                }

                // TODO: type binding
                if (!equal_types(c_p_tp->skip_type_alias(), c_p_tp->skip_type_alias())) {
                    error(call, EC_ARG_TYPE_MISMATCH);
                } else {
                    IType::Modifiers p_modifiers = p_type->get_type_modifiers();
                    if (p_modifiers & IType::MK_UNIFORM) {
                        // argument must be uniform
                        if (!arg_is_uniform) {
                            error(call, EC_ARG_NON_UNIFORM);
                        }
                    } else if ((p_modifiers & IType::MK_VARYING) == 0) {
                        // auto typed parameter
                        all_auto_are_uniform &= arg_is_uniform;
                    }
                }
            }
        }
    }

    // check return type
    IType const *call_type = call->get_type();
    IType const *ret_type  = has_def ? ftype->get_return_type() : call_type->skip_type_alias();

    if (!equal_types(ret_type->skip_type_alias(), call_type->skip_type_alias())) {
        error(call, EC_WRONG_RET_TYPE);
    } else {
        if (call_type->get_type_modifiers() & IType::MK_UNIFORM) {
            bool is_uniforn = (ret_type->get_type_modifiers() & IType::MK_UNIFORM) != 0;

            if (!is_uniforn && (def_is_uniform || def->get_property(IDefinition::DP_IS_UNIFORM))) {
                is_uniforn = all_auto_are_uniform;
            }

            if (!is_uniforn) {
                error(call, EC_RET_TYPE_NON_UNIFORM);
            }
        }
    }
    return m_errors;
}

// Check a parameter.
size_t DAG_ir_checker::check_parameter(DAG_parameter const *param)
{
    Store<size_t> stored_errors(m_errors, 0);

    if (param == NULL) {
        error(param, EC_NULL_DAG);
        return m_errors;
    }

    IType const *type = param->get_type();
    if (type == NULL)
        error(param, EC_NULL_TYPE);
    else if (!m_tf->is_owner(type))
        error(param, EC_TYPE_NOT_OWNED);
    if (!m_node_fact->is_owner(param))
        error(param, EC_DAG_NOT_OWNED);

    if (m_allow_parameters) {

    } else {
        error(param, EC_PARAMETER_NOT_ALLOWED);
    }
    return m_errors;
}

// Check a temporary.
size_t DAG_ir_checker::check_temp(DAG_temporary const *tmp)
{
    Store<size_t> stored_errors(m_errors, 0);

    if (tmp == NULL) {
        error(tmp, EC_NULL_DAG);
        return m_errors;
    }

    IType const *type = tmp->get_type();
    if (type == NULL)
        error(tmp, EC_NULL_TYPE);
    else if (!m_tf->is_owner(type))
        error(tmp, EC_TYPE_NOT_OWNED);
    if (!m_node_fact->is_owner(tmp))
        error(tmp, EC_DAG_NOT_OWNED);

    if (m_allow_temporary) {
        DAG_node const *expr = tmp->get_expr();

        if (expr == NULL) {
            error(tmp, EC_NULL_TEMP);
        }

        if (m_collect_temporary) {
            size_t idx = tmp->get_index();

            if (idx >= m_temporaries.size()) {
                size_t rsize = (idx + 16) & size_t(~0xF);
                m_temporaries.reserve(rsize);

                m_temporaries.resize(idx + 1, (DAG_node const *)NULL);
            }

            if (m_temporaries[idx] != NULL && m_temporaries[idx] != expr) {
                error(tmp, EC_TEMP_INDEX_USED_TWICE);
            }
            m_temporaries[idx] = expr;
        }
    } else {
        error(tmp, EC_TEMPORARY_NOT_ALLOWED);
    }
    return m_errors;
}

// Report an error detected on given node.
void DAG_ir_checker::error(DAG_node const *node, Error_code code)
{
    switch (code) {
    case EC_OK:
        break;
    case EC_PARAMETER_NOT_ALLOWED:
        MDL_ASSERT(!"Parameter now allowed here");
        break;
    case EC_TEMPORARY_NOT_ALLOWED:
        MDL_ASSERT(!"Temporary not allowed here");
        break;
    case EC_UNRESOLVED_MODULE_NAME:
        MDL_ASSERT(!"module name of a call could not be resolved");
        break;
    case EC_UNDECLARED_NAME:
        MDL_ASSERT(!"undeclared name");
        break;
    case EC_PARAM_COUNT:
        MDL_ASSERT(!"Mismatch in the number of parameters");
        break;
    case EC_PARAM_NAME_MISMATCH:
        MDL_ASSERT(!"Parameter name does not match");
        break;
    case EC_NULL_ARG:
        MDL_ASSERT(!"Call argument is NULL");
    case EC_ARG_TYPE_MISMATCH:
        MDL_ASSERT(!"Call argument type mismatch");
        break;
    case EC_ARG_NON_UNIFORM:
//        MDL_ASSERT(!"Call argument must be uniform");
        break;
    case EC_WRONG_RET_TYPE:
        MDL_ASSERT(!"Call return type mismatch");
        break;
    case EC_RET_TYPE_NON_UNIFORM:
        MDL_ASSERT(!"Call return is not uniform");
        break;
    case EC_NULL_TEMP:
        MDL_ASSERT(!"NULL temporary");
        break;
    case EC_SEMANTIC_MISMATCH:
        MDL_ASSERT(!"Wrong call semantic");
        break;
    case EC_NULL_DAG:
        MDL_ASSERT(!"NULL DAG node");
        break;
    case EC_NULL_TYPE:
        MDL_ASSERT(!"NULL type");
        break;
    case EC_NULL_VALUE:
        MDL_ASSERT(!"NULL value");
        break;
    case EC_VALUE_NOT_OWNED:
        MDL_ASSERT(!"Value not owned");
        break;
    case EC_TYPE_NOT_OWNED:
        MDL_ASSERT(!"Type not owned");
        break;
    case EC_DAG_NOT_OWNED:
        MDL_ASSERT(!"DAG node not owned");
        break;
    case EC_TEMP_INDEX_USED_TWICE:
        MDL_ASSERT(!"Temporary index used more then once");
        break;
    case EC_WRONG_TEMP:
        MDL_ASSERT(!"Computed temporary is wrong");
        break;
    case EC_TEMP_INDEX_TOO_HIGH:
        MDL_ASSERT(!"Temporary index to high");
        break;
    }
    if (code != EC_OK)
        ++m_errors;
}

// Post-visit a Constant.
void DAG_ir_checker::visit(DAG_constant *cnst)
{
    m_errors += check_const(cnst);
}

// Post-visit a Temporary.
void DAG_ir_checker::visit(DAG_temporary *tmp)
{
    m_errors += check_temp(tmp);
}

// Post-visit a call.
void DAG_ir_checker::visit(DAG_call *call)
{
    m_errors += check_call(call);
}

// Post-visit a Parameter.
void DAG_ir_checker::visit(DAG_parameter *param)
{
    m_errors += check_parameter(param);
}

// Post-visit a temporary initializer.
void DAG_ir_checker::visit(int index, DAG_node *init)
{

}

// Check that two symbols (potentially from different modules) are equal.
bool DAG_ir_checker::equal_symbols(ISymbol const *s1, ISymbol const *s2) const
{
    if (s1 == s2)
        return true;
    return strcmp(s1->get_name(), s2->get_name()) == 0;
}

// Check that two types (potentially from different modules) are equal.
bool DAG_ir_checker::equal_types(IType const *t1, IType const *t2) const
{
    if (t1 == t2)
        return true;

    t1 = m_tf->get_equal(t1);
    t2 = m_tf->get_equal(t2);

    return t1 != NULL && t1 == t2;
}

// Set the node owner.
DAG_node_factory_impl const *DAG_ir_checker::set_owner(DAG_node_factory_impl const *owner)
{
    DAG_node_factory_impl const *res = m_node_fact;

    if (owner != NULL) {
        m_node_fact = owner;
        m_tf        = &owner->get_type_factory();
        m_vf        = &owner->get_value_factory();
    } else {
        m_node_fact = NULL;
        m_tf        = NULL;
        m_vf        = NULL;
    }
    return res;
}

#endif

} // mdl
} // mi
