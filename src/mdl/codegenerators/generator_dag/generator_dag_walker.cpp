/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/mdl/mdl_generated_dag.h>

#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_assert.h"
#include "mdl/compiler/compilercore/compilercore_bitset.h"
#include "mdl/compiler/compilercore/compilercore_hash.h"

#include "generator_dag_walker.h"
#include "generator_dag_generated_dag.h"
#include "generator_dag_tools.h"

namespace mi {
namespace mdl {

// Constructor.
DAG_ir_walker::DAG_ir_walker(
    IAllocator *alloc,
    bool       as_tree)
: m_alloc(alloc)
, m_as_tree(as_tree)
{
}

// Walk the expressions of a material.
void DAG_ir_walker::walk_material(
    Generated_code_dag *dag,
    int                mat_index,
    IDAG_ir_visitor    *visitor)
{
    DAG_node *expr = const_cast<DAG_node *>(dag->get_material_value(mat_index));

    Memory_arena arena(m_alloc);
    Visited_node_set marker(
        0, Visited_node_set::hasher(), Visited_node_set::key_equal(), &arena);
    Temp_queue queue(m_alloc);

    do_walk_node(marker, queue, expr, visitor);

    Bitset visited_temps(m_alloc, dag->get_material_temporary_count(mat_index));

    while (!queue.empty()) {
        int temp = queue.front();
        queue.pop_front();

        if (visited_temps.test_bit(temp))
            continue;
        visited_temps.set_bit(temp);

        DAG_node *tmp_init = const_cast<DAG_node *>(dag->get_material_temporary(mat_index, temp));
        do_walk_node(marker, queue, tmp_init, visitor);
        visitor->visit(temp, tmp_init);
    }
}

// Walk the expressions of an instance, including temporaries.
void DAG_ir_walker::walk_instance(
    Generated_code_dag::Material_instance *instance,
    IDAG_ir_visitor                       *visitor)
{
    DAG_node *node = const_cast<DAG_call *>(instance->get_constructor());

    Memory_arena arena(m_alloc);
    Visited_node_set marker(
        0, Visited_node_set::hasher(), Visited_node_set::key_equal(), &arena);
    Temp_queue queue(m_alloc);

    do_walk_node(marker, queue, node, visitor);

    Arena_Bitset visited_temps(arena, instance->get_temporary_count());

    while (!queue.empty()) {
        int temp = queue.front();
        queue.pop_front();

        if (visited_temps.test_bit(temp))
            continue;
        visited_temps.set_bit(temp);

        DAG_node *tmp_init = const_cast<DAG_node *>(instance->get_temporary_value(temp));
        do_walk_node(marker, queue, tmp_init, visitor);
        visitor->visit(temp, tmp_init);
    }
}

/// Skip a temporaries if necessary.
static DAG_node const *skip_temporary(DAG_node const *node)
{
    if (DAG_temporary const *tmp = as<DAG_temporary>(node))
        node = tmp->get_expr();
    return node;
}

// Walk the IR nodes of an instance material slot, including temporaries.
void DAG_ir_walker::walk_instance_slot(
    Generated_code_dag::Material_instance       *instance,
    Generated_code_dag::Material_instance::Slot slot,
    IDAG_ir_visitor                             *visitor)
{
    struct Locator {
        char const *first_name;
        char const *second_name;
        char const *third_name;
    };

    static Locator const locators[] = {
        { "thin_walled", NULL,                     NULL },
        { "surface",     "scattering",             NULL },
        { "surface",     "emission",               "emission" },
        { "surface",     "emission",               "intensity" },
        { "backface",    "scattering",             NULL },
        { "backface",    "emission",               "emission" },
        { "backface",    "emission",               "intensity" },
        { "ior",         NULL,                     NULL },
        { "volume",      "scattering",             NULL },
        { "volume",      "absorption_coefficient", NULL },
        { "volume",      "scattering_coefficient", NULL },
        { "geometry",    "displacement",           NULL },
        { "geometry",    "cutout_opacity",         NULL },
        { "geometry",    "normal",                 NULL },
        { "hair",        NULL,                     NULL }
    };

    DAG_node     *node = NULL;
    IValue const *v    = NULL;

    Locator const &locator = locators[slot];

    DAG_call const *constr = instance->get_constructor();
    for (int i = 0, n = constr->get_argument_count(); i < n; ++i) {
        const char *pname = constr->get_parameter_name(i);
        if (strcmp(pname, locator.first_name) == 0) {
            // found the first component
            DAG_node const *f_comp = constr->get_argument(i);

            if (locator.second_name == NULL) {
                // ready
                node = const_cast<DAG_node *>(f_comp);
            } else {
                // extract second
                if (DAG_constant const *cnst = as<DAG_constant>(f_comp)) {
                    // this component is folded into a constant
                    IValue_struct const *strct = cast<IValue_struct>(cnst->get_value());
                    v = strct->get_field(locator.second_name);

                    if (locator.third_name != NULL) {
                        // extract third
                        IValue_struct const *s_strct = cast<IValue_struct>(v);
                        v = s_strct->get_field(locator.third_name);
                    }
                } else {
                    f_comp = skip_temporary(f_comp);
                    if (is<DAG_parameter>(f_comp)) {
                        // we cannot dive further, because we stopped at a parameter
                        node = const_cast<DAG_node *>(f_comp);
                        break;
                    }

                    DAG_call const *call = cast<DAG_call>(f_comp);
                    if (call->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR) {
                        // not a constructor, we cannot dive further
                        node = const_cast<DAG_call *>(call);
                        break;
                    }

                    // this component is the result of a constructor
                    for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
                        const char *pname = call->get_parameter_name(i);
                        if (strcmp(pname, locator.second_name) == 0) {
                            // found the second component
                            DAG_node const *s_comp = call->get_argument(i);

                            if (locator.third_name == NULL) {
                                // ready
                                node = const_cast<DAG_node *>(s_comp);
                            } else {
                                // extract third
                                if (DAG_constant const *cnst = as<DAG_constant>(s_comp)) {
                                    // this component is folded into a constant
                                    IValue_struct const *strct =
                                        cast<IValue_struct>(cnst->get_value());
                                    v = strct->get_field(locator.third_name);
                                } else {
                                    s_comp = skip_temporary(s_comp);
                                    if (is<DAG_parameter>(s_comp)) {
                                        // we cannot dive further, because we stopped at a parameter
                                        node = const_cast<DAG_node *>(s_comp);
                                        break;
                                    }

                                    DAG_call const *c_call = cast<DAG_call>(s_comp);
                                    if (c_call->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR)
                                    {
                                        // not a constructor, we cannot dive further
                                        node = const_cast<DAG_call *>(c_call);
                                        break;
                                    }

                                    // this component is the result of a constructor
                                    for (int i = 0, n = c_call->get_argument_count(); i < n; ++i) {
                                        const char *pname = c_call->get_parameter_name(i);
                                        if (strcmp(pname, locator.third_name) == 0) {
                                            // found the third component
                                            DAG_node const *t_comp = c_call->get_argument(i);

                                            node = const_cast<DAG_node *>(t_comp);
                                            break;
                                        }
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
            break;
        }
    }
    MDL_ASSERT((node != NULL || v != NULL) && "material component could not be located");

    if (v != NULL) {
        // create a temporary Const node, so we can visit it.
        node = const_cast<DAG_constant *>(instance->create_temp_constant(v));
    }
    Memory_arena arena(m_alloc);
    Visited_node_set marker(
        0, Visited_node_set::hasher(), Visited_node_set::key_equal(), &arena);
    Temp_queue queue(m_alloc);

    do_walk_node(marker, queue, node, visitor);

    Arena_Bitset visited_temps(arena, instance->get_temporary_count());

    while (!queue.empty()) {
        int temp = queue.front();
        queue.pop_front();

        if (visited_temps.test_bit(temp))
            continue;
        visited_temps.set_bit(temp);

        DAG_node *tmp_init = const_cast<DAG_node *>(instance->get_temporary_value(temp));
        do_walk_node(marker, queue, tmp_init, visitor);
        visitor->visit(temp, tmp_init);
    }
}

// Walk the expressions of a function.
void DAG_ir_walker::walk_function(
    Generated_code_dag *dag,
    int                func_index,
    IDAG_ir_visitor    *visitor)
{
    DAG_node *expr = const_cast<DAG_node *>(dag->get_function_body(func_index));

    Memory_arena arena(m_alloc);
    Visited_node_set marker(
        0, Visited_node_set::hasher(), Visited_node_set::key_equal(), &arena);
    Temp_queue queue(m_alloc);

    do_walk_node(marker, queue, expr, visitor);

    Bitset visited_temps(m_alloc, dag->get_function_temporary_count(func_index));

    while (!queue.empty()) {
        int temp = queue.front();
        queue.pop_front();

        if (visited_temps.test_bit(temp))
            continue;
        visited_temps.set_bit(temp);

        DAG_node *tmp_init = const_cast<DAG_node *>(dag->get_function_temporary(func_index, temp));
        do_walk_node(marker, queue, tmp_init, visitor);
        visitor->visit(temp, tmp_init);
    }
}

// Walk a DAG IR node.
void DAG_ir_walker::walk_node(
    DAG_node        *node,
    IDAG_ir_visitor *visitor)
{
    Memory_arena arena(m_alloc);
    Visited_node_set marker(
        0, Visited_node_set::hasher(), Visited_node_set::key_equal(), &arena);
    Temp_queue queue(m_alloc);

    do_walk_node(marker, queue, node, visitor);
}

// Walk an DAG IR node.
void DAG_ir_walker::do_walk_node(
    Visited_node_set &marker,
    Temp_queue       &queue,
    DAG_node         *node,
    IDAG_ir_visitor  *visitor)
{
    if (!m_as_tree) {
        if (marker.find(node) != marker.end())
            return;
        marker.insert(node);
    }

    switch (node->get_kind()) {
    case DAG_node::EK_CONSTANT:
        {
            DAG_constant *c = cast<DAG_constant>(node);
            visitor->visit(c);
            return;
        }
    case DAG_node::EK_TEMPORARY:
        {
            DAG_temporary *t = cast<DAG_temporary>(node);
            visitor->visit(t);

            queue.push_back(t->get_index());
            return;
        }
    case DAG_node::EK_CALL:
        {
            DAG_call *c = cast<DAG_call>(node);

            for (int i = 0, n = c->get_argument_count(); i < n; ++i) {
                DAG_node *arg = const_cast<DAG_node *>(c->get_argument(i));

                do_walk_node(marker, queue, arg, visitor);
            }
            visitor->visit(c);
            return;
        }
    case DAG_node::EK_PARAMETER:
        {
            DAG_parameter *p = cast<DAG_parameter>(node);
            visitor->visit(p);
            return;
        }
    }
    MDL_ASSERT(!"Unsupported DAG node kind");
}

// Constructor.
Dag_hasher::Dag_hasher(
    MD5_hasher &hasher)
: m_hasher(hasher)
{
}

// Post-visit a Constant.
void Dag_hasher::visit(DAG_constant *cnst)
{
    m_hasher.update('C');
    IValue const *v = cnst->get_value();

    hash(v);
}

// Post-visit a variable (temporary).
void Dag_hasher::visit(DAG_temporary *tmp)
{
    m_hasher.update('T');
    m_hasher.update(tmp->get_index());
}

// Post-visit a call.
void Dag_hasher::visit(DAG_call *call)
{
    m_hasher.update('C');
    IDefinition::Semantics sema = call->get_semantic();

    if (sema != IDefinition::DS_UNKNOWN && sema != IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS) {
        // semantic is enough
        m_hasher.update(sema);
    } else {
        // name is needed
        m_hasher.update(call->get_name());
    }
    m_hasher.update(call->get_argument_count());

    // assume at this point that argument order is "safe", i.e.
    // all calls are ordered "by position"
}

// Post-visit a Parameter.
void Dag_hasher::visit(DAG_parameter *param)
{
    m_hasher.update('P');
    m_hasher.update(param->get_index());
}

// Post-visit a Temporary.
void Dag_hasher::visit(int index, DAG_node *init)
{
}

// Hash a parameter.
void Dag_hasher::hash_parameter(char const *name, IType const *type)
{
    m_hasher.update(name);
    hash(type);
}

// Hash a type.
void Dag_hasher::hash(IType const *tp) {
    IType::Kind kind = tp->get_kind();
    m_hasher.update(kind);

    switch (kind) {
    case IType::TK_ALIAS:
        {
            IType_alias const *a_tp = cast<IType_alias>(tp);
            m_hasher.update(a_tp->get_type_modifiers());
            if (ISymbol const *sym = a_tp->get_symbol())
                m_hasher.update(sym->get_name());
            hash(a_tp->get_aliased_type());
        }
        break;
    case IType::TK_BOOL:
    case IType::TK_INT:
        break;
    case IType::TK_ENUM:
        {
            IType_enum const *et = cast<IType_enum>(tp);

            m_hasher.update(et->get_symbol()->get_name());
        }
        break;
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_STRING:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
        break;
    case IType::TK_VECTOR:
        {
            IType_vector const *vt = cast<IType_vector>(tp);

            m_hasher.update(vt->get_size());
            hash(vt->get_element_type());
        }
        break;
    case IType::TK_MATRIX:
        {
            IType_matrix const *mt = cast<IType_matrix>(tp);

            m_hasher.update(mt->get_columns());
            hash(mt->get_element_type());
        }
        break;
    case IType::TK_ARRAY:
        {
            IType_array const *at = cast<IType_array>(tp);

            if (at->is_immediate_sized()) {
                m_hasher.update(at->get_size());
            } else {
                IType_array_size const *sz = at->get_deferred_size();

                m_hasher.update(sz->get_name()->get_name());
            }
            hash(at->get_element_type());
        }
        break;
    case IType::TK_COLOR:
        break;
    case IType::TK_FUNCTION:
        {
            IType_function const *ft = cast<IType_function>(tp);

            if (IType const *ret_type = ft->get_return_type()) {
                m_hasher.update('R');
                hash(ret_type);
            } else {
                m_hasher.update('N');
            }

            int n_params = ft->get_parameter_count();
            m_hasher.update(n_params);

            for (int i = 0; i < n_params; ++i) {
                IType const *p_tp;
                ISymbol const *p_sym;

                ft->get_parameter(i, p_tp, p_sym);

                m_hasher.update(p_sym->get_name());
                hash(p_tp);
            }
        }
        break;
    case IType::TK_STRUCT:
        {
            IType_struct const *st = cast<IType_struct>(tp);

            m_hasher.update(st->get_symbol()->get_name());
        }
        break;
    case IType::TK_TEXTURE:
        {
            IType_texture const *tt = cast<IType_texture>(tp);

            m_hasher.update(tt->get_shape());
        }
        break;
    case IType::TK_BSDF_MEASUREMENT:
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
        break;
    }
}

// Hash a value.
void Dag_hasher::hash(IValue const *v) {
    IValue::Kind kind = v->get_kind();
    m_hasher.update(kind);

    switch (kind) {
    case IValue::VK_BAD:
        break;
    case IValue::VK_BOOL:
        {
            IValue_bool const *bv = cast<IValue_bool>(v);
            m_hasher.update(bv->get_value() ? 'T' : 'F');
        }
        break;
    case IValue::VK_INT:
        {
            IValue_int const *iv = cast<IValue_int>(v);
            m_hasher.update(iv->get_value());
        }
        break;
    case IValue::VK_ENUM:
        {
            IValue_enum const *ev = cast<IValue_enum>(v);
            IType_enum const  *et = ev->get_type();

            m_hasher.update(et->get_symbol()->get_name());
            m_hasher.update(ev->get_value());
        }
        break;
    case IValue::VK_FLOAT:
        {
            IValue_float const *fv = cast<IValue_float>(v);
            m_hasher.update(fv->get_value());
        }
        break;
    case IValue::VK_DOUBLE:
        {
            IValue_double const *dv = cast<IValue_double>(v);
            m_hasher.update(dv->get_value());
        }
        break;
    case IValue::VK_STRING:
        {
            IValue_string const *sv = cast<IValue_string>(v);
            m_hasher.update(sv->get_value());
        }
        break;
    case IValue::VK_STRUCT:
        {
            IValue_struct const *sv = cast<IValue_struct>(v);
            IType_struct const  *st = sv->get_type();

            m_hasher.update(st->get_symbol()->get_name());
        }
        // fallthrough
    case IValue::VK_VECTOR:
    case IValue::VK_MATRIX:
    case IValue::VK_ARRAY:
    case IValue::VK_RGB_COLOR:
        {
            IValue_compound const *cv = cast<IValue_compound>(v);

            for (int i = 0, n = cv->get_component_count(); i < n; ++i) {
                IValue const *child = cv->get_value(i);
                hash(child);
            }
        }
        break;
    case IValue::VK_INVALID_REF:
        {
            IValue_invalid_ref const *iv = cast<IValue_invalid_ref>(v);
            IType_reference const    *it = iv->get_type();

            int tkind = it->get_kind();
            m_hasher.update(tkind);
        }
        break;
    case IValue::VK_TEXTURE:
        {
            IValue_texture const *tv = cast<IValue_texture>(v);
            m_hasher.update(tv->get_string_value());
            m_hasher.update(tv->get_gamma_mode());
            m_hasher.update(tv->get_tag_value());
            m_hasher.update(tv->get_tag_version());
        }
        break;
    case IValue::VK_LIGHT_PROFILE:
        {
            IValue_light_profile const *lv = cast<IValue_light_profile>(v);
            m_hasher.update(lv->get_string_value());
            m_hasher.update(lv->get_tag_value());
            m_hasher.update(lv->get_tag_version());
        }
        break;
    case IValue::VK_BSDF_MEASUREMENT:
        {
            IValue_bsdf_measurement const *lv = cast<IValue_bsdf_measurement>(v);
            m_hasher.update(lv->get_string_value());
            m_hasher.update(lv->get_tag_value());
            m_hasher.update(lv->get_tag_version());
        }
        break;
    }
}


} // mdl
} // mi
