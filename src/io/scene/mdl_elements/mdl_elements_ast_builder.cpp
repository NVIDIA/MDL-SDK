/***************************************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Source for building MDL AST from integration expressions/types

#include "pch.h"

#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_modules.h>

#include <base/lib/log/i_log_assert.h>
#include <base/lib/log/i_log_logger.h>

#include <base/data/db/i_db_transaction.h>
#include <base/data/db/i_db_access.h>

#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_modules.h>
#include <mdl/compiler/compilercore/compilercore_symbols.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>

#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/texture/i_texture.h>


#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_function_definition.h"
#include "i_mdl_elements_utilities.h"
#include "mdl_elements_ast_builder.h"

namespace MI {
namespace MDL {

using mi::mdl::as;
using mi::mdl::as_or_null;
using mi::mdl::cast;
using mi::mdl::dimension_of;
using mi::mdl::impl_cast;
using mi::mdl::is;

using mi::base::Handle;

Mdl_ast_builder::Mdl_ast_builder(
    mi::mdl::IModule *owner,
    DB::Transaction *transaction,
    bool traverse_ek_parameter,
    const IExpression_list* args,
    Name_mangler& name_mangler,
    bool avoid_resource_urls)
: m_owner(impl_cast<mi::mdl::Module>(owner))
, m_trans(transaction)
, m_traverse_ek_parameter(traverse_ek_parameter)
, m_nf(*m_owner->get_name_factory())
, m_vf(*m_owner->get_value_factory())
, m_ef(*m_owner->get_expression_factory())
, m_tf(*m_owner->get_type_factory())
, m_st(m_owner->get_symbol_table())
, m_int_tf(MDL::get_type_factory())
, m_tmp_idx(0u)
, m_args(args, mi::base::DUP_INTERFACE)
, m_owner_version(m_owner->get_mdl_version())
, m_name_mangler(name_mangler)
, m_avoid_resource_urls(avoid_resource_urls)
{
}

const mi::mdl::ISimple_name* Mdl_ast_builder::create_simple_name( const std::string& name)
{
    const mi::mdl::ISymbol* symbol = m_st.get_symbol( strip_deprecated_suffix( name).c_str());
    return m_nf.create_simple_name( symbol);
}

mi::mdl::IQualified_name* Mdl_ast_builder::create_qualified_name( const std::string& name)
{
    mi::mdl::IQualified_name* qualified_name = m_nf.create_qualified_name();
    if( is_absolute( name) && !is_in_module( name, get_builtins_module_mdl_name()))
        qualified_name->set_absolute();

    std::string module_name     = get_mdl_module_name( name);
    std::string definition_name = get_mdl_simple_definition_name( name);

    std::vector<std::string> package_components = get_mdl_package_component_names( module_name);
    std::string simple_module_name              = get_mdl_simple_module_name( module_name);

    const mi::mdl::ISimple_name* simple_name = nullptr;

    for( const auto& pc: package_components) {
        simple_name = create_simple_name( m_name_mangler.mangle( pc.c_str()));
        qualified_name->add_component( simple_name);
    }

    if( simple_module_name != get_builtins_module_simple_name()) {
        simple_name = create_simple_name( m_name_mangler.mangle( simple_module_name.c_str()));
        qualified_name->add_component( simple_name);
    }

    simple_name = create_simple_name( definition_name);
    qualified_name->add_component( simple_name);

    return qualified_name;
}

mi::mdl::IQualified_name* Mdl_ast_builder::create_scope_name( const std::string& name)
{
    mi::mdl::IQualified_name* qualified_name = m_nf.create_qualified_name();

    if( is_absolute( name) && !is_in_module( name, get_builtins_module_mdl_name()))
        qualified_name->set_absolute();

    std::string module_name     = get_mdl_module_name( name);
    std::string definition_name = get_mdl_simple_definition_name( name);

    std::vector<std::string> package_components = get_mdl_package_component_names( module_name);
    std::string simple_module_name              = get_mdl_simple_module_name( module_name);

    for( const auto& pc: package_components) {
        const mi::mdl::ISimple_name* simple_name
            = create_simple_name( m_name_mangler.mangle( pc.c_str()));
        qualified_name->add_component( simple_name);
    }

    if( simple_module_name != get_builtins_module_simple_name()) {
        const mi::mdl::ISimple_name* simple_name
            = create_simple_name( m_name_mangler.mangle( simple_module_name.c_str()));
        qualified_name->add_component( simple_name);
    }

    return qualified_name;
}

namespace {

/// Creates an MDL AST expression reference for a given MDL type.
///
/// TODO unify with create_to_type_name()
///
/// \param module                 The module on which the qualified name is created.
/// \param type                   The type.
/// \param name_mangler           The name mangler.
/// \return                       The MDL AST expression reference for the type, or for arrays the
///                               MDL AST expression reference for the corresponding array
///                               constructor.
mi::mdl::IType_name* type_to_type_name(
    mi::mdl::IModule* module, const mi::mdl::IType* type, Name_mangler* name_mangler)
{
    char buf[32];
    const char* s = nullptr;

    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case mi::mdl::IType::TK_ALIAS:
    case mi::mdl::IType::TK_PTR:
    case mi::mdl::IType::TK_REF:
    case mi::mdl::IType::TK_AUTO:
    case mi::mdl::IType::TK_ERROR:
    case mi::mdl::IType::TK_FUNCTION:
        ASSERT(M_SCENE, !"unexpected MDL type kind");
        return nullptr;

    case mi::mdl::IType::TK_VOID:
        ASSERT(M_SCENE, !"unexpected void type");
        s = "void";
        break;
    case mi::mdl::IType::TK_BOOL:
        s = "bool";
        break;
    case mi::mdl::IType::TK_INT:
        s = "int";
        break;
    case mi::mdl::IType::TK_ENUM:
        s = as<mi::mdl::IType_enum>(type)->get_symbol()->get_name();
        break;
    case mi::mdl::IType::TK_FLOAT:
        s = "float";
        break;
    case mi::mdl::IType::TK_DOUBLE:
        s = "double";
        break;
    case mi::mdl::IType::TK_STRING:
        s = "string";
        break;
    case mi::mdl::IType::TK_LIGHT_PROFILE:
        s = "light_profile";
        break;
    case mi::mdl::IType::TK_BSDF:
        s = "bsdf";
        break;
    case mi::mdl::IType::TK_HAIR_BSDF:
        s = "hair_bsdf";
        break;
    case mi::mdl::IType::TK_EDF:
        s = "edf";
        break;
    case mi::mdl::IType::TK_VDF:
        s = "vdf";
        break;
    case mi::mdl::IType::TK_VECTOR:
        {
            const auto* v_type = as<mi::mdl::IType_vector>(type);
            const mi::mdl::IType_atomic* a_type = v_type->get_element_type();
            int size = v_type->get_size();

            switch (a_type->get_kind()) {
            case mi::mdl::IType::TK_BOOL:
                switch (size) {
                case 2: s = "bool2"; break;
                case 3: s = "bool3"; break;
                case 4: s = "bool4"; break;
                }
                break;
            case mi::mdl::IType::TK_INT:
                switch (size) {
                case 2: s = "int2"; break;
                case 3: s = "int3"; break;
                case 4: s = "int4"; break;
                }
                break;
            case mi::mdl::IType::TK_FLOAT:
                switch (size) {
                case 2: s = "float2"; break;
                case 3: s = "float3"; break;
                case 4: s = "float4"; break;
                }
                break;
            case mi::mdl::IType::TK_DOUBLE:
                switch (size) {
                case 2: s = "double2"; break;
                case 3: s = "double3"; break;
                case 4: s = "double4"; break;
                }
                break;
            default:
                ASSERT(M_SCENE, !"Unexpected type kind");
            }
        }
        break;
    case mi::mdl::IType::TK_MATRIX:
        {
            const auto *m_type = as<mi::mdl::IType_matrix>(type);
            const mi::mdl::IType_vector *e_type = m_type->get_element_type();
            const mi::mdl::IType_atomic *a_type = e_type->get_element_type();

            snprintf(buf, sizeof(buf), "%s%dx%d",
                a_type->get_kind() == mi::mdl::IType::TK_FLOAT ? "float" : "double",
                m_type->get_columns(),
                e_type->get_size());
            buf[sizeof(buf) - 1] = '\0';
            s = buf;
        }
        break;
    case mi::mdl::IType::TK_ARRAY:
        {
            const auto *a_type = as<mi::mdl::IType_array>(type);

            mi::mdl::IType_name* tn
                = type_to_type_name(module, a_type->get_element_type(), name_mangler);

            if (a_type->is_immediate_sized()) {
                int size = a_type->get_size();
                mi::mdl::IValue const *v = module->get_value_factory()->create_int(size);
                mi::mdl::IExpression const *lit =
                    module->get_expression_factory()->create_literal(v);
                tn->set_array_size(lit);
            } else {
                // we should not be here, but if, we create an incomplete array
                tn->set_incomplete_array();
            }
            return tn;
        }
    case mi::mdl::IType::TK_COLOR:
        s = "color";
        break;
    case mi::mdl::IType::TK_STRUCT:
        s = mi::mdl::as<mi::mdl::IType_struct>(type)->get_symbol()->get_name();
        break;
    case mi::mdl::IType::TK_TEXTURE:
        {
            auto const *t_type = as<mi::mdl::IType_texture>(type);

            switch (t_type->get_shape()) {
            case mi::mdl::IType_texture::TS_2D:
                s = "texture_2d";
                break;
            case mi::mdl::IType_texture::TS_3D:
                s = "texture_3d";
                break;
            case mi::mdl::IType_texture::TS_CUBE:
                s = "texture_cube";
                break;
            case mi::mdl::IType_texture::TS_PTEX:
                s = "texture_ptex";
                break;
            case mi::mdl::IType_texture::TS_BSDF_DATA:
                ASSERT(M_SCENE, !"bsdf data textures cannot be expression in MDL source");
                break;
            }
        }
        break;
    case mi::mdl::IType::TK_BSDF_MEASUREMENT:
        s = "bsdf_measurement";
        break;
    }
    ASSERT(M_SCENE, s);

    mi::mdl::IName_factory* nf = module->get_name_factory();
    mi::mdl::IQualified_name* qualified_name
        = signature_to_qualified_name( nf, s, name_mangler);

    return nf->create_type_name(qualified_name);
}

} // namespace

// Construct a Type_name AST element for a neuray type.
//
// TODO unify with type_to_type_name()
mi::mdl::IType_name *Mdl_ast_builder::create_type_name(
    Handle<IType const> const &t)
{
    mi::Uint32 modifiers = t->get_all_type_modifiers();
    Handle<IType const> const type(t->skip_all_type_aliases());

    switch (type->get_kind()) {
    case IType::TK_ALIAS:
        // should not happen
        ASSERT( M_SCENE, !"unexpected type kind");
        return nullptr;

    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_STRING:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
    case IType::TK_COLOR:
    case IType::TK_STRUCT:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
        {
            std::string name = m_int_tf->get_mdl_type_name( type.get());
            name = decode_name_without_signature( name);
            name = remove_qualifiers_if_from_module( name, m_owner->get_name());

            mi::mdl::IQualified_name *qname = create_qualified_name(name);
            mi::mdl::IType_name      *tn    = m_nf.create_type_name(qname);

            if (modifiers & IType::MK_UNIFORM) {
                tn->set_qualifier(mi::mdl::FQ_UNIFORM);
            } else if (modifiers & IType::MK_VARYING) {
                tn->set_qualifier(mi::mdl::FQ_VARYING);
            }
            return tn;
        }
    case IType::TK_ARRAY:
        {
            Handle<const IType_array> a_tp(type->get_interface<IType_array>());
            Handle<const IType>       e_tp(a_tp->get_element_type());

            mi::mdl::IType_name *tn = create_type_name(e_tp);

            if (modifiers & IType::MK_UNIFORM) {
                tn->set_qualifier(mi::mdl::FQ_UNIFORM);
            } else if (modifiers & IType::MK_VARYING) {
                tn->set_qualifier(mi::mdl::FQ_VARYING);
            }

            if (a_tp->is_immediate_sized()) {
                size_t size = a_tp->get_size();
                mi::mdl::Value_factory *vf     = m_owner->get_value_factory();
                mi::mdl::IValue const  *v_size = vf->create_int(int(size));
                mi::mdl::IExpression_literal *lit = m_ef.create_literal(v_size);
                tn->set_array_size(lit);
            } else {
                // Note: in the MDL AST, a deferred size has two "names".
                // The fully qualified name, the identifies every deferred size symbol uniquely
                // and the symbol itself, which is typically the last component of the
                // fully qualified name.
                //
                // In neuray, only the symbol itself exists.
                // We have to create a new symbol here, with the following properties:
                // - all deferred size symbols with the same name must map to the same symbol
                // - all deferred size symbols with different names must map to *different*
                //   symbol, even if there last component is identical, because we do not have
                //   here enough context to decide if otherwise clashes can be avoided.
                //
                // We have to avoid clashes, because that would mean *the same size*.
                //
                // The whole task would be much simpler if the real core types would exists here.
                // Note further: this "local" technique cannot avoid clashes at all, because we
                // have no way to find a unique symbol here. For that, all symbols must be known
                // in advance.
                //
                // A possible solution would be to "defer" the name until all symbols of the whole
                // module are created, then "fix" them by replacing with unique ones.
                //
                // So as a "mostly working" work-around we just take the symbol itself and hope
                // the best that no complex operation merges different types.
                char const *size = a_tp->get_deferred_size();
                mi::mdl::ISymbol const      *sym   = m_st.create_symbol(size);
                mi::mdl::ISimple_name const *sname = m_nf.create_simple_name(sym);
                tn->set_size_name(sname);
            }
            return tn;
        }
        break;
    }
    ASSERT( M_SCENE, !"unexpected type kind");
    return nullptr;
}

// Retrieve the field symbol from an unmangled DS_INTRINSIC_DAG_FIELD_ACCESS call, i.e., without
// parameter types.
const mi::mdl::ISymbol* Mdl_ast_builder::get_field_sym( const std::string& def)
{
    ASSERT(M_SCENE, !def.empty() && def.back() != ')');
    std::string field = get_mdl_field_name(def);
    return m_st.get_symbol(field.c_str());
}

static size_t const zero_size = 0u;

// Transform a call.
mi::mdl::IExpression const *Mdl_ast_builder::transform_call(
    const IType                          *ret_type,
    mi::mdl::IDefinition::Semantics      sema,
    std::string const                    &callee_name,
    mi::Size                             n_params,
    const IExpression_list               *args,
    bool                                 named_args)
{
    // Skip decl_cast operator.
    if (sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_DECL_CAST) {
        mi::base::Handle<const IExpression> arg(args->get_expression(mi::Size(0)));
        return transform_expr(arg.get());
    }

    bool is_standard_material_type = false;
    Handle<IType const> const type(ret_type->skip_all_type_aliases());
    if (type->get_kind() == IType::TK_STRUCT) {

        Handle<IType_struct const> s_tp(type->get_interface<IType_struct>());
        IType_struct::Predefined_id id = s_tp->get_predefined_id();
        if (id == IType_struct::SID_USER) {
            m_used_user_types.insert(decode_name_without_signature(s_tp->get_symbol()));
        } else if (id == IType_struct::SID_MATERIAL) {
            is_standard_material_type = true;
        }

    } else if (type->get_kind() == IType::TK_ENUM) {

        Handle<IType_enum const> e_tp(type->get_interface<IType_enum>());
        if (e_tp->get_predefined_id() != IType_enum::EID_INTENSITY_MODE) {
            // only intensity_mode is predefined
            m_used_user_types.insert(decode_name_without_signature(e_tp->get_symbol()));
        }

    }

    if (semantic_is_operator(sema)) {
        mi::mdl::IExpression::Operator op = semantic_to_operator(sema);

        if (mi::mdl::is_unary_operator(op)) {

            Handle<IExpression const> un_arg(args->get_expression(mi::Size(0)));
            mi::mdl::IExpression const *arg = transform_expr(un_arg.get());

            mi::mdl::IExpression_unary* res = m_ef.create_unary(
                mi::mdl::IExpression_unary::Operator(op), arg);

            if (op == mi::mdl::IExpression::OK_CAST) {
                const mi::mdl::IType* tp = int_type_to_core_type(
                    ret_type,
                    m_tf);

                const mi::mdl::IType_name* tn = type_to_type_name(m_owner, tp, &m_name_mangler);
                res->set_type_name(tn);
            }
            return res;
        } else if (mi::mdl::is_binary_operator(op)) {
            auto bop = mi::mdl::IExpression_binary::Operator(op);

            Handle<IExpression const> l_arg(args->get_expression(mi::Size(0)));
            Handle<IExpression const> r_arg(args->get_expression(mi::Size(1)));

            mi::mdl::IExpression const *l = transform_expr(l_arg.get());
            mi::mdl::IExpression const *r = transform_expr(r_arg.get());

            return m_ef.create_binary(bop, l, r);
        } else if (op == mi::mdl::IExpression::OK_TERNARY) {
            // C-like ternary operator with lazy evaluation
            Handle<IExpression const> cond_arg(args->get_expression(mi::Size(0)));
            Handle<IExpression const> true_arg(args->get_expression(mi::Size(1)));
            Handle<IExpression const> false_arg(args->get_expression(mi::Size(2)));

            mi::mdl::IExpression const *cond      = transform_expr(cond_arg.get());
            mi::mdl::IExpression const *true_res  = transform_expr(true_arg.get());
            mi::mdl::IExpression const *false_res = transform_expr(false_arg.get());

            return m_ef.create_conditional(cond, true_res, false_res);
        }
    }

    // do MDL 1.X => MDL 1.Y conversion here
    switch (sema) {
    case mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR:
        if (   m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_4
            && n_params == 6
            && is_standard_material_type) {
            // MDL 1.4 -> 1.5: add default hair bsdf
            mi::mdl::IQualified_name *tu_qname = create_qualified_name("hair_bsdf");
            mi::mdl::IExpression_reference *tu_ref = to_reference(tu_qname);
            mi::mdl::IExpression_call *tu_call = m_ef.create_call(tu_ref);

            mi::mdl::IQualified_name *qname
                = create_qualified_name(strip_deprecated_suffix(callee_name));
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            for (mi::Size i = 0; i < n_params; ++i) {
                mi::mdl::IArgument const *arg = nullptr;

                Handle<IExpression const> nr_arg(args->get_expression(i));
                mi::mdl::IExpression const *expr = transform_expr(nr_arg.get());

                if (named_args) {
                    arg = m_ef.create_named_argument(to_simple_name(args->get_name(i)), expr);
                } else {
                    arg = m_ef.create_positional_argument(expr);
                }
                call->add_argument(arg);

                if (i == 5) {
                    // insert the hair parameter
                    if (named_args) {
                        arg = m_ef.create_named_argument(to_simple_name("hair"), tu_call);
                    } else {
                        arg = m_ef.create_positional_argument(tu_call);
                    }
                    call->add_argument(arg);
                }
            }
            return call;
        }
        break;

    case mi::mdl::IDefinition::DS_TEXTURE_CONSTRUCTOR:
        {
            mi::mdl::IQualified_name *qname = create_qualified_name(callee_name);
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            mi::mdl::IExpression const *tex_expr = nullptr;

            ASSERT( M_SCENE, n_params > 0);
            for (mi::Size i = 0, n = n_params; i < n; ++i) {
                Handle<IExpression const> arg(args->get_expression(i));

                mi::mdl::IExpression const *expr = transform_expr(arg.get());
                if (i == 0) {
                    tex_expr = expr;
                }

                if (named_args) {
                    mi::mdl::ISimple_name const *sname = to_simple_name(args->get_name(i));
                    call->add_argument(
                        m_ef.create_named_argument(sname, expr));
                } else {
                    call->add_argument(
                        m_ef.create_positional_argument(expr));
                }
            }

            if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_6) {
                mi::mdl::IType const *tex = tex_expr->get_type();
                if (n_params == 2 && (mi::mdl::is_tex_2d(tex) || mi::mdl::is_tex_3d(tex))) {
                    // MDL 1.6 -> 1.7: insert the selector parameter
                    mi::mdl::IExpression const *expr =
                        m_ef.create_literal(m_vf.create_string(""));

                    mi::mdl::IArgument const *arg = nullptr;
                    if (named_args) {
                        arg = m_ef.create_named_argument(to_simple_name("selector"), expr);
                    } else {
                        arg = m_ef.create_positional_argument(expr);
                    }
                    call->add_argument(arg);
                }
            }
            return call;
        }
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_DF_MEASURED_EDF:
        if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_1 && n_params == 4) {
            // MDL 1.0 -> 1.2: insert the multiplier and tangent_u parameters
            mi::mdl::IQualified_name *tu_qname = create_qualified_name("state::texture_tangent_u");
            mi::mdl::IExpression_reference *tu_ref = to_reference(tu_qname);
            mi::mdl::IExpression_call *tu_call = m_ef.create_call(tu_ref);

            tu_call->add_argument(
                m_ef.create_positional_argument(
                    m_ef.create_literal(
                        m_vf.create_int(0))));

            mi::mdl::IQualified_name *qname
                = create_qualified_name(strip_deprecated_suffix(callee_name));
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            for (mi::Size i = 0, j = 0; i < n_params; ++i, ++j) {
                mi::mdl::IArgument const *arg = nullptr;

                if (j == 1) {
                    // add multiplier
                    if (named_args) {
                        arg = m_ef.create_named_argument(
                            to_simple_name("multiplier"),
                            m_ef.create_literal(m_vf.create_float(1.0f)));
                    } else {
                        arg = m_ef.create_positional_argument(
                            m_ef.create_literal(m_vf.create_float(1.0f)));
                    }
                    call->add_argument(arg);
                    ++j;
                } else if (j == 4) {
                    // add tangent_u
                    if (named_args) {
                        arg = m_ef.create_named_argument(to_simple_name("tangent_u"), tu_call);
                    } else {
                        arg = m_ef.create_positional_argument(tu_call);
                    }
                    call->add_argument(arg);
                    ++j;
                }

                Handle<IExpression const> nr_arg(args->get_expression(i));
                mi::mdl::IExpression const *expr = transform_expr(nr_arg.get());

                if (named_args) {
                    arg = m_ef.create_named_argument(to_simple_name(args->get_name(i)), expr);
                } else {
                    arg = m_ef.create_positional_argument(expr);
                }
                call->add_argument(arg);
            }
            return call;
        } else if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_1 && n_params == 5) {
            // MDL 1.1 -> 1.2: insert tangent_u parameter
            mi::mdl::IQualified_name *tu_qname = create_qualified_name("state::texture_tangent_u");
            mi::mdl::IExpression_reference *tu_ref = to_reference(tu_qname);
            mi::mdl::IExpression_call *tu_call = m_ef.create_call(tu_ref);

            tu_call->add_argument(
                m_ef.create_positional_argument(
                    m_ef.create_literal(
                        m_vf.create_int(0))));

            mi::mdl::IQualified_name *qname
                = create_qualified_name(strip_deprecated_suffix(callee_name));
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            for (mi::Size i = 0, j = 0; i < n_params; ++i, ++j) {
                mi::mdl::IArgument const *arg = nullptr;

                if (j == 4) {
                    // add tangent_u
                    if (named_args) {
                        arg = m_ef.create_named_argument(to_simple_name("tangent_u"), tu_call);
                    } else {
                        arg = m_ef.create_positional_argument(tu_call);
                    }
                    call->add_argument(arg);
                    ++j;
                }

                Handle<IExpression const> nr_arg(args->get_expression(i));
                mi::mdl::IExpression const *expr = transform_expr(nr_arg.get());

                if (named_args) {
                    arg = m_ef.create_named_argument(to_simple_name(args->get_name(i)), expr);
                } else {
                    arg = m_ef.create_positional_argument(expr);
                }
                call->add_argument(arg);
            }
            return call;
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
        {
            if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_3) {
                if (is_deprecated(callee_name)) {
                    // MDL 1.3 -> 1.4: convert "half-colored" to full colored

                    mi::mdl::IQualified_name *qname = create_qualified_name(
                        "::df::color_fresnel_layer");
                    mi::mdl::IExpression_reference *ref = to_reference(qname);
                    mi::mdl::IExpression_call *call = m_ef.create_call(ref);

                    for (mi::Size i = 0; i < n_params; ++i) {
                        mi::mdl::IArgument const *arg = nullptr;

                        Handle<IExpression const> nr_arg(args->get_expression(i));
                        mi::mdl::IExpression const *expr = transform_expr(nr_arg.get());

                        if (i == 1) {
                            // wrap by color constructor
                            mi::mdl::IQualified_name *qname = create_qualified_name("color");
                            mi::mdl::IExpression_reference *ref = to_reference(qname);
                            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

                            call->add_argument(m_ef.create_positional_argument(expr));
                            expr = call;
                        }

                        if (named_args) {
                            arg = m_ef.create_named_argument(
                                to_simple_name(args->get_name(i)), expr);
                        } else {
                            arg = m_ef.create_positional_argument(expr);
                        }
                        call->add_argument(arg);
                    }
                    return call;
                }
            }
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DF_SPOT_EDF:
        if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_0 && n_params == 4) {
            // MDL 1.0 -> 1.1: insert spread parameter
            mi::mdl::IQualified_name *qname
                = create_qualified_name(strip_deprecated_suffix(callee_name));
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            for (mi::Size i = 0; i < n_params; ++i) {
                mi::mdl::IArgument const *arg = nullptr;

                if (i == 1) {
                    // insert the spread parameter
                    mi::mdl::IExpression const *expr =
                        m_ef.create_literal(m_vf.create_float(float(M_PI)));
                    if (named_args) {
                        arg = m_ef.create_named_argument(to_simple_name("spread"), expr);
                    } else {
                        arg = m_ef.create_positional_argument(expr);
                    }
                    call->add_argument(arg);
                }

                Handle<IExpression const> nr_arg(args->get_expression(i));
                mi::mdl::IExpression const *expr = transform_expr(nr_arg.get());

                if (named_args) {
                    arg = m_ef.create_named_argument(to_simple_name(args->get_name(i)), expr);
                } else {
                    arg = m_ef.create_positional_argument(expr);
                }
                call->add_argument(arg);
            }
            return call;
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
    case mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
    case mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
    case mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
    case mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
        if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_5 && n_params == 6) {
            return add_multiscatter_param(callee_name, n_params, named_args, args);
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
    case mi::mdl::IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
        if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_5 && n_params == 5) {
            return add_multiscatter_param(callee_name, n_params, named_args, args);
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL:
        if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_2 && n_params == 2) {
            // MDL 1.2 -> 1.3: insert the roundness parameter
            mi::mdl::IQualified_name *qname
                = create_qualified_name(strip_deprecated_suffix(callee_name));
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            for (mi::Size i = 0; i < n_params; ++i) {
                Handle<IExpression const> nr_arg(args->get_expression(i));
                mi::mdl::IExpression const *expr = transform_expr(nr_arg.get());

                mi::mdl::IArgument const *arg = nullptr;
                if (named_args) {
                    arg = m_ef.create_named_argument(to_simple_name(args->get_name(i)), expr);
                } else {
                    arg = m_ef.create_positional_argument(expr);
                }
                call->add_argument(arg);
            }

            mi::mdl::IArgument const   *arg = nullptr;
            mi::mdl::IExpression const *expr = m_ef.create_literal(m_vf.create_float(1.0f));
            if (named_args) {
                arg = m_ef.create_named_argument(to_simple_name("roundness"), expr);
            } else {
                arg = m_ef.create_positional_argument(expr);
            }
            call->add_argument(arg);
            return call;
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_WIDTH:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_HEIGHT:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_DEPTH:
        {
            mi::mdl::IQualified_name       *qname =
                create_qualified_name(strip_deprecated_suffix(callee_name));
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call      *call = m_ef.create_call(ref);

            Handle<IExpression const> nr_arg(args->get_expression(zero_size));
            mi::mdl::IExpression const *tex = transform_expr(nr_arg.get());

            mi::mdl::IArgument const *arg = nullptr;
            if (named_args) {
                arg = m_ef.create_named_argument(to_simple_name(args->get_name(0)), tex);
            } else {
                arg = m_ef.create_positional_argument(tex);
            }
            call->add_argument(arg);

            if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_3 && n_params == 1) {
                if (mi::mdl::is_tex_2d(tex->get_type())) {
                    // MDL 1.3 -> 1.4: insert the uv_tile parameter for width/height(tex_2d)
                    mi::mdl::IExpression const *expr =
                        m_ef.create_literal(mi::mdl::create_int2_zero(m_vf));
                    if (named_args) {
                        arg = m_ef.create_named_argument(to_simple_name("uv_tile"), expr);
                    } else {
                        arg = m_ef.create_positional_argument(expr);
                    }
                    call->add_argument(arg);
                    ++n_params;
                }
            }
            if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_6) {
                if ((
                        /* width/height(tex_2d tex, uv_tile) */
                        n_params == 2 && mi::mdl::is_tex_2d(tex->get_type())
                    )
                    ||
                    (
                        /* width/height/depth(tex_3d tex) */
                        n_params == 1 && mi::mdl::is_tex_3d(tex->get_type())
                    )) {
                    // MDL 1.6 -> 1.7: insert the frame parameter for width/height/depth()
                    mi::mdl::IExpression const *expr =
                        m_ef.create_literal(m_vf.create_float(0.0f));
                    mi::mdl::IArgument const *arg = nullptr;
                    if (named_args) {
                        arg = m_ef.create_named_argument(to_simple_name("frame"), expr);
                    } else {
                        arg = m_ef.create_positional_argument(expr);
                    }
                    call->add_argument(arg);
                }
            }
            return call;
        }
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT2:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT3:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT4:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_LOOKUP_COLOR:
        {
            mi::mdl::IQualified_name *qname = create_qualified_name(callee_name);
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            mi::mdl::IExpression const *tex_expr = nullptr;

            ASSERT( M_SCENE, n_params > 0);
            for (mi::Size i = 0, n = n_params; i < n; ++i) {
                Handle<IExpression const> arg(args->get_expression(i));

                mi::mdl::IExpression const *expr = transform_expr(arg.get());
                if (i == 0) {
                    tex_expr = expr;
                }

                if (named_args) {
                    mi::mdl::ISimple_name const *sname = to_simple_name(args->get_name(i));
                    call->add_argument(
                        m_ef.create_named_argument(sname, expr));
                } else {
                    call->add_argument(
                        m_ef.create_positional_argument(expr));
                }
            }

            if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_6) {
                if ((n_params == 6 && mi::mdl::is_tex_2d(tex_expr->get_type())) ||
                    (n_params == 8 && mi::mdl::is_tex_3d(tex_expr->get_type())))
                {
                    // MDL 1.6 -> 1.7: insert the frame parameter
                    mi::mdl::IExpression const *expr =
                        m_ef.create_literal(m_vf.create_float(0.0));

                    mi::mdl::IArgument const *arg = nullptr;
                    if (named_args) {
                        arg = m_ef.create_named_argument(to_simple_name("frame"), expr);
                    } else {
                        arg = m_ef.create_positional_argument(expr);
                    }
                    call->add_argument(arg);
                }
            }
            return call;
        }
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT2:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT3:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT4:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXEL_COLOR:
        {
            mi::mdl::IQualified_name *qname
                = create_qualified_name(strip_deprecated_suffix(callee_name));
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            mi::mdl::IExpression const *tex_expr;
            {
                Handle<IExpression const> nr_arg(args->get_expression(zero_size));
                tex_expr = transform_expr(nr_arg.get());

                mi::mdl::IArgument const *arg = nullptr;
                if (named_args) {
                    arg = m_ef.create_named_argument(to_simple_name(args->get_name(0)), tex_expr);
                } else {
                    arg = m_ef.create_positional_argument(tex_expr);
                }
                call->add_argument(arg);
            }

            {
                Handle<IExpression const> nr_arg(args->get_expression(1));
                mi::mdl::IExpression const *expr = transform_expr(nr_arg.get());

                mi::mdl::IArgument const *arg = nullptr;
                if (named_args) {
                    arg = m_ef.create_named_argument(to_simple_name(args->get_name(1)), expr);
                } else {
                    arg = m_ef.create_positional_argument(expr);
                }
                call->add_argument(arg);
            }

            if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_3 && n_params == 2) {
                if (mi::mdl::is_tex_2d(tex_expr->get_type())) {
                    // MDL 1.3 -> 1.4: insert the uv_tile parameter
                    mi::mdl::IExpression const *expr =
                        m_ef.create_literal(mi::mdl::create_int2_zero(m_vf));

                    mi::mdl::IArgument const *arg = nullptr;
                    if (named_args) {
                        arg = m_ef.create_named_argument(to_simple_name("uv_tile"), expr);
                    } else {
                        arg = m_ef.create_positional_argument(expr);
                    }
                    call->add_argument(arg);
                }
            }

            if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_6) {
                if ((n_params == 3 && mi::mdl::is_tex_2d(tex_expr->get_type())) ||
                    (n_params == 2 && mi::mdl::is_tex_3d(tex_expr->get_type())))
                {
                    // MDL 1.6 -> 1.7: insert the frame parameter
                    mi::mdl::IExpression const *expr =
                        m_ef.create_literal(m_vf.create_float(0.0));

                    mi::mdl::IArgument const *arg = nullptr;
                    if (named_args) {
                        arg = m_ef.create_named_argument(to_simple_name("frame"), expr);
                    } else {
                        arg = m_ef.create_positional_argument(expr);
                    }
                    call->add_argument(arg);
                }
            }
            return call;
        }
        break;

    default:
        // no changes
        break;
    }

    switch (sema) {
    case mi::mdl::IDefinition::DS_CONV_CONSTRUCTOR:
    case mi::mdl::IDefinition::DS_CONV_OPERATOR:
    {
        // Create constructor of return type's type
        Handle<IType const> const type(ret_type->skip_all_type_aliases());
        mi::mdl::IType_name const *tn = create_type_name(type);
        mi::mdl::IExpression_reference *tn_ref = m_ef.create_reference(tn);
        mi::mdl::IExpression_call *tn_call = m_ef.create_call(tn_ref);

        Handle<IExpression const> arg(args->get_expression(mi::Size(0)));
        mi::mdl::IExpression const *arg_mdl = transform_expr(arg.get());

        tn_call->add_argument(m_ef.create_positional_argument(arg_mdl));

        return tn_call;
    }
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
        {
            Handle<IExpression const> comp_arg(args->get_expression(mi::Size(0)));
            mi::mdl::IExpression const *compound = transform_expr(comp_arg.get());

            mi::mdl::ISymbol const *f_sym = get_field_sym(callee_name);

            if (f_sym) {
                mi::mdl::IExpression const *member = to_reference(f_sym);
                return m_ef.create_binary(
                    mi::mdl::IExpression_binary::OK_SELECT, compound, member);
            }
            ASSERT( M_SCENE, !"could not retrieve the field from a DAG_FIELD_ACCESS");
            return m_ef.create_invalid();
        }
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR:
        {
            Handle<IType_array const> a_tp(ret_type->get_interface<IType_array>());
            Handle<IType const>       e_tp(a_tp->get_element_type());

            mi::mdl::IType_name *tn = create_type_name(a_tp);
            mi::mdl::IExpression_reference *ref = m_ef.create_reference(tn);
            ref->set_array_constructor();

            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            // for array constructors, we need to take the size from the provided argument list
            for (mi::Size i = 0, n = args->get_size(); i < n; ++i) {
                Handle<IExpression const> arg(
                    args->get_expression(("value" + std::to_string(i)).c_str()));
                mi::mdl::IExpression const *expr = transform_expr(arg.get());
                call->add_argument(m_ef.create_positional_argument(expr));
            }
            return call;
        }
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
        {
            Handle<IExpression const> arg(args->get_expression(mi::Size(0)));
            Handle<IType const>       tp(arg->get_type());
            Handle<IType_array const> a_tp(tp->get_interface<IType_array>());

            if (!a_tp) {
                ASSERT( M_SCENE, false);
                return m_ef.create_invalid();
            }

            if (a_tp->is_immediate_sized()) {
                mi::Size size = a_tp->get_size();

                mi::mdl::IValue_int const *v = m_vf.create_int(int(size));
                return m_ef.create_literal(v);
            } else {
                mi::mdl::IQualified_name *qname = create_qualified_name(a_tp->get_deferred_size());
                return to_reference(qname);
            }
        }
        break;

    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID:
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS:
        // should not occur in a material, reserved for lambdas
        ASSERT( M_SCENE, !"unexpected DAG intrinsic");
        return m_ef.create_invalid();

    case mi::mdl::IDefinition::DS_UNKNOWN:
    default:
        {
            // all other cases:

            mi::mdl::IQualified_name *qname = create_qualified_name(callee_name);
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            for (mi::Size i = 0, n = n_params; i < n; ++i) {
                Handle<IExpression const> arg(args->get_expression(i));

                mi::mdl::IExpression const *expr = transform_expr(arg.get());

                if (named_args) {
                    mi::mdl::ISimple_name const *sname = to_simple_name(args->get_name(i));
                    call->add_argument(
                        m_ef.create_named_argument(sname, expr));
                } else {
                    call->add_argument(
                        m_ef.create_positional_argument(expr));
                }
            }
            return call;
        }
    }
    // gcc believes this is not dead :(
    return m_ef.create_invalid();
}

// Transform a MDL expression from neuray representation to MDL representation.
const mi::mdl::IExpression* Mdl_ast_builder::transform_expr( const IExpression* expr)
{
    auto it( m_param_map.find( make_handle_dup( expr)));
    if( it != m_param_map.end()) {
        // must be mapped
        return to_reference( it->second);
    }

    switch( expr->get_kind()) {

        case IExpression::EK_CONSTANT: {

            Handle<const IExpression_constant> constant(
                expr->get_interface<IExpression_constant>());
            Handle<const IValue> v(constant->get_value());
            return transform_value(v.get());
        }

        case IExpression::EK_CALL: {

            Handle<const IExpression_call> call( expr->get_interface<IExpression_call>());

            DB::Tag tag = call->get_call();
            if( !tag)
                return m_ef.create_invalid();

            Call_stack_guard guard( m_set_indirect_calls, tag);
            if( guard.last_frame_creates_cycle())
                return m_ef.create_invalid();

            Handle<const IType> type( call->get_type());
            mi::mdl::IDefinition::Semantics sema = mi::mdl::IDefinition::DS_UNKNOWN;
            Handle<const IExpression_list> args;
            std::string def;
            bool named_args = false;
            mi::Size n_params = 0;

            SERIAL::Class_id class_id = m_trans->get_class_id( tag);
            if( class_id != ID_MDL_FUNCTION_CALL) {
                ASSERT( M_SCENE, !"invalid type of DB element referenced by indirect call");
                return m_ef.create_invalid();
            }

            DB::Access<Mdl_function_call> fcall( tag, m_trans);
            DB::Tag def_tag = fcall->get_function_definition( m_trans);
            if( !def_tag)
                return m_ef.create_invalid();
            DB::Access<Mdl_function_definition> fdef( def_tag, m_trans);
            def = fdef->get_mdl_name_without_parameter_types();

            // if reexported, use the original name
            const char* original_mdl_name = fdef->get_mdl_original_name();
            if( original_mdl_name) {
                DB::Tag orig_tag = m_trans->name_to_tag(
                    get_db_name( original_mdl_name).c_str());
                DB::Access<Mdl_function_definition> orig_fdef( orig_tag, m_trans);
                def = orig_fdef->get_mdl_name_without_parameter_types();
            }

            def        = decode_name_without_signature( def);
            named_args = fdef->is_material();
            sema       = fdef->get_core_semantic();
            args       = fcall->get_arguments();
            n_params   = fcall->get_parameter_count();

            def = remove_qualifiers_if_from_module( def, m_owner->get_name());
            return transform_call( type.get(), sema, def, n_params, args.get(), named_args);
        }

        case IExpression::EK_DIRECT_CALL: {

            Handle<const IExpression_direct_call> dcall(
                expr->get_interface<IExpression_direct_call>());

            DB::Tag tag = dcall->get_definition( m_trans);
            if( !tag)
                return m_ef.create_invalid();

            Handle<const IType> type( dcall->get_type());
            mi::mdl::IDefinition::Semantics sema = mi::mdl::IDefinition::DS_UNKNOWN;
            Handle<const IExpression_list> args( dcall->get_arguments());
            std::string def;
            bool named_args = false;
            mi::Size n_params = 0;

            DB::Access<Mdl_function_definition> fdef( tag, m_trans);
            def = fdef->get_mdl_name_without_parameter_types();

            // if reexported, use the original
            const char* original_mdl_name = fdef->get_mdl_original_name();
            if( original_mdl_name) {
                DB::Tag orig_tag = m_trans->name_to_tag(
                    get_db_name( original_mdl_name).c_str());
                DB::Access<Mdl_function_definition> orig_fdef( orig_tag, m_trans);
                def = orig_fdef->get_mdl_name_without_parameter_types();

            }

            def        = decode_name_without_signature( def);
            named_args = fdef->is_material();
            sema       = fdef->get_core_semantic();
            n_params   = fdef->get_parameter_count();

            def = remove_qualifiers_if_from_module( def, m_owner->get_name());
            return transform_call( type.get(), sema, def, n_params, args.get(), named_args);
        }

        case IExpression::EK_PARAMETER: {

            Handle<const IExpression_parameter> parameter(
                expr->get_interface<IExpression_parameter>());
            mi::Size index = parameter->get_index();

            if( m_traverse_ek_parameter) {
                Handle<const IExpression> arg( m_args->get_expression(index));
                if( !arg) {
                    ASSERT( M_SCENE, !"parameter has no argument");
                    break;
                }
                return transform_expr(arg.get());
            } else {
                if( index >= m_param_vector.size()) {
                    ASSERT( M_SCENE, !"parameter has no argument");
                    break;
                }
                return to_reference( m_param_vector[index]);
            }
        }

        case IExpression::EK_TEMPORARY: {

            Handle<const IExpression_temporary> temporary(
                expr->get_interface<IExpression_temporary>());
            mi::Size index = temporary->get_index();
            if( index >= m_temporaries.size()) {
                ASSERT( M_SCENE, !"invalid temporary reference");
                break;
            }

            return to_reference( m_temporaries[index]);
        }
    }

    ASSERT( M_SCENE, !"unexpected expression kind");
    return m_ef.create_invalid();
}

namespace {

/// Get the texture resource name of a tag.
std::string get_texture_resource_name_gamma_selector(
    DB::Transaction* trans,
    DB::Tag tag,
    mi::mdl::IValue_texture::gamma_mode &gamma_mode,
    std::string& selector)
{
    gamma_mode = mi::mdl::IValue_texture::gamma_default;
    selector = "";

    SERIAL::Class_id class_id = trans->get_class_id( tag);
    if( class_id != TEXTURE::Texture::id) {
        const char* name = trans->tag_to_name( tag);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Incorrect type for texture resource \"%s\".", name ? name : "");
        return {};
    }

    DB::Access<TEXTURE::Texture> texture( tag, trans);
    DB::Tag image_tag( texture->get_image());

    if( image_tag.is_valid()) {
        class_id = trans->get_class_id( image_tag);
        if( class_id != DBIMAGE::Image::id) {
            const char* name = trans->tag_to_name( image_tag);
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "Incorrect type for image resource \"%s\".", name ? name : "");
            return {};
        }

        DB::Access<DBIMAGE::Image> image( image_tag, trans);

        // try to convert gamma value into the MDL enum
        gamma_mode = convert_gamma_float_to_enum( texture->get_gamma());
        selector = image->get_selector();

        const std::string& s1 = image->get_mdl_file_path();
        if( !s1.empty())
            return s1;
        const std::string& s2 = image->get_filename( 0, 0);
        if( s2.empty())
            return {};
        const std::string& s3 = get_file_path(
            s2, mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_FIRST);
        // Do not use the filename for animated textures/uvtiles since it only identifies the first
        // frame/uvtile.
        if( !s3.empty() && !image->is_uvtile() && !image->is_animated())
            return s3;
        return {};
    }
    return {};
}

/// Get the light_profile resource name of a tag.
std::string get_light_profile_resource_name(
    DB::Transaction* trans,
    DB::Tag tag)
{
    SERIAL::Class_id class_id = trans->get_class_id( tag);
    if( class_id != LIGHTPROFILE::Lightprofile::id) {
        const char* name = trans->tag_to_name( tag);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Incorrect type for light profile resource \"%s\".", name ? name : "");
        return {};
    }
    DB::Access<LIGHTPROFILE::Lightprofile> lightprofile( tag, trans);
    const std::string& s1 = lightprofile->get_mdl_file_path();
    if( !s1.empty())
        return s1;
    const std::string& s2 = lightprofile->get_filename();
    if( s2.empty())
        return {};
    const std::string& s3 = get_file_path(
        s2, mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_FIRST);
    if( !s3.empty())
        return s3;
    return {};
}

/// Get the bsdf_measurement resource name of a tag.
std::string get_bsdf_measurement_resource_name(
    DB::Transaction* trans,
    DB::Tag tag)
{
    SERIAL::Class_id class_id = trans->get_class_id( tag);
    if( class_id != BSDFM::Bsdf_measurement::id) {
        const char* name = trans->tag_to_name( tag);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Incorrect type for BSDF measurement resource \"%s\".", name ? name : "");
        return {};
    }
    DB::Access<BSDFM::Bsdf_measurement> bsdf_measurement( tag, trans);
    const std::string& s1 = bsdf_measurement->get_mdl_file_path();
    if( !s1.empty())
        return s1;
    const std::string& s2 = bsdf_measurement->get_filename();
    if( s2.empty())
        return {};
    const std::string& s3 = get_file_path(
        s2, mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_FIRST);
    if( !s3.empty())
        return s3;
    return {};
}

} // namespace

// Transform a MDL expression from neuray representation to MDL representation.
mi::mdl::IExpression const *Mdl_ast_builder::transform_value( const IValue* value)
{
    IValue::Kind kind = value->get_kind();
    switch (kind) {
    case IValue::VK_BOOL:
        {
            Handle<IValue_bool const> v(value->get_interface<IValue_bool>());

            mi::mdl::IValue const *vv = m_vf.create_bool(v->get_value());
            return m_ef.create_literal(vv);
        }
    case IValue::VK_INT:
        {
            Handle<IValue_int const> v(value->get_interface<IValue_int>());

            mi::mdl::IValue const *vv = m_vf.create_int(v->get_value());
            return m_ef.create_literal(vv);
        }
    case IValue::VK_ENUM:
        {
            Handle<IValue_enum const> v(value->get_interface<IValue_enum>());
            Handle<IType_enum const> e_tp(v->get_type());

            mi::Size index = v->get_index();
            char const *v_name = e_tp->get_value_name(index);
            mi::mdl::ISimple_name const *sname = create_simple_name(v_name);

            std::string symbol = decode_name_without_signature(e_tp->get_symbol());
            // work-around: do not qualify enum value from the current module
            symbol = remove_qualifiers_if_from_module(symbol, m_owner->get_name());

            mi::mdl::IQualified_name *qname = create_scope_name(symbol);
            qname->add_component(sname);

            mi::mdl::IType_enum const *core_e_tp = convert_enum_type(e_tp.get());
            return to_reference(qname, core_e_tp);
        }
    case IValue::VK_FLOAT:
        {
            Handle<IValue_float const> v(value->get_interface<IValue_float>());

            mi::mdl::IValue const *vv = m_vf.create_float(v->get_value());
            return m_ef.create_literal(vv);
        }
    case IValue::VK_DOUBLE:
        {
            Handle<IValue_double const> v(value->get_interface<IValue_double>());

            mi::mdl::IValue const *vv = m_vf.create_double(v->get_value());
            return m_ef.create_literal(vv);
        }
    case IValue::VK_STRING:
        {
            {
                Handle<IValue_string_localized const> v(
                    value->get_interface<IValue_string_localized>());
                if (v) {
                    mi::mdl::IValue const *vv = m_vf.create_string(v->get_original_value());
                    return m_ef.create_literal(vv);
                }
            }
            Handle<IValue_string const> v(value->get_interface<IValue_string>());

            mi::mdl::IValue const *vv = m_vf.create_string(v->get_value());
            return m_ef.create_literal(vv);
        }
    case IValue::VK_VECTOR:
    case IValue::VK_MATRIX:
    case IValue::VK_COLOR:
    case IValue::VK_STRUCT:
        // handle compound types as calls
        {
            Handle<IValue_compound const> v(value->get_interface<IValue_compound>());
            Handle<IType_compound const> c_tp(v->get_type());

            mi::mdl::IType_name *tn             = create_type_name(c_tp);
            mi::mdl::IExpression_reference *ref = m_ef.create_reference(tn);

            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            mi::Size n = v->get_size();
            if (kind == IValue::VK_STRUCT) {
                mi::mdl::IMDL::MDL_version version = m_owner->get_mdl_version();

                if (version < mi::mdl::IMDL::MDL_VERSION_1_7) {
                    Handle<IValue_struct const> v(value->get_interface<IValue_struct>());
                    Handle<IType_struct const> s_tp(v->get_type());

                    if (s_tp->get_predefined_id() == IType_struct::SID_MATERIAL_VOLUME) {
                        // material_volume has no emission_intensity field
                        --n;
                    }
                    if (version < mi::mdl::IMDL::MDL_VERSION_1_1 &&
                        s_tp->get_predefined_id() == IType_struct::SID_MATERIAL_EMISSION)
                    {
                        // material_emission has no intensity_mode field
                        --n;
                    }
                }
            }

            for (mi::Size i = 0; i < n; ++i) {
                Handle<IValue const> e_v(v->get_value(i));

                call->add_argument(m_ef.create_positional_argument(transform_value(e_v.get())));
            }
            return call;
        }
    case IValue::VK_ARRAY:
        // create an array constructor
        {
            Handle<IValue_array const> v(value->get_interface<IValue_array>());
            Handle<IType_array const> a_tp(v->get_type());
            Handle<IType const> e_tp(a_tp->get_element_type());

            mi::mdl::IType_name *tn = create_type_name(e_tp);
            tn->set_incomplete_array();

            mi::mdl::IExpression_reference *ref  = m_ef.create_reference(tn);
            mi::mdl::IExpression_call      *call = m_ef.create_call(ref);

            for (mi::Size i = 0, n = v->get_size(); i < n; ++i) {
                Handle<IValue const> e_v(v->get_value(i));

                call->add_argument(m_ef.create_positional_argument(transform_value(e_v.get())));
            }
            return call;
        }
        break;
    case IValue::VK_INVALID_DF:
        {
            Handle<IValue_invalid_df const> v(value->get_interface<IValue_invalid_df>());
            Handle<IType_reference const> type(v->get_type());

            auto const *r_tp = cast<mi::mdl::IType_reference>(transform_type(type.get()));
            mi::mdl::IValue_invalid_ref const *vv = m_vf.create_invalid_ref(r_tp);
            return m_ef.create_literal(vv);
        }
    case IValue::VK_TEXTURE:
        // create an texture constructor
        {
            Handle<IValue_texture const> v(value->get_interface<IValue_texture>());
            Handle<IType_texture const> type(v->get_type());
            mi::mdl::IType_name            *tn = create_type_name(type);
            mi::mdl::IExpression_reference *ref  = m_ef.create_reference(tn);
            mi::mdl::IExpression_call      *call = m_ef.create_call(ref);

            DB::Tag tag = v->get_value();
            SERIAL::Class_id class_id = tag ? m_trans->get_class_id( tag) : 0;

            // neuray sometimes creates wrong textures with TAG 0, handle them
            if (tag.is_invalid() || class_id != TEXTURE::Texture::id) {
                auto const *r_tp = cast<mi::mdl::IType_reference>(transform_type(type.get()));
                mi::mdl::IValue_invalid_ref const *vv = m_vf.create_invalid_ref(r_tp);
                return m_ef.create_literal(vv);
            }

            mi::mdl::IValue_texture::gamma_mode gamma = mi::mdl::IValue_texture::gamma_default;
            std::string selector;
            std::string url = get_texture_resource_name_gamma_selector(
                m_trans, tag, gamma, selector);
            if (m_avoid_resource_urls || url.empty()) {
                // map to IValue with tag
                DB::Access<TEXTURE::Texture> texture(tag, m_trans);
                DB::Tag image_volume_tag(
                     texture->get_image() ? texture->get_image() : texture->get_volume_data());
                DB::Tag_version image_volume_tag_version
                    = m_trans->get_tag_version(image_volume_tag);
                auto const *t_tp = cast<mi::mdl::IType_texture>(transform_type(type.get()));
                DB::Tag_version tag_version = m_trans->get_tag_version(tag);
                mi::mdl::IValue_texture const *vv = m_vf.create_texture(
                    t_tp, "", gamma, selector.c_str(), tag.get_uint(),
                    get_hash(/*mdl_file_path*/ nullptr, 0.0f, tag_version,
                    image_volume_tag_version));
                return m_ef.create_literal(vv);
            }

            // create arg0: url
            {
                mi::mdl::IValue const *s   = m_vf.create_string(url.c_str());
                mi::mdl::IExpression  *lit = m_ef.create_literal(s);

                call->add_argument(m_ef.create_positional_argument(lit));
            }

            // create arg1: gamma
            {
                mi::mdl::ISymbol const *sym = nullptr;
                switch (gamma) {
                case mi::mdl::IValue_texture::gamma_default:
                    sym = m_st.create_symbol("gamma_default");
                    break;
                case mi::mdl::IValue_texture::gamma_linear:
                    sym = m_st.create_symbol("gamma_linear");
                    break;
                case mi::mdl::IValue_texture::gamma_srgb:
                    sym = m_st.create_symbol("gamma_srgb");
                    break;
                }
                if (!sym) {
                    ASSERT( M_SCENE, !"unexpected gamma mode");
                    sym = m_st.get_error_symbol();
                }

                mi::mdl::ISymbol const      *t_sym   = m_st.create_symbol("tex");
                mi::mdl::ISimple_name const *t_sname = m_nf.create_simple_name(t_sym);
                mi::mdl::ISimple_name const *g_sname = m_nf.create_simple_name(sym);
                mi::mdl::IQualified_name    *qname   = m_nf.create_qualified_name();

                // ::tex::gamma_*
                qname->add_component(t_sname);
                qname->add_component(g_sname);
                qname->set_absolute();

                // set the type so the name importer can handle it
                mi::mdl::IType_enum const      *e_tp =
                    m_tf.get_predefined_enum(mi::mdl::IType_enum::EID_TEX_GAMMA_MODE);
                mi::mdl::IExpression_reference *ref  = to_reference(qname, e_tp);

                call->add_argument(m_ef.create_positional_argument(ref));
            }

            // optionally create arg2: selector
            if( !selector.empty()) {
                mi::mdl::IValue const *s   = m_vf.create_string(selector.c_str());
                mi::mdl::IExpression  *lit = m_ef.create_literal(s);

                call->add_argument(m_ef.create_positional_argument(lit));
            }

            return call;
        }
    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        // create an resource constructor
        {
            Handle<IValue_resource const> v(value->get_interface<IValue_resource>());
            Handle<IType_resource const> type(v->get_type());

            mi::mdl::IType_name            *tn = create_type_name(type);
            mi::mdl::IExpression_reference *ref  = m_ef.create_reference(tn);
            mi::mdl::IExpression_call      *call = m_ef.create_call(ref);

            // neuray sometimes creates invalid resources with TAG 0, handle them
            DB::Tag tag = v->get_value();
            if (tag.is_invalid()) {
                auto const *r_tp = cast<mi::mdl::IType_reference>(transform_type(type.get()));
                mi::mdl::IValue_invalid_ref const *vv = m_vf.create_invalid_ref(r_tp);
                return m_ef.create_literal(vv);
            }

            std::string url(kind == IValue::VK_LIGHT_PROFILE ?
                get_light_profile_resource_name(m_trans, tag) :
                get_bsdf_measurement_resource_name(m_trans, tag));
            if (m_avoid_resource_urls || url.empty()) {
                // map to IValue with tag
                DB::Tag_version tag_version = m_trans->get_tag_version(tag);
                if (kind == IValue::VK_LIGHT_PROFILE) {
                    auto const *lp_tp =
                        cast<mi::mdl::IType_light_profile>(transform_type(type.get()));
                    mi::mdl::IValue_light_profile const *vv = m_vf.create_light_profile(
                        lp_tp, "", tag.get_uint(),
                        get_hash( /*mdl_file_path*/ nullptr, tag_version));
                    return m_ef.create_literal(vv);
                } else {
                    auto const *bm_tp =
                        cast<mi::mdl::IType_bsdf_measurement>(transform_type(type.get()));
                    mi::mdl::IValue_bsdf_measurement const *vv = m_vf.create_bsdf_measurement(
                        bm_tp, "", tag.get_uint(),
                        get_hash( /*mdl_file_path*/ nullptr, tag_version));
                    return m_ef.create_literal(vv);
                }
            }

            // create arg0: url
            {
                mi::mdl::IValue const *s = m_vf.create_string(url.c_str());
                mi::mdl::IExpression  *lit = m_ef.create_literal(s);

                call->add_argument(m_ef.create_positional_argument(lit));
            }
            return call;
        }
    }
    ASSERT( M_SCENE, !"unexpected value kind");
    return m_ef.create_invalid();
}

// Transform a (non-user defined) MDL type from neuray representation to MDL representation.
mi::mdl::IType const *Mdl_ast_builder::transform_type( const IType* type)
{
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
    case IType::TK_ENUM:
    case IType::TK_ARRAY:
    case IType::TK_STRUCT:
        // user defined types should not be used here
        ASSERT( M_SCENE, !"user defined types not allowed here");
        return nullptr;
    case IType::TK_BOOL:
        return m_tf.create_bool();
    case IType::TK_INT:
        return m_tf.create_int();
    case IType::TK_FLOAT:
        return m_tf.create_float();
    case IType::TK_DOUBLE:
        return m_tf.create_double();
    case IType::TK_STRING:
        return m_tf.create_string();
    case IType::TK_VECTOR:
        {
            Handle<IType_vector const> v_tp(type->get_interface<IType_vector>());
            Handle<IType const>        e_tp(v_tp->get_element_type());

            auto const *a_tp = cast<mi::mdl::IType_atomic>(transform_type(e_tp.get()));
            return m_tf.create_vector(a_tp, int(v_tp->get_size()));
        }
    case IType::TK_MATRIX:
        {
            Handle<IType_matrix const> m_tp(type->get_interface<IType_matrix>());
            Handle<IType const>        e_tp(m_tp->get_element_type());

            auto const *v_tp = cast<mi::mdl::IType_vector>(transform_type(e_tp.get()));
            return m_tf.create_matrix(v_tp, int(m_tp->get_size()));
        }
    case IType::TK_COLOR:
        return m_tf.create_color();
    case IType::TK_TEXTURE:
        {
            Handle<IType_texture const> t_tp(type->get_interface<IType_texture>());

            switch (t_tp->get_shape()) {
            case IType_texture::TS_2D:
                return m_tf.create_texture(mi::mdl::IType_texture::TS_2D);
            case IType_texture::TS_3D:
                return m_tf.create_texture(mi::mdl::IType_texture::TS_3D);
            case IType_texture::TS_CUBE:
                return m_tf.create_texture(mi::mdl::IType_texture::TS_CUBE);
            case IType_texture::TS_PTEX:
                return m_tf.create_texture(mi::mdl::IType_texture::TS_PTEX);
            case IType_texture::TS_BSDF_DATA:
                return m_tf.create_texture(mi::mdl::IType_texture::TS_BSDF_DATA);
            }
        }
        break;
    case IType::TK_LIGHT_PROFILE:
        return m_tf.create_light_profile();
    case IType::TK_BSDF_MEASUREMENT:
        return m_tf.create_bsdf_measurement();
    case IType::TK_BSDF:
        return m_tf.create_bsdf();
    case IType::TK_HAIR_BSDF:
        return m_tf.create_hair_bsdf();
    case IType::TK_EDF:
        return m_tf.create_edf();
    case IType::TK_VDF:
        return m_tf.create_vdf();
    }
    ASSERT( M_SCENE, !"unsupported type kind");
    return m_tf.create_error();
}

// Create a new temporary name.
mi::mdl::ISymbol const *Mdl_ast_builder::get_temporary_symbol()
{
    char buffer[32];

    snprintf(buffer, dimension_of(buffer), "tmp%u", m_tmp_idx++);
    buffer[dimension_of(buffer) - 1] = '\0';

    // TODO check for name clashes here
    return m_st.get_symbol(buffer);
}

// Create a new temporary name.
mi::mdl::ISimple_name const *Mdl_ast_builder::to_simple_name(mi::mdl::ISymbol const *sym)
{
    return m_nf.create_simple_name(sym);
}

// Create a simple name for a given name.
mi::mdl::ISimple_name const *Mdl_ast_builder::to_simple_name(char const *name)
{
    mi::mdl::ISymbol const *sym = m_st.get_symbol(name);
    return to_simple_name(sym);
}

// Create a reference expression for a qualified name.
mi::mdl::IExpression_reference *Mdl_ast_builder::to_reference(
    mi::mdl::IQualified_name *qname,
    const mi::mdl::IType* type)
{
    mi::mdl::IType_name *tn = m_nf.create_type_name(qname);
    if (type)
        tn->set_type(type);
    mi::mdl::IExpression_reference *ref = m_ef.create_reference(tn);
    if (type)
        ref->set_type(type);
    return ref;
}

// Create a reference expression for a given Symbol.
mi::mdl::IExpression_reference *Mdl_ast_builder::to_reference(mi::mdl::ISymbol const *sym)
{
    mi::mdl::ISimple_name const *sname = to_simple_name(sym);
    mi::mdl::IQualified_name *qname = m_nf.create_qualified_name();

    qname->add_component(sname);
    return to_reference(qname);
}

// Declare a parameter.
void Mdl_ast_builder::declare_parameter(
    mi::mdl::ISymbol const *sym,
    const IExpression *init)
{
    m_param_map[make_handle_dup(init)] = sym;
    m_param_vector.push_back(sym);
}

// Remove all declared parameter mappings.
void Mdl_ast_builder::remove_parameters()
{
    m_param_map.clear();
    m_param_vector.clear();
}

void Mdl_ast_builder::add_temporary( const mi::mdl::ISymbol* sym)
{
    m_temporaries.push_back( sym);
}

// Convert an neuray enum type into a MDL enum type.
mi::mdl::IType_enum const *Mdl_ast_builder::convert_enum_type(
    IType_enum const *e_tp)
{
    switch (e_tp->get_predefined_id()) {
    case IType_enum::EID_USER:
        {
            std::string symbol = decode_name_without_signature(e_tp->get_symbol());
            symbol = m_name_mangler.mangle_scoped_name(symbol);
            if (mi::mdl::IType_enum const *et = m_tf.lookup_enum(symbol.c_str())) {
                // an enum with this name already exists, assume it's the right one
                return et;
            }

            mi::mdl::ISymbol const *sym = m_st.get_user_type_symbol(symbol.c_str());
            size_t n = e_tp->get_size();
            MDL::Small_VLA<mi::mdl::IType_enum::Value, 8> values(n);
            for (mi::Size i = 0; i < n; ++i) {
                mi::mdl::ISymbol const *v_sym = m_st.get_symbol(e_tp->get_value_name(i));
                mi::Sint32             v_code = e_tp->get_value_code(i, nullptr);

                values[i] = mi::mdl::IType_enum::Value(v_sym, v_code);
            }
            return m_tf.create_enum(sym, values.data(), values.size());
    }
    case IType_enum::EID_TEX_GAMMA_MODE:
        return m_tf.get_predefined_enum(mi::mdl::IType_enum::EID_TEX_GAMMA_MODE);
    case IType_enum::EID_INTENSITY_MODE:
        return m_tf.get_predefined_enum(mi::mdl::IType_enum::EID_INTENSITY_MODE);
    }
    ASSERT( M_SCENE, !"unexpected enum type ID");
    return nullptr;
}

mi::mdl::IExpression const *Mdl_ast_builder::add_multiscatter_param(
     std::string const     &callee_name,
     mi::Size              n_params,
     bool                  named_args,
     const IExpression_list *args)
 {
     // MDL 1.5 -> 1.6: insert multiscatter_tint parameter to all glossy BSDFs
     mi::mdl::IQualified_name *qname
         = create_qualified_name(strip_deprecated_suffix(callee_name));
     mi::mdl::IExpression_reference *ref = to_reference(qname);
     mi::mdl::IExpression_call *call = m_ef.create_call(ref);

     for (mi::Size i = 0; i < n_params; ++i) {
         mi::mdl::IArgument const *arg = nullptr;

         if (i == 3) {
             // insert the multiscatter_tint parameter
             mi::mdl::IValue_float const *zero = m_vf.create_float(0.0f);
             mi::mdl::IExpression const *expr =
                 m_ef.create_literal(m_vf.create_rgb_color(zero, zero, zero));
             if (named_args) {
                 arg = m_ef.create_named_argument(to_simple_name("multiscatter_tint"), expr);
             } else {
                 arg = m_ef.create_positional_argument(expr);
             }
             call->add_argument(arg);
         }

         Handle<IExpression const> nr_arg(args->get_expression(i));
         mi::mdl::IExpression const *expr = transform_expr(nr_arg.get());

         if (named_args) {
             arg = m_ef.create_named_argument(to_simple_name(args->get_name(i)), expr);
         } else {
             arg = m_ef.create_positional_argument(expr);
         }
         call->add_argument(arg);
     }
     return call;
 }

} // namespace MDL
} // namespace MI

