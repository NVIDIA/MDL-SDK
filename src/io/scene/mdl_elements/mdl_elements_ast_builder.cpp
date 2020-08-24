/***************************************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Source for building MDL AST from neuray expressions/types

#include "pch.h"

#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_modules.h>

#include "base/lib/log/i_log_assert.h"
#include "base/lib/log/i_log_logger.h"

#include "base/data/db/i_db_transaction.h"
#include "base/data/db/i_db_access.h"
#include "base/data/db/i_db_tag.h"

#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_modules.h"
#include "mdl/compiler/compilercore/compilercore_symbols.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"

#include "io/scene/texture/i_texture.h"
#include "io/scene/dbimage/i_dbimage.h"
#include "io/scene/lightprofile/i_lightprofile.h"
#include "io/scene/bsdf_measurement/i_bsdf_measurement.h"

#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_function_definition.h"
#include "i_mdl_elements_material_definition.h"
#include "i_mdl_elements_material_instance.h"
#include "mdl_elements_ast_builder.h"

// Disable false positives (claiming expressions involving "class_id" always being true or false)
//-V:class_id:547 PVS

namespace MI {
namespace MDL {

using mi::mdl::dimension_of;
using mi::mdl::impl_cast;
using mi::mdl::cast;

using mi::base::Handle;

/// Get the name of an neuray type.
///
/// \param type  the type
static std::string get_type_name(
    const Handle<const IType>& type)
{
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
    case IType::TK_FORCE_32_BIT:
        // should not happen
        ASSERT( M_SCENE, !"unexpected type kind");
        return "";

    case IType::TK_BOOL:
        return "bool";
    case IType::TK_INT:
        return "int";
    case IType::TK_ENUM:
        {
            Handle<const IType_enum> e_tp(type->get_interface<IType_enum>());
            return e_tp->get_symbol();
        }
        break;
    case IType::TK_FLOAT:
        return "float";
    case IType::TK_DOUBLE:
        return "double";
    case IType::TK_STRING:
        return "string";
    case IType::TK_LIGHT_PROFILE:
        return "light_profile";
    case IType::TK_BSDF:
        return "bsdf";
    case IType::TK_HAIR_BSDF:
        return "hair_bsdf";
    case IType::TK_EDF:
        return "edf";
    case IType::TK_VDF:
        return "vdf";
    case IType::TK_VECTOR:
        {
            Handle<const IType_vector> v_tp(type->get_interface<IType_vector>());
            Handle<const IType>        a_tp(v_tp->get_element_type());

            std::string res = get_type_name(a_tp);
            char buf[16];
            snprintf(buf, dimension_of(buf), "%u", unsigned(v_tp->get_size()));

            return res + buf;
        }
        break;
    case IType::TK_MATRIX:
        {
            Handle<const IType_matrix> m_tp(type->get_interface<IType_matrix>());
            Handle<const IType_vector> v_tp(m_tp->get_element_type());
            Handle<const IType>        a_tp(v_tp->get_element_type());

            std::string res = get_type_name(a_tp);

            char buf[32];
            snprintf(buf, dimension_of(buf), "%ux%u",
                unsigned(m_tp->get_size()),  // cols
                unsigned(v_tp->get_size())); // rows

            return res + buf;
        }
        break;
    case IType::TK_COLOR:
        return "color";
    case IType::TK_STRUCT:
        {
            Handle<const IType_struct> s_tp(type->get_interface<IType_struct>());

            switch (s_tp->get_predefined_id()) {
            default:
            case IType_struct::SID_USER:
                return s_tp->get_symbol();
            case IType_struct::SID_MATERIAL_EMISSION:
                return "material_emission";
            case IType_struct::SID_MATERIAL_SURFACE:
                return "material_surface";
            case IType_struct::SID_MATERIAL_VOLUME:
                return "material_volume";
            case IType_struct::SID_MATERIAL_GEOMETRY:
                return "material_geometry";
            case IType_struct::SID_MATERIAL:
                return "material";
            }
        }
        break;
    case IType::TK_TEXTURE:
        {
            Handle<const IType_texture> t_tp(type->get_interface<IType_texture>());

            switch (t_tp->get_shape()) {
            case IType_texture::TS_2D:
                return "texture_2d";
            case IType_texture::TS_3D:
                return "texture_3d";
            case IType_texture::TS_CUBE:
                return "texture_cube";
            case IType_texture::TS_PTEX:
                return "texture_ptex";
            case IType_texture::TS_BSDF_DATA:
                ASSERT( M_SCENE, !"bsdf data tables cannot be expressed in MDL source");
                return "";
            case IType_texture::TS_FORCE_32_BIT:
                break;
            }
            ASSERT( M_SCENE, !"unexpected texture shape");
            return "";
        }
        break;
    case IType::TK_BSDF_MEASUREMENT:
        return "bsdf_measurement";
    case IType::TK_ARRAY:
        {
            Handle<const IType_array> a_tp(type->get_interface<IType_array>());
            Handle<const IType>       e_tp(a_tp->get_element_type());

            std::string res = get_type_name(e_tp) + '[';

            if (a_tp->is_immediate_sized()) {
                char buf[16];
                snprintf(buf, dimension_of(buf), "%u", unsigned(a_tp->get_size()));
                res += buf;
            } else {
                res += a_tp->get_deferred_size();
            }
            return res + ']';
        }
        break;
    }
    ASSERT( M_SCENE, !"unexpected type kind");
    return "";
}

Mdl_ast_builder::Mdl_ast_builder(
    mi::mdl::IModule *owner,
    DB::Transaction *transaction,
    const mi::base::Handle<IExpression_list const>& args,
    Name_mangler& name_mangler)
: m_owner(impl_cast<mi::mdl::Module>(owner))
, m_trans(transaction)
, m_nf(*m_owner->get_name_factory())
, m_vf(*m_owner->get_value_factory())
, m_ef(*m_owner->get_expression_factory())
, m_tf(*m_owner->get_type_factory())
, m_st(m_owner->get_symbol_table())
, m_tmp_idx(0u)
, m_param_map()
, m_args(args)
, m_used_user_types()
, m_owner_version(m_owner->get_mdl_version())
, m_name_mangler(name_mangler)
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
    if( is_absolute( name) && !is_in_module( name, "::<builtins>"))
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

    if( simple_module_name != "<builtins>") {
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

    if( is_absolute( name) && !is_in_module( name, "::<builtins>"))
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

    if( simple_module_name != "<builtins>") {
        const mi::mdl::ISimple_name* simple_name
            = create_simple_name( m_name_mangler.mangle( simple_module_name.c_str()));
        qualified_name->add_component( simple_name);
    }

    return qualified_name;
}

// Construct a Type_name AST element for a neuray type.
mi::mdl::IType_name *Mdl_ast_builder::create_type_name(
    Handle<IType const> const &t)
{
    mi::Uint32 modifiers = t->get_all_type_modifiers();
    Handle<IType const> const type(t->skip_all_type_aliases());

    switch (type->get_kind()) {
    case IType::TK_ALIAS:
    case IType::TK_FORCE_32_BIT:
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
            std::string name(get_type_name(type));

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
                char const *size = a_tp->get_deferred_size();

                // Note: in the MDL AST, a deferred size has two "names".
                // The fully qualified name, the identifies every deferred size symbol uniquely
                // and the symbol itself, which is typically the last component of the
                // fully qualified name.
                //
                // In neuray, only the fully qualified name exists.
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
                // have to way to find a unique symbol here. For that, all symbols must be known
                // in advance.
                //
                // A possible solution would be to "defer" the name until all symbols of the whole
                // module are created, then "fix" them by replacing with unique ones.
                //
                // So as a "mostly working" work--around we just take the last component and hope
                // the best that no complex operation merges different types...
                char const *p = strrchr(size, ':');
                if (p != nullptr) {
                    size = p + 1;
                }

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
    ASSERT(M_SCENE, def.size() > 0 && def[def.size()-1] != ')');
    std::string field = get_mdl_field_name(def);
    return m_st.get_symbol(field.c_str());
}

static size_t const zero_size = 0u;

// Transform a call.
mi::mdl::IExpression const *Mdl_ast_builder::transform_call(
    Handle<IType const> const            &ret_type,
    mi::mdl::IDefinition::Semantics      sema,
    std::string const                    &callee_name,
    mi::Size                             n_params,
    Handle<IExpression_list const> const &args,
    bool                                 named_args)
{
    bool is_material_type = false;
    Handle<IType const> const type(ret_type->skip_all_type_aliases());
    if (type->get_kind() == IType::TK_STRUCT) {
        Handle<IType_struct const> s_tp(type->get_interface<IType_struct>());

        IType_struct::Predefined_id id = s_tp->get_predefined_id();
        if (id == IType_struct::SID_USER) {
            m_used_user_types.push_back(s_tp->get_symbol());
        } else if (id == IType_struct::SID_MATERIAL) {
            is_material_type = true;
        }
    } else if (type->get_kind() == IType::TK_ENUM) {
        Handle<IType_enum const> e_tp(type->get_interface<IType_enum>());

        if (e_tp->get_predefined_id() != IType_enum::EID_INTENSITY_MODE) {
            // only intensity_mode is predefined
            m_used_user_types.push_back(e_tp->get_symbol());
        }
    }

    if (semantic_is_operator(sema)) {
        mi::mdl::IExpression::Operator op = semantic_to_operator(sema);

        if (mi::mdl::is_unary_operator(op)) {

            Handle<IExpression const> un_arg(args->get_expression(mi::Size(0)));
            mi::mdl::IExpression const *arg = transform_expr(un_arg);

            mi::mdl::IExpression_unary* res = m_ef.create_unary(
                mi::mdl::IExpression_unary::Operator(op), arg);

            if (op == mi::mdl::IExpression::OK_CAST) {
                const mi::mdl::IType* type = int_type_to_mdl_type(
                    ret_type.get(),
                    m_tf);

                const mi::mdl::IType_name* tn = type_to_type_name(m_owner, type);
                res->set_type_name(tn);
            }
            return res;
        } else if (mi::mdl::is_binary_operator(op)) {
            mi::mdl::IExpression_binary::Operator bop = mi::mdl::IExpression_binary::Operator(op);

            Handle<IExpression const> l_arg(args->get_expression(mi::Size(0)));
            Handle<IExpression const> r_arg(args->get_expression(mi::Size(1)));

            mi::mdl::IExpression const *l = transform_expr(l_arg);
            mi::mdl::IExpression const *r = transform_expr(r_arg);

            return m_ef.create_binary(bop, l, r);
        } else if (op == mi::mdl::IExpression::OK_TERNARY) {
            // C-like ternary operator with lazy evaluation
            Handle<IExpression const> cond_arg(args->get_expression(mi::Size(0)));
            Handle<IExpression const> true_arg(args->get_expression(mi::Size(1)));
            Handle<IExpression const> false_arg(args->get_expression(mi::Size(2)));

            mi::mdl::IExpression const *cond      = transform_expr(cond_arg);
            mi::mdl::IExpression const *true_res  = transform_expr(true_arg);
            mi::mdl::IExpression const *false_res = transform_expr(false_arg);

            return m_ef.create_conditional(cond, true_res, false_res);
        }
    }

    // do MDL 1.X => MDL 1.Y conversion here
    switch (sema) {
    case mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR:
        if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_4 && n_params == 6 && is_material_type) {
            // MDL 1.4 -> 1.5: add default hair bsdf
            mi::mdl::IQualified_name *tu_qname = create_qualified_name("hair_bsdf");
            mi::mdl::IExpression_reference *tu_ref = to_reference(tu_qname);
            mi::mdl::IExpression_call *tu_call = m_ef.create_call(tu_ref);

            mi::mdl::IQualified_name *qname = create_qualified_name(strip_deprecated_suffix(callee_name));
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            for (mi::Size i = 0; i < n_params; ++i) {
                mi::mdl::IArgument const *arg = nullptr;

                Handle<IExpression const> nr_arg(args->get_expression(i));
                mi::mdl::IExpression const *expr = transform_expr(nr_arg);

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

            mi::mdl::IQualified_name *qname = create_qualified_name(strip_deprecated_suffix(callee_name));
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
                mi::mdl::IExpression const *expr = transform_expr(nr_arg);

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

            mi::mdl::IQualified_name *qname = create_qualified_name(strip_deprecated_suffix(callee_name));
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
                mi::mdl::IExpression const *expr = transform_expr(nr_arg);

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
                        mi::mdl::IExpression const *expr = transform_expr(nr_arg);

                        if (i == 1) {
                            // wrap by color constructor
                            mi::mdl::IQualified_name *qname = create_qualified_name("color");
                            mi::mdl::IExpression_reference *ref = to_reference(qname);
                            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

                            call->add_argument(m_ef.create_positional_argument(expr));
                            expr = call;
                        }

                        if (named_args) {
                            arg = m_ef.create_named_argument(to_simple_name(args->get_name(i)), expr);
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
            mi::mdl::IQualified_name *qname = create_qualified_name(strip_deprecated_suffix(callee_name));
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
                mi::mdl::IExpression const *expr = transform_expr(nr_arg);

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
            mi::mdl::IQualified_name *qname = create_qualified_name(strip_deprecated_suffix(callee_name));
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            for (mi::Size i = 0; i < n_params; ++i) {
                Handle<IExpression const> nr_arg(args->get_expression(i));
                mi::mdl::IExpression const *expr = transform_expr(nr_arg);

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
        if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_3 && n_params == 1) {
            mi::mdl::IQualified_name *qname = create_qualified_name(strip_deprecated_suffix(callee_name));
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            Handle<IExpression const> nr_arg(args->get_expression(zero_size));
            mi::mdl::IExpression const *expr = transform_expr(nr_arg);

            mi::mdl::IArgument const *arg = nullptr;
            if (named_args) {
                arg = m_ef.create_named_argument(to_simple_name(args->get_name(0)), expr);
            } else {
                arg = m_ef.create_positional_argument(expr);
            }
            call->add_argument(arg);

            if (mi::mdl::is_tex_2d(expr->get_type())) {
                // MDL 1.3 -> 1.4: insert the uv_tile parameter
                mi::mdl::IExpression const *expr =
                    m_ef.create_literal(mi::mdl::create_int2_zero(m_vf));
                if (named_args) {
                    arg = m_ef.create_named_argument(to_simple_name("uv_tile"), expr);
                } else {
                    arg = m_ef.create_positional_argument(expr);
                }
                call->add_argument(arg);
            }
            return call;
        }
        break;
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT2:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT3:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT4:
    case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXEL_COLOR:
        if (m_owner_version > mi::mdl::IMDL::MDL_VERSION_1_3 && n_params == 2) {
            mi::mdl::IQualified_name *qname = create_qualified_name(strip_deprecated_suffix(callee_name));
            mi::mdl::IExpression_reference *ref = to_reference(qname);
            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            mi::mdl::IExpression const *tex_expr;
            {
                Handle<IExpression const> nr_arg(args->get_expression(zero_size));
                tex_expr = transform_expr(nr_arg);

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
                mi::mdl::IExpression const *expr = transform_expr(nr_arg);

                mi::mdl::IArgument const *arg = nullptr;
                if (named_args) {
                    arg = m_ef.create_named_argument(to_simple_name(args->get_name(1)), expr);
                } else {
                    arg = m_ef.create_positional_argument(expr);
                }
                call->add_argument(arg);
            }

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
        mi::mdl::IExpression const *arg_mdl = transform_expr(arg);

        tn_call->add_argument(m_ef.create_positional_argument(arg_mdl));

        return tn_call;
    }
    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
        {
            Handle<IExpression const> comp_arg(args->get_expression(mi::Size(0)));
            mi::mdl::IExpression const *compound = transform_expr(comp_arg);

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

            mi::mdl::IType_name *tn = create_type_name(e_tp);
            mi::mdl::IExpression_reference *ref = m_ef.create_reference(tn);
            ref->set_array_constructor();

            mi::mdl::IExpression_call *call = m_ef.create_call(ref);

            for (mi::Size i = 0, n = n_params; i < n; ++i) {
                Handle<IExpression const> arg(args->get_expression(i));

                mi::mdl::IExpression const *expr = transform_expr(arg);

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

            if (!a_tp.is_valid_interface()) {
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

                mi::mdl::IExpression const *expr = transform_expr(arg);

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
const mi::mdl::IExpression* Mdl_ast_builder::transform_expr( const Handle<const IExpression>& expr)
{
    Param_map::const_iterator it( m_param_map.find( expr));
    if( it != m_param_map.end()) {
        // must be mapped
        return to_reference( it->second);
    }

    switch( expr->get_kind()) {

        case IExpression::EK_CONSTANT: {

            Handle<const IExpression_constant> constant(expr->get_interface<IExpression_constant>());
            Handle<const IValue> v(constant->get_value());
            return transform_value(v);
        }

        case IExpression::EK_CALL: {

            Handle<const IExpression_call> call( expr->get_interface<IExpression_call>());

            DB::Tag tag = call->get_call();
            if( !tag)
                return m_ef.create_invalid();

            Handle<const IType> type( call->get_type());
            mi::mdl::IDefinition::Semantics sema = mi::mdl::IDefinition::DS_UNKNOWN;
            Handle<const IExpression_list> args;
            std::string def;
            bool named_args = false;
            mi::Size n_params = 0;

            SERIAL::Class_id class_id = m_trans->get_class_id( tag);

            if( class_id == Mdl_function_call::id) {

                // handle function calls
                DB::Access<Mdl_function_call> fcall( tag, m_trans);
                DB::Tag def_tag = fcall->get_function_definition( m_trans);
                if( !def_tag)
                    return m_ef.create_invalid();
                DB::Access<Mdl_function_definition> fdef( def_tag, m_trans);
                def = fdef->get_mdl_name_without_parameter_types();

                // if reexported, use the original
                const char* original_mdl_name = fdef->get_mdl_original_name();
                if( original_mdl_name) {
                    DB::Tag orig_tag = m_trans->name_to_tag( get_db_name( original_mdl_name).c_str());
                    DB::Access<Mdl_function_definition> orig_fdef( orig_tag, m_trans);
                    def = orig_fdef->get_mdl_name_without_parameter_types();
                }

                named_args = false;
                sema       = fdef->get_mdl_semantic();
                args       = fcall->get_arguments();
                n_params   = fcall->get_parameter_count();

            } else if( class_id == Mdl_material_instance::id) {

                // handle material instances
                DB::Access<Mdl_material_instance> mat_inst( tag, m_trans);
                DB::Tag def_tag = mat_inst->get_material_definition( m_trans);
                if( !def_tag)
                    return m_ef.create_invalid();
                DB::Access<Mdl_material_definition> mat_def( def_tag, m_trans);
                def = mat_def->get_mdl_name_without_parameter_types();

                // if reexported, use the original
                const char* original_mdl_name = mat_def->get_mdl_original_name();
                if( original_mdl_name) {
                    DB::Tag orig_tag = m_trans->name_to_tag( get_db_name( original_mdl_name).c_str());
                    DB::Access<Mdl_material_definition> orig_mat_def( orig_tag, m_trans);
                    def = orig_mat_def->get_mdl_name_without_parameter_types();
                }

                named_args = true;
                sema       = mi::mdl::IDefinition::DS_UNKNOWN;
                args       = mat_inst->get_arguments();
                n_params   = mat_def->get_parameter_count();

            } else {

                // unsupported
                ASSERT( M_SCENE, !"unsupported callee kind");
                return m_ef.create_invalid();

            }

            return transform_call(type, sema, def, n_params, args, named_args);
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

            SERIAL::Class_id class_id = m_trans->get_class_id( tag);

            if( class_id == Mdl_function_definition::id) {

                // handle function calls
                DB::Access<Mdl_function_definition> fdef( tag, m_trans);
                def = fdef->get_mdl_name_without_parameter_types();

                // if reexported, use the original
                const char* original_mdl_name = fdef->get_mdl_original_name();
                if( original_mdl_name) {
                    DB::Tag orig_tag = m_trans->name_to_tag( get_db_name( original_mdl_name).c_str());
                    DB::Access<Mdl_function_definition> orig_fdef( orig_tag, m_trans);
                    def = orig_fdef->get_mdl_name_without_parameter_types();
                }

                named_args = false;
                sema       = fdef->get_mdl_semantic();
                n_params   = fdef->get_parameter_count();

            } else if( class_id == Mdl_material_definition::id) {

                // handle material instances
                DB::Access<Mdl_material_definition> mat_def( tag, m_trans);
                def = mat_def->get_mdl_name_without_parameter_types();

                // if reexported, use the original
                const char* original_mdl_name = mat_def->get_mdl_original_name();
                if( original_mdl_name) {
                    DB::Tag orig_tag = m_trans->name_to_tag( get_db_name( original_mdl_name).c_str());
                    DB::Access<Mdl_function_definition> c( orig_tag, m_trans);
                    def = mat_def->get_mdl_name_without_parameter_types();
                }

                named_args = true;
                sema       = mi::mdl::IDefinition::DS_UNKNOWN;
                n_params   = mat_def->get_parameter_count();

            } else {

                // unsupported
                ASSERT( M_SCENE, !"unsupported callee kind");
                return m_ef.create_invalid();

            }

            return transform_call(type, sema, def, n_params, args, named_args);
        }

        case IExpression::EK_PARAMETER: {

            Handle<const IExpression_parameter> parameter( expr->get_interface<IExpression_parameter>());
            mi::Size index = parameter->get_index();

            Handle<const IExpression> arg( m_args->get_expression(index));
            if( !arg) {
                ASSERT( M_SCENE, !"parameter has no argument");
                break;
            }

            return transform_expr(arg);
        }

    case IExpression::EK_TEMPORARY:
        // should not occur for AST builder
        ASSERT( M_SCENE, !"unexpected temporary");
        break;

    case IExpression::EK_FORCE_32_BIT:
        // not a real type
        break;
    }

    ASSERT( M_SCENE, !"unexpected expression kind");
    return m_ef.create_invalid();
}

/// Get the texture resource name of a tag.
static std::string get_texture_resource_name_and_gamma(
    DB::Transaction* trans,
    DB::Tag tag,
    mi::mdl::IValue_texture::gamma_mode &gamma_mode)
{
    gamma_mode = mi::mdl::IValue_texture::gamma_default;

    SERIAL::Class_id class_id = trans->get_class_id( tag);
    if( class_id != TEXTURE::Texture::id) {
        const char* name = trans->tag_to_name( tag);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Incorrect type for texture resource \"%s\".", name ? name : "");
        return std::string();
    }

    DB::Access<TEXTURE::Texture> texture( tag, trans);
    DB::Tag image_tag( texture->get_image());
    if( !image_tag)
        return std::string();

    class_id = trans->get_class_id( image_tag);
    if( class_id != DBIMAGE::Image::id) {
        const char* name = trans->tag_to_name( image_tag);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Incorrect type for image resource \"%s\".", name ? name : "");
        return std::string();
    }

    // try to convert gamma value into the MDL constant
    mi::Float32 gamma_override = texture->get_gamma();
    if( gamma_override == 1.0f)
        gamma_mode = mi::mdl::IValue_texture::gamma_linear;
    else if( gamma_override == 2.2f)
        gamma_mode = mi::mdl::IValue_texture::gamma_srgb;
    else
        gamma_mode = mi::mdl::IValue_texture::gamma_default;

    DB::Access<DBIMAGE::Image> image( image_tag, trans);
    return image->get_original_filename();
}

/// Get the light_profile resource name of a tag.
static std::string get_light_profile_resource_name(
    DB::Transaction* trans,
    DB::Tag tag)
{
    SERIAL::Class_id class_id = trans->get_class_id( tag);
    if( class_id != LIGHTPROFILE::Lightprofile::id) {
        const char* name = trans->tag_to_name( tag);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Incorrect type for light profile resource \"%s\".", name?name:"");
        return std::string();
    }
    DB::Access<LIGHTPROFILE::Lightprofile> lightprofile( tag, trans);
    return lightprofile->get_original_filename();
}

/// Get the bsdf_measurement resource name of a tag.
static std::string get_bsdf_measurement_resource_name(
    DB::Transaction* trans,
    DB::Tag tag)
{
    SERIAL::Class_id class_id = trans->get_class_id( tag);
    if( class_id != BSDFM::Bsdf_measurement::id) {
        const char* name = trans->tag_to_name( tag);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Incorrect type for BSDF measurement resource \"%s\".", name?name:"");
        return std::string();
    }
    DB::Access<BSDFM::Bsdf_measurement> bsdf_measurement( tag, trans);
    return bsdf_measurement->get_original_filename();
}

// Transform a MDL expression from neuray representation to MDL representation.
mi::mdl::IExpression const *Mdl_ast_builder::transform_value(
    const Handle<IValue const>& value)
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

            mi::mdl::IQualified_name *qname = create_scope_name(e_tp->get_symbol());
            qname->add_component(sname);

            mi::mdl::IType_enum const *mdl_tp = convert_enum_type(e_tp.get());
            return to_reference(qname, mdl_tp);
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
                Handle<IValue_string_localized const> v(value->get_interface<IValue_string_localized>());
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

            for (mi::Size i = 0, n = v->get_size(); i < n; ++i) {
                Handle<IValue const> e_v(v->get_value(i));

                call->add_argument(m_ef.create_positional_argument(transform_value(e_v)));
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

                call->add_argument(m_ef.create_positional_argument(transform_value(e_v)));
            }
            return call;
        }
        break;
    case IValue::VK_INVALID_DF:
        {
            Handle<IValue_invalid_df const> v(value->get_interface<IValue_invalid_df>());
            Handle<IType_reference const> type(v->get_type());

            mi::mdl::IType_reference const *r_tp =
                cast<mi::mdl::IType_reference>(transform_type(type));
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
                mi::mdl::IType_reference const *r_tp =
                    cast<mi::mdl::IType_reference>(transform_type(type));
                mi::mdl::IValue_invalid_ref const *vv = m_vf.create_invalid_ref(r_tp);
                return m_ef.create_literal(vv);
            }

            mi::mdl::IValue_texture::gamma_mode gamma = mi::mdl::IValue_texture::gamma_default;
            std::string url( get_texture_resource_name_and_gamma(m_trans, tag, gamma));
            if (url.empty()) {
                // no file, map to IValue with tag
                DB::Access<TEXTURE::Texture> texture(tag, m_trans);
                DB::Tag image_tag(texture->get_image());
                DB::Tag_version image_tag_version = m_trans->get_tag_version(image_tag);
                mi::mdl::IType_texture const *t_tp =
                    cast<mi::mdl::IType_texture>(transform_type(type));
                DB::Tag_version tag_version = m_trans->get_tag_version(tag);
                mi::mdl::IValue_texture const *vv = m_vf.create_texture(
                    t_tp, "", gamma, tag.get_uint(),
                    get_hash(/*mdl_file_path*/ nullptr, 0.0f, tag_version, image_tag_version));
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
                mi::mdl::IType_reference const *r_tp =
                    cast<mi::mdl::IType_reference>(transform_type(type));
                mi::mdl::IValue_invalid_ref const *vv = m_vf.create_invalid_ref(r_tp);
                return m_ef.create_literal(vv);
            }

            std::string url(kind == IValue::VK_LIGHT_PROFILE ?
                get_light_profile_resource_name(m_trans, tag) :
                get_bsdf_measurement_resource_name(m_trans, tag));
            if (url.empty()) {
                // no file, map to IValue with tag
                DB::Tag_version tag_version = m_trans->get_tag_version(tag);
                if (kind == IValue::VK_LIGHT_PROFILE) {
                    mi::mdl::IType_light_profile const *lp_tp =
                        cast<mi::mdl::IType_light_profile>(transform_type(type));
                    mi::mdl::IValue_light_profile const *vv = m_vf.create_light_profile(
                        lp_tp, "", tag.get_uint(), get_hash(/*mdl_file_path*/ nullptr, tag_version));
                    return m_ef.create_literal(vv);
                } else {
                    mi::mdl::IType_bsdf_measurement const *bm_tp =
                        cast<mi::mdl::IType_bsdf_measurement>(transform_type(type));
                    mi::mdl::IValue_bsdf_measurement const *vv = m_vf.create_bsdf_measurement(
                        bm_tp, "", tag.get_uint(), get_hash(/*mdl_file_path*/ nullptr, tag_version));
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
    case IValue::VK_FORCE_32_BIT:
        // not a real type
        break;
    }
    ASSERT( M_SCENE, !"unexpected value kind");
    return m_ef.create_invalid();
}

// Transform a (non-user defined) MDL type from neuray representation to MDL representation.
mi::mdl::IType const *Mdl_ast_builder::transform_type(
    mi::base::Handle<IType const> const &type)
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

            mi::mdl::IType_atomic const *a_tp = cast<mi::mdl::IType_atomic>(transform_type(e_tp));
            return m_tf.create_vector(a_tp, int(v_tp->get_size()));
        }
    case IType::TK_MATRIX:
        {
            Handle<IType_matrix const> m_tp(type->get_interface<IType_matrix>());
            Handle<IType const>        e_tp(m_tp->get_element_type());

            mi::mdl::IType_vector const *v_tp = cast<mi::mdl::IType_vector>(transform_type(e_tp));
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
            case IType_texture::TS_FORCE_32_BIT:
                // not a real shape
                break;
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
    case IType::TK_FORCE_32_BIT:
        // not a real type
        break;
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

    // FIXME: check for name clashes here
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
    mi::base::Handle<const IExpression> const &init)
{
    m_param_map[init] = sym;
}

// Remove all declared parameter mappings.
void Mdl_ast_builder::remove_parameters()
{
    m_param_map.clear();
}

// Convert an neuray enum type into a MDL enum type.
mi::mdl::IType_enum const *Mdl_ast_builder::convert_enum_type(
    IType_enum const *e_tp)
{
    switch (e_tp->get_predefined_id()) {
    case IType_enum::EID_USER:
        {
            mi::mdl::ISymbol const *sym = m_st.get_user_type_symbol(e_tp->get_symbol());
            mi::mdl::IType_enum *res = m_tf.create_enum(sym);

            for (mi::Size i = 0, n = e_tp->get_size(); i < n; ++i) {
                mi::mdl::ISymbol const *v_sym = m_st.get_symbol(e_tp->get_value_name(i));
                mi::Sint32             v_code = e_tp->get_value_code(i, nullptr);

                res->add_value(v_sym, v_code);
            }
            return res;
        }
    case IType_enum::EID_TEX_GAMMA_MODE:
        return m_tf.get_predefined_enum(mi::mdl::IType_enum::EID_TEX_GAMMA_MODE);
    case IType_enum::EID_INTENSITY_MODE:
        return m_tf.get_predefined_enum(mi::mdl::IType_enum::EID_INTENSITY_MODE);
    case IType_enum::EID_FORCE_32_BIT:
        break;
    }
    ASSERT( M_SCENE, !"unexpected enum type ID");
    return nullptr;
}

mi::mdl::IExpression const *Mdl_ast_builder::add_multiscatter_param(
     std::string const                              &callee_name,
     mi::Size                                       n_params,
     bool                                           named_args,
     mi::base::Handle<IExpression_list const> const &args)
 {
     // MDL 1.5 -> 1.6: insert multiscatter_tint parameter to all glossy BSDFs
     mi::mdl::IQualified_name *qname = create_qualified_name(strip_deprecated_suffix(callee_name));
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
         mi::mdl::IExpression const *expr = transform_expr(nr_arg);

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
