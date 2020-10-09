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

#include "pch.h"

#include <cctype>

#include <mi/base/handle.h>
#include <base/system/main/types.h>

#include "compilercore_allocator.h"
#include "compilercore_analysis.h"
#include "compilercore_assert.h"
#include "compilercore_mdl.h"
#include "compilercore_modules.h"
#include "compilercore_module_transformer.h"
#include "compilercore_tools.h"
#include "compilercore_string.h"
#include "compilercore_symbols.h"

namespace mi {
namespace mdl {

typedef Store<mi::base::Handle<Module const> > Module_scope;

/// Checks, if the given type must be imported if used.
static bool need_type_import(IType const *t)
{
    if (IType_enum const *te = as<IType_enum>(t)) {
        // i,port all user defined enums AND tex:gamma_mode
        IType_enum::Predefined_id id = te->get_predefined_id();
        return id == IType_enum::EID_USER || id == IType_enum::EID_TEX_GAMMA_MODE;
    } else if (IType_struct const *ts = as<IType_struct>(t)) {
        // import all user defined structs
        return ts->get_predefined_id() == IType_struct::SID_USER;
    }
    return false;
}

/// Get type definition for the given type.
///
/// \param module  the owner module of the type
/// \param type    the type
static Definition const *get_type_definition(
    Module const *module,
    IType const  *type)
{
    Definition_table const &dt = module->get_definition_table();
    Scope *tp_scope = dt.get_type_scope(type);

    MDL_ASSERT(tp_scope != NULL);
    Definition const *type_def = tp_scope->get_owner_definition();
    return type_def;
}

/// Promote a call from a MDL version to the target MDL version.
///
/// \param[in]  mod       the target module
/// \param[in]  major     the major version of the source module (that owns the call)
/// \param[in]  minor     the major version of the source module (that owns the call)
/// \param[in]  ref       the callee reference
/// \param[in]  modifier  a clone modifier for cloning expressions
/// \param[out] rules     output: necessary rules to modify the arguments of the call
///
/// \return the (potentially modified) callee reference
IExpression_reference const *Module_inliner::promote_call_reference(
    Module                      &mod,
    int                         major,
    int                         minor,
    IExpression_reference const *ref,
    IClone_modifier             *modifier,
    unsigned                     &rules)
{
    IType_name const *tn = ref->get_name();
    rules = Module::PR_NO_CHANGE;
    if (tn->is_array() || ref->is_array_constructor()) {
        return ref;
    }

    IDefinition const      *def = ref->get_definition();
    IDefinition::Semantics sema = def->get_semantics();

    IQualified_name const *qn = tn->get_qualified_name();
    int                   n_components = qn->get_component_count();
    ISimple_name const    *sn = qn->get_component(n_components - 1);
    ISymbol const         *sym = sn->get_symbol();
    char const            *name = sym->get_name();

    int mod_major = 0, mod_minor = 0;
    mod.get_version(mod_major, mod_minor);

    string n(name, mod.get_allocator());

    if (major < mod_major || (major == mod_major && minor < mod_minor)) {
        // the symbol was removed BEFORE the module version, we need promotion
        // which might change the name
        if (major == 1) {
            if (minor == 0) {
                switch (sema) {
                case Definition::DS_INTRINSIC_DF_SPOT_EDF:
                    rules = Module::PR_SPOT_EDF_ADD_SPREAD_PARAM;
                    break;
                case Definition::DS_INTRINSIC_DF_MEASURED_EDF:
                    if (mod_major > 1 || (mod_major == 1 && mod_minor >= 1))
                        rules |= Module::PC_MEASURED_EDF_ADD_MULTIPLIER;
                    if (mod_major > 1 || (mod_major == 1 && mod_minor >= 2))
                        rules |= Module::PR_MEASURED_EDF_ADD_TANGENT_U;
                    break;
                default:
                    break;
                }
            } else if (minor == 1) {
                switch (sema) {
                case Definition::DS_INTRINSIC_DF_MEASURED_EDF:
                    rules = Module::PR_MEASURED_EDF_ADD_TANGENT_U;
                    break;
                default:
                    break;
                }
            }
            if (mod_minor >= 3 && minor < 3) {
                switch (sema) {
                case Definition::DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL:
                    rules = Module::PR_ROUNDED_CORNER_ADD_ROUNDNESS;
                    break;
                default:
                    break;
                }
            }
            if (mod_minor >= 4 && minor < 4) {
                switch (sema) {
                case Definition::DS_INTRINSIC_DF_FRESNEL_LAYER:
                    rules = Module::PR_FRESNEL_LAYER_TO_COLOR;
                    n = "color_fresnel_layer";
                    break;
                case Definition::DS_INTRINSIC_TEX_WIDTH:
                case Definition::DS_INTRINSIC_TEX_HEIGHT:
                    rules = Module::PR_WIDTH_HEIGHT_ADD_UV_TILE;
                    break;
                case Definition::DS_INTRINSIC_TEX_TEXEL_COLOR:
                case Definition::DS_INTRINSIC_TEX_TEXEL_FLOAT:
                case Definition::DS_INTRINSIC_TEX_TEXEL_FLOAT2:
                case Definition::DS_INTRINSIC_TEX_TEXEL_FLOAT3:
                case Definition::DS_INTRINSIC_TEX_TEXEL_FLOAT4:
                    rules = Module::PR_TEXEL_ADD_UV_TILE;
                    break;
                default:
                    break;
                }
            }
            if (mod_minor >= 5 && minor < 5) {
                if (sema == IDefinition::DS_ELEM_CONSTRUCTOR) {
                    IType const *t = tn->get_type();
                    if (t != NULL && is_material_type(t)) {
                        rules = Module::PR_MATERIAL_ADD_HAIR;
                    }
                }
            }
            if (mod_minor >= 6 && minor < 6) {
                if (sema == IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF ||
                    sema == IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF||
                    sema == IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF ||
                    sema == IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF ||
                    sema == IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF ||
                    sema == IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF ||
                    sema == IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF)
                {
                    rules = Module::PR_GLOSSY_ADD_MULTISCATTER;
                }
            }
        }
    }
    if (rules == Module::PR_NO_CHANGE) {
        return ref;
    }

    Name_factory *name_fact = mod.get_name_factory();

    sym = name_fact->create_symbol(n.c_str());

    IQualified_name *n_qn = name_fact->create_qualified_name();

    for (int i = 0; i < n_components - 1; ++i) {
        n_qn->add_component(mod.clone_name(qn->get_component(i)));
    }

    sn = name_fact->create_simple_name(sym);
    n_qn->add_component(sn);

    if (qn->is_absolute()) {
        n_qn->set_absolute();
    }
    n_qn->set_definition(qn->get_definition());

    IType_name *n_tn = name_fact->create_type_name(n_qn);

    if (tn->is_absolute()) {
        n_tn->set_absolute();
    }
    if (IExpression const *array_size = tn->get_array_size()) {
        n_tn->set_array_size(mod.clone_expr(array_size, modifier));
    }
    if (tn->is_incomplete_array()) {
        n_tn->set_incomplete_array();
    }
    n_tn->set_qualifier(tn->get_qualifier());
    if (ISimple_name const *size_name = tn->get_size_name()) {
        n_tn->set_size_name(size_name);
    }
    n_tn->set_type(tn->get_type());
    return mod.get_expression_factory()->create_reference(n_tn);
}

// Constructor
Module_inliner::Module_inliner(
    IAllocator              *alloc,
    Module const            *module,
    Module                  *target_module,
    IInline_import_callback *inline_imports,
    bool                    omit_anno_origin,
    bool                    is_root_module,
    Reference_map           &references,
    Import_set              &imports,
    Def_set                 &visible_definitions,
    size_t                  &counter)
: m_module(mi::base::make_handle_dup(module))
, m_target_module(base::make_handle_dup(target_module))
, m_inline_imports(inline_imports)
, m_omit_anno_origin(omit_anno_origin)
, m_is_root(is_root_module)
, m_references(references)
, m_imports(imports)
, m_exports(visible_definitions)
, m_alloc(alloc)
, m_af(*m_target_module->get_annotation_factory())
, m_sf(*m_target_module->get_statement_factory())
, m_ef(*m_target_module->get_expression_factory())
, m_df(*m_target_module->get_declaration_factory())
, m_nf(*m_target_module->get_name_factory())
, m_vf(*m_target_module->get_value_factory())
, m_counter(counter)
, m_keep_non_root_parameter_defaults(false)
{
    if (m_is_root) {
        Exports_collector(module, m_exports, /*is_root=*/true).collect();
    }
}

// Destructor.
Module_inliner::~Module_inliner()
{
}

// Runs the inliner on the src module passed to the constructor.
void Module_inliner::run()
{
    // if we have to export hidden helper definitions, add required import upfront
    if (!m_exports.empty()) {
        register_import("::anno", "hidden");
    }
    // display name is always required
    register_import("::anno", "display_name");
    // register ::anno::origin if required
    if (!m_omit_anno_origin) {
        register_import("::anno", "origin");
    }

    // inline the module
    visit(m_module.get());

    // add imports
    for (Import_set::iterator it(m_imports.begin()), end(m_imports.end()); it != end; ++it) {
        m_target_module->add_import((*it).c_str());
    }
}

// Called, before an annotation block is visited.
bool Module_inliner::pre_visit(IAnnotation_block *anno)
{
    // traverse children (aka annotations) only for the root module
    return m_is_root;
}

// Called, whenever a reference has been visited.
IExpression *Module_inliner::post_visit(IExpression_reference *expr)
{
    if (expr->is_array_constructor()) {
        return expr;
    }

    Definition const *def = impl_cast<Definition>(expr->get_definition());
    Definition::Kind kind = def->get_kind();

    switch (kind) {
    case IDefinition::DK_ENUM_VALUE:
        {
            // we cannot import enum values, only enum types
            IType const *e_type = def->get_type();
            def = get_type_definition(m_module.get(), e_type);
        }
        break;
    case IDefinition::DK_CONSTRUCTOR:
        {
            // constructors are automatically imported, if the type is imported
            IType_function const *f_type   = cast<IType_function>(def->get_type());
            IType const          *ret_type = f_type->get_return_type();
            def = get_type_definition(m_module.get(), ret_type);
        }
        break;
    case IDefinition::DK_FUNCTION:
    case IDefinition::DK_CONSTANT:
        // import them
        break;
    default:
        // ignore all others so far
        return expr;
    }

    if (!m_is_root || def->get_original_import_idx() != 0) {
        // Definition was imported
        mi::base::Handle<Module const> orig_module(m_module->get_owner_module(def));
        if (orig_module->is_builtins()) {
            return expr;
        }
        Definition const *orig_def = m_module->get_original_definition(def);
        if (!needs_inline(orig_module.get())) {
            register_import(orig_module->get_name(), orig_def->get_symbol()->get_name());

            {
                Module_scope scope(m_module, orig_module);
                register_child_types(orig_def->get_type());
            }
            return expr;
        }
        if (m_references.find(orig_def) != m_references.end()) {
            // the reference was already collected
            return expr;
        }

        IDeclaration const *orig_decl = orig_def->get_declaration();
        if (orig_def->get_kind() == IDefinition::DK_TYPE && orig_decl == NULL) {
            // not all types have a declaration
            return expr;
        }

        MDL_ASSERT(orig_decl != NULL);

        // Traverse original declaration
        Module_inliner inliner(
            m_alloc,
            orig_module.get(),
            m_target_module.get(),
            m_inline_imports,
            m_omit_anno_origin,
            /*is_root_module=*/false,
            m_references,
            m_imports,
            m_exports,
            m_counter);
        inliner.visit(orig_decl);
    }
    return expr;
}

// End of a literal expression.
IExpression *Module_inliner::post_visit(IExpression_literal *expr)
{
    do_type(expr->get_type());
    return expr;
}

// End of a binary expression.
IExpression *Module_inliner::post_visit(IExpression_binary *expr)
{
    if (expr->get_operator() == IExpression_binary::OK_SELECT) {
        // we select something from the left subexpression. Its type must be known thats why.
        do_type(expr->get_left_argument()->get_type());
    }
    return expr;
}

void Module_inliner::post_visit(IType_name *tn)
{
    do_type(tn->get_type());
}

void Module_inliner::post_visit(IAnnotation *anno)
{
    Definition const *def = impl_cast<Definition>(anno->get_name()->get_definition());
    if (!m_is_root || def->get_original_import_idx() != 0) {
        mi::base::Handle<Module const> orig_module(m_module->get_owner_module(def));
        if (orig_module->is_builtins()) {
            return;
        }

        Definition const *orig_def = m_module->get_original_definition(def);
        if (!needs_inline(orig_module.get())) {
            unsigned flags = orig_def->get_version_flags();
            unsigned rem   = mdl_removed_version(flags);
            if (rem > unsigned(m_target_module->get_version())) {
                register_import(orig_module->get_name(), orig_def->get_symbol()->get_name());

                {
                    Module_scope scope(m_module, orig_module);
                    register_child_types(orig_def->get_type());
                }
            }
            return;
        }
        if (m_references.find(orig_def) != m_references.end()) {
            return;
        }

        IDeclaration const *decl = orig_def->get_declaration();
        MDL_ASSERT(decl);

        // Traverse original declaration
        Module_inliner inliner(
            m_alloc,
            orig_module.get(),
            m_target_module.get(),
            m_inline_imports,
            m_omit_anno_origin,
            /*is_root_module=*/false,
            m_references,
            m_imports,
            m_exports,
            m_counter);
        inliner.visit(decl);
    }
}

// Checks, if the non-root declaration has already been copied.
bool Module_inliner::pre_visit(IDeclaration_type_struct *decl)
{
    if (!m_is_root) {
        MDL_ASSERT(m_references.find(decl->get_definition()) == m_references.end());
    }
    return true;
}

// Clones a struct declaration
void Module_inliner::post_visit(IDeclaration_type_struct *decl)
{
    clone_declaration(decl, "s_");
}

// Checks, if the non-root declaration has already been copied.
bool Module_inliner::pre_visit(IDeclaration_constant *decl)
{
    if (!m_is_root) {
        for (int i = 0, n = decl->get_constant_count(); i < n; ++i) {
            MDL_ASSERT(
                m_references.find(
                    decl->get_constant_name(i)->get_definition()) == m_references.end());
        }
    }
    return true;
}

// Clones the given declaration and puts it into the reference map.
void Module_inliner::post_visit(IDeclaration_constant *decl)
{
    clone_declaration(decl, "c_");
}

// Checks, if the non-root declaration has already been copied.
bool Module_inliner::pre_visit(IDeclaration_type_enum *decl)
{
    if (!m_is_root) {
        MDL_ASSERT(m_references.find(decl->get_definition()) == m_references.end());
    }
    return true;
}

// Clones an enum declaration
void Module_inliner::post_visit(IDeclaration_type_enum *decl)
{
    clone_declaration(decl, "e_");
}

// Checks, if the non-root declaration has already been copied.
bool Module_inliner::pre_visit(IDeclaration_function *decl)
{
    if (m_is_root) {
        if (IAnnotation_block const *anno = decl->get_annotations()) {
            visit(anno);
        }
        if (IAnnotation_block const *ranno = decl->get_return_annotations()) {
            visit(ranno);
        }
    } else {
        MDL_ASSERT(m_references.find(
            decl->get_definition()) == m_references.end() &&
            "visited function is already inlined");
    }

    bool is_exported = (m_exports.find(decl->get_definition()) != m_exports.end());
    for (int i = 0, n = decl->get_parameter_count(); i < n; ++i) {

        IParameter const *param = decl->get_parameter(i);
        if (m_is_root) {
            visit(param);
        } else {

            // visit type name
            visit(const_cast<IType_name*>(param->get_type_name()));

            if (m_keep_non_root_parameter_defaults || is_exported) {

                IExpression const *init = param->get_init_expr();
                if (init != NULL) {
                    visit(init);
                }
            }
        }
    }

    visit(decl->get_return_type_name());

    visit(decl->get_body());

    return false;
}

// Clones the given declaration and puts it into the reference map.
void Module_inliner::post_visit(IDeclaration_function *decl)
{
    clone_declaration(decl, "f_");
}

// Checks, if the non-root declaration has already been copied.
bool Module_inliner::pre_visit(IDeclaration_annotation *decl)
{
    if (!m_is_root) {
        MDL_ASSERT(m_references.find(decl->get_definition()) == m_references.end());
    }
    return true;
}

// Clones the given declaration and puts it into the reference map.
void Module_inliner::post_visit(IDeclaration_annotation *decl)
{
    clone_declaration(decl, "a_");
}

// If the referenced function can be found in the reference map, it is replaced
// by a reference to the inlined function.
IExpression *Module_inliner::clone_expr_reference(IExpression_reference const *ref)
{
    if (ref->is_array_constructor()) {
        IType_name const *name = m_target_module->clone_name(ref->get_name(), this);
        IExpression_reference *new_ref = m_ef.create_reference(name);
        new_ref->set_array_constructor();
        return new_ref;
    } else {
        Definition const *def = impl_cast<Definition>(ref->get_definition());
        Definition::Kind kind = def->get_kind();

        ISymbol const *new_local_sym = NULL;
        if (kind == IDefinition::DK_FUNCTION) {
            Definition const *orig_def = def;
            bool is_imported = def->get_original_import_idx() != 0;

            if (is_imported) {
                orig_def = m_module->get_original_definition(def);
            }

            Reference_map::iterator const &p = m_references.find(orig_def);
            if (p != m_references.end()) {
                new_local_sym = p->second;
            } else if (is_imported) {
                // always rewrite the reference
                return create_imported_reference(def, orig_def);
            }
        } else if (kind == IDefinition::DK_CONSTRUCTOR) {
            IType_function const *tf = cast<IType_function>(def->get_type());
            IType const          *t  = tf->get_return_type();
            if (need_type_import(t)) {
                Definition const *type_def      = get_type_definition(m_module.get(), t);
                Definition const *orig_type_def = type_def;

                bool is_imported = type_def->has_flag(Definition::DEF_IS_IMPORTED);
                if (is_imported) {
                    orig_type_def = m_module->get_original_definition(type_def);
                }
                Reference_map::iterator const &p = m_references.find(orig_type_def);
                if (p != m_references.end()) {
                    new_local_sym = p->second;
                } else if (is_imported) {
                    // always rewrite the reference
                    return create_imported_reference(type_def, orig_type_def);
                }
            }
        } else if (kind == IDefinition::DK_ENUM_VALUE || kind == IDefinition::DK_CONSTANT) {
            IType_name *tname = m_target_module->clone_name(ref->get_name(), this);
            return m_ef.create_reference(tname);
        }

        if (new_local_sym != NULL) {
            ISimple_name    *sname = m_nf.create_simple_name(new_local_sym);
            IQualified_name *qname = m_nf.create_qualified_name();
            IType_name      *tname = m_nf.create_type_name(qname);

            qname->add_component(sname);

            return m_ef.create_reference(tname);
        }
    }
    return m_target_module->clone_expr(ref, /*modifier=*/NULL);
}

// Clone a call.
IExpression *Module_inliner::clone_expr_call(IExpression_call const *c_expr)
{
    // check if the call needs promotion
    unsigned int rules = Module::PR_NO_CHANGE;
    if (IExpression_reference const *ref = as<IExpression_reference>(c_expr->get_reference())) {
        int major = 1, minor = 0;
        m_module->get_version(major, minor);

        ref = promote_call_reference(*m_target_module, major, minor, ref, this, rules);

        IExpression const *new_ref;
        if (rules != Module::PR_NO_CHANGE) {
            // was modified
            new_ref = ref;

            if (rules &  Module::PR_FRESNEL_LAYER_TO_COLOR)
                register_import("::df", "color_fresnel_layer");
            if (rules & Module::PR_MEASURED_EDF_ADD_TANGENT_U)
                register_import("::state", "texture_tangent_u");
        } else {
            // no changes, just clone it
            new_ref = m_target_module->clone_expr(ref, this);
        }
        IExpression_call *call = m_ef.create_call(new_ref);

        for (int i = 0, j = 0, n = c_expr->get_argument_count(); i < n; ++i, ++j) {
            IArgument const *arg = m_target_module->clone_arg(c_expr->get_argument(i), this);
            call->add_argument(arg);
            j = m_target_module->promote_call_arguments(call, arg, j, rules);
        }
        return call;
    }
    return m_target_module->clone_expr(c_expr, /*modifier=*/NULL);
}

// Clone a literal.
IExpression *Module_inliner::clone_literal(IExpression_literal const *lit)
{
    IType const *lit_type = lit->get_type()->skip_type_alias();
    if (IType_array const *ta = as<IType_array>(lit_type)) {
        lit_type = ta->get_element_type();
    }
    if (need_type_import(lit_type)) {
        // we must handle two cases here:
        // 1) a literal of the original module is cloned
        // 2) an expression of the original module was cloned, resulting in a literal
        //    after a fold operation
        // in case 1), type is owned by m_module, in case 2) by m_target_module
        // Fortunately the owner test for types is O(1)
        Type_factory const *tf = m_target_module->get_type_factory();

        if (tf->is_owner(lit_type)) {
            // the type was already imported into the target module, but the definition
            // of the type exists only in the original module. Hence convert it back to the
            // original module's type
            lit_type = m_module->get_type_factory()->get_equal(lit_type);
        }

        Definition const *type_def = get_type_definition(m_module.get(), lit_type);
        if (type_def->has_flag(Definition::DEF_IS_IMPORTED)) {
            mi::base::Handle<const IModule> mod_ori(m_module->get_owner_module(type_def));
            if (!needs_inline(mod_ori.get())) {
                return m_target_module->clone_expr(lit, /*modifier=*/NULL);
            }

            type_def = m_module->get_original_definition(type_def);
        }
        // handle enum values
        if (IValue_enum const *ve = as<IValue_enum>(lit->get_value())) {
            ISimple_name const *sn = cast<IDeclaration_type_enum>(
                type_def->get_declaration())->get_value_name(ve->get_index());

            IDefinition const *vndef = sn->get_definition();
            Reference_map::iterator const &it = m_references.find(vndef);
            if (it == m_references.end()) {
                return simple_name_to_reference(m_target_module->clone_name(sn));
            } else {
                ISymbol const *sym = it->second;
                ISimple_name *sn = m_nf.create_simple_name(sym);
                return simple_name_to_reference(sn);
            }
        } else if (IValue_compound const *vs = as<IValue_compound>(lit->get_value())) {
            ISymbol const *sym;
            Reference_map::iterator const &it = m_references.find(type_def);
            if (it == m_references.end()) {
                sym = m_nf.create_symbol(type_def->get_symbol()->get_name());
            } else {
                sym = it->second;
            }
            return make_constructor(vs, sym);
        }
    }
    return m_target_module->clone_expr(lit, /*modifier=*/NULL);
}

// Clone a qualified name.
IQualified_name *Module_inliner::clone_name(IQualified_name const *qname)
{
    Definition const *def = impl_cast<Definition>(qname->get_definition());

    string orig_module_name(m_alloc);
    if (def->get_original_import_idx() != 0) {
        Definition const *def_ori = m_module->get_original_definition(def);
        mi::base::Handle<Module const> mod_ori(m_module->get_owner_module(def));
        orig_module_name = mod_ori->get_name();
        def = def_ori;
    }

    Reference_map::iterator const& it = m_references.find(def);
    if (it == m_references.end()) {
        if (!qname->is_absolute() && !orig_module_name.empty()) {
            IQualified_name *res = m_nf.create_qualified_name();
            int j = 2;
            for (int i = 2, n = orig_module_name.size() - 1; i < n; ++i) {
                if (orig_module_name[i] == ':' && orig_module_name[i + 1] == ':') {
                    orig_module_name[i] = '\0';
                    ISimple_name *sn = m_nf.create_simple_name(
                        m_nf.create_symbol(&orig_module_name.c_str()[j]));
                    res->add_component(sn);
                    j = i + 2;
                }
            }
            ISimple_name *sn = m_nf.create_simple_name(
                m_nf.create_symbol(&orig_module_name.c_str()[j]));
            res->add_component(sn);

            MDL_ASSERT(qname->get_component_count() >= 1);
            ISimple_name const *sname = m_target_module->clone_name(
                qname->get_component(qname->get_component_count() - 1));
            res->add_component(sname);
            return res;
        }
        return m_target_module->clone_name(qname, /*modifier=*/NULL);
    }
    ISimple_name    *sname = m_nf.create_simple_name(it->second);
    IQualified_name *res   = m_nf.create_qualified_name();

    res->add_component(sname);
    return res;
}

// Generates a call to the cloned struct constructor from the values of the original struct.
IExpression_call *Module_inliner::make_constructor(
    IValue_compound const   *vs_ori,
    ISymbol const         *new_sym)
{
    ISimple_name    *sname = m_nf.create_simple_name(new_sym);
    IQualified_name *qname = m_nf.create_qualified_name();
    IType_name      *tname = m_nf.create_type_name(qname);

    if (is<IType_array>(vs_ori->get_type())) {
        IExpression_literal *array_size = m_ef.create_literal(
            m_vf.create_int(vs_ori->get_component_count()));
        tname->set_array_size(array_size);
    }

    qname->add_component(sname);

    IExpression_reference *ref_new = m_ef.create_reference(tname);

    IExpression_call *c = m_ef.create_call(ref_new);
    IValue const * const *values = vs_ori->get_values();

    for (int i = 0, n = vs_ori->get_component_count(); i < n; ++i) {
        IValue const *old_value = values[i];
        IExpression *arg_expr = NULL;

        IType const *type = old_value->get_type();
        MDL_ASSERT(type);
        type = type->skip_type_alias();
        if (IType_array const *ta = as<IType_array>(type)) {
            type = ta->get_element_type();
        }
        if (need_type_import(type)) {
            Definition const *type_def = get_type_definition(m_module.get(), type);
            if (type_def->has_flag(Definition::DEF_IS_IMPORTED)) {
                type_def = m_module->get_original_definition(type_def);
            }
            if (is<IValue_compound>(old_value)) {
                Reference_map::iterator const &it = m_references.find(type_def);
                MDL_ASSERT(it != m_references.end());
                arg_expr = make_constructor(cast<IValue_compound>(old_value), it->second);
            } else if (is<IValue_enum>(old_value)) {
                ISimple_name const *sn = cast<IDeclaration_type_enum>(
                    type_def->get_declaration())->get_value_name(cast<IValue_enum>(old_value)->get_index());

                IDefinition const *vndef = sn->get_definition();
                Reference_map::iterator const &it = m_references.find(vndef);
                if (it == m_references.end()) {
                    arg_expr = simple_name_to_reference(m_target_module->clone_name(sn));
                } else {
                    ISymbol const *sym = it->second;
                    ISimple_name  *sn  = m_nf.create_simple_name(sym);
                    arg_expr =  simple_name_to_reference(sn);
                }
            }
        } else {
            IValue const *new_value = m_target_module->import_value(old_value);
            arg_expr = m_ef.create_literal(new_value);
        }
        c->add_argument(m_ef.create_positional_argument(arg_expr));
    }
    return c;
}

// Generates a new name from a simple name and a prefix.
ISimple_name const *Module_inliner::generate_name(
    ISimple_name const *simple_name,
    char const         *prefix)
{
    string name(prefix != NULL ? prefix : "", m_alloc);
    name.append(simple_name->get_symbol()->get_name());

    // find an unused name
    string test = name;
    Symbol_table const &st = m_target_module->get_symbol_table();
    for (;;) {
        if (st.lookup_symbol(test.c_str()) == NULL) {
            name = test;
            break;
        }
        test = name;
        test.append('_');

        char buf[32];
        snprintf(buf, sizeof(buf), "%" FMT_SIZE_T, ++m_counter);
        buf[sizeof(buf) - 1] = '\0';

        test.append(buf);
    }
    return m_nf.create_simple_name(m_nf.create_symbol(name.c_str()));
}

// Clones a statement.
IStatement *Module_inliner::clone_statement(IStatement const *stmt)
{
    if (stmt == NULL) {
        return NULL;
    }

    switch (stmt->get_kind()) {
    case IStatement::SK_EXPRESSION:
        {
            IStatement_expression const *stmt_expr = cast<IStatement_expression>(stmt);
            IExpression const *expr = stmt_expr->get_expression();
            IExpression const *new_expr = m_target_module->clone_expr(expr, this);
            return m_sf.create_expression(new_expr);
        }
    case IStatement::SK_COMPOUND:
        {
            IStatement_compound const *stmt_cmp = cast<IStatement_compound>(stmt);

            IStatement_compound *new_cmp = m_sf.create_compound();
            for (int i = 0, n = stmt_cmp->get_statement_count(); i < n; ++i) {
                IStatement *stmt = clone_statement(stmt_cmp->get_statement(i));
                if (stmt != NULL) {
                    new_cmp->add_statement(stmt);
                }
            }
            return new_cmp;
        }
    case IStatement::SK_RETURN:
        {
            IStatement_return const *stmt_ret = cast<IStatement_return>(stmt);

            IExpression const *new_expr = NULL;
            if (IExpression const *expr = stmt_ret->get_expression()) {
                new_expr = m_target_module->clone_expr(expr, this);
            }

            return m_sf.create_return(new_expr);
        }
    case IStatement::SK_BREAK:
        return m_sf.create_break();
    case IStatement::SK_CASE:
        {
            IStatement_case const *stmt_case = cast<IStatement_case>(stmt);
            IExpression const *new_label = NULL;
            if (IExpression const *label = stmt_case->get_label()) {
                new_label = m_target_module->clone_expr(label, this);
            }
            IStatement_case *new_stmt_case = m_sf.create_switch_case(new_label);
            for (int i = 0, n = stmt_case->get_statement_count(); i < n; ++i) {
                IStatement *new_stmt = clone_statement(stmt_case->get_statement(i));
                new_stmt_case->add_statement(new_stmt);
            }
            return new_stmt_case;
        }
    case IStatement::SK_CONTINUE:
        return m_sf.create_continue();
    case IStatement::SK_DECLARATION:
        {
            IStatement_declaration const *stmt_decl = cast<IStatement_declaration>(stmt);
            IDeclaration const *decl = stmt_decl->get_declaration();
            if (IDeclaration_variable const *var_decl = as<IDeclaration_variable>(decl)) {
                IDeclaration *new_decl = m_target_module->clone_decl(var_decl, this);
                return m_sf.create_declaration(new_decl);
            }
            if (is<IDeclaration_constant>(decl)) {
                return NULL;
               /* IDeclaration *new_decl = m_target_module->clone_decl(const_decl, this);
                return m_sf.create_declaration(new_decl);*/
            }
            return m_sf.create_declaration(m_df.create_invalid());
        }
    case IStatement::SK_DO_WHILE:
        {
            IStatement_do_while const *stmt_do = cast<IStatement_do_while>(stmt);
            IExpression const *new_cond = NULL;
            if (IExpression const *cond = stmt_do->get_condition()) {
                new_cond = m_target_module->clone_expr(cond, this);
            }
            IStatement *new_body = clone_statement(stmt_do->get_body());

            return m_sf.create_do_while(new_cond, new_body);
        }
    case IStatement::SK_FOR:
        {
            IStatement_for const* stmt_for = cast<IStatement_for>(stmt);
            IExpression const *new_update = NULL, *new_cond = NULL;
            if (IExpression const *update = stmt_for->get_update()) {
                new_update = m_target_module->clone_expr(update, this);
            }
            if (IExpression const *cond = stmt_for->get_condition()) {
                new_cond = m_target_module->clone_expr(cond, this);
            }
            IStatement *new_init = clone_statement(stmt_for->get_init());
            IStatement *new_body = clone_statement(stmt_for->get_body());

            return m_sf.create_for(new_init, new_cond, new_update, new_body);
        }
    case IStatement::SK_IF:
        {
            IStatement_if const *stmt_if = cast<IStatement_if>(stmt);
            IExpression const *new_cond = NULL;
            if (IExpression const *cond = stmt_if->get_condition()) {
                new_cond = m_target_module->clone_expr(cond, this);
            }
            IStatement *new_then = clone_statement(stmt_if->get_then_statement());
            IStatement *new_else = clone_statement(stmt_if->get_else_statement());

            return m_sf.create_if(new_cond, new_then, new_else);
        }
    case IStatement::SK_SWITCH:
        {
            IStatement_switch const *stmt_sw = cast<IStatement_switch>(stmt);
            IExpression const *new_cond = NULL;
            if (IExpression const *cond = stmt_sw->get_condition()) {
                new_cond = m_target_module->clone_expr(cond, this);
            }
            IStatement_switch *new_sw = m_sf.create_switch(new_cond);
            for (int i = 0, n = stmt_sw->get_case_count(); i < n; ++i) {
                new_sw->add_case(clone_statement(stmt_sw->get_case(i)));
            }
            return new_sw;
        }
    case IStatement::SK_WHILE:
        {
            IStatement_while const *stmt_w = cast<IStatement_while>(stmt);
            IExpression const *new_cond = NULL;
            if (IExpression const *cond = stmt_w->get_condition()) {
                new_cond = m_target_module->clone_expr(cond, this);
            }
            IStatement *new_body = clone_statement(stmt_w->get_body());

            return m_sf.create_while(new_cond, new_body);
        }
    default:
        break;
    }
    return m_sf.create_invalid();
}

IAnnotation *Module_inliner::create_anno(
    char const *anno_name) const
{
    IQualified_name *qn = m_nf.create_qualified_name();
    qn->add_component(m_nf.create_simple_name(m_nf.create_symbol("anno")));
    qn->add_component(m_nf.create_simple_name(m_nf.create_symbol(anno_name)));

    IAnnotation *anno = m_af.create_annotation(qn);
    return anno;
}

IAnnotation *Module_inliner::create_string_anno(
    char const *anno_name,
    char const *value) const
{
    IValue_string const *s   = m_target_module->get_value_factory()->create_string(value);
    IExpression_literal *lit = m_ef.create_literal(s);
    IArgument_positional const *arg = m_ef.create_positional_argument(lit);

    IAnnotation *anno = create_anno(anno_name);
    anno->add_argument(arg);

    return anno;
}

// Creates or alters the annotation block of a definition.
IAnnotation_block *Module_inliner::create_annotation_block(
    IDefinition const       *def,
    IAnnotation_block const *anno_block)
{
    if (m_is_root && anno_block != NULL) {
        return m_target_module->clone_annotation_block(anno_block, this);
    }
    bool is_anno = def->get_declaration()->get_kind() == IDeclaration::DK_ANNOTATION;

    IAnnotation_block *new_block = m_af.create_annotation_block();
    // create a "hidden" annotation for exported non-root entities
    if (!is_anno && m_exports.find(def) != m_exports.end()) {
        IQualified_name *qn = m_nf.create_qualified_name();
        qn->add_component(m_nf.create_simple_name(m_nf.create_symbol("anno")));
        qn->add_component(m_nf.create_simple_name(m_nf.create_symbol("hidden")));
        IAnnotation *anno_hidden = m_af.create_annotation(qn);
        new_block->add_annotation(anno_hidden);
    }

    IAnnotation *anno_display_name = NULL;
    IAnnotation *anno_origin       = NULL;

    if (anno_block != NULL) {
        for (int i = 0, n = anno_block->get_annotation_count(); i < n; ++i) {
            IAnnotation const     *anno = anno_block->get_annotation(i);
            IQualified_name const *qn   = anno->get_name();
            IDefinition const     *def  = qn->get_definition();

            if (!is_anno) {
                // only for non-annotations
                switch (def->get_semantics()) {
                case IDefinition::DS_DISPLAY_NAME_ANNOTATION:
                    // copy and remember the display_name annotation
                    anno_display_name = m_target_module->clone_annotation(anno, /*modifier=*/NULL);
                    continue;
                case IDefinition::DS_DESCRIPTION_ANNOTATION:
                    // copy the description annotation
                    register_import("::anno", "description");
                    new_block->add_annotation(
                        m_target_module->clone_annotation(anno, /*modifier=*/NULL));
                    continue;
                case IDefinition::DS_NOINLINE_ANNOTATION:
                    // copy the noinline annotation, this is necessary for the function hash
                    // based replacement
                    register_import("::anno", "noinline");
                    new_block->add_annotation(
                        m_target_module->clone_annotation(anno, /*modifier=*/NULL));
                    continue;
                default:
                    // ignore
                    break;
                }
            }

            // for all entities
            if (def->get_semantics() == IDefinition::DS_ORIGIN_ANNOTATION) {
                // copy and remember the origin annotation
                anno_origin = m_target_module->clone_annotation(anno, /*modifier=*/ NULL);
            }
        }
    }

    if (anno_origin == NULL) {
        if (!m_omit_anno_origin) {
            string name_ori(m_alloc);
            name_ori += m_module->get_name();
            name_ori += "::";
            name_ori += def->get_symbol()->get_name();

            anno_origin = create_string_anno("origin", name_ori.c_str());
            new_block->add_annotation(anno_origin);
        }
    } else {
        new_block->add_annotation(anno_origin);
    }

    if (!is_anno) {
        if (anno_display_name == NULL) {
            anno_display_name = create_string_anno("display_name", def->get_symbol()->get_name());
        }
        new_block->add_annotation(anno_display_name);
    }
    return new_block;
}

// Clones parameter annotations.
// For non-root material parameters, annotations are dropped, except for anno::unused.
IAnnotation_block const *Module_inliner::clone_parameter_annotations(
    IAnnotation_block const *anno_block,
    char const              *display_name)
{
    if (m_is_root) {
        // simply clone the block
        return m_target_module->clone_annotation_block(anno_block, this);
    }

    IAnnotation_block *new_block = m_af.create_annotation_block();
    bool has_display_name = false;
    if (anno_block != NULL) {
        for (int i = 0, n = anno_block->get_annotation_count(); i < n; ++i) {
            IAnnotation const     *anno = anno_block->get_annotation(i);
            IQualified_name const *qn   = anno->get_name();
            IDefinition const     *def  = qn->get_definition();

            if (def->get_semantics() == IDefinition::DS_UNUSED_ANNOTATION) {
                register_import("::anno", "unused");
                new_block->add_annotation(
                    m_target_module->clone_annotation(anno, this));
            } else if (def->get_semantics() == IDefinition::DS_DISPLAY_NAME_ANNOTATION) {
                new_block->add_annotation(
                    m_target_module->clone_annotation(anno, /*modifier=*/ this));
                has_display_name = true;
            } else if (def->get_semantics() == IDefinition::DS_DESCRIPTION_ANNOTATION) {
                register_import("::anno", "description");
                new_block->add_annotation(
                    m_target_module->clone_annotation(anno, /*modifier=*/ this));
            }
        }
    }

    if (!has_display_name && display_name != NULL) {
        new_block->add_annotation(create_string_anno("display_name", display_name));
    }

    return new_block->get_annotation_count() > 0 ? new_block : NULL;
}

// Clones an annotation block completely.
IAnnotation_block const *Module_inliner::clone_annotations(
    IAnnotation_block const *anno_block)
{
    return m_target_module->clone_annotation_block(anno_block, this);
}

// Clones a function declaration.
IDeclaration_function *Module_inliner::clone_declaration(
    IDeclaration_function const *decl,
    ISimple_name const          *function_name,
    bool                        is_exported)
{
    // clone return type
    IType_name *ret_tn =
        m_target_module->clone_name(decl->get_return_type_name(), this);

    // clone body
    IStatement const *stmt = decl->get_body();
    IStatement *new_stmt = clone_statement(stmt);

    IAnnotation_block *new_fct_annos =
        create_annotation_block(decl->get_definition(), decl->get_annotations());
    IAnnotation_block *new_ret_annos = NULL;
    if (m_is_root) {
        new_ret_annos = m_target_module->clone_annotation_block(
            decl->get_return_annotations(), this);
    }

    IDeclaration_function *new_decl = m_df.create_function(
        ret_tn, new_ret_annos, function_name, decl->is_preset(),
        new_stmt, new_fct_annos, /*is_exported=*/is_exported);
    new_decl->set_qualifier(decl->get_qualifier());

    bool clone_init = (m_exports.find(decl->get_definition()) != m_exports.end() ||
        m_keep_non_root_parameter_defaults || m_is_root);
    for (int i = 0, n = decl->get_parameter_count(); i < n; ++i) {
        new_decl->add_parameter(
            clone_parameter(decl->get_parameter(i), clone_init));
    }
    return new_decl;
}

// Clone the given parameter.
IParameter const *Module_inliner::clone_parameter(
    IParameter const *param,
    bool             clone_init)
{
    IType_name         *tn = m_target_module->clone_name(param->get_type_name(), this);
    ISimple_name const *sn = m_target_module->clone_name(param->get_name());

    IExpression *new_init = NULL;
    if (clone_init) {
        if (IExpression const *init_expr = param->get_init_expr()) {
            new_init = m_target_module->clone_expr(init_expr, this);
        }
    }
    IAnnotation_block const *new_annos = NULL;

    if (m_is_root) {
        new_annos = clone_parameter_annotations(
            param->get_annotations(), /*display_name=*/NULL);
    }
    return m_df.create_parameter(tn, sn, new_init, new_annos);
}

// Clones an enum declaration.
IDeclaration_type_enum *Module_inliner::clone_declaration(
    IDeclaration_type_enum const *decl,
    ISimple_name const           *enum_name,
    bool                         is_exported)
{
    IAnnotation_block *new_enum_annos = create_annotation_block(
        decl->get_definition(),
        decl->get_annotations());
    IDeclaration_type_enum *new_enum = m_df.create_enum(
        enum_name, new_enum_annos, is_exported, decl->is_enum_class());

    for (int i = 0, n = decl->get_value_count(); i < n; ++i) {
        ISimple_name const *value_name = decl->get_value_name(i);
        ISimple_name const *new_name;
        if (m_is_root) {
            new_name = m_target_module->clone_name(value_name);
        } else {
            new_name = generate_name(value_name, "ev_");
            m_references.insert(Reference_map::value_type(
                value_name->get_definition(),
                new_name->get_symbol()));
        }
        IExpression *new_init = NULL;
        if (IExpression const *init_expr = decl->get_value_init(i)) {
            new_init = m_target_module->clone_expr(init_expr, this);
        }
        IAnnotation_block const *new_annos = clone_parameter_annotations(
            decl->get_annotations(i), value_name->get_symbol()->get_name());
        new_enum->add_value(new_name, new_init, new_annos);
    }
    return new_enum;
}

// Clones an struct declaration.
IDeclaration_type_struct *Module_inliner::clone_declaration(
    IDeclaration_type_struct const *decl,
    ISimple_name const             *struct_name,
    bool                           is_exported)
{
    IAnnotation_block *new_struct_annos = create_annotation_block(
            decl->get_definition(),
            decl->get_annotations());

    IDeclaration_type_struct *new_struct = m_df.create_struct(
        struct_name, new_struct_annos, is_exported);

    for (int i = 0, n = decl->get_field_count(); i < n; ++i) {
        ISimple_name const *new_name = m_target_module->clone_name(decl->get_field_name(i));
        IExpression        *new_init = NULL;
        if (IExpression const *init_expr = decl->get_field_init(i)) {
            new_init = m_target_module->clone_expr(init_expr, this);
        }
        IType_name const *new_tn =
            m_target_module->clone_name(decl->get_field_type_name(i), this);
        IAnnotation_block const *new_annos = clone_parameter_annotations(
            decl->get_annotations(i), /*display_name=*/NULL);
        new_struct->add_field(new_tn, new_name, new_init, new_annos);
    }
    return new_struct;
}

// Clones an annotation declaration.
IDeclaration_annotation *Module_inliner::clone_declaration(
    IDeclaration_annotation const *decl,
    ISimple_name const            *anno_name,
    bool                          is_exported)
{
    IAnnotation_block const *annos = create_annotation_block(
        decl->get_definition(),
        decl->get_annotations());

    IDeclaration_annotation *new_decl = m_df.create_annotation(
        anno_name, annos, is_exported);

    for (int i = 0, n = decl->get_parameter_count(); i < n; ++i) {
        new_decl->add_parameter(m_target_module->clone_param(
            decl->get_parameter(i),
            /*clone_init=*/true,
            /*clone_anno=*/true,
            this));
    }
    return new_decl;
}

// Clones a constant declaration.
void Module_inliner::clone_declaration(
    IDeclaration_constant *decl,
    char const            *prefix)
{
    if (m_is_root) {
        bool is_exported = decl->is_exported();

        IDeclaration_constant *cst_decl = m_df.create_constant(
            m_target_module->clone_name(decl->get_type_name(), this),
            is_exported);

        for (int i = 0, n = decl->get_constant_count(); i < n; ++i) {
            ISimple_name const *c_name = decl->get_constant_name(i);

            ISimple_name const *new_name = m_target_module->clone_name(c_name);
            IExpression const  *c_expr =
                m_target_module->clone_expr(decl->get_constant_exp(i), this);

            cst_decl->add_constant(new_name, c_expr);
        }
        m_target_module->add_declaration(cst_decl);
    } else {
        for (int i = 0, n = decl->get_constant_count(); i < n; ++i) {
            ISimple_name const *c_name = decl->get_constant_name(i);
            IDefinition const  *def    = c_name->get_definition();

            MDL_ASSERT(m_references.find(def) == m_references.end());

            ISimple_name const *new_name = generate_name(c_name, prefix);
            IExpression const  *c_expr   =
                m_target_module->clone_expr(decl->get_constant_exp(i), this);

            bool is_exported = m_exports.find(def) != m_exports.end();

            IDeclaration_constant *cst_decl = m_df.create_constant(
                m_target_module->clone_name(decl->get_type_name(), this),
                is_exported);

            cst_decl->add_constant(new_name, c_expr);

            m_target_module->add_declaration(cst_decl);

            // Enter it into the references map.
            m_references.insert(Reference_map::value_type(def, new_name->get_symbol()));
        }
    }

}

// Creates a reference from a simple name.
IExpression_reference* Module_inliner::simple_name_to_reference(
    ISimple_name const* name)
{
    IQualified_name *qualified_name = m_nf.create_qualified_name();
    qualified_name->add_component(name);

    IType_name const *type_name = m_nf.create_type_name(qualified_name);
    return m_ef.create_reference(type_name);
}

// Adds the given entity to the import set.
void Module_inliner::register_import(char const *module_name, char const *symbol_name)
{
    string import(module_name, m_alloc);
    import += "::";
    import += symbol_name;
    m_imports.insert(import);
}

void Module_inliner::register_child_types(
    IType const *tp)
{
    tp = tp->skip_type_alias();

    switch (tp->get_kind()) {
    case IType::TK_FUNCTION:
        {
            IType_function const *f_tp = cast<IType_function>(tp);

            if (IType const *ret_tp = f_tp->get_return_type()) {
                do_type(ret_tp);
            }
            for (int i = 0, n = f_tp->get_parameter_count(); i < n; ++i) {
                ISymbol const *p_sym;
                IType const   *p_tp;

                f_tp->get_parameter(i, p_tp, p_sym);
                do_type(p_tp);
            }
        }
        break;
    case IType::TK_STRUCT:
        {
            IType_struct const *s_tp = cast<IType_struct>(tp);

            for (int i = 0, n = s_tp->get_field_count(); i < n; ++i) {
                ISymbol const *f_sym;
                IType const   *f_tp;

                s_tp->get_field(i, f_tp, f_sym);
                do_type(f_tp);
            }
        }
        break;
    default:
        // do nothing
        break;
    }
}

// Handles a type.
void Module_inliner::do_type(IType const *t)
{
    if (t != NULL) {
        t = t->skip_type_alias();
        if (is<IType_texture>(t)) {
            register_import("::tex", "gamma_mode");
            return;
        }

        if (IType_array const *ta = as<IType_array>(t)) {
            t = ta->get_element_type()->skip_type_alias();
        }

        Definition const *type_def = get_type_definition(m_module.get(), t);
        if (!m_is_root || type_def->has_flag(Definition::DEF_IS_IMPORTED)) {
            mi::base::Handle<Module const> orig_module(m_module->get_owner_module(type_def));
            Definition const *orig_type_def = m_module->get_original_definition(type_def);
            if (!needs_inline(orig_module.get())) {
                if (need_type_import(t)) {
                    register_import(orig_module->get_name(), type_def->get_symbol()->get_name());
                }

                {
                    Module_scope scope(m_module, orig_module);
                    register_child_types(orig_type_def->get_type());
                }
                return;
            }

            if (!need_type_import(t)) {
                return;
            }

            if (m_references.find(orig_type_def) != m_references.end()) {
                return;
            }

            IDeclaration const *type_decl = orig_type_def->get_declaration();
            MDL_ASSERT(type_decl);

            // Traverse original declaration
            Module_inliner inliner(
                m_alloc,
                orig_module.get(),
                m_target_module.get(),
                m_inline_imports,
                m_omit_anno_origin,
                /*is_root_module=*/false,
                m_references,
                m_imports,
                m_exports,
                m_counter);
            inliner.visit(type_decl);
        }
    }
}

bool Module_inliner::needs_inline(IModule const *module) const
{
    return m_inline_imports->inline_import(module);
}

// Creates a reference to an imported entity.
mi::mdl::IExpression *Module_inliner::create_imported_reference(
    Definition const *imported_def,
    Definition const *orig_def)
{
    // construct qualified name for reference
    mi::base::Handle<Module const> orig_module(m_module->get_owner_module(imported_def));
    MDL_ASSERT(!needs_inline(orig_module.get()));

    IQualified_name       *qn_new = m_nf.create_qualified_name();
    IQualified_name const *mqn    = orig_module->get_qualified_name();

    for (int i = 0, n = mqn->get_component_count(); i < n; ++i) {
        qn_new->add_component(m_target_module->clone_name(mqn->get_component(i)));
    }
    ISymbol const *sym_new = m_nf.create_symbol(orig_def->get_symbol()->get_name());
    qn_new->add_component(m_nf.create_simple_name(sym_new));

    IType_name const *type_name = m_nf.create_type_name(qn_new);
    return m_ef.create_reference(type_name);
}

// Constructor.
Exports_collector::Exports_collector(
    Module const *module,
    Def_set      &def_set,
    bool         is_root)
: m_module(mi::base::make_handle_dup(module))
, m_def_set(def_set)
, m_is_root(is_root)
{
}

// Collect entities.
void Exports_collector::collect()
{
    visit(m_module.get());
}

bool Exports_collector::pre_visit(IDeclaration_function *decl)
{
    if (m_is_root) {
        if (IAnnotation_block const *anno = decl->get_annotations()) {
            visit(anno);
        }
        if (IAnnotation_block const *ranno = decl->get_return_annotations()) {
            visit(ranno);
        }
    }

    for (int i = 0, n = decl->get_parameter_count(); i < n; ++i) {
        visit(decl->get_parameter(i));
    }

    visit(decl->get_return_type_name());

    if (decl->is_preset()) {
        IStatement_expression const *preset_body
            = cast<IStatement_expression>(decl->get_body());
        IExpression const           *expr = preset_body->get_expression();

        // skip let expressions
        while (IExpression_let const *let = as<IExpression_let>(expr)) {
            expr = let->get_expression();
        }
        IExpression_call const      *inst = cast<IExpression_call>(expr);

        // visit the call to collect possible exports
        visit(inst);
    }

    return false;
}

bool Exports_collector::pre_visit(IParameter *param)
{
    if (m_is_root) {
        if (IAnnotation_block const *anno = param->get_annotations()) {
            visit(anno);
        }
    }

    visit(param->get_type_name());

    if (IExpression const *init = param->get_init_expr()) {
        visit(init);
    }

    return false;
}

void Exports_collector::post_visit(IAnnotation *anno)
{
    IDefinition const *d = anno->get_name()->get_definition();
    if (impl_cast<Definition>(d)->get_original_import_idx() != 0) {
        mi::base::Handle<Module const> mod_ori(m_module->get_owner_module(d));
        if (mod_ori->is_builtins() || mod_ori->is_stdlib()) {
            return;
        }
        m_def_set.insert(m_module->get_original_definition(d));
    }
}

bool Exports_collector::pre_visit(IDeclaration_type_struct *decl)
{
    if (m_is_root)
        if (IAnnotation_block const *anno = decl->get_annotations()) {
            visit(anno);
        }

    for (int i = 0, n = decl->get_field_count(); i < n; ++i) {
        visit(decl->get_field_type_name(i));
        if (decl->get_field_init(i)) {
            visit(decl->get_field_init(i));
        }
        if (m_is_root) {
            if (IAnnotation_block const *anno = decl->get_annotations(i)) {
                visit(anno);
            }
        }
    }
    return false;
}

IExpression *Exports_collector::post_visit(IExpression_reference *ref)
{
    if (ref->is_array_constructor()) {
        return ref;
    }
    Definition const *def = impl_cast<Definition>(ref->get_definition());

    if (def->get_kind() != Definition::DK_FUNCTION &&
        def->get_kind() != IDefinition::DK_CONSTANT)
    {
        return ref;
    }

    mi::base::Handle<Module const> mod_ori(mi::base::make_handle_dup(m_module.get()));
    bool is_root = m_is_root;
    if (def->get_original_import_idx() != 0) {
        mod_ori = m_module->get_owner_module(def);
        if (mod_ori->is_builtins() || mod_ori->is_stdlib()) {
            return ref;
        }
        def = m_module->get_original_definition(def);
        is_root = false;
    }
    if (m_def_set.insert(def).second) {
        // traverse reference declaration
        Exports_collector c(mod_ori.get(), m_def_set, is_root);
        c.visit(def->get_declaration());
    }
    return ref;
}

IExpression *Exports_collector::post_visit(IExpression_literal *lit)
{
    handle_type(lit->get_type());
    return lit;
}

void Exports_collector::post_visit(IType_name *tn)
{
    if (IType const *t = tn->get_type()) {
        // this typename names a real type
        handle_type(t);
    } else {
        // this typename names a non-type entity
        IQualified_name const *qn = tn->get_qualified_name();
        if (Definition const *d = impl_cast<Definition>(qn->get_definition())) {
            if (d->get_kind() == IDefinition::DK_ENUM_VALUE) {
                handle_type(d->get_type());
            } else if (d->get_kind() == IDefinition::DK_MEMBER) {
                // handle potentially nested struct member access
                while (Definition const *p = d->get_def_scope()->get_owner_definition()) {
                    handle_type(p->get_type());
                    d = p;
                }
            }
        }
    }
}

// Collect user type definition.
void Exports_collector::handle_type(IType const *t) {
    t = t->skip_type_alias();
    if (IType_array const *ta = as<IType_array>(t)) {
        t = ta->get_element_type();
    }
    if (need_type_import(t)) {
        Definition const *type_def = get_type_definition(m_module.get(), t);
        mi::base::Handle<Module const> mod_ori(mi::base::make_handle_dup(m_module.get()));

        bool is_root = m_is_root;
        if (type_def->has_flag(Definition::DEF_IS_IMPORTED)) {
            mod_ori = m_module->get_owner_module(type_def);
            if (mod_ori->is_builtins() || mod_ori->is_stdlib()) {
                return;
            }
            type_def = m_module->get_original_definition(type_def);
            is_root = false;
        }
        if (m_def_set.insert(type_def).second && t->get_kind() != IType::TK_ENUM) {
            // traverse struct declaration
            Exports_collector c(mod_ori.get(), m_def_set, is_root);
            c.visit(type_def->get_declaration());
        }
    }
}

static bool is_mdle(char const *module_name)
{
    if (module_name[0] == ':' && module_name[1] == ':' &&
        (module_name[2] == '/' ||
        (isalpha(module_name[2]) && module_name[3] == ':' && module_name[4] == '/')))
    {
        return true;
    }

    return false;
}

class Inline_import_callback : public IInline_import_callback
{
public:
    Inline_import_callback(bool inline_mdle) : m_inline_mdle(inline_mdle) { }

    bool inline_import(IModule const *module)
    {
        if (m_inline_mdle) {
            return is_mdle(module->get_name());
        }
        if (module->is_stdlib() || module->is_builtins()) {
            // never inline the standard modules (besides it will fail)
            return false;
        }

        return true;
    }

private:
    bool m_inline_mdle;
};

// Creates a new error.
// Constructor.
MDL_module_transformer::MDL_module_transformer(
    IAllocator *alloc,
    MDL        *compiler)
: Base(alloc)
, m_compiler(mi::base::make_handle_dup(compiler))
, m_msg_list(alloc, /*owner_fname=*/"")
, m_last_msg_idx(0)
{
}

// Inline all imports of a module, creating a new one.
IModule const *MDL_module_transformer::inline_imports(IModule const *imodule)
{
    return inline_module(imodule, false);
}

// Inline all MDLE imports of a module, creating a new one.
IModule const *MDL_module_transformer::inline_mdle(IModule const *imodule)
{
    return inline_module(imodule, true);
}

// Compute the resulting MDL version.
IMDL::MDL_version MDL_module_transformer::compute_mdl_version(
    Module const *root)
{
    IMDL::MDL_version v = root->get_mdl_version();

    for (int i = 0, n = root->get_import_count(); i < n; ++i) {
        base::Handle<Module const> import(root->get_import(i));
        if (import->is_builtins() || import->is_stdlib()) {
            continue;
        }

        IMDL::MDL_version imp_v = import->get_mdl_version();
        if (imp_v > v) {
            v = imp_v;
        }
    }

    // MDL 1.5 is required for
    // - the cast<> operator (needed by MDLE)
    // - the ::anno::origin anotation
    if (v < IMDL::MDL_VERSION_1_5) {
        v = IMDL::MDL_VERSION_1_5;
    }
    return v;
}

// Inline all imports of a module, creating a new one.
IModule const *MDL_module_transformer::inline_module(IModule const *imodule, bool inline_mdle)
{
    if (!imodule->is_valid()) {
        // only valid modules can be inlined
        error(SOURCE_MODULE_INVALID, Error_params(get_allocator()));
        return NULL;
    }

    Module const *module = impl_cast<Module>(imodule);

    bool user_imports = false;
    for (int i = 0, n = module->get_import_count(); i < n; ++i) {
        base::Handle<IModule const> im(module->get_import(i));
        if (!(im->is_builtins() || im->is_stdlib())) {
            user_imports = true;
            break;
        }
    }
    if (!user_imports) {
        // Nothing to be inlined
        // TODO: should it be copied anyway?
        module->retain();
        return module;
    }

    // Create new module
    string module_name(imodule->get_name(), get_allocator());

    IMDL::MDL_version v = compute_mdl_version(module);

    base::Handle<Module> inlined_module(
        m_compiler->create_module(
            /*context=*/NULL,
            module_name.c_str(),
            v));

    // setup inliner ...
    Inline_import_callback callback(inline_mdle);

    Def_set exports(
        0, Def_set::hasher(), Def_set::key_equal(), get_allocator());

    Module_inliner::Reference_map references(
        Module_inliner::Reference_map::key_compare(), get_allocator());

    Module_inliner::Import_set imports(
        Module_inliner::Import_set::key_compare(), get_allocator());

    size_t counter = 0;
    Module_inliner inliner(
        get_allocator(),
        module,
        inlined_module.get(),
        &callback,
        /*omit_anno_origin=*/false,
        /*is_root_module=*/true,
        references,
        imports,
        exports,
        counter);

    // and run it.
    inliner.run();

    // check it
    MDL_ASSERT(AST_checker::check_module(inlined_module.get()) && "inlined AST containes errors");

    // Finalize the new module
    inlined_module->analyze(/*cache=*/NULL, /*context=*/NULL);
    if (!inlined_module->is_valid()) {
        error(
            INLINING_MODULE_FAILED,
            Error_params(get_allocator())
                .add(imodule->get_name()));
        Messages const &messages = inlined_module->access_messages();
        for (size_t i = 0, n = messages.get_message_count(); i < n; ++i) {
            m_msg_list.add_imported(m_last_msg_idx, 0, messages.get_message(i));
        }
        return NULL;
    }

    inlined_module->retain();
    return inlined_module.get();
}

// Access messages of last operation.
Messages const &MDL_module_transformer::access_messages() const
{
    return m_msg_list;
}

// Creates a new error.
void MDL_module_transformer::error(int code, Error_params const &params)
{
    Position_impl zero(0, 0, 0, 0);

    string msg(m_msg_list.format_msg(code, MESSAGE_CLASS, params));
    m_last_msg_idx = m_msg_list.add_error_message(
        code, MESSAGE_CLASS, 0, &zero, msg.c_str());
}

// Creates a new warning.
void MDL_module_transformer::warning(int code, Error_params const &params)
{
    Position_impl zero(0, 0, 0, 0);

    string msg(m_msg_list.format_msg(code, MESSAGE_CLASS, params));
    m_last_msg_idx = m_msg_list.add_warning_message(
        code, MESSAGE_CLASS, 0, &zero, msg.c_str());
}

}  // mdl
}  // mi
