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

#ifndef MDL_COMPILERCORE_MODULE_TRANSFORMER_H
#define MDL_COMPILERCORE_MODULE_TRANSFORMER_H 1

#include <mi/base/handle.h>

#include <mi/mdl/mdl_annotations.h>
#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_module_transformer.h>
#include <mi/mdl/mdl_types.h>

#include "compilercore_allocator.h"
#include "compilercore_mdl.h"
#include "compilercore_messages.h"
#include "compilercore_modules.h"
#include "compilercore_options.h"
#include "compilercore_visitor.h"

namespace mi {
namespace mdl {

typedef ptr_hash_set<IDefinition const>::Type Def_set;

/// An interface that decides whether to inline a given module.
class IInline_import_callback
{
public:
    /// Return whether to inline the given module or not.
    virtual bool inline_import(IModule const *module) = 0;
};

/// Traverses a module and copies all entities into the target module.
class Module_inliner : protected Module_visitor, public IClone_modifier
{
public:

    typedef map<IDefinition const *, ISymbol const* >::Type Reference_map;
    typedef set<string >::Type Import_set;

    /// Constructor
    ///
    /// \param alloc               The allocator.
    /// \param module              The current module.
    /// \param target_module       The target module.
    /// \param inline_imports      The callback that decides whether to inline a given module.
    /// \param omit_anno_origin    True, if no ::anno::origin annotations should be added.
    /// \param is_root_module      True, if 'module' is the root module of the traversal.
    /// \param references          A map holding the references that have already been cloned.
    /// \param imports             A set holding standard modules to be imported.
    /// \param visible_definitions Definitions visible at the interface of the root module.
    /// \param counter             Used to generate unique names.
    Module_inliner(
        IAllocator                                *alloc,
        Module const                              *module,
        Module                                    *target_module,
        IInline_import_callback                   *inline_imports,
        bool                                      omit_anno_origin,
        bool                                      is_root_module,
        Reference_map                             &references,
        Import_set                                &imports,
        Def_set                                   &visible_definitions,
        size_t                                    &counter
    );

    ~Module_inliner();

    /// Runs the inliner on the src module passed to the constructor.
    void run();

    /// If the referenced function can be found in the reference map, it is replaced
    /// by a reference to the inlined function.
    IExpression *clone_expr_reference(IExpression_reference const *ref) MDL_FINAL;

    /// Creates a reference to an imported entity.
    ///
    /// \param imported_def  the definition of the imported entity
    /// \param orig_def      the original definition (from the owner module)
    ///
    /// \note We always create fully specified imports in the module inliner, hence the created
    ///       reference uses a full qualified name here
    IExpression *create_imported_reference(
        Definition const *imported_def,
        Definition const *orig_def);

    /// Clone a call.
    IExpression *clone_expr_call(IExpression_call const *c_expr) MDL_FINAL;

    /// Clone a literal.
    IExpression *clone_literal(IExpression_literal const *lit) MDL_FINAL;

    /// Clone a qualified name.
    IQualified_name *clone_name(IQualified_name const *qname) MDL_FINAL;

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
    static IExpression_reference const *promote_call_reference(
        Module                      &mod,
        int                         major,
        int                         minor,
        IExpression_reference const *ref,
        IClone_modifier             *modifier,
        unsigned                     &rules);

protected:

    /// Called, before an annotation block is visited.
    ///
    bool pre_visit(IAnnotation_block *anno) MDL_FINAL;

    /// Called, whenever a reference has been visited.
    ///
    /// If the referenced element originates from a different module, its
    /// declaration is traversed and eventually copied.
    IExpression *post_visit(IExpression_reference *expr) MDL_FINAL;

    /// End of a literal expression.
    IExpression *post_visit(IExpression_literal *expr) MDL_FINAL;

    /// End of a binary expression.
    IExpression *post_visit(IExpression_binary *expr) MDL_FINAL;

    void post_visit(IType_name *tn) MDL_FINAL;

    void post_visit(IAnnotation *anno) MDL_FINAL;

    template <typename T>
    void clone_declaration(T* decl, char const *prefix)
    {
        if (m_is_root) {
            m_target_module->add_declaration(clone_declaration(
                decl,
                m_target_module->clone_name(decl->get_name()),
                decl->is_exported()));
        } else {
            IDefinition const *def = decl->get_definition();
            MDL_ASSERT(m_references.find(def) == m_references.end());

            ISimple_name const *new_name = generate_name(decl->get_name(), prefix);

            bool is_exported = m_exports.find(def) != m_exports.end();
            m_target_module->add_declaration(
                clone_declaration(decl, new_name, is_exported));

            // Enter it into the references map.
            m_references.insert(Reference_map::value_type(def, new_name->get_symbol()));
        }
    }

    /// Checks, if the non-root declaration has already been copied.
    bool pre_visit(IDeclaration_type_struct *decl) MDL_FINAL;

    /// Clones a struct declaration
    void post_visit(IDeclaration_type_struct *decl) MDL_FINAL;

    /// Checks, if the non-root declaration has already been copied.
    bool pre_visit(IDeclaration_type_enum *decl) MDL_FINAL;

    /// Clones an enum declaration
    void post_visit(IDeclaration_type_enum *decl) MDL_FINAL;

    /// Checks, if the non-root declaration has already been copied.
    bool pre_visit(IDeclaration_function *decl) MDL_FINAL;

    /// Clones the given declaration and puts it into the reference map.
    void post_visit(IDeclaration_function *decl) MDL_FINAL;

    /// Checks, if the non-root declaration has already been copied.
    bool pre_visit(IDeclaration_annotation *decl) MDL_FINAL;

    /// Clones the given declaration and puts it into the reference map.
    void post_visit(IDeclaration_annotation *decl) MDL_FINAL;

    /// Checks, if the non-root declaration has already been copied.
    bool pre_visit(IDeclaration_constant *decl) MDL_FINAL;

    /// Clones the given declaration and puts it into the reference map.
    void post_visit(IDeclaration_constant *decl) MDL_FINAL;

private:

    /// Generates a call to the cloned constructor from the values of the original compound.
    IExpression_call *make_constructor(
        IValue_compound const *vs_ori,
        ISymbol const         *new_sym);

    /// Generates a new name from a simple name and a prefix.
    ISimple_name const *generate_name(
        ISimple_name const *sn,
        char const         *prefix);

    /// Clones a statement.
    IStatement *clone_statement(IStatement const *stmt);

    /// Creates or alters the annotation block of a definition.
    IAnnotation_block *create_annotation_block(
        IDefinition const       *def,
        IAnnotation_block const *anno_block);

    /// Clones parameter annotations.
    ///
    /// \param anno_block   the annotation block to clone
    /// \param display_name if non-NULL, add a display_name annotation
    ///
    /// For non-root material parameters, annotations are dropped, except for anno::unused,
    /// description and display_name. if no display name exists in the original annotation
    /// block, but the passed display_name is non-NULL, a new annotation is added.
    IAnnotation_block const *clone_parameter_annotations(
        IAnnotation_block const *anno_block,
        char const              *display_name);

    /// Clones an annotation block completely.
    ///
    /// \param anno_block   the annotation block to clone
    IAnnotation_block const *clone_annotations(
        IAnnotation_block const *anno_block);

    /// Clones a function declaration.
    ///
    /// \param decl             The declaration to clone.
    /// \param function_name    The new function name.
    /// \param is_exported      True, if the new declaration is supposed to be exported
    ///                         from the module.
    ///
    /// \return The cloned declaration.
    IDeclaration_function *clone_declaration(
        IDeclaration_function const *decl,
        ISimple_name const          *function_name,
        bool                        is_exported);

    /// Clones the given parameter.
    ///
    /// \param param            The parameter to clone.
    /// \param clone_init       True, if init expressions should be cloned.
    IParameter const *clone_parameter(
        IParameter const *param,
        bool clone_init);

    /// Clones an enum declaration.
    ///
    /// \param decl             The declaration to clone.
    /// \param enum_name        The new enum name.
    /// \param is_exported      True, if the new declaration is supposed to be exported
    ///                         from the module.
    ///
    /// \return The cloned declaration.
    IDeclaration_type_enum *clone_declaration(
        IDeclaration_type_enum const *decl,
        ISimple_name const           *enum_name,
        bool                         is_exported);

    /// Clones an struct declaration.
    ///
    /// \param decl             The declaration to clone.
    /// \param struct_name      The new struct name.
    /// \param is_exported      True, if the new declaration is supposed to be exported
    ///                         from the module.
    ///
    /// \return The cloned declaration.
    IDeclaration_type_struct *clone_declaration(
        IDeclaration_type_struct const *decl,
        ISimple_name const             *struct_name,
        bool                           is_exported);

    /// Clones an annotation declaration.
    ///
    /// \param decl             The declaration to clone.
    /// \param anno_name        The new annotation name.
    /// \param is_exported      True, if the new declaration is supposed to be exported
    ///                         from the module.
    ///
    /// \return The cloned declaration.
    IDeclaration_annotation *clone_declaration(
        IDeclaration_annotation const *decl,
        ISimple_name const            *anno_name,
        bool                          is_exported);

    /// Clones a constant declaration.
    ///
    /// \param decl             The declaration to clone.
    /// \param prefix           The new name prefix.
    ///
    /// \return The cloned declaration.
    void clone_declaration(
        IDeclaration_constant *decl,
        char const            *prefix);

    /// Creates a reference from a simple name.
    IExpression_reference *simple_name_to_reference(
        ISimple_name const* name);

    /// Creates a "::anno" annotation.
    /// \param anno_name    annotation name
    IAnnotation *create_anno(
        char const *anno_name) const;

    /// Creates a one parameter "::anno" string annotation annotation.
    /// \param anno_name    annotation name
    /// \param value        annotation value
    IAnnotation *create_string_anno(
        char const *anno_name,
        char const *value) const;

    /// Adds the given entity to the import set.
    ///
    /// \param module_name  the imported module name
    /// \param symbol_name  the name of the imported entity
    void register_import(
        char const *module_name,
        char const *symbol_name);

    void register_child_types(IType const *tp);

    /// Handles a type.
    void do_type(IType const *t);

    bool needs_inline(IModule const *module) const;

private:
    /// The current module.
    base::Handle<Module const> m_module;

    /// The target module.
    base::Handle<Module> m_target_module;

    /// Callback to decide whether a given module is inlined.
    IInline_import_callback *m_inline_imports;

    /// True, if no ::anno::origin annotations should be added.
    bool m_omit_anno_origin;

    /// True, if the current module is the root of the traversal.
    bool m_is_root;

    /// Map holding the inlined functions and its new symbol.
    Reference_map &m_references;

    /// Set holding all imports.
    Import_set &m_imports;

    /// Set holding all definitions that need to be exported
    Def_set &m_exports;

    /// The Allocator.
    IAllocator *m_alloc;

    // The zoo of factories from the target module.
    IAnnotation_factory  &m_af;
    IStatement_factory   &m_sf;
    Expression_factory   &m_ef;
    IDeclaration_factory &m_df;
    IName_factory        &m_nf;
    IValue_factory       &m_vf;

    /// Current counter for adding number suffixes to unique symbols.
    size_t &m_counter;

    /// True, if non-root definition parameter defaults are to be kept.
    bool m_keep_non_root_parameter_defaults;
};

/// Helper class which traverses a module to find all imported entities
/// that require to be exported from the module.
class Exports_collector : protected Module_visitor
{
public:
    /// Constructor.
    ///
    /// \param[in]  module       current module.
    /// \param[out] def_set      definition set to fill
    /// \param[in]  is_root      true, if the given module is at the root of the traversal.
    Exports_collector(
        Module const *module,
        Def_set      &def_set,
        bool         is_root);

    ///
    void collect();

private:
    bool pre_visit(IDeclaration_function *decl) MDL_FINAL;

    bool pre_visit(IParameter *param) MDL_FINAL;

    bool pre_visit(IDeclaration_type_struct *decl) MDL_FINAL;

    void post_visit(IAnnotation *anno) MDL_FINAL;

    IExpression *post_visit(IExpression_reference *ref) MDL_FINAL;

    void post_visit(IType_name *tn) MDL_FINAL;

    IExpression *post_visit(IExpression_literal *lit) MDL_FINAL;

    /// Collect user type definition.
    ///
    /// For structs, recurse into the type declaration to also collect
    /// user defined members. Those have to be visible, too.
    void handle_type(IType const *t);

    /// Module to traverse.
    mi::base::Handle<Module const> m_module;

    /// Definition set to fill.
    Def_set &m_def_set;

    /// Indicates, if this module represents the root of the traversal.
    bool m_is_root;
};

/// Implementation of the IMDL_module_transformer interface.
class MDL_module_transformer : public Allocator_interface_implement<IMDL_module_transformer>
{
    typedef Allocator_interface_implement<IMDL_module_transformer> Base;
    friend class Allocator_builder;

public:
    static char const MESSAGE_CLASS = 'T';

public:
    /// Inline all imports of a module, creating a new one.
    ///
    /// \param module       the module
    ///
    /// This function inlines ALL except standard library imports and produces a new module.
    /// The imported functions, materials, and types are renamed and only exported if visible
    /// in the interface.
    ///
    /// The annotation "origin(string)" holds the original full qualified name.
    IModule const *inline_imports(IModule const *module) MDL_FINAL;

    /// Inline all MDLE imports of a module, creating a new one.
    ///
    /// \param module       the module
    ///
    /// This function inlines ALL MDLE imports and produces a new module.
    /// The imported functions, materials, and types are renamed and only exported if visible
    /// in the interface.
    ///
    /// The annotation "origin(string)" holds the original full qualified name.
    IModule const *inline_mdle(IModule const *module) MDL_FINAL;

    /// Access messages of the last operation.
    Messages const &access_messages() const MDL_FINAL;

protected:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param compiler  the MDL compiler
    MDL_module_transformer(
        IAllocator *alloc,
        MDL        *compiler);

    /// Creates a new error.
    ///
    /// \param code    the error code
    /// \param params  additional parameters
    void error(int code, Error_params const &params);

    /// Creates a new warning.
    ///
    /// \param code    the error code
    /// \param params  additional parameters
    void warning(int code, Error_params const &params);

    /// Compute the resulting MDL version.
    ///
    /// \param root  the root module
    IMDL::MDL_version compute_mdl_version(
        Module const *root);

    /// Inlines the given module.
    ///
    /// \param module       the module
    /// \param inline_mdle  true, if only MDLE definitions should be inlined
    /// \return the inlined module or NULL in case of failure.
    IModule const *inline_module(IModule const *imodule, bool inline_mdle);

private:
    /// The MDL compiler.
    mi::base::Handle<mi::mdl::MDL> m_compiler;

    /// The message list.
    Messages_impl m_msg_list;

    /// Last message index.
    size_t m_last_msg_idx;
};

}  // mdl
}  // mi

#endif
