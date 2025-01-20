/***************************************************************************************************
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_BUILDER_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_BUILDER_H

#include <memory>

#include <mi/base/handle.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/neuraylib/iexpression.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_tag.h>

#include "i_mdl_elements_type.h"

namespace mi {

namespace mdl {
class IAnnotation;
class IAnnotation_block;
class IAnnotation_factory;
class IDeclaration_factory;
class IExpression_factory;
class IName_factory;
class IStatement_factory;
class IType_factory;
class IValue_factory;
}

}

namespace MI {

namespace DB { class Transaction; }

namespace MDL {

class Execution_context;
class IAnnotation;
class IAnnotation_block;
class IAnnotation_list;
class IExpression;
class IExpression_direct_call;
class IExpression_factory;
class IExpression_list;
class IType;
class IType_factory;
class IType_list;
class IValue;
class Mdl_function_definition;
class Name_mangler;
class Symbol_importer;

/// Optimization ideas:
/// - A mode where DAG creation and export to DB is only done on request.
/// - An incremental analyze() implementation (would not work for all operations, but for typical
///   ones).
class Mdl_module_builder
{
public:
    /// Initializes the module builder with an empty module of the given name and version.
    ///
    /// \param transaction             The DB transaction to use.
    /// \param db_module_name          The DB name of the new module.
    /// \param min_mdl_version         The initial MDL version of the new module. Ignored if the
    ///                                module exists already.
    /// \param max_mdl_version         The maximal desired MDL version of the module. If higher
    ///                                than the current MDL version of the module, then the module
    ///                                builder will upgrade the MDL version as necessary to handle
    ///                                request requiring newer features.
    /// \param export_to_db            If \c true, the module builder exports the built core module
    ///                                to the database. Otherwise, callers have to use
    ///                                #get_module() to get access to the built module and process
    ///                                it themselves, e.g., for MDLE creation.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c nullptr.
    Mdl_module_builder(
        DB::Transaction* transaction,
        const char* db_module_name,
        mi::mdl::IMDL::MDL_version min_mdl_version,
        mi::mdl::IMDL::MDL_version max_mdl_version,
        bool export_to_db,
        Execution_context* context);

    ~Mdl_module_builder();

    // public API methods

    mi::Sint32 add_variant(
        const char* name,
        DB::Tag prototype_tag,
        const IExpression_list* defaults,
        const IAnnotation_block* annotations,
        const IAnnotation_block* return_annotations,
        bool is_exported,
        bool is_declarative,
        Execution_context* context);

    mi::Sint32 add_function(
        const char* name,
        const IExpression* body,
        const IExpression_list* temporaries,
        const IType_list* parameters,
        const IExpression_list* defaults,
        const IAnnotation_list* parameter_annotations,
        const IAnnotation_block* annotations,
        const IAnnotation_block* return_annotations,
        bool is_exported,
        bool is_declarative,
        IType::Modifier frequency_qualifier,
        Execution_context* context);

    mi::Sint32 add_annotation(
        const char* name,
        const IType_list* parameters,
        const IExpression_list* defaults,
        const IAnnotation_list* parameter_annotations,
        const IAnnotation_block* annotations,
        bool is_exported,
        Execution_context* context);

    mi::Sint32 add_struct_category(
        const char* name,
        const IAnnotation_block* annotations,
        bool is_exported,
        Execution_context* context);

    mi::Sint32 add_enum_type(
        const char* name,
        const IExpression_list* enumerators,
        const IAnnotation_list* enumerator_annotations,
        const IAnnotation_block* annotations,
        bool is_exported,
        Execution_context* context);

    mi::Sint32 add_struct_type(
        const char* name,
        const IType_list* fields,
        const IExpression_list* field_defaults,
        const IAnnotation_list* field_annotations,
        const IAnnotation_block* annotations,
        bool is_exported,
        bool is_declarative,
        const IStruct_category* struct_category,
        Execution_context* context);

    mi::Sint32 add_constant(
        const char* name,
        const IExpression* expr,
        const IAnnotation_block* annotations,
        bool is_exported,
        Execution_context* context);

    mi::Sint32 set_module_annotations(
        const IAnnotation_block* annotations,
        Execution_context* context);

    mi::Sint32 remove_entity(
        const char* name,
        mi::Size index,
        Execution_context* context);

    mi::Sint32 clear_module(
        Execution_context* context);

    std::vector<bool> analyze_uniform(
        const IExpression* root_expr,
        bool root_expr_uniform,
        Execution_context* context);

    // internal methods

    /// Adds a prototype-based function or material to the module.
    ///
    /// This method is used by IMdle_api::export_mdle().
    mi::Sint32 add_function(
        const char* name,
        DB::Tag prototype_tag,
        const IExpression_list* defaults,
        const IAnnotation_list* parameter_annotations,
        const IAnnotation_block* annotations,
        const IAnnotation_block* return_annotations,
        bool is_exported,
        bool is_declarative,
        IType::Modifier frequency_qualifier,
        Execution_context* context);

    /// Returns the current module (or \c nullptr if there is no valid module).
    const mi::mdl::IModule* get_module() const;

    /// Analyzes which parameters or query expressions need to be uniform.
    ///
    /// In addition, checks the uniform constraints for each node, and reports the first
    /// encountered node that violates it.
    ///
    /// \param transaction               The DB transaction to use.
    /// \param root_expr                 Root expression of the graph.
    /// \param root_expr_uniform         Indicates whether the root expression should be uniform.
    /// \param query_expr                A node of the expression graph for which the uniform
    ///                                  property is queried. This expression is \em only used to
    ///                                  identify the corresponding node in the graph, i.e., it
    ///                                  even makes sense to pass constant expressions (which by
    ///                                  themselves are always uniform) to determine whether a
    ///                                  to-be-connected call expression has to be uniform. Can be
    ///                                  \c nullptr.
    /// \param[out] uniform_parameters   Indicates which parameters need to be uniform. The vector
    ///                                  might be shorter than expected if trailing parameters are
    ///                                  not referenced by \p root_expr.
    /// \param[out] uniform_query_expr   Indicates whether \p query_expr needs to be uniform (or \c
    ///                                  false if \p query_expr is \c nullptr, or in case of
    ///                                  errors).
    /// \param[out] error_path           Path to a node of the graph that violates the uniform
    ///                                  constraints (or the empty string if there is no such node,
    ///                                  or in case of errors). Parameters do \em not count as
    ///                                  violation here. Such violations are also reported via
    ///                                  \p context.
    static void analyze_uniform(
        DB::Transaction* transaction,
        const IExpression* root_expr,
        bool root_expr_uniform,
        const IExpression* query_expr,
        std::vector<bool>& uniform_parameters,
        bool& uniform_query_expr,
        std::string& error_path,
        Execution_context* context);

private:
    /// Clears the execution context and checks whether the module is valid.
    bool check_valid( Execution_context* context);

    /// Adds a prototype-based material/function or variant to the module.
    ///
    /// Parameter annotations without corresponding default are ignored.
    ///
    /// Used by add_function(..., DB::Tag, ...) and add_variant().
    ///
    /// \param for_mdle   This affects how resources are converted to the AST representation. MDLE
    ///                   requires tag-based resources, all other use cases require string-based
    ///                   resources.
    mi::Sint32 add_prototype_based(
        const char* name,
        const Mdl_function_definition* prototype,
        const IExpression_list* defaults,
        const IAnnotation_list* parameter_annotations,
        const IAnnotation_block* annotations,
        const IAnnotation_block* return_annotations,
        bool is_variant,
        bool is_exported,
        bool is_declarative,
        bool for_mdle,
        Execution_context* context);

    /// Upgrades the module to \p version, but not beyond \c m_max_mdl_version.
    void upgrade_mdl_version(
        mi::neuraylib::Mdl_version version, Execution_context* context);

    /// Converts an IAnnotation to an mi::mdl::IAnnotation.
    ///
    /// Returns \c nullptr in case of errors.
    mi::mdl::IAnnotation* int_anno_to_core_anno(
        const IAnnotation* annotation, Execution_context* context);

    /// Converts an IAnnotation_block to an mi::mdl::IAnnotation_block.
    ///
    /// Returns \c nullptr in case of errors, or if \p annotation_block is \c nullptr.
    mi::mdl::IAnnotation_block* int_anno_block_to_core_anno_block(
        const IAnnotation_block* annotation_block,
        bool skip_anno_unused,
        Execution_context* context);

    /// Populates \c m_module (all cases).
    void create_module( Execution_context* context);

    /// Populates \c m_module from the DB (tag derived from \c m_db_module_name).
    void sync_from_db();

    /// Populates \c m_module from the DB (tag known).
    ///
    /// Used by create_module() after construction, and possibly by public API methods if the
    /// builder needs to be re-initialized after export to the DB.
    void sync_from_db( DB::Tag tag);

    /// Updates various members bound to the module.
    void update_module();

    /// Analyses the module (and inlines it if \c m_inline_mdle is set).
    ///
    /// Also exports it to the DB depending on \c m_export_to_db.
    void analyze_module( Execution_context* context);

    /// Checks that the given name is a valid MDL identifier.
    ///
    /// \param name                      The intended name of the function, variant, annotation,
    ///                                  enum, struct, or constant.
    bool validate_name( const char* name, Execution_context* context);

    /// Checks that the given expression is well-formed.
    ///
    /// \param transaction               The DB transaction to use.
    /// \param expr                      The expression to check.
    /// \param allow_calls               Indicates whether call expressions are allowed.
    /// \param allow_direct_calls        Indicates whether direct call expressions are allowed.
    /// \param allowed_parameter_count   Indicates the number of parameters (including 0 if no
    ///                                  parameter expressions are allowed).
    /// \param allowed_temporary_count   Indicates the number of temporaries (including 0 for no
    ///                                  temporary expressions are allowed).
    bool validate_expression(
        DB::Transaction* transaction,
        const IExpression* expr,
        bool allow_calls,
        bool allow_direct_calls,
        mi::Size allowed_parameter_count,
        mi::Size allowed_temporary_count,
        Execution_context* context);

    /// Checks that the given expression list is well-formed.
    ///
    /// \param transaction               The DB transaction to use.
    /// \param expr_list                 The expression list to check.
    /// \param allow_calls               Indicates whether call expressions are allowed.
    /// \param allow_direct_calls        Indicates whether direct call expressions are allowed.
    /// \param allowed_parameter_count   Indicates the number of parameters (including 0 if no
    ///                                  parameter expressions are allowed).
    /// \param allowed_temporary_count   Indicates the number of temporaries (including 0 for no
    ///                                  temporary expressions are allowed).
    bool validate_expression_list(
        DB::Transaction* transaction,
        const IExpression_list* expr_list,
        bool allow_calls,
        bool allow_direct_calls,
        mi::Size allowed_parameter_count,
        mi::Size allowed_temporary_count,
        Execution_context* context);

    /// Returns the argument if \p expr is a direct or (indirect) call to the decl_cast operator.
    /// Otherwise, returns \p expr itself.
    const IExpression* skip_decl_cast_operator( const IExpression* expr);

    /// The MDL interface.
    mi::base::Handle<mi::mdl::IMDL> m_mdl;

    /// The DB transaction to use.
    DB::Transaction* m_transaction;

    /// The context the module is created in.
    mi::base::Handle<mi::mdl::IThread_context> m_thread_context;

    /// DB name of the module.
    std::string m_db_module_name;

    /// MDL name of the module.
    std::string m_mdl_module_name;

    /// Initial MDL version of the module. Ignored if the module exists already.
    mi::mdl::IMDL::MDL_version m_min_mdl_version;

    /// The maximal desired MDL version of the module.
    mi::mdl::IMDL::MDL_version m_max_mdl_version;

    /// The underlying MDL module.
    mi::base::Handle<mi::mdl::IModule> m_module;

    /// Indicates whether the built module will be exported to the DB.
    bool m_export_to_db;

    /// Cached setting from the MDL configuration.
    bool m_implicit_cast_enabled;

    /// Indicates whether the module being worked on needs to be re-initialized from the DB.
    bool m_needs_sync_from_db = false;

    std::unique_ptr<Symbol_importer> m_symbol_importer;
    std::unique_ptr<Name_mangler> m_name_mangler;

    /// Various MDL factories.
    mi::mdl::IAnnotation_factory*  m_af;
    mi::mdl::IDeclaration_factory* m_df;
    mi::mdl::IExpression_factory*  m_ef;
    mi::mdl::IName_factory*        m_nf;
    mi::mdl::IStatement_factory*   m_sf;
    mi::mdl::IType_factory*        m_tf;
    mi::mdl::IValue_factory*       m_vf;

    mi::base::Handle<IType_factory>       m_int_tf;
    mi::base::Handle<IExpression_factory> m_int_ef;

    /// Empty lists/blocks as replacement for \c nullptr pointers.
    mi::base::Handle<const IType_list>        m_empty_type_list;
    mi::base::Handle<const IExpression_list>  m_empty_expression_list;
    mi::base::Handle<const IAnnotation_list>  m_empty_annotation_list;
    mi::base::Handle<const IAnnotation_block> m_empty_annotation_block;
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_BUILDER_H
