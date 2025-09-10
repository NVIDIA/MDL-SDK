/***************************************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_H

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/neuraylib/imodule.h> // for mi::neuraylib::Mdl_version

#include <vector>

#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_tag.h>
#include <io/scene/scene/i_scene_scene_element.h>

#include "i_mdl_elements_utilities.h"
#include "i_mdl_elements_resource_tag_tuple.h"

namespace mi {

namespace mdl {
class IAnnotation_block;
class IGenerated_code_dag;
class IMDL;
class IModule;
}

namespace neuraylib { class IReader; }

}

namespace MI {

namespace DB { class Transaction; }

namespace MDL {

class IAnnotation_definition;
class IAnnotation_definition_list;
class IAnnotation_block;
class IExpression_factory;
class IExpression_list;
class IType_factory;
class IType_list;
class IValue_factory;
class IValue_resource;
class Symbol_importer;

using Mdl_ident = mi::Uint64;
using Mdl_tag_ident = std::pair<DB::Tag, Mdl_ident>;

/// The class ID for the #Mdl_module class.
static const SERIAL::Class_id ID_MDL_MODULE = 0x5f4d6d6f; // '_Mmo'

class Mdl_module : public SCENE::Scene_element<Mdl_module, ID_MDL_MODULE>
{
public:

    /// Factory (public, loads the module from file and creates the DB element if needed).
    ///
    /// Looks up the DB element for \p module_name. If it exists, the method returns 0. Otherwise,
    /// the method loads the MDL module from file, creates the DB element, and stores it in the DB
    /// (storing it is required since the method also creates DB elements for all contained
    /// definitions which need the tag of their module). If necessary, DB elements for imported
    /// modules are created recursively, too.
    ///
    /// \param transaction     The DB transaction to use.
    /// \param argument        The MDL module name (for non-MDLE modules), or an MDLE file path
    ///                        (absolute or relative to the current working directory).
    /// \param[inout] context  Execution context used to pass options to and store messages from
    ///                        the MDL compiler.
    /// \return
    ///           -  1: Success (module exists already, loading from file was skipped).
    ///           -  0: Success (module was actually loaded from file).
    ///           - -1: The module name/MDLE file path \p argument is invalid.
    ///           - -2: Failed to find or to compile the module \p argument.
    ///           - -3: The DB name for an imported module is already in use but is not an MDL
    ///                 module, or the DB name for a definition in this module is already in use.
    ///           - -4: Initialization of an imported module failed.
    static mi::Sint32 create_module(
        DB::Transaction* transaction,
        const char* argument,
        Execution_context* context);

    /// Factory (public, loads the module from string and creates the DB element if needed).
    ///
    /// Looks up the DB element for \p module_name. If it exists, the method returns 0. Otherwise,
    /// the method loads the MDL module from string, creates the DB element, and stores it in the DB
    /// (storing it is required since the method also creates DB elements for all contained
    /// definitions which need the tag of their module). If necessary, DB elements for imported
    /// modules are created recursively, too.
    ///
    /// \param transaction     The DB transaction to use.
    /// \param module_name     The MDL module name.
    /// \param module_source   The source code of the MDL module.
    /// \param[inout] context  Execution context used to pass options to and store messages from
    ///                        the MDL compiler.
    /// \return
    ///           -  1: Success (module exists already, creating from \p module_source was skipped).
    ///           -  0: Success (module was actually created from \p module_source).
    ///           - -1: The module name \p module_name is invalid.
    ///           - -2: Failed to find or to compile the module \p module_name.
    ///           - -3: The DB name for an imported module is already in use but is not an MDL
    ///                 module, or the DB name for a definition in this module is already in use.
    ///           - -4: Initialization of an imported module failed.
    static mi::Sint32 create_module(
        DB::Transaction* transaction,
        const char* module_name,
        mi::neuraylib::IReader* module_source,
        Execution_context* context);

    /// Default constructor.
    ///
    /// Does not create a valid instance, to be used by the deserializer only. Use one of the
    /// factories above instead.
    Mdl_module();

    /// Copy constructor.
    ///
    /// The copy constructor does the same as the default one. It is just explicit to avoid pulling
    /// in mi/mdl/mdl_generated_dag.h, mi/mdl/mdl_mdl.h, and mi/mdl/mdl_modules.h for Visual Studio.
    Mdl_module( const Mdl_module& other);

    Mdl_module& operator=( const Mdl_module&) = delete;

    // methods corresponding to mi::neuraylib::IModule

    const char* get_filename() const;

    const char* get_api_filename() const;

    const char* get_mdl_name() const;

    const char* get_mdl_simple_name() const;

    mi::Size get_mdl_package_component_count() const;

    const char* get_mdl_package_component_name( mi::Size index) const;

    mi::neuraylib::Mdl_version get_mdl_version() const;

    mi::Size get_import_count() const;

    DB::Tag get_import( mi::Size index) const;

    const IStruct_category_list* get_struct_categories() const;

    const IType_list* get_types() const;

    const IValue_list* get_constants() const;

    mi::Size get_function_count() const;

    DB::Tag get_function( mi::Size index) const;

    const char* get_function_name( DB::Transaction* transaction, mi::Size index) const;

    mi::Size get_material_count() const;

    DB::Tag get_material( mi::Size index) const;

    const char* get_material_name( DB::Transaction* transaction, mi::Size index) const;

    const IAnnotation_block* get_annotations() const;

    mi::Size get_annotation_definition_count() const;

    const IAnnotation_definition* get_annotation_definition( mi::Size index) const;

    const IAnnotation_definition* get_annotation_definition( const char* name) const;

    bool is_standard_module() const;

    bool is_mdle_module() const;

    std::vector<std::string> get_function_overloads(
        DB::Transaction* transaction,
        const char* name,
        const IExpression_list* arguments = nullptr) const;

    std::vector<std::string> get_function_overloads_by_signature(
        const char* name,
        const std::vector<const char*>& parameter_types) const;

    /// Does not contain all/any resources if context options
    /// MDL_CTX_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS or MDL_CTX_OPTION_RESOLVE_RESOURCES are
    /// set.
    mi::Size get_resources_count() const;

    /// Does not contain all/any resources if context options
    /// MDL_CTX_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS or MDL_CTX_OPTION_RESOLVE_RESOURCES are
    /// set.
    const IValue_resource* get_resource( mi::Size index) const;

    mi::Sint32 reload(
        DB::Transaction* transaction,
        bool recursive,
        Execution_context* context);

    mi::Sint32 reload_from_string(
        DB::Transaction* transaction,
        mi::neuraylib::IReader* module_source,
        bool recursive,
        Execution_context* context);

    // internal methods

    /// Returns the underlying core module.
    ///
    /// Never returns \c nullptr.
    const mi::mdl::IModule* get_core_module() const;

    /// Returns the DAG representation of this module.
    const mi::mdl::IGenerated_code_dag* get_code_dag() const;

    /// Returns true if the tag versions of all imported modules still match
    /// the tag versions stored in this module.
    bool is_valid(
        DB::Transaction* transaction,
        Execution_context* context) const;

    /// Improved version of SERIAL::Serializable::dump().
    ///
    /// \param transaction   The DB transaction (for name lookups and tag versions). Can be
    ///                      \c nullptr.
    void dump( DB::Transaction* transaction) const;

    // methods of SERIAL::Serializable

    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const;

    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer);

    void dump() const  { dump( /*transaction*/ nullptr); }

    // methods of DB::Element_base

    size_t get_size() const;

    DB::Journal_type get_journal_flags() const;

    Uint bundle( DB::Tag* results, Uint size) const;

    // methods of SCENE::Scene_element_base

    void get_scene_element_references( DB::Tag_set* result) const;

    /// Check if the given signature exists in the module.
    ///
    /// \param def_name   The DB name of the function definition.
    ///
    /// \return
    ///         -   0 the signature exists and matches the given \p def_ident
    ///         -  -1 the signature does not exist
    ///         -  -2 the signature exists but its \p def_ident has changed
    mi::Sint32 has_definition(
        bool is_material, const std::string& def_name, Mdl_ident def_ident) const;

    /// Returns the index of the given definition.
    ///
    /// \param is_material      Material or function.
    /// \param def_name         DB name of the definition,
    /// \param def_ident        Definition identifier.
    /// \return                 The index or -1 in case the definition does no longer exist or has
    ///                         an outdated identifier.
    mi::Size get_definition_index(
        bool is_material, const std::string& def_name, Mdl_ident def_ident) const;

    /// Returns the definition indicated by \p is_material and \p index.
    ///
    /// Wrapper for #get_function() and #get_material().
    DB::Tag get_definition( bool is_material, mi::Size index) const;

    /// Returns the identifier of this module.
    Mdl_ident get_ident() const;

    /// Indicates whether the module supports reloading (or editing).
    ///
    /// Reloading is not supported for standard or builtin modules plus ::base and
    /// ::nvidia::distilling_support.
    bool supports_reload() const;

    /// Returns the resource tag tuple for a given resource (low-level access to the resource
    /// vector).
    ///
    /// Returns \c nullptr for invalid indices.
    ///
    /// Does not contain all/any resources if context options
    /// MDL_CTX_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS or MDL_CTX_OPTION_RESOLVE_RESOURCES are
    /// set.
    ///
    /// \see #get_resource_count(), #get_resource()
    const Resource_tag_tuple_ext* get_resource_tag_tuple( mi::Size index) const;

    /// Factory (public, takes an mi::mdl::IModule and creates the DB element if needed).
    ///
    /// Used by Mdl_module_builder.
    ///
    /// Looks up the DB element for \p module. If it exists, the method stores its tag in
    /// \p module_tag and returns 1. Otherwise, the method creates the DB element, and stores it in
    /// the DB (storing it is required since the method also creates DB elements for all contained
    /// definitions which need the tag of their module). If necessary, DB elements for imported
    /// modules are created recursively, too.
    ///
    /// \param transaction             The DB transaction to use.
    /// \param mdl                     The IMDL instance.
    /// \param module                  The corresponding MDL module.
    /// \param[inout] context          Execution context used to pass options to and store messages
    ///                                from the compiler.
    /// \param[out] module_tag_ident   The identifier of the already existing or just created DB
    ///                                element (only valid if the return value is 0 or 1).

    /// \return
    ///
    ///           -  1: Success (module exists already, was not created from \p module).
    ///           -  0: Success (module was actually created from \p module).
    ///           - -2: \p module is an invalid module.
    ///           - -3: The DB name for an imported module is already in use but is not an MDL
    ///                 module, or the DB name for a definition in this module is already in use.
    ///           - -4: Initialization of an imported module failed.
    static mi::Sint32 create_module_internal(
        DB::Transaction* transaction,
        mi::mdl::IMDL* mdl,
        const mi::mdl::IModule* module,
        Execution_context* context,
        Mdl_tag_ident* module_tag_ident = nullptr);

private:
    /// Constructor.
    ///
    /// This constructor is used by the factory #create_module_internal(). The parameters are used
    /// to initialize the parameters in the obvious way (m_mdl_name and m_filename are taken from
    /// module). The annotations are taken from m_code_dag and converted using the transaction.
    Mdl_module(
        DB::Transaction* transaction,
        Mdl_ident module_ident,
        mi::mdl::IMDL* mdl,
        const mi::mdl::IModule* module,
        mi::mdl::IGenerated_code_dag* code_dag,
        std::vector<Mdl_tag_ident> imports,
        std::vector<Mdl_tag_ident> functions,
        std::vector<Mdl_tag_ident> materials,
        std::vector<DB::Tag> annotation_proxies,
        Execution_context* context);

    /// Performs a post-order DFS traversal of the DAG of imports and invokes reload() on all nodes.
    ///
    /// \param transaction       The DB transaction to use.
    /// \param module_tag        The module where the traversal should start.
    /// \param top_level         Indicates whether this call is the top-level call or not. For the
    ///                          top-level module, the reloading is handled by the caller, for all
    ///                          other modules, it is triggered by this method.
    /// \param done              The set of already handled modules. Pass the empty set on top
    ///                          level.
    /// \param[inout] context    Execution context used to pass options to and store messages from
    ///                          the compiler.
    /// \return
    ///           -    0: Success (imports have been reloaded).
    ///           -  - 1: Reload failed. Refer to the \p context for details.
    static mi::Sint32 reload_imports(
        DB::Transaction* transaction,
        DB::Tag module_tag,
        bool top_level,
        ankerl::unordered_dense::set<DB::Tag>& done,
        Execution_context* context);

public:
    /// Replaces the module of this DB element with the given \p module.
    ///
    /// \param transaction       The DB transaction to use.
    /// \param mdl               The IMDL instance.
    /// \param module            The corresponding MDL module.
    /// \param[inout] context    Execution context used to pass options to and store messages from
    ///                          the compiler.
    /// \return
    ///           -    0: Success (module was actually recreated from \p module).
    ///           -  - 1: Reload failed. Refer to the \p context for details.
    mi::Sint32 reload_module_internal(
        DB::Transaction* transaction,
        mi::mdl::IMDL* mdl,
        const mi::mdl::IModule* module,
        Execution_context* context);

private:
    /// Initializes the module by converting and storing types, constants and annotations.
    ///
    /// Contains code shared by the constructor and reload_module_internal().
    ///
    /// \param transaction       The DB transaction to use.
    /// \param[inout] context    Execution context used to pass options to and store messages from
    ///                          the compiler.
    void init_module( DB::Transaction* transaction, Execution_context* context);

    /// The main MDL interface.
    mi::base::Handle<mi::mdl::IMDL> m_mdl;
    /// The underlying core module.
    mi::base::Handle<const mi::mdl::IModule> m_module;
    /// The DAG representation of this module.
    mi::base::Handle<const mi::mdl::IGenerated_code_dag> m_code_dag;

    mi::base::Handle<IType_factory> m_tf;        ///< The type factory.
    mi::base::Handle<IValue_factory> m_vf;       ///< The value factory.
    mi::base::Handle<IExpression_factory> m_ef;  ///< The expression factory.

    std::string m_mdl_name;                             ///< The MDL name.
    std::string m_simple_name;                          ///< The simple (MDL) name.
    std::vector<std::string> m_package_component_names; ///< The package component (MDL) names.

    /// The filename of the module (might be empty). Contains archive and member names in case of
    /// archives.
    std::string m_file_name;

    /// The filename of the module (might be empty). Contains only the archive name in case of
    /// archives.
    std::string m_api_file_name;

    Mdl_ident m_ident;                                  ///< This module's current identifier.

    std::vector<Mdl_tag_ident> m_imports;               ///< The imported modules.

    mi::base::Handle<IStruct_category_list> m_struct_categories; ///< The struct categories.
    mi::base::Handle<IType_list> m_exported_types;      ///< The exported user defined types.
    mi::base::Handle<IType_list> m_local_types;         ///< The local user defined types.
    mi::base::Handle<IValue_list> m_constants;          ///< The constants.
    mi::base::Handle<IAnnotation_block> m_annotations;  ///< Module annotations.

    /// This module's annotation definitions.
    mi::base::Handle<IAnnotation_definition_list> m_annotation_definitions;

    std::vector<Mdl_tag_ident> m_functions;     ///< Tags of the contained function definitions.
    std::vector<Mdl_tag_ident> m_materials;     ///< Tags of the contained material definitions.
    std::vector<DB::Tag> m_annotation_proxies;  ///< Tags of the contained annotation def. proxies.

    /// Resources of this module.
    std::vector<Resource_tag_tuple_ext> m_resources;

    /// Maps functions definition DB names to indices as used in #m_functions.
    std::map<std::string, mi::Size> m_function_name_to_index;

    /// Maps material definition DB names to indices as used in #m_materials.
    std::map<std::string, mi::Size> m_material_name_to_index;

    /// Maps annotation definition DB names to indices as used in #m_annotation_proxies.
    std::map<std::string, mi::Size> m_annotation_name_to_index;
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MODULE_H
