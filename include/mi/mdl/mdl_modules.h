/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/mdl/mdl_modules.h
/// \brief Interfaces for compiled MDL modules
#ifndef MDL_MODULES_H
#define MDL_MODULES_H 1

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

#include <mi/mdl/mdl_values.h>
#include <mi/mdl/mdl_names.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_statements.h>
#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_messages.h>

namespace mi {
namespace mdl {

class IModule;
class IThread_context;
class IQualified_name;

/// Handle that is passed to \c IModule_cache::lookup in order to identify an requested module from
/// first successful file resolution to notifications about success or failure.
class IModule_cache_lookup_handle
{
public:
    /// Get an identifier to be used throughout the loading of a module.
    virtual const char *get_lookup_name() const = 0;

    /// Returns true if this handle belongs to context that loads module.
    /// If the module is already loaded or the handle belongs to a waiting context, false will be
    /// returned.
    virtual bool is_processing() const = 0;
};

/// The Interface of a module loaded callback.
///
/// An implementation can be registered on the IModule_cache to get notified when a module is
/// successfully loaded. It is used to communicate to the cache that a new entry can be made.
class IModule_loaded_callback
{
public:
    /// Function that is called when the \c module was loaded successfully so that it can be cached.
    ///
    /// \param module   The loaded, valid, module.
    virtual bool register_module(
        IModule const *module) = 0;

    /// Function that is called when a module was not found or when loading failed.
    ///
    /// \param handle   A handle created by \c create_lookup_handle which is used throughout the
    ///                 loading process of a module
    virtual void module_loading_failed(
        IModule_cache_lookup_handle const &handle) = 0;

    /// Called while loading a module to check if the built-in modules are already registered.
    ///
    /// \param absname  the absolute name of the built-in module to check.
    /// \return         If false, the built-in will be registered shortly after.
    virtual bool is_builtin_module_registered(
        char const *absname) const = 0;
};

/// The Interface of a module cache.
///
/// The MDL compiler itself does neither take ownership of compiled MDL modules nor
/// does it remember which modules are already compiled (except if a module is already imported,
/// in that case it is not compiled again).
/// Hence, compiling a MDL module a second time results in a new (semantically equal)
/// MDL module. This can be avoided by implementing this interface on the user side.
/// Before the MDL compiler tries to compile an unknown import, it interrogates the
/// IModule_cache if one was passed.
class IModule_cache {
public:
    /// Create an \c IModule_cache_lookup_handle for this \c IModule_cache implementation.
    /// Has to be freed using \c free_lookup_handle.
    virtual IModule_cache_lookup_handle *create_lookup_handle() const = 0;

    /// Free a handle created by \c create_lookup_handle.
    /// \param handle       a handle created by this module cache.
    virtual void free_lookup_handle(
        IModule_cache_lookup_handle *handle) const = 0;

    /// Lookup a module.
    ///
    /// \param absname      the absolute name of a MDL module as returned by the module resolver
    /// \param handle       a handle created by \c create_lookup_handle which is used throughout the
    ///                     loading process of a module or NULL in case the goal is to just check
    ///                     if a module is loaded.
    ///
    /// \return             If this module is already known, return it, otherwise NULL.
    ///
    /// \note  The module must be returned with increased reference count.
    virtual IModule const *lookup(
        char const                  *absname,
        IModule_cache_lookup_handle *handle) const = 0;

    /// Get the module loading callback which is used to notify the cache or the integration
    /// about successfully loaded modules.
    ///
    /// \return                 The callback implementation.
    virtual IModule_loaded_callback *get_module_loading_callback() const = 0;
};

/// Helper interface to represent an overload resolution result set returned by
/// IModule::find_overload_by_signature().
class IOverload_result_set : public
    mi::base::Interface_declare<0xa475df83,0x306d,0x4793,0x95,0xe7,0xdb,0x23,0xfc,0x88,0xb4,0x75,
    mi::base::IInterface>
{
public:
    /// Get the first result or NULL if no results.
    virtual IDefinition const *first() const = 0;

    /// Get the next result or NULL if no more results
    virtual IDefinition const *next() const = 0;

    /// Get the first result as a DAG signature.
    virtual char const *first_signature() const = 0;

    /// Get the first result as a DAG signature.
    virtual char const *next_signature() const = 0;
};

/// A Semantic version.
class ISemantic_version : public Interface_owned {
public:
    /// Get the major version.
    virtual int get_major() const = 0;

    /// Get the minor version.
    virtual int get_minor() const = 0;

    /// Get the patch version.
    virtual int get_patch() const = 0;

    /// Get the pre-release string.
    virtual char const *get_prerelease() const = 0;
};

/// The representation of a compiled  MDL module.
///
/// This interface presents a view of a compiled MDL module.
class IModule : public
    mi::base::Interface_declare<0xf2b748b8,0x92b4,0x4ed1,0xa3,0x14,0x0c,0xe4,0x70,0x0c,0x5c,0x4b,
    mi::base::IInterface>
{
public:
    /// A function hash for hashed modules.
    struct Function_hash {
        unsigned char hash[16];
    };

public:
    /// Get the absolute name of the module.
    ///
    /// \return The absolute name of this MDL module.
    virtual char const *get_name() const = 0;

    /// Get the qualified name of the module.
    ///
    /// \return The qualfief name of this MDL module.
    virtual IQualified_name const *get_qualified_name() const = 0;

    /// Get the absolute name of the file from which the module was loaded.
    ///
    /// \returns    The absolute path of the file from which the module was loaded,
    ///             or the empty string if no such file exists.
    virtual char const *get_filename() const = 0;

    /// Get the language version as specified in the MDL module.
    ///
    /// \param[out] major  get the major version number
    /// \param[out] minor  get the minor version number
    virtual void get_version(int &major, int &minor) const = 0;

    /// Analyze the module.
    ///
    /// \param cache  If non-NULL, a cache of already loaded modules.
    /// \param ctx    The thread context or NULL.
    ///
    /// \returns      True if the module is valid and false otherwise.
    ///
    /// This runs the MDL compiler's semantical analysis on this module.
    virtual bool analyze(
        IModule_cache   *cache,
        IThread_context *ctx) = 0;

    /// Check if the module has been analyzed.
    virtual bool is_analyzed() const = 0;

    /// Check if the module contents are valid.
    virtual bool is_valid() const = 0;

    /// Get the number of imported modules.
    virtual int get_import_count() const = 0;

    /// Get the imported module at index.
    ///
    /// \param index    The index of the imported module.
    ///
    /// \returns        The imported module.
    virtual IModule const *get_import(int index) const = 0;

    /// Get the number of exported definitions.
    ///
    /// \returns The number of exported definitions.
    virtual int get_exported_definition_count() const = 0;

    /// Get the exported definition at index.
    ///
    /// \param index    The index of the exported definition.
    ///
    /// \returns        The exported definition at index.
    virtual IDefinition const *get_exported_definition(int index) const = 0;

    /// Get the number of declarations.
    virtual int get_declaration_count() const = 0;

    /// Get the declaration at index.
    ///
    /// \param index  the index
    virtual IDeclaration const *get_declaration(int index) const = 0;

    /// Add a declaration (to the end of the declaration list).
    ///
    /// \param decl  the declaration to add
    virtual void add_declaration(IDeclaration const *decl) = 0;

    /// Add an "import name" declaration to the current module.
    ///
    /// \param name  an absolute MDL module name
    virtual void add_import(char const *name) = 0;

    /// Get the name factory of this module.
    virtual IName_factory *get_name_factory() const = 0;

    /// Get the expression factory of this module.
    virtual IExpression_factory *get_expression_factory() const = 0;

    /// Get the statement factory of this module.
    virtual IStatement_factory *get_statement_factory() const = 0;

    /// Get the declaration factory of this module.
    virtual IDeclaration_factory *get_declaration_factory() const = 0;

    /// Get the type factory of this module.
    virtual IType_factory *get_type_factory() const = 0;

    /// Get the value factory of this module.
    virtual IValue_factory *get_value_factory() const = 0;

    /// Get the annotation factory of this module.
    virtual IAnnotation_factory *get_annotation_factory() const = 0;

    /// Access messages of this module.
    virtual Messages const &access_messages() const = 0;

    /// Get the absolute name of the module a definitions belongs to.
    ///
    /// \param def  the definition
    ///
    /// \note  def must be owned by this module, i.e. a definition of this module OR
    ///        an imported definition of this module.
    virtual char const *get_owner_module_name(IDefinition const *def) const = 0;

    /// Get the module a definition belongs to.
    ///
    /// \param def  the definition
    ///
    /// \return the owner module, refcount increased
    ///
    /// \note  def must be owned by this module, i.e. a definition of this module OR
    ///        an imported definition of this module.
    virtual IModule const *get_owner_module(IDefinition const *def) const = 0;

    /// Get the original definition (if imported from another module).
    ///
    /// \param def  the definition
    ///
    /// \return the original definition, owned by get_owner_module(def)
    ///
    /// \note  If def is originated by this module, def itself is returned.
    virtual IDefinition const *get_original_definition(IDefinition const *def) const = 0;

    /// Get the the number of constructors of a given type.
    ///
    /// \param type  the type
    ///
    /// The type must exists in this module, or -1 will be returned.
    virtual int get_type_constructor_count(IType const *type) const = 0;

    /// Get the i'th constructor of a type or NULL.
    ///
    /// \param type   the type
    /// \param index  the index of the requested constructor
    virtual IDefinition const *get_type_constructor(
        IType const *type,
        int         index) const = 0;

    /// Get the the number of conversion operators of a given type.
    ///
    /// \param type  the type
    ///
    /// The type must exists in this module, or -1 will be returned.
    virtual int get_conversion_operator_count(IType const *type) const = 0;

    /// Get the i'th conversion operator of a type or NULL.
    ///
    /// \param type   the type
    /// \param index  the index of the requested conversion operator
    virtual IDefinition const *get_conversion_operator(
        IType const *type,
        int         index) const = 0;

    /// Check if a given identifier is defined at global scope in this module.
    ///
    /// \param name  an unqualified MDL name
    ///
    /// \return true  if name names an entity in the module as analyzed during
    ///               the last call to analyze()
    ///         false otherwise
    virtual bool is_name_defined(char const *name) const = 0;

    /// Get the number of builtin definitions.
    ///
    /// \returns The number of builtin definitions if this is the builtin module, else 0.
    virtual int get_builtin_definition_count() const = 0;

    /// Get the builtin definition at index.
    ///
    /// \param index    The index of the builtin definition.
    /// \returns        The exported definition at index.
    virtual IDefinition const *get_builtin_definition(int index) const = 0;

    /// Returns true if this is a module from the standard library.
    virtual bool is_stdlib() const = 0;

    /// Returns true if this is a module is the one and only builtins module.
    virtual bool is_builtins() const = 0;

    /// Returns true if this is an MDLE module.
    virtual bool is_mdle() const = 0;

    /// Returns the amount of used memory by this module.
    virtual size_t get_memory_size() const = 0;

    /// Drop all import entries of this module.
    virtual void drop_import_entries() const = 0;

    /// Restore all import entries using a module cache.
    ///
    /// \param cache  A module cache used to restore all import entries.
    ///
    /// \returns true on success, false if at least one import was not restored.
    ///
    /// \note Note that modules with missing import entries cannot be compiled.
    virtual bool restore_import_entries(IModule_cache *cache) const = 0;

    /// Lookup a function definition given by its name and an array of (positional) parameter
    /// types. Apply overload rules.
    ///
    /// \param func_name              the name of the function
    /// \param param_type_names       the parameter type names
    /// \param num_param_type_names   the number of parameter type names
    ///
    /// \return the found overload set if there is at least one match, NULL otherwise
    virtual IOverload_result_set const *find_overload_by_signature(
        char const         *func_name,
        char const * const param_type_names[],
        size_t             num_param_type_names) const = 0;

    /// Returns the mangled MDL name of a definition that is owned by the current module
    /// if one exists.
    ///
    /// \param def      a definition
    /// \param context  a thread context
    ///
    /// This method is thread safe, if every thread uses its own thread context object.
    /// A name mangling scheme similar to the Itanium C++ name mangling is used.
    virtual char const *mangle_mdl_name(
        IDefinition const *def,
        IThread_context   *context) const = 0;

    /// Returns the mangled DAG name of a definition that is owned by the current module
    /// if one exists.
    ///
    /// \param def      a definition
    /// \param context  a thread context
    ///
    /// This method is thread safe, if every thread uses its own thread context object.
    /// The simple name mangling of the MDL DAG backend is used.
    virtual char const *mangle_dag_name(
        IDefinition const *def,
        IThread_context   *context) const = 0;

    /// Lookup an exact annotation definition given by its name and an array of all (positional)
    /// parameter types.
    ///
    /// \param anno_name              the name of the annotation
    /// \param param_type_names       the parameter type names
    /// \param num_param_type_names   the number of parameter type names
    ///
    /// \return the definition of the function if there is exactly one match, NULL otherwise
    ///
    /// \note This method currently does not support the lookup of annotations with deferred size
    ///       parameters.
    virtual IDefinition const *find_annotation(
        char const         *anno_name,
        char const * const param_type_names[],
        size_t             num_param_type_names) const = 0;

    /// Lookup an annotation definition given by its name and an array of (positional) parameter
    /// types. Apply overload rules.
    ///
    /// This function uses the MDL overload rules to find definitions if the
    /// type signature is not complete.
    ///
    /// \param anno_name              the name of the annotation
    /// \param param_type_names       the parameter type names
    /// \param num_param_type_names   the number of parameter type names
    ///
    /// \return the definition of the function if there is exactly one match, NULL otherwise
    virtual IOverload_result_set const *find_annotation_by_signature(
        char const         *anno_name,
        char const * const param_type_names[],
        size_t             num_param_type_names) const = 0;

    /// Get the module declaration of this module if any.
    ///
    /// \return the module declaration or NULL if there was none.
    virtual IDeclaration_module const *get_module_declaration() const = 0;

    /// Get the semantic version of this module if one was set.
    virtual ISemantic_version const *get_semantic_version() const = 0;

    /// Get the number of referenced resources in this module.
    virtual size_t get_referenced_resources_count() const = 0;

    /// Get the absolute URL of the i'th referenced resource.
    ///
    /// \param i  the index of the referenced resource
    virtual char const *get_referenced_resource_url(size_t i) const = 0;

    /// Get the type of the i'th referenced resource.
    ///
    /// \param i  the index of the referenced resource
    virtual IType_resource const *get_referenced_resource_type(size_t i) const = 0;

    /// Get the exists flag of the i'th referenced resource.
    ///
    /// \param i  the index of the referenced resource
    ///
    /// \note During compilation the compiler checks, if a referenced resource exists.
    ///       It is legal to reference non-existing resources in MDL.
    virtual bool get_referenced_resource_exists(size_t i) const = 0;

    /// Get the OS-specific filename of the i'th referenced resource if any.
    ///
    /// \param i  the index of the referenced resource
    ///
    /// \note This file name is OS and machine specific, hence it is not serialized
    ///       and set to NULL after deserialization.
    ///
    /// \return The OS-specific file name of NULL f the name is not known.
    virtual char const *get_referenced_resource_file_name(size_t i) const = 0;

    /// Returns true if this module supports function hashes.
    virtual bool has_function_hashes() const = 0;

    /// Get the function hash for a given function definition if any.
    ///
    /// \param def  the function definition (must be owned by this module)
    virtual Function_hash const *get_function_hash(IDefinition const *def) const = 0;
};

}  // mdl
}  // mi

#endif
