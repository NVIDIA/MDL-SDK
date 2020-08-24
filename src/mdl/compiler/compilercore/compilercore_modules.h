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

#ifndef MDL_COMPILERCORE_MODULES_H
#define MDL_COMPILERCORE_MODULES_H 1

#include <mi/base/handle.h>
#include <mi/base/lock.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_streams.h>
#include <mi/mdl/mdl_modules.h>

#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"
#include "compilercore_memory_arena.h"
#include "compilercore_factories.h"
#include "compilercore_messages.h"
#include "compilercore_symbols.h"
#include "compilercore_names.h"
#include "compilercore_def_table.h"

namespace mi {
namespace mdl {

class Analysis;
class File_resolver;
class MDL;
class Module_serializer;
class Module_deserializer;
class IResource_restriction_handler;

/// An interface that allows to modify cloned AST.
class IClone_modifier
{
public:
    /// Clone a reference expression.
    ///
    /// \param ref   the expression to clone
    virtual IExpression *clone_expr_reference(IExpression_reference const *ref) = 0;

    /// Clone a call expression.
    ///
    /// \param call  the expression to clone
    virtual IExpression *clone_expr_call(IExpression_call const *call) = 0;

    /// Clone a literal.
    ///
    /// \param lit  the literal to clone
    virtual IExpression *clone_literal(IExpression_literal const *lit) = 0;

    /// Clone a qualified name.
    ///
    /// \param name  the name to clone
    virtual IQualified_name *clone_name(IQualified_name const *qname) = 0;
};

/// Implementation of the semantic version.
class Semantic_version : public ISemantic_version {
public:
    /// Get the major version.
    int get_major() const MDL_FINAL;

    /// Get the minor version.
    int get_minor() const MDL_FINAL;

    /// Get the patch version.
    int get_patch() const MDL_FINAL;

    /// Get the pre-release string.
    char const *get_prerelease() const MDL_FINAL;

    // ------------------- non interface -------------------

    /// Compare for less or equal.
    bool operator<=(Semantic_version const &o) const;

public:
    /// Constructor.
    ///
    /// \param major       major number
    /// \param minor       minor number
    /// \param patch       patch number
    /// \param prerelease  optional pre-release
    ///
    /// \note Does not copy the prerelease string, this must be placed at a arena by the caller
    Semantic_version(
        int        major,
        int        minor,
        int        patch,
        char const *prerelease);

    /// Constructor.
    explicit Semantic_version(
        ISemantic_version const &ver);

private:
    int m_major;
    int m_minor;
    int m_patch;
    char const *m_prelease;
};

/// Implementation of a module.
class Module : public Allocator_interface_implement<IModule>
{
    typedef Allocator_interface_implement<IModule> Base;
    friend class Allocator_builder;
    friend class Analysis;
    friend class NT_analysis;
    friend class Expression_call;
    friend class MDL;

public:
    /// Special module property flags.
    enum Module_flags {
        MF_STANDARD   = 0,      ///< Standard flags.
        MF_IS_STDLIB  = 1 << 0, ///< This is a standard library module.
        MF_IS_BUILTIN = 1 << 1, ///< This is the builtins module.
        MF_IS_OWNED   = 1 << 2, ///< This module is owned by the compiler.
        MF_IS_DEBUG   = 1 << 3, ///< This is the debug module.
        MF_IS_NATIVE  = 1 << 4, ///< This module is native.
        MF_IS_HASHED  = 1 << 5, ///< This module has function hashes.
        MF_IS_MDLE    = 1 << 6, ///< This is an MDLE module.
        MF_IS_FOREIGN = 1 << 7, ///< This module is converted from a foreign language.
    };  // can be or'ed

    typedef set<Function_hash>::Type Function_hash_set;

    /// An archive version.
    class Archive_version {
    public:
        /// Constructor.
        Archive_version(
            char const              *archive_name,
            ISemantic_version const *version)
        : m_archive_name(archive_name)
        , m_version(*version)
        {
        }

        /// Get the archive name.
        char const *get_name() const { return m_archive_name; }

        /// Get the archive version.
        Semantic_version const &get_version() const { return m_version; }

    private:
        char const       *m_archive_name;
        Semantic_version m_version;
    };

    /// Helper POTS entry of the resource table.
    struct Resource_entry {
        //// Constructor.
        Resource_entry(
            char const           *url,
            char const           *filename,
            IType_resource const *type,
            bool                 exists)
        : m_url(url)
        , m_filename(filename)
        , m_type(type)
        , m_exists(exists)
        {
        }

    public:
        char const           *m_url;
        char const           *m_filename;
        IType_resource const *m_type;
        bool                 m_exists;
    };

public:
    /// Get the absolute name of the module.
    char const *get_name() const MDL_FINAL;

    /// Get the absolute name of the file from which the module was loaded.
    ///
    /// \returns    The absolute path of the file from which the module was loaded,
    ///             or null if no such file exists.
    char const *get_filename() const MDL_FINAL;

    /// Get the language version.
    ///
    /// \param[out] major  get the major version number
    /// \param[out] minor  get the minor version number
    void get_version(int &major, int &minor) const MDL_FINAL;

    /// Analyze the module.
    ///
    /// \param cache  if non-NULL, a cache of already loaded modules
    /// \param ctx    the thread context
    ///
    /// \returns      True if the module is valid and false otherwise.
    ///
    /// This runs the MDL compiler's semantical analysis on this module.
    bool analyze(
        IModule_cache   *cache,
        IThread_context *ctx) MDL_FINAL;

    /// Check if the module has been analyzed.
    bool is_analyzed() const MDL_FINAL;

    /// Check if the module contents are valid.
    bool is_valid() const MDL_FINAL;

    /// Get the number of imported modules.
    int get_import_count() const MDL_FINAL;

    /// Get the imported module at index.
    ///
    /// \param index        The index of the imported module.
    /// \returns            The imported Module.
    ///
    Module const *get_import(int index) const MDL_FINAL MDL_WARN_UNUSED_RESULT;

    /// Get the number of exported definitions.
    ///
    /// \returns The number of exported definitions.
    ///
    int get_exported_definition_count() const MDL_FINAL;

    /// Get the exported definition at index.
    ///
    /// \param index    The index of the exported definition.
    /// \returns        The exported definition at index.
    ///
    Definition const *get_exported_definition(int index) const MDL_FINAL;

    /// Get the number of declarations.
    int get_declaration_count() const MDL_FINAL;

    /// Get the declaration at index.
    IDeclaration const *get_declaration(int index) const MDL_FINAL;

    /// Add a declaration.
    void add_declaration(IDeclaration const *decl) MDL_FINAL;

    /// Add an import at the end of all other imports or namespace aliases.
    void add_import(char const *name);

    /// Get the name factory.
    Name_factory *get_name_factory() const MDL_FINAL;

    /// Get the expression factory.
    Expression_factory *get_expression_factory() const MDL_FINAL;

    /// Get the statement factory.
    Statement_factory *get_statement_factory() const MDL_FINAL;

    /// Get the declaration factory.
    Declaration_factory *get_declaration_factory() const MDL_FINAL;

    /// Get the type factory.
    Type_factory *get_type_factory() const MDL_FINAL;

    /// Get the value factory.
    Value_factory *get_value_factory() const MDL_FINAL;

    /// Get the annotation factory.
    Annotation_factory *get_annotation_factory() const MDL_FINAL;

    /// Access messages.
    Messages const &access_messages() const MDL_FINAL;

    /// Get the absolute name of the module a definition belongs to.
    ///
    /// \param def  the definition
    /// 
    /// \note  def must belong to this module, else this function returns NULL.
    char const *get_owner_module_name(IDefinition const *def) const MDL_FINAL;

    /// Get the module a definitions belongs to.
    ///
    /// \param def  the definition
    ///
    /// \return the owner module, refcount increased
    ///
    /// \note  def must belong to this module, else this function returns NULL
    Module const *get_owner_module(IDefinition const *def) const MDL_FINAL;

    /// Get the original definition (if imported from another module).
    ///
    /// \param def  the definition
    ///
    /// \return the original definition, owned by get_owner_module(def)
    ///
    /// \note  If def is originated by this module, the definite definition of def is returned
    ///        if available.
    Definition const *get_original_definition(IDefinition const *def) const MDL_FINAL;

    /// Get the the number of constructors of a given type.
    /// The type must exists in this module, or -1 will be returned.
    ///
    /// \param type  the type
    int get_type_constructor_count(IType const *type) const MDL_FINAL;

    /// Get the i'th constructor of a type or NULL.
    ///
    /// \param type   the type
    /// \param index  the index of the requested constructor
    IDefinition const *get_type_constructor(IType const *type, int index) const MDL_FINAL;

    /// Get the the number of conversion operators of a given type.
    /// The type must exists in this module, or -1 will be returned.
    ///
    /// \param type  the type
    int get_conversion_operator_count(IType const *type) const MDL_FINAL;

    /// Get the i'th conversion operator of a type or NULL.
    ///
    /// \param type   the type
    /// \param index  the index of the requested conversion operator
    IDefinition const *get_conversion_operator(
        IType const *type,
        int         index) const MDL_FINAL;

    /// Check if a given identifier is defined at global scope in this module.
    ///
    /// \param name  an unqualified MDL name
    ///
    /// \return true  if name names an entity in the module as analyzed during
    ///               the last call to analyze()
    ///         false otherwise
    bool is_name_defined(char const *name) const MDL_FINAL;

    /// Get the number of builtin definitions.
    ///
    /// \returns The number of builtin definitions if this is the builtin module, else 0.
    int get_builtin_definition_count() const MDL_FINAL;

    /// Get the builtin definition at index.
    ///
    /// \param index    The index of the builtin definition.
    /// \returns        The exported definition at index.
    IDefinition const *get_builtin_definition(int index) const MDL_FINAL;

    /// Returns true if this is a module from the standard library.
    bool is_stdlib() const MDL_FINAL;

    /// Returns true if this is a module is the one and only builtins module.
    bool is_builtins() const MDL_FINAL;

    /// Returns true if this is an MDLE module.
    bool is_mdle() const MDL_FINAL;

    /// Returns the amount of used memory by this module.
    size_t get_memory_size() const MDL_FINAL;

    /// Drop all import entries.
    void drop_import_entries() const MDL_FINAL;

    /// Restore all import entries using a module cache.
    ///
    /// \param cache  A module cache used to restore all import entries.
    ///
    /// \returns true on success, false if at least one import was not restored.
    ///
    /// \note Note that modules with missing import entries cannot be compiled.
    bool restore_import_entries(IModule_cache *cache) const MDL_FINAL;

    /// Lookup a function definition given by its name and an array of (positional) parameter
    /// types. Apply overload rules.
    ///
    /// \param func_name              the name of the function
    /// \param param_type_names       the parameter type names
    /// \param num_param_type_names   the number of parameter type names
    ///
    /// \return the found overload set if there is at least one match, NULL otherwise
    IOverload_result_set const *find_overload_by_signature(
        char const         *func_name,
        char const * const param_type_names[],
        size_t             num_param_type_names) const MDL_FINAL;

    /// Returns the mangled MDL name of a definition that is owned by the current module
    /// if one exists.
    ///
    /// \param def      a definition
    /// \param context  a thread context
    ///
    /// This method is thread safe, if every thread uses its own thread context object.
    /// A name mangling scheme similar to the Itanium C++ name mangling is used.
    char const *mangle_mdl_name(
        IDefinition const *def,
        IThread_context   *context) const MDL_FINAL;

    /// Returns the mangled DAG name of a definition that is owned by the current module
    /// if one exists.
    ///
    /// \param def      a definition
    /// \param context  a thread context
    ///
    /// This method is thread safe, if every thread uses its own thread context object.
    /// The simple name mangling of the MDL DAG backend is used.
    char const *mangle_dag_name(
        IDefinition const *def,
        IThread_context   *context) const MDL_FINAL;

    /// Lookup an exact annotation definition given by its name and an array of of all (positional
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
    IDefinition const *find_annotation(
        char const         *anno_name,
        char const * const param_type_names[],
        size_t             num_param_type_names) const MDL_FINAL;

    /// Lookup an annotation definition given by its name and an array of (positional) parameter
    /// types. Apply overload rules.
    ///
    /// \param anno_name              the name of the annotation
    /// \param param_type_names       the parameter type names
    /// \param num_param_type_names   the number of parameter type names
    ///
    /// \return the found overload set if there is at least one match, NULL otherwise
    IOverload_result_set const *find_annotation_by_signature(
        char const         *anno_name,
        char const * const param_type_names[],
        size_t             num_param_type_names) const MDL_FINAL;

    /// Get the module declaration of this module if any.
    ///
    /// \return the module declaration or NULL if there was none.
    IDeclaration_module const *get_module_declaration() const MDL_FINAL;

    /// Get the semantic version of this module if one was set.
    ISemantic_version const *get_semantic_version() const MDL_FINAL;

    /// Get the number of referenced resources in this module.
    size_t get_referenced_resources_count() const MDL_FINAL;

    /// Get the absolute URL of the i'th referenced resource.
    ///
    /// \param i  the index of the referenced resource
    char const *get_referenced_resource_url(size_t i) const MDL_FINAL;

    /// Get the type of the i'th referenced resource.
    ///
    /// \param i  the index of the referenced resource
    IType_resource const *get_referenced_resource_type(size_t i) const MDL_FINAL;

    /// Get the exists flag of the i'th referenced resource.
    ///
    /// \param i  the index of the referenced resource
    bool get_referenced_resource_exists(size_t i) const MDL_FINAL;

    /// Get the OS-specific filename of the i'th referenced resource if any.
    ///
    /// \param i  the index of the referenced resource
    ///
    /// \note This file name is OS and machine specific, hence it is not serialized
    ///       and set to NULL after deserialisation.
    ///
    /// \return The OS-specific file name of NULL f the name is not known.
    char const *get_referenced_resource_file_name(size_t i) const MDL_FINAL;

    /// Returns true if this module supports function hashes.
    bool has_function_hashes() const MDL_FINAL;

    /// Get the function hash for a given function definition if any.
    ///
    /// \param def  the function definition (must be owned by this module)
    Function_hash const *get_function_hash(IDefinition const *def) const MDL_FINAL;

    // --------------------------- non interface methods ---------------------------

    /// Set the absolute name of the file from which the module was loaded.
    ///
    /// \param  name    The absolute path of the file from which the module was loaded.
    void set_filename(char const *name);

    /// Get the original definition (if imported from another module).
    ///
    /// \param[in]  def    the definition
    /// \param[out] owner  if def was imported, the owner module (refcount NOT increased)
    ///
    /// \return the original definition, owned by get_owner_module(def)
    ///
    /// \note  If def is originated by this module, the definite definition of def is returned
    ///        if available.
    Definition const *get_original_definition(
        IDefinition const *def,
        Module const      *&owner) const;

    /// Returns true if this is a native module.
    bool is_native() const {
        return m_is_native;
    }

    /// Returns true if this is a owned module (by the compiler).
    bool is_compiler_owned() const {
        return m_is_compiler_owned;
    }

    /// Get the owner file name of the message list.
    char const *get_msg_name() const;

    /// Set the owner file name of the message list.
    void set_msg_name(char const *msg_name);

    /// Get the name of the module as a qualified name.
    IQualified_name const *get_qualified_name() const;

    /// Get the language version.
    IMDL::MDL_version get_version() const { return m_mdl_version; }

    /// Convert a MDL version into major, minor pair.
    ///
    /// \param[in] version  the MDL version
    /// \param[out] major   the major version
    /// \param[out] minor   the minor version
    static void get_version(IMDL::MDL_version version, int &major, int &minor);

    /// Set the language version if possible.
    ///
    /// \param compiler                      the MDL compiler
    /// \param major                         desired major version
    /// \param minor                         desired minor version
    /// \param enable_experimental_features  if true, allow experimental MDL features
    /// 
    /// \return true on success, false on error
    bool set_version(MDL *compiler, int major, int minor, bool enable_experimental_features);

    /// Access messages.
    Messages_impl &access_messages_impl();

    /// Access messages.
    Messages_impl const &access_messages_impl() const;

    /// Get the symbol table of this module.
    Symbol_table &get_symbol_table() { return m_sym_tab; }

    /// Get the symbol table of this module.
    Symbol_table const &get_symbol_table() const { return m_sym_tab; }

    /// Get the definition table of this module.
    Definition_table &get_definition_table() { return m_def_tab; }

    /// Get the definition table of this module.
    Definition_table const &get_definition_table() const { return m_def_tab; }

    /// Get the module symbol for a foreign symbol if it exists.
    ///
    /// \param sym  the symbol to lookup
    ///
    /// \return the module's own symbol or NULL if the name
    ///         is not defined in this module.
    /// \note   in contrast to \see import_symbol() a new symbol is not
    ///         created if it does not exists.
    const ISymbol *lookup_symbol(const ISymbol *sym) const;

    /// Import a symbol from another module.
    ///
    /// \param sym  the symbol to import
    ///
    /// \return The module's symbol for the given name.
    ///         Creates one if none exist.
    /// \see    lookup_symbol()
    const ISymbol *import_symbol(const ISymbol *sym);

    /// Import a type from another module.
    ///
    /// \param type  the type to import
    const IType *import_type(const IType *type);

    /// Import a value from another module.
    ///
    /// \param value  the value to import
    IValue const *import_value(IValue const *value) const;

    /// Import a position from another module.
    ///
    /// \param pos  the position to import
    Position *import_position(const Position *pos);

    /// Register a module to be imported into this module.
    ///
    /// \param[in]  imp_mod  the module to import
    /// \param[out] first    if non-NULL, will be set to TRUE if this is the first time imp_mod
    ///                      was registered
    ///
    /// \return the import index of imp_mod
    size_t register_import(
        Module const *imp_mod,
        bool         *first = NULL);

    /// Register a module to be imported into this module lazy.
    ///
    /// \param abs_name   the absolute name of the module to import
    /// \param fname      the file name of the module to import
    /// \param is_stdlib  this is a standard library module
    ///
    /// \return the import index of this module
    size_t register_import(char const *abs_name, char const *fname, bool is_stdlib);

    /// Register an exported entity.
    ///
    /// \param def  the definition of the exported entity
    void register_exported(Definition const *def);

    /// Get the unique id of this module.
    size_t get_unique_id() const { return m_unique_id; }

    /// Set the analyzed and valid states.
    ///
    /// \param is_valid   true if the module is error free, false otherwise
    void set_analyze_result(bool is_valid) { m_is_analyzed = true; m_is_valid = is_valid; }

    /// Allocate initializers for a function definition.
    ///
    /// \param def  the definition
    /// \param num  number of needed initializers
    void allocate_initializers(Definition *def, size_t num);

    /// Clone the given expression.
    ///
    /// \param expr      the expression to clone
    /// \param modifier  an optional clone modifier, may be NULL
    IExpression *clone_expr(
        IExpression const *expr,
        IClone_modifier   *modifier);

    /// Clone the given argument.
    ///
    /// \param arg       the argument to clone
    /// \param modifier  an optional clone modifier, may be NULL
    IArgument const *clone_arg(
        IArgument const *arg,
        IClone_modifier *modifier);

    /// Clone the given type name.
    ///
    /// \param type_name  the type name to clone
    /// \param modifier   an optional clone modifier, may be NULL
    IType_name *clone_name(
        IType_name const *type_name,
        IClone_modifier  *modifier);

    /// Clone the given simple name.
    ///
    /// \param sname  the name to clone
    ISimple_name const *clone_name(ISimple_name const *sname);

    /// Clone the given qualified name.
    ///
    /// \param qname  the qualified name
    /// \param modifier   an optional clone modifier, may be NULL
    IQualified_name *clone_name(
        IQualified_name const *qname,
        IClone_modifier       *modifier);

    /// Clone the given parameter.
    ///
    /// \param param      the parameter to clone
    /// \param clone_init if true, the init expression is cloned
    /// \param clone_anno if true, the annotations are cloned
    /// \param modifier   an optional clone modifier, may be NULL
    IParameter const *clone_param(
        IParameter const   *param,
        bool               clone_init,
        bool               clone_anno,
        IClone_modifier    *modifier);

    /// Clone the given variable declaration.
    ///
    /// \param decl       the variable declaration to clone
    /// \param modifier   an optional clone modifier, may be NULL
    IDeclaration *clone_decl(
        IDeclaration_variable const *decl,
        IClone_modifier             *modifier);

    /// Clone the given annotation block.
    ///
    /// \param anno_block the annotation block to clone
    /// \param modifier   an optional clone modifier, may be NULL
    IAnnotation_block *clone_annotation_block(
        IAnnotation_block const *anno_block,
        IClone_modifier         *modifier);

    /// Clone the given annotation.
    ///
    /// \param anno       the annotation to clone
    /// \param modifier   an optional clone modifier, may be NULL
    IAnnotation *clone_annotation(
        IAnnotation const *anno,
        IClone_modifier   *modifier);

    /// Get the first constructor of a given type.
    /// The type must exists in this module, or NULL will be returned.
    ///
    /// \param type  the type
    Definition const *get_first_constructor(IType const *type) const;

    /// Get the next constructor of a type or NULL.
    ///
    /// \param constr_def  a previous constructor definition
    Definition const *get_next_constructor(Definition const *constr_def) const;

    /// Get the first conversion operator of a given type.
    /// The type must exists in this module, or NULL will be returned.
    ///
    /// \param type  the type
    Definition const *get_first_conversion_operator(IType const *type) const;

    /// Get the next conversion operator of a type or NULL.
    ///
    /// \param op_def  a previous conversion operator definition
    Definition const *get_next_conversion_operator(Definition const *op_def) const;

    /// Return the filename of an imported module.
    ///
    /// \param mod_id  the module ID of the searched module
    char const *get_import_fname(size_t mod_id) const;

    /// Return the value create by a const default constructor of a type.
    ///
    /// \param factory  the factory to create the value
    /// \param type     the type
    IValue const *create_default_value(
        IValue_factory *factory,
        IType const    *type) const;

    /// Replace the global declarations by new collection.
    ///
    /// \param decls  array of new declarations
    /// \param len    length of decls
    void replace_declarations(IDeclaration const * const decls[], size_t len);

    /// Get the imported definition for an outside definition if it is imported.
    ///
    /// \param def   the definition
    ///
    /// \return If def is imported in this module, return its imported definition.
    ///         If def is from this module, returns def itself.
    ///         In all other cases returns NULL.
    Definition const *find_imported_definition(Definition const *def) const;

    /// Get the imported definition for an outside type if it is imported.
    ///
    /// \param type   the type
    ///
    /// \return If type is imported in this module, return its imported definition.
    ///         If type is from this module, returns def itself.
    ///         In all other cases returns NULL.
    Definition const *find_imported_type_definition(IType const *type) const;

    /// Get the module for a given module id.
    ///
    /// \param id      the ID of the module
    /// \param direct  is set to true if this module already imports the searched module
    ///
    /// \note: The module must be either directly or indirectly imported by the current module
    ///        (or it must be a standard module).
    ///        Does NOT increase the module reference count!
    Module const *find_imported_module(size_t id, bool &direct) const;

    /// Get the module for a given module name.
    ///
    /// \param absname  the absolute name of the module
    /// \param direct   is set to true if this module already imports the searched module
    ///
    /// \note: The module must be either directly or indirectly imported by the current module
    ///        (or it must be a standard module).
    ///        Does NOT increase the module reference count!
    Module const *find_imported_module(char const *absname, bool &direct) const;

    /// Add an auto-import.
    ///
    /// \param import  the import declaration to add
    void add_auto_import(IDeclaration_import const *import);

    /// Find the definition of a standard library entity.
    ///
    /// \param mod_name  the name of the stdlib module
    /// \param sym       the name of the entity to lookup
    ///
    /// \return the definition or NULL if this symbol was not found
    ///
    /// \note  The returned definition is owned by its original module
    ///        even if it is already imported into the current one!
    Definition const *find_stdlib_symbol(
        char const    *mod_name,
        ISymbol const *sym) const;

    /// Create a literal or a scoped literal from a value.
    ///
    /// \param value  The literal value.
    /// \param pos    If non-NULL, the position for the literal.
    ///
    /// \return The newly created literal.
    IExpression_literal *create_literal(IValue const *value, Position const *pos);

    /// Lookup the original definition on an imported one.
    ///
    /// \param def  the imported definition
    ///
    /// \return the original definition
    Definition const *lookup_original_definition(Definition const *def) const;

    /// Insert a constant declaration after all other constant declarations.
    void insert_constant_declaration(IDeclaration_constant *decl);

    /// Find the definition of a signature.
    ///
    /// \param signature      a (function) signature
    /// \param only_exported  if true, only exported function are found, else
    ///                       local ones are allowed
    IDefinition const *find_signature(
        char const *signature,
        bool       only_exported) const;

    /// Find the definition of a signature of a standard library function.
    ///
    /// \param module_name  the absolute name of a standard library module
    /// \param signature    a (function) signature
    IDefinition const *find_stdlib_signature(
        char const *module_name,
        char const *signature) const;

    /// Set a deprecated message for a given definition.
    ///
    /// \param def  the definition
    /// \param msg  the message
    void set_deprecated_message(Definition const *def, IValue_string const *msg);

    /// Get the deprecated message for a given definition if any.
    ///
    /// \param def  the definition
    ///
    /// \return the message or NULL if no message was set
    IValue_string const *get_deprecated_message(Definition const *def) const;

    /// Find function overloads.
    void find_overload(Definition const *def, IType const *param_types[], size_t num_params);

    /// Returns true if this is the debug module.
    bool is_debug() const { return m_is_debug; }

    /// Fix the state namespace after the deserialization.
    void fix_state_import_after_deserialization();

    /// Serialize this module.
    ///
    /// \param serializer  the module serializer
    void serialize(Module_serializer &serializer) const;

    /// Check that owned imports match the builtin version after deserialization.
    void check_owned_imports();

    /// Deserialize a module.
    ///
    /// \param deserializer  the module deserializer
    ///
    /// \return the deserialized module
    static Module const *deserialize(Module_deserializer &deserializer);

    /// Get the MDL version of this module.
    IMDL::MDL_version get_mdl_version() const { return m_mdl_version; }

    /// Get the compiler of this module.
    ///
    /// \return the compiler, refcount increased
    MDL *get_compiler() const;

    /// Enter a new semantic version for given module.
    ///
    /// \param major       the major version
    /// \param minor       the minor version
    /// \param patch       the patch version
    /// \param prerelease  the pre-release string
    bool set_semantic_version(
        int            major,
        int            minor,
        int            patch,
        char const     *prelease);

    /// Set the module info from an archive manifest.
    void set_archive_info(
        IArchive_manifest const *manifest);

    /// Get the owner archive version if any.
    Archive_version const *get_owner_archive_version() const;

    /// Get the number of archive dependencies.
    size_t get_archive_dependencies_count() const;

    /// Get the i'th archive dependency.
    Archive_version const *get_archive_dependency(size_t i) const;

    /// Add a new resource entry.
    ///
    /// \param url         an absolute MDL url
    /// \param file_name   the OS specific file name of the url
    /// \param type        the type of the resource
    /// \param exists      if FALSE, the resource does not exists
    void add_resource_entry(
        char const           *url,
        char const           *file_name,
        IType_resource const *type,
        bool                 exists);


    /// Possible MDL version promotion rules.
    enum Promotion_rules {
        PR_NO_CHANGE                    = 0x000,
        PR_SPOT_EDF_ADD_SPREAD_PARAM    = 0x001, ///< add a spread param to spot_edf()
        PC_MEASURED_EDF_ADD_MULTIPLIER  = 0x002, ///< add a multiplier param to measured_edf()
        PR_MEASURED_EDF_ADD_TANGENT_U   = 0x004, ///< add a tangent_u param to measured_edf()
        PR_FRESNEL_LAYER_TO_COLOR       = 0x008, ///< convert fresnel_layer() to color_*()
        PR_WIDTH_HEIGHT_ADD_UV_TILE     = 0x010, ///< add an uv_tile param to width()/height()
        PR_TEXEL_ADD_UV_TILE            = 0x020, ///< add an uv_tile param to texel_*()
        PR_ROUNDED_CORNER_ADD_ROUNDNESS = 0x040, ///< add roundness param to rounded_corner_normal
        PR_MATERIAL_ADD_HAIR            = 0x080, ///< add hair bsdf to material constructor
        PR_GLOSSY_ADD_MULTISCATTER      = 0x100, ///< add a multiscatter_tint param to
                                                 ///  all glossy bsdfs()
    };

    /// Alters one call argument according to the given promotion rules.
    /// 
    /// \param call         the call to alter
    /// \param arg          the argument at current parameter index
    /// \param param_index  index of the current parameter
    /// \param rules        the set of transformation rules to be applied
    /// 
    /// \return The index of the parameter that was modified, e.g. inserted.
    int promote_call_arguments(
        IExpression_call *call,
        IArgument const  *arg,
        int              param_index,
        unsigned         rules);

    /// Clear all function hashes.
    void clear_function_hashes() { m_func_hashes.clear(); }

    /// Add a function hash.
    void add_function_hash(IDefinition const *def, IModule::Function_hash const &hash) {
        m_func_hashes[def] = hash;
    }

    /// Get all known function hashes.
    ///
    /// \param[out]  a set that will be filled with all hashes
    void get_all_function_hashes(Function_hash_set &hashes) const;

private:
    class Import_entry {
    public:
        /// Constructor.
        ///
        /// \param id        the unique ID of the module to import
        /// \param mod       the module, the entry does NOT take ownership
        /// \param fname     the file name of the module
        /// \param abs_name  the absolute name of the module
        Import_entry(
            size_t       id,
            Module const *mod,
            char const   *fname,
            char const   *abs_name)
        : id(id)
        , h_mod(mod, mi::base::DUP_INTERFACE)
        , f_name(fname)
        , abs_name(abs_name)
        , m_refcnt(1)
        , is_stdlib(mod->is_stdlib())
        {
        }

        /// Constructor of a weak entry.
        ///
        /// \param fname     the file name of the module
        /// \param abs_name  the absolute name of the module
        Import_entry(char const *fname, char const *abs_name, bool is_stdlib)
        : id(0)
        , h_mod()
        , f_name(fname)
        , abs_name(abs_name)
        , m_refcnt(0)
        , is_stdlib(is_stdlib)
        {
        }

        /// Get the unique id of the imported module.
        size_t get_id() const { return id; }

        /// Get the module handle, not retained.
        ///
        /// \note Does NOT increase the reference count of the returned
        ///       module, do NOT decrease it just because of this call.
        Module const *get_module() const { return h_mod.get(); }

        /// Get the module handle, not retained.
        ///
        /// \note Does NOT increase the reference count of the returned
        ///       module, do NOT decrease it just because of this call.
        Module const *lock_module(mi::base::Lock &lock) const {
            if (is_stdlib)
                return h_mod.get();
            mi::base::Lock::Block block(&lock);

            Module const *mod = h_mod.get();
            if (mod != NULL)
                ++m_refcnt;
            return mod;
        }

        /// Get the file name of this module.
        char const *get_file_name() const { return f_name; }

        /// Get the absolute module name of this module.
        char const *get_absolute_name() const { return abs_name; }

        /// Drop the imported module from this entry.
        Uint32 drop_module(mi::base::Lock &lock) {
            if (!is_stdlib) {
                mi::base::Lock::Block block(&lock);

                Uint32 refcount = --m_refcnt;
                if (refcount == 0) {
                    h_mod.reset();
                    id = 0;
                }
                return refcount;
            }
            return 1;
        }

        /// Re-enter a module.
        void enter_module(mi::base::Lock &lock, Module const *mod) {
            mi::base::Lock::Block block(&lock);

            ++m_refcnt;
            h_mod = mi::base::make_handle_dup(mod);
            id    = mod->get_unique_id();
        }

    private:
        size_t                         id;       ///< The unique id of the imported module.
        mi::base::Handle<Module const> h_mod;    ///< The module itself.
        char const *                   f_name;   ///< The file name of this module.
        char const *                   abs_name; ///< The absolute name of this module.
        mutable mi::base::Atom32       m_refcnt; ///< The weak reference count.
        bool                           is_stdlib;///< True, if this is a standard lib entry.
    };

    friend size_t dynamic_memory_consumption(Import_entry const &s);
    friend bool   has_dynamic_memory_consumption(Import_entry const &s);

    typedef vector<Import_entry>::Type Import_vector;

private:
    /// Set the absolute name of the module.
    ///
    /// \param  name    The absolute name of the module in MDL notation.
    void set_name(char const *name);

    /// Get the import entry for a given import index.
    ///
    /// \param idx  the import index
    ///
    /// \returns the import entry, NULL if idx is out of range.
    Import_entry const *get_import_entry(size_t idx) const;

    /// Get the import index plus 1 for a given (already imported) module.
    ///
    /// \param mod  the imported module
    ///
    /// \returns the import index plus 1, or 0 if mod is not part of the import table.
    size_t get_import_index(Module const *mod) const;

public:
    /// Get the import index plus 1 for a given (already imported) module.
    ///
    /// \param abs_name  the absolute module name of the imported module
    ///
    /// \returns the import index plus 1, or 0 if abs_name is not part of the import table.
    size_t get_import_index(char const *abs_name) const;

private:
    /// Get the unique id of the original owner module of a definition.
    ///
    /// \param def  the definition, must be owned by this module
    ///
    /// \returns the unique ID of the module where a (possible imported) definition
    ///          was defined originally
    size_t get_original_owner_id(Definition const *def) const;

    /// Find the import entry for a given module ID.
    ///
    /// \param id  the module ID
    ///
    /// \returns the import entry if the module with ID id is imported, NULL otherwise.
    Import_entry const *find_import_entry(size_t module_id) const;

    /// Find the import entry for a given module name.
    ///
    /// \param absname  the absolute module name
    ///
    /// \returns the import entry if the module with name id is imported, NULL otherwise.
    Import_entry const *find_import_entry(char const *absname) const;

    /// Creates a qualified name from a C-string.
    IQualified_name *qname_from_cstring(char const *name);

    /// Serialize the AST of this module.
    ///
    /// \param serializer  the module serializer
    void serialize_ast(Module_serializer &serializer) const;

    /// Deserialize the AST of this module.
    ///
    /// \param deserializer  the module deserializer
    void deserialize_ast(Module_deserializer &deserializer);

    /// Get the module for a given module id.
    ///
    /// \param id      the ID of the module
    /// \param direct  is set to true if this module already imports the searched module
    ///
    /// \note: The module must be either directly or indirectly imported by the current module.
    Module const *do_find_imported_module(size_t id, bool &direct) const;

    /// Get the module for a given module name.
    ///
    /// \param absname  the absolute name of the module
    /// \param direct   is set to true if this module already imports the searched module
    ///
    /// \note: The module must be either directly or indirectly imported by the current module.
    Module const *do_find_imported_module(char const *absname, bool &direct) const;


    /// Helper function to parse definition and the parameter types from a signature.
    Definition const *parse_annotation_params(
        char const                  *anno_name,
        char const * const          param_type_names[],
        int                         num_param_type_names,
        vector<IType const *>::Type &arg_types) const;

    /// Check all referenced resources by this module for restrictions.
    ///
    /// \param ana           the calling analysis
    /// \param res_resolver  the file resolver used to resolve resources
    /// \param rrh           the resource restriction handler
    void check_referenced_resources(
        Analysis                      &ana,
        File_resolver                 &res_resolver,
        IResource_restriction_handler &rrh);

private:
    /// Constructor.
    ///
    /// \param alloc        the allocator
    /// \param compiler     the compiler that creates this module
    /// \param unique_id    the unique id for this module, assigned by the compiler
    /// \param module_name  the name of the module
    /// \param file_name    the file name of the module
    /// \param version      the MDL language level of this module
    /// \param flags        Module property flags
    explicit Module(
        IAllocator        *alloc,
        MDL               *compiler,
        size_t            unique_id,
        char const        *module_name,
        char const        *file_name,
        IMDL::MDL_version version,
        unsigned          flags);

    /// Destructor.
    ~Module() MDL_FINAL;

private:
    // non copyable
    Module(Module const &) MDL_DELETED_FUNCTION;
    Module &operator=(Module const &) MDL_DELETED_FUNCTION;

private:
    /// The memory arena of this module, use to allocate all elements of this module.
    Memory_arena m_arena;

    /// The compiler interface that "owns" this module.
    MDL *m_compiler;

    /// An unique number for this module in (unique across all modules of the same compiler).
    const size_t m_unique_id;

    /// The absolute name of the module.
    char const *m_absname;

    /// The name of the file from which the module was loaded.
    char const *m_filename;

    /// The name of the module as a qualified name (deprecated).
    IQualified_name const *m_qual_name;

    /// Set if this module was analyzed.
    bool m_is_analyzed;

    /// Set if this module is valid.
    bool m_is_valid;

    /// Set if this module is a module from the standard library.
    bool m_is_stdlib;

    /// Set if this module is the one and only builtins module.
    bool m_is_builtins;

    /// Set if this is an MDLE module.
    bool m_is_mdle;

    /// Set if this module is native.
    bool m_is_native;

    /// Set if this module is a owned by the compiler.
    bool m_is_compiler_owned;

    /// Set if this module is only visible in debug configuration.
    bool m_is_debug;

    /// Set if this module has function hashes.
    bool m_is_hashed;

    /// The semantic version if any.
    Semantic_version const *m_sema_version;

    /// The MDL language version.
    IMDL::MDL_version m_mdl_version;

    /// The compiler Messages of this module
    Messages_impl m_msg_list;

    /// The symbol table of this module.
    Symbol_table m_sym_tab;

    /// The name factory of this module.
    mutable Name_factory m_name_factory;

    /// The declaration factory of this module.
    mutable Declaration_factory m_decl_factory;

    /// The expression factory of this module.
    mutable Expression_factory m_expr_factory;

    /// The statement factory of this module.
    mutable Statement_factory m_stmt_factory;

    /// The type factory of this module.
    mutable Type_factory m_type_factory;

    /// The value factory of this module.
    mutable Value_factory m_value_factory;

    /// The annotation factory of this module.
    mutable Annotation_factory m_anno_factory;

    /// The definition table of this module;
    Definition_table m_def_tab;

    /// The type of vectors of declarations.
    typedef Arena_vector<IDeclaration const *>::Type Declaration_vector;

    /// The vector of declarations.
    Declaration_vector m_declarations;

    /// The vector of all imported other modules.
    mutable Import_vector m_imported_modules;

    typedef vector<Definition const *>::Type Export_vector;

    /// The vector of all exported definitions of the module.
    Export_vector m_exported_definitions;

    /// If vector of all builtins, if this is the ::std module.
    Export_vector m_builtin_definitions;

    typedef map<Definition const *, IValue_string const *>::Type Deprecated_msg_map;

    /// Contains deprecated messages for entities defined or imported in this module.
    Deprecated_msg_map m_deprecated_msg_map;

    typedef Arena_vector<Archive_version>::Type Arc_version_vec;

    // ----- archive info -----

    /// The MDL version of the archive.
    IMDL::MDL_version m_arc_mdl_version;

    /// The archive versions if any.
    Arc_version_vec m_archive_versions;

    // ----- resource table -----
    typedef Arena_vector<Resource_entry>::Type Res_table;

    /// The resource table.
    Res_table m_res_table;

    // ----- function hashes -----
    typedef ptr_map<IDefinition const, Function_hash>::Type Func_hash_map;

    /// The function hash map.
    Func_hash_map m_func_hashes;
};

/// Construct a Type_name AST element for an MDL type.
///
/// \param type   the MDL type
/// \param owner  the MDL module that will own the newly constructed type name
IType_name *create_type_name(
    IType const *type,
    IModule     *owner);

/// Promote a given expression to the MDL version of the owner module.
///
/// \param owner  the owner module
/// \param expr   the expression
///
/// This function modifies a given expression to the MDL version of the owner module.
/// Several restrictions apply:
/// - only forward promotion is supported, i.e. only promotion to "newer" version
/// - the detection of "deprecated" entities works solely on the names, i.e. this function
///   expects that the "$version" prefixes are used.
IExpression const *promote_expressions_to_mdl_version(
    IModule           *owner,
    IExpression const *expr);

// Helper shims for calculation the dynamic memory consumption
inline size_t dynamic_memory_consumption(Module::Import_entry const &s) { return 0; }
inline bool has_dynamic_memory_consumption(Module::Import_entry const &s) { return false; }

/// Compare two function hashes.
bool operator<(IModule::Function_hash const &a, IModule::Function_hash const &b);

}  // mdl
}  // mi

#endif

