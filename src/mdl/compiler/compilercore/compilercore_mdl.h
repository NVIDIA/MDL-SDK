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

#ifndef MDL_COMPILERCORE_MDL_H
#define MDL_COMPILERCORE_MDL_H 1

#include <mi/base/handle.h>
#include <mi/base/lock.h>
#include <mi/mdl/mdl_mdl.h>

#include "compilercore_allocator.h"
#include "compilercore_memory_arena.h"
#include "compilercore_factories.h"
#include "compilercore_modules.h"
#include "compilercore_options.h"
#include "compilercore_printers.h"
#include "compilercore_cstring_hash.h"
#include "compilercore_thread_context.h"

namespace mi {
namespace mdl {

class Analysis;
class IMDL_import_result;
class File_resolver;
class Jitted_code;
class Messages_impl;

// Evaluates an intrinsic function called on constant arguments.
extern IValue const *evaluate_intrinsic_function(
    IValue_factory         *value_factory,
    IDefinition::Semantics sema,
    IValue const * const   arguments[],
    size_t                 n_arguments);

/// RAII-like helper class to simplify IModule_loaded_callback and IModule_cache_lookup_handle
//7 handling.
class Module_callback {
public:
    /// Constructor.
    /*implicit*/ Module_callback(
        IModule_cache *module_cache)
    : m_module_cache(module_cache)
    , m_cb(module_cache != NULL ? module_cache->get_module_loading_callback() : NULL)
    , m_cache_lookup_handle(m_cb != NULL ? module_cache->create_lookup_handle() : NULL)
    {
    }

    /// Destructor.
    ~Module_callback()
    {
        if (m_module_cache != NULL && m_cache_lookup_handle != NULL) {
            m_module_cache->free_lookup_handle(m_cache_lookup_handle);
        }
    }

    /// check if valid.
    bool is_valid() const { return m_cache_lookup_handle != NULL; }

    /// Function that is called when the \c module was loaded successfully so that it can be cached.
    ///
    /// \param module   The loaded, valid, module.
    bool register_module(
        IModule const *module)
    {
        return m_cb != NULL ? m_cb->register_module(module) : true;
    }

    /// Function that is called when a module was not found or when loading failed.
    void module_loading_failed()
    {
        if (m_cb != NULL) {
            m_cb->module_loading_failed(*m_cache_lookup_handle);
        }
    }

    /// Called while loading a module to check if the built-in modules are already registered.
    ///
    /// \param absname  the absolute name of the built-in module to check.
    /// \return         If false, the built-in will be registered shortly after.
    bool is_builtin_module_registered(
        char const *absname) const
    {
        return m_cb != NULL ? m_cb->is_builtin_module_registered(absname) : false;
    }

    /// Get an identifier to be used throughout the loading of a module.
    char const *get_lookup_name() const
    {
        return m_cache_lookup_handle != NULL ? m_cache_lookup_handle->get_lookup_name(): NULL;
    }

    /// Returns true if this handle belongs to context that loads module.
    /// If the module is already loaded or the handle belongs to a waiting context, false will be
    /// returned.
    bool is_processing() const
    {
        return m_cache_lookup_handle != NULL ? m_cache_lookup_handle->is_processing() : false;
    }

    /// Conversion operator.
    operator IModule_cache_lookup_handle *()
    {
        return m_cache_lookup_handle;
    }

    /// Conversion operator.
    operator IModule_loaded_callback *()
    {
        return m_cb;
    }

private:
    /// The module cache if any.
    IModule_cache *m_module_cache;

    /// The module loaded callback if any.
    IModule_loaded_callback *m_cb;

    /// The lookup handle if any.
    IModule_cache_lookup_handle *m_cache_lookup_handle;
};

/// RAII-like helper class for IModule_cache_lookup_handle.
class Module_cache_lookup_handle {
public:
    /// Constructor.
    /*implicit*/ Module_cache_lookup_handle(
        IModule_loaded_callback *cb)
    {
    }

private:
};


/// Implementation of the IMDL interface.
class MDL : public Allocator_interface_implement<IMDL>
{
    typedef Allocator_interface_implement<IMDL> Base;
public:

    /// The name of the option to dump the auto-typing dependence graph for every
    /// function of a compiled module.
    static char const *option_dump_dependence_graph;

    /// The name of the option to dump the call graph for every compiled module.
    static char const *option_dump_call_graph;

    /// The name of the option that steers warnings.
    static char const *option_warn;

    /// The optimization level, 0 disables all optimizations
    static char const *option_opt_level;

    /// The name of the option that switches the strict compilation mode.
    static char const *option_strict;

    /// The name of the option that enables undocumented experimental MDL features.
    static char const *option_experimental_features;

    /// The name of the option that controls, if resources are resolved by the compiler.
    static char const *option_resolve_resources;

    /// The value of limits::FLOAT_MIN.
    static char const *option_limits_float_min;

    /// The value of limits::FLOAT_MAX.
    static char const *option_limits_float_max;

    /// The value of limits::DOUBLE_MIN.
    static char const *option_limits_double_min;

    /// The value of limits::DOUBLE_MAX.
    static char const *option_limits_double_max;

    /// The value of state::WAVELENGTH_BASE_MAX.
    static char const *option_state_wavelength_base_max;


    /// The name of the option to keep resource file paths as is.
    static char const *option_keep_original_resource_file_paths;

    /// Get the type factory.
    Type_factory *get_type_factory() const MDL_FINAL;

    /// Create a module.
    ///
    /// \param context      the thread context for this operation
    /// \param module_name  the name of the new module
    /// \param version      the MDL language level of this module
    ///
    /// \returns an empty new module or NULL if the given name already exists
    Module *create_module(
        IThread_context   *context,
        char const        *module_name,
        IMDL::MDL_version version = IMDL::MDL_DEFAULT_VERSION) MDL_FINAL;

    /// Install a MDL search path helper.
    ///
    /// \param search_path  the new search path helper to install, takes ownership
    ///
    /// The new search path helper will be released at this interface
    /// life time. Any previously set helper will be released now.
    void install_search_path(IMDL_search_path *search_path) MDL_FINAL;

    /// Load a module with a given name.
    ///
    /// \param context       The thread context for this operation.
    /// \param  module_name  The absolute module name.
    /// \param  cache        If non-NULL, a module cache.
    ///
    /// \returns             An interface to the loaded module.
    /// If the module is already loaded, no new module will be created,
    /// but an interface to the already loaded module will be returned.
    Module const *load_module(
        IThread_context *context,
        char const      *module_name,
        IModule_cache   *cache) MDL_FINAL;

    /// Load a module with a given name from a given string.
    ///
    /// \param context                 the thread context for this operation
    /// \param cache                   If non-NULL, a module cache.
    /// \param module_name             The absolute module name.
    /// \param utf8_buffer             An UTF8 buffer (must start with a BOM) or an ASCII buffer.
    /// \param length                  The length of the buffer in bytes.
    ///
    /// \returns                       An interface to the loaded module.
    /// If the module name already exists, returns NULL.
    IModule const *load_module_from_string(
        IThread_context *context,
        IModule_cache   *cache,
        char const      *module_name,
        char const      *utf8_buffer,
        size_t          length) MDL_FINAL;

    /// Load a module with a given name from a given input stream.
    ///
    /// \param context                 the thread context for this operation
    /// \param cache                   If non-NULL, a module cache.
    /// \param module_name             The absolute module name.
    /// \param stream                  An input stream.
    ///
    /// \returns                       An interface to the loaded module.
    /// If the module name already exists, returns NULL.
    IModule const *load_module_from_stream(
        IThread_context *context,
        IModule_cache   *cache,
        char const      *module_name,
        IInput_stream   *stream) MDL_FINAL;

    /// Load a code generator.
    /// \param  target_language The name of the target language for which to generate code.
    /// \returns                An interface to the loaded code generator.
    /// If the code generator is already loaded, no new code generator will be created,
    /// but an interface to the already loaded code generator will be returned.
    ICode_generator *load_code_generator(const char *target_language) MDL_FINAL;

    /// Create a printer.
    ///
    /// \param stream  an output stream the new printer will print to
    ///
    /// Pass an IOutput_stream_colored for colored output.
    Printer *create_printer(IOutput_stream *stream) const MDL_FINAL;

    /// Access options.
    Options &access_options() MDL_FINAL;

    /// Return a builtin semantic for a given absolute intrinsic function name.
    ///
    /// \param name  the name of a builtin intrinsic function
    ///
    /// \returns  the semantic for the given intrinsic function name
    ///           DS_UNKNOWN otherwise
    IDefinition::Semantics get_builtin_semantic(char const *name) const MDL_FINAL;

    /// Evaluates an intrinsic function called on constant arguments.
    ///
    /// \param value_factory  The value factory used to create new values
    /// \param sema           The semantic of the intrinsic function
    /// \param arguments      The values of the function arguments
    /// \param n_arguments    The number of arguments
    ///
    /// \return The function result or IValue_bad if the function could
    ///         not be evaluated
    IValue const *evaluate_intrinsic_function(
        IValue_factory         *value_factory,
        IDefinition::Semantics sema,
        IValue const * const   arguments[],
        size_t                 n_arguments) const MDL_FINAL;

    /// Serialize a module to the given serializer.
    ///
    /// \param module                the module to serialize
    /// \param is                    the serializer data is written to
    /// \param include_dependencies  if true, the module is written including all
    ///                              modules it imports (self-contained binary)
    void serialize_module(
        IModule const*module,
        ISerializer  *is,
        bool         include_dependencies) const MDL_FINAL;

    /// Deserialize a module from a given deserializer.
    ///
    /// \param ds  the deserializer data is read from
    ///
    /// \return the module
    Module const *deserialize_module(IDeserializer *ds) MDL_FINAL;

    /// Create an IOutput_stream standard stream.
    ///
    /// \param kind  a standard stream kind
    ///
    /// \return an IOutzput stream stream
    IOutput_stream *create_std_stream(Std_stream kind) const MDL_FINAL;

    /// Create an IOutput_stream from a file.
    ///
    /// \param filename  the file name
    ///
    /// \return an IOutput_stream stream or NULL if the file could not be created
    IOutput_stream *create_file_output_stream(char const *filename) const MDL_FINAL;

    /// Create an IInput_stream from a file.
    ///
    /// \param filename  the file name
    ///
    /// \return an IInput_stream stream or NULL if the file could not be opened
    IInput_stream *create_file_input_stream(char const *filename) const MDL_FINAL;

    /// Serialize a code DAG to the given serializer.
    ///
    /// \param code                  the code DAG to serialize
    /// \param is                    the serializer data is written to
    void serialize_code_dag(
        IGenerated_code_dag const *code,
        ISerializer               *is) const MDL_FINAL;

    /// Deserialize a code DAG from a given deserializer.
    ///
    /// \param ds  the deserializer data is read from
    ///
    /// \return the code DAG
    IGenerated_code_dag const *deserialize_code_dag(IDeserializer *ds) MDL_FINAL;

    /// Create a new MDL lambda function.
    ///
    /// \param context  the execution context for this lambda function.
    ///
    /// \returns  a new lambda function.
    ILambda_function *create_lambda_function(
        ILambda_function::Lambda_execution_context context) MDL_FINAL;

    /// Create a new MDL distribution function.
    ///
    /// \returns  a new distribution function.
    IDistribution_function *create_distribution_function() MDL_FINAL;

    /// Serialize a lambda function to the given serializer.
    ///
    /// \param lambda                the lambda function to serialize
    /// \param is                    the serializer data is written to
    void serialize_lambda(
        ILambda_function const *lambda,
        ISerializer            *is) MDL_FINAL;

    /// Deserialize a lambda function from a given deserializer.
    ///
    /// \param ds  the deserializer data is read from
    ///
    /// \return the lambda function
    ILambda_function *deserialize_lambda(IDeserializer *ds) MDL_FINAL;

    /// Check if the given absolute module name name a builtin MDL module.
    ///
    /// \param absname  an absolute module name
    ///
    /// \return true if absname names a builtin module, false otherwise
    bool is_builtin_module(char const *absname) const MDL_FINAL;

    /// Add a new builtin module to the MDL compiler.
    ///
    /// \param abs_name    the absolute name of the module to add
    /// \param buffer      the buffer containing the source of the builtin module
    /// \param buf_len     the length of the buffer
    /// \param is_encoded  if true, buffer points to an encoded buffer
    /// \param is_native   if true, the added module is native
    ///
    /// \note this call must be issued before any real MDL module is compiled, otherwise it
    ///       will fail
    ///
    /// \return true on success, false otherwise
    bool add_builtin_module(
        char const *abs_name,
        char const *buffer,
        size_t     buf_len,
        bool       is_encoded,
        bool       is_native) MDL_FINAL;

    /// Get the used allocator.
    mi::base::IAllocator *get_mdl_allocator() const MDL_FINAL;

    /// Get the MDL version of a given Module.
    ///
    /// \param module  the module
    MDL_version get_module_version(IModule const *module) const MDL_FINAL;

    /// Creates a new thread context for this compiler.
    Thread_context *create_thread_context() MDL_FINAL;

    /// Create an MDL exporter.
    MDL_exporter *create_exporter() const MDL_FINAL;

    /// Check if a given identifier is a valid MDL identifier.
    ///
    /// \param ident  the identifier to check
    ///
    /// \note The set of keywords depends on the MDL version. Hence a name that is valid for
    ///       one version of MDL might be invalid for another. This function checks against a
    ///       superset of all possible keywords, so it might deny valid names for some MDL
    ///       versions.
    bool is_valid_mdl_identifier(char const *ident) const MDL_FINAL;

    /// Create an MDL entity resolver.
    ///
    /// \param module_cache  If non-NULL, a module cache.
    IEntity_resolver *create_entity_resolver(
        IModule_cache *module_cache) const MDL_FINAL;

    /// Return the current MDL entity resolver.
    ///
    /// If an external resolver is installed, it is returned. Otherwise, a newly created instance
    /// of the built-in resolver using the provided module cache is returned.
    IEntity_resolver *get_entity_resolver(
        IModule_cache *module_cache) const MDL_FINAL;

    /// Create an MDL archive tool using this compiler.
    IArchive_tool *create_archive_tool() MDL_FINAL;

    // Create an MDL encapsulate tool using this compiler.
    IEncapsulate_tool *create_encapsulate_tool() MDL_FINAL;

    /// Create an MDL comparator tool using this compiler.
    IMDL_comparator *create_mdl_comparator() MDL_FINAL;

    /// Create an MDL module transformer using this compiler.
    ///
    /// The module transformer operates on modules, transforming them into semantically equivalent
    /// modules.
    IMDL_module_transformer *create_module_transformer() MDL_FINAL;

    /// Sets a resolver interface that will be used to lookup MDL modules and resources.
    ///
    /// \param resolver  the resolver
    ///
    /// \note This disables the built-it resolver currently.
    void set_external_entity_resolver(IEntity_resolver *resolver) MDL_FINAL;

    /// Check if an external entity resolver is installed.
    bool uses_external_entity_resolver() const MDL_FINAL;

    /// Add a foreign module translator.
    ///
    /// \param translator  the translator
    void add_foreign_module_translator(
        IMDL_foreign_module_translator *translator) MDL_FINAL;

    /// Remove a foreign module translator.
    ///
    /// \param translator  the translator
    ///
    /// \return true on success, false if the given translator was not found
    bool remove_foreign_module_translator(
        IMDL_foreign_module_translator *translator) MDL_FINAL;

    // ------------------- non interface methods ---------------------------

    /// Create an empty module.
    ///
    /// \param module_name   the (absolute) name of the module
    /// \param file_name     the file name of the module
    /// \param version       the MDL language level of this module
    /// \param flags         module property flags
    Module *create_module(
        char const        *module_name,
        char const        *file_name,
        IMDL::MDL_version version,
        unsigned          flags);

    /// Check if a given identifier is a valid MDL identifier.
    ///
    /// \param ident  the identifier to check
    ///
    /// \note The set of keywords depends on the MDL version. Hence a name that is valid for
    ///       one version of MDL might be invalid for another. This function checks against a
    ///       superset of all possible keywords, so it might deny valid names for some MDL
    ///       versions.
    static bool valid_mdl_identifier(char const *ident);

    /// Check if the compiler supports a requested MDL version.
    ///
    /// \param major                         major number
    /// \param minor                         minor number
    /// \param version                       the accepted MDL version on success
    /// \param enable_experimental_features  if true, allow experimental MDL features
    /// 
    /// \return true on success
    bool check_version(
        int major,
        int minor,
        MDL_version &version,
        bool enable_experimental_features);

    /// Register a builtin module and take ownership of it.
    ///
    /// \param module  the builtin module
    void register_builtin_module(const Module *module);

    /// Find a builtin module by name.
    ///
    /// \param name  the absolute module name
    ///
    /// \return the builtin module or NULL in no such module exists.
    ///
    /// \note Does NOT increase the reference count of the returned
    ///       module, do NOT decrease it just because of this call.
    Module const *find_builtin_module(string const &name) const;

    /// Find a builtin module by its ID.
    ///
    /// \param id  the module id
    ///
    /// \return the builtin module or NULL in no such module exists.
    ///
    /// \note Does NOT increase the reference count of the returned
    ///       module, do NOT decrease it just because of this call.
    Module const *find_builtin_module(size_t id) const;

    /// Find the definition of a signature of a standard library function.
    ///
    /// \param module_name  the absolute name of a standard library module
    /// \param signature    a (function) signature
    IDefinition const *find_stdlib_signature(
        char const *module_name,
        char const *signature) const;

    /// Resolve an import (module) name to the corresponding absolute module name.
    ///
    /// \param resolver                The file resolver to be used
    /// \param import_name             The relative module name to import.
    /// \param owner_module            The owner MDL module that tries to resolve the import.
    ///                                This may be NULL for the top-level import.
    /// \param pos                     The position where the import must be resolved for an
    ///                                error message.
    ///                                This may be NULL for the top-level import.
    /// \returns                       The import result or null if the module does not exist.
    IMDL_import_result *resolve_import(
        File_resolver  &resolver,
        char const     *import_name,
        Module         *owner_module,
        Position const *pos);

    /// Compile a module with a given name.
    ///
    /// \param ctx           The thread context.
    /// \param module_name   The module name (absolute or relative).
    /// \param module_cache  If non-NULL, a module cache.
    /// \returns             An interface to the compiled module.
    ///
    /// If the module is already loaded, no new module will be created,
    /// but an interface to the already loaded module will be returned.
    Module const *compile_module(
        Thread_context &ctx,
        char const     *module_name,
        IModule_cache  *module_cache);

    /// Compile a foreign module with a given name.
    ///
    /// \param translator    The foreign module translator.
    /// \param ctx           The thread context.
    /// \param module_name   The module name (absolute or relative).
    /// \param module_cache  If non-NULL, a module cache.
    /// \returns             An interface to the compiled module.
    ///
    /// If the module is already loaded, no new module will be created,
    /// but an interface to the already loaded module will be returned.
    Module const *compile_foreign_module(
        IMDL_foreign_module_translator &translator,
        Thread_context                 &ctx,
        char const                     *module_name,
        IModule_cache                  *module_cache);

    /// Compile a module with a given name from a input stream.
    ///
    /// \param ctx           The thread context.
    /// \param module_cache  If non-NULL, a module cache.
    /// \param module_name   The absolute module name.
    /// \param input         A UTF8 input stream.
    /// \param msg_name      The file name of the string used for error reports.
    ///
    /// \returns             An interface to the loaded module.
    /// If the module name already exists, returns NULL.
    Module const *compile_module_from_stream(
        Thread_context &ctx,
        IModule_cache  *cache,
        char const     *module_name,
        IInput_stream  *input,
        char const     *msg_name);

    /// Get an option value.
    ///
    /// \param ctx   if non-NULL, the current thread context
    /// \param name  the name of the requested option
    ///
    /// \return the option value or NULL of the option is not set
    char const *get_compiler_option(
        Thread_context const *ctx,
        char const           *name) const;

    /// Get a bool option.
    ///
    /// \param ctx        if non-NULL, the current thread context
    /// \param name       the name of the requested option
    /// \param def_value  will be returned if this option is not set
    ///
    /// \return the boolean option value or def_value
    bool get_compiler_bool_option(
        Thread_context const *ctx,
        char const           *name,
        bool                 def_value) const;

    /// Get an integer option.
    ///
    /// \param ctx        if non-NULL, the current thread context
    /// \param name       the name of the requested option
    /// \param def_value  will be returned if this option is not set
    ///
    /// \return the integer option value or def_value
    int get_compiler_int_option(
        Thread_context const *ctx,
        char const           *name,
        int                  def_value) const;

    /// Get a float option.
    ///
    /// \param ctx        if non-NULL, the current thread context
    /// \param name       the name of the requested option
    /// \param def_value  will be returned if this option is not set
    ///
    /// \return the float option value or def_value
    float get_compiler_float_option(
        Thread_context const *ctx,
        char const           *name,
        float                def_value) const;

    /// Get a double option.
    ///
    /// \param ctx        if non-NULL, the current thread context
    /// \param name       the name of the requested option
    /// \param def_value  will be returned if this option is not set
    ///
    /// \return the double option value or def_value
    double get_compiler_double_option(
        Thread_context const *ctx,
        char const           *name,
        double               def_value) const;

    /// Return the number of builtin modules.
    size_t get_builtin_module_count() const;

    /// Get the builtin module of given index.
    ///
    /// \param idx  the index
    ///
    /// \note Does NOT increase the reference count of the returned
    ///       module, do NOT decrease it just because of this call.
    Module const *get_builtin_module(size_t idx) const;

    /// Returns true if predefined types must be build, false otherwise.
    bool build_predefined_types();

    /// Get the "weak module reference lock".
    mi::base::Lock &get_weak_module_lock() const;

    /// Get the search path lock.
    mi::base::Lock &get_search_path_lock() const;

    /// Get the Jitted code singleton.
    ///
    /// \note Does NOT increase the reference count of the returned
    ///       object, do NOT decrease it just because of this call.
    Jitted_code *get_jitted_code();

    /// Get the search path helper.
    mi::base::Handle<IMDL_search_path> const &get_search_path() const { return m_search_path; }

    /// Get the external entity resolver.
    mi::base::Handle<IEntity_resolver> const &get_external_resolver() const {
        return m_external_resolver;
    }

    /// Parse a string to an MDL expression.
    ///
    /// \param expr_str                      the expression string to parse
    /// \param start_line                    the starting line of the string
    /// \param start_col                     the starting column of the string
    /// \param module                        the module receiving any error messages
    /// \param enable_experimental_features  if true, allow experimental MDL features
    /// \param msgs                          a message list
    ///
    /// \return the parsed expression.
    ///         Error messages will be added to the messages passed.
    IExpression const *parse_expression(
        char const    *expr_str,
        int           start_line,
        int           start_col,
        Module        *module,
        bool          enable_experimental_features,
        Messages_impl &msgs);

    /// Creates a new thread context from current analysis settings.
    ///
    /// \param analysis    the current analysis
    /// \param front_path  a front path to set if any
    Thread_context *create_thread_context(
        Analysis const &analysis,
        char const    *front_path);

    /// If the given module name names a foreign module, return the matching translator.
    ///
    /// \param module_name  the module name
    IMDL_foreign_module_translator *is_foreign_module(
        char const *module_name);

public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    explicit MDL(IAllocator *alloc);

private:
    /// Destructor.
    ~MDL();

private:
    /// Register built-in modules at a module cache
    void register_builtin_module_at_cache(IModule_cache *cache);

    /// Create all builtin semantics.
    void create_builtin_semantics();

    /// Create all options (and default values) of the compiler.
    void create_options();

    /// Load a module from a stream.
    ///
    /// \param cache        if non-NULL, a module cache of already loaded modules
    /// \param ctx          the thread context or NULL
    /// \param module_name  the absolute module name
    /// \param s            the input stream of the module
    /// \param flags        module property flags
    /// \param msg_name     if non-NULL, use this name for reporting compiler messages
    Module *load_module(
        IModule_cache   *cache,
        IThread_context *ctx,
        char const      *module_name,
        IInput_stream   *s,
        unsigned        flags,
        char const      *msg_name = NULL);

    /// Serialize a module and all its imported modules in bottom-up order.
    ///
    /// \param module                the module to serialize
    /// \param is                    the serializer for the output data
    /// \param bin_serializer        the binary serializer to handle uniqueness
    /// \param is_root               true if mod is the root module
    void serialize_module_with_imports(
        Module const          *mod,
        ISerializer           *is,
        MDL_binary_serializer &bin_serializer,
        bool                  is_root) const;

    /// Copy all messages from the given module to the given message list.
    ///
    /// \param dst  destination message list
    /// \param mod  if non-NULL, all compiler messages are copied
    void copy_message(Messages_impl &dst, Module const *mod);

private:
    /// The builder for all created interface.
    mutable Allocator_builder m_builder;

    /// Next unique module id.
    size_t m_next_module_id;

    /// Arena for the compiler.
    Memory_arena m_arena;

    /// The global type factory of this compiler.
    mutable Type_factory m_type_factory;

    /// The options table.
    Options_impl m_options;

    typedef hash_map<string, size_t, string_hash<string> >::Type Module_map;

    /// The map of builtin modules names to indexes in the builtin vector.
    Module_map m_builtin_module_indexes;

    /// The builtin modules.
    vector<mi::base::Handle<Module const> >::Type m_builtin_modules;

    typedef hash_map<
        char const *,
        IDefinition::Semantics,
        cstring_hash,
        cstring_equal_to
    >::Type Sema_map;

    /// The map of builtin semantics.
    Sema_map m_builtin_semantics;

    /// The search path helper.
    mi::base::Handle<IMDL_search_path> m_search_path;

    /// If set, use this external entity resolver instead of the search path.
    mi::base::Handle<IEntity_resolver> m_external_resolver;

    /// Global compiler lock.
    mi::base::Lock m_global_lock;

    /// The search path lock for this compiler.
    mutable mi::base::Lock m_search_path_lock;

    /// The shared lock for all module's weak import tables.
    mutable mi::base::Lock m_weak_module_lock;

    /// Set to true after predefined types are created.
    bool m_predefined_types_build;

    /// The Jitted code singleton if any.
    Jitted_code *m_jitted_code;

    typedef list<mi::base::Handle<IMDL_foreign_module_translator> >::Type Translator_list;

    /// The list of registered translators.
    Translator_list m_translator_list;
};

/// Implementation of the factory function mi_mdl_factory().
///
/// \param allocator  An allocator interface that will be used for all
///                   memory allocations in this compiler.
///
/// \return A new MDL compiler interface.
mi::mdl::IMDL *initialize(IAllocator *allocator = NULL);

} // mdl
} // mi

#endif

