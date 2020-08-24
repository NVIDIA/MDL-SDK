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
/// \file mi/mdl/mdl_mdl.h
/// \brief Interfaces for the MDL core compiler.
#ifndef MDL_MDL_H
#define MDL_MDL_H 1

#include <mi/base/types.h>
#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_options.h>

namespace mi {

namespace base { class IAllocator; }

namespace mdl {

class IArchive_tool;
class IEncapsulate_tool;
class IInput_stream;
class IModule;
class IModule_cache;
class IEntity_resolver;
class IOutput_stream;
class IPrinter;
class IType_factory;
class IThread_context;
class IValue;
class IValue_factory;
class IGenerated_code_dag;
class ISerializer;
class IDeserializer;
class IMDL_comparator;
class IMDL_exporter;
class IMDL_foreign_module_translator;
class IMDL_module_transformer;

/// An interface handling MDL search paths.
///
/// This interface must be implemented by the user application. Note that an application should not
/// modify the content of a search path while compilation is active. If necessary, the application
/// must ensure that access is locked in that case.
class IMDL_search_path : public
    mi::base::Interface_declare<0x2ad74621,0x160f,0x45ff,0xbe,0xaf,0x73,0x31,0xb6,0x3e,0xad,0x1b,
    mi::base::IInterface>
{
public:
    /// Supported path sets.
    ///
    /// For historic reasons, the MDL entity resolver supports two sets of
    /// search paths. \c MDL_SEARCH_PATH is the specification conform
    /// set of paths, as described by the MDL specification.
    ///
    /// If a resource is not found there, the current entity resolver also
    /// looks up paths in the \c MDL_RESOURCE_SET. This feature is deprecated
    /// and will be removed in the future, do _not_ use it.
    /// It is legal to return 0 for the \c MDL_RESOURCE_SET count, do _not_
    /// copy the \c MDL_SEARCH_PATH paths here.
    enum Path_set {
        MDL_SEARCH_PATH,    ///< The MDL search path as defined by the MDL spec.
        MDL_RESOURCE_PATH,  ///< Deprecated resource path, _Non_ MDL spec compliant.
    };

public:
    /// Get the number of search paths.
    ///
    /// \param set  the path set
    virtual size_t get_search_path_count(Path_set set) const = 0;

    /// Get the i'th search path.
    ///
    /// \param set  the path set
    /// \param i    index of the path
    virtual char const *get_search_path(Path_set set, size_t i) const = 0;
};

/// The primary interface to the MDL core compiler and the MDL Core API.
///
/// For historical reasons, this interface is not only the MDL compiler, but also the
/// primary entry point to the MDL Core API. This will be subject to change later.
///
/// An IMDL interface can be obtained by calling the mi_mdl_factory() function.
class IMDL : public
    mi::base::Interface_declare<0x6af90e47,0xb232,0x4567,0xa7,0x54,0x5d,0xce,0x7b,0x3e,0x0a,0x76,
    mi::base::IInterface>
{
public:
    /// The MDL version of an MDL module.
    enum MDL_version {
        MDL_VERSION_1_0,                        ///< compile MDL 1.0
        MDL_VERSION_1_1,                        ///< compile MDL 1.1
        MDL_VERSION_1_2,                        ///< compile MDL 1.2
        MDL_VERSION_1_3,                        ///< compile MDL 1.3
        MDL_VERSION_1_4,                        ///< compile MDL 1.4
        MDL_VERSION_1_5,                        ///< compile MDL 1.5
        MDL_VERSION_1_6,                        ///< compile MDL 1.6
        MDL_VERSION_1_7,                        ///< compile MDL 1.7
        MDL_LATEST_VERSION = MDL_VERSION_1_6,   ///< always the latest supported version
        MDL_DEFAULT_VERSION = MDL_VERSION_1_0,  ///< The default compiler version.
    };

    /// The name of the option to dump the auto-typing dependence graph for every
    /// function of a compiled module.
    #define MDL_OPTION_DUMP_DEPENDENCE_GRAPH "dump_dependence_graph"

    /// The name of the option to dump the call graph for every compiled module.
    #define MDL_OPTION_DUMP_CALL_GRAPH "dump_call_graph"

    /// The name of the option that steers warnings.
    #define MDL_OPTION_WARN "warning"

    /// The optimization level, 0 disables all optimizations
    #define MDL_OPTION_OPT_LEVEL "opt_level"

    /// The name of the option that switches the strict compilation mode.
    #define MDL_OPTION_STRICT "strict"

    /// The name of the option that enables undocumented experimental MDL features.
    #define MDL_OPTION_EXPERIMENTAL_FEATURES "experimental"

    /// The name of the option that controls, if resources are to be resolved by the compiler.
    #define MDL_OPTION_RESOLVE_RESOURCES "resolve_resources"

    /// The value of \c limits::FLOAT_MIN.
    #define MDL_OPTION_LIMITS_FLOAT_MIN "limits::FLOAT_MIN"

    /// The value of \c limits::FLOAT_MAX.
    #define MDL_OPTION_LIMITS_FLOAT_MAX "limits::FLOAT_MAX"

    /// The value of \c limits::DOUBLE_MIN.
    #define MDL_OPTION_LIMITS_DOUBLE_MIN "limits::DOUBLE_MIN"

    /// The value of \c limits::DOUBLE_MAX.
    #define MDL_OPTION_LIMITS_DOUBLE_MAX "limits::DOUBLE_MAX"

    /// The value of \c state::WAVELENGTH_BASE_MAX.
    #define MDL_OPTION_STATE_WAVELENGTH_BASE_MAX "state::WAVELENGTH_BASE_MAX"


    /// The name of the option to keep resource file paths as is.
    #define MDL_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS "keep_original_resource_file_paths"

public:
    /// Get the type factory of the compiler.
    ///
    /// This returns the type factory of the MDL compiler itself, which is different from all other
    /// type factories. It owns the built-in types but cannot create user types, i.e. neither
    /// new structs, nor new enums, arrays or aliases.
    virtual IType_factory *get_type_factory() const = 0;

    /// Create a new empty MDL module.
    ///
    /// \param context      the threat context for this operation
    /// \param module_name  the name of the new module
    /// \param version      the MDL language level of this module
    ///
    /// \returns an empty new module or NULL if the given name already exists
    ///
    /// Empty modules can be created to construct ASTs in them. To do this, get the factories
    /// from the newly created IModule, use them to construct the AST and add the top-level
    /// entities to the module via IModule::add_declaration().
    /// Once ready, a call to IModule::analyze() will check and finalize the created module,
    /// making it usable for the DAG backend.
    virtual IModule *create_module(
        IThread_context   *context,
        char const        *module_name,
        IMDL::MDL_version version = IMDL::MDL_DEFAULT_VERSION) = 0;

    /// Install an MDL search path helper.
    ///
    /// \param search_path  the new search path helper to install, takes ownership
    ///
    /// The new search path helper will be released at this interface
    /// life time. Any previously set helper will be released now.
    /// Note that this method does not increase the reference count of the objects,
    /// it just "takes" it.
    ///
    /// It is legal to pass NULL here, this sets the search path to empty.
    virtual void install_search_path(IMDL_search_path *search_path) = 0;

    /// Load a module with a given name.
    ///
    /// \param context      if non-NULL, the thread context for this operation
    /// \param module_name  the absolute module name
    /// \param cache        if non-NULL, a module cache
    ///
    /// \returns            an interface to the loaded module
    ///
    /// This is the central entry point to "compile" an MDL module.
    ///
    /// If a module cache is provided and it already contains the module, no new module will be
    /// created, but an interface to the already cached module will be returned.
    virtual IModule const *load_module(
        IThread_context *context,
        char const      *module_name,
        IModule_cache   *cache) = 0;

    /// Load a module with a given name from a given string.
    ///
    /// \param context                 If non-NULL, the threat context for this operation.
    /// \param cache                   If non-NULL, a module cache.
    /// \param module_name             The absolute module name.
    /// \param utf8_buffer             An UTF8 buffer (must start with a BOM) or an ASCII buffer.
    /// \param length                  The length of the buffer in bytes.
    ///
    /// \returns                       An interface to the loaded module or NULL
    ///                                if the module name already exists.
    ///
    /// This method works like load_module(), but reads the module content from a string.
    /// Note that the MDL specification does not define string modules, and they can easily
    /// add harm to your ecosystem by "stealing and hiding" valid modules. Try to avoid them.
    virtual IModule const *load_module_from_string(
        IThread_context *context,
        IModule_cache   *cache,
        char const      *module_name,
        char const      *utf8_buffer,
        size_t          length) = 0;

    /// Load a module with a given name from a given input stream.
    ///
    /// \param context                 If non-NULL, the threat context for this operation
    /// \param cache                   If non-NULL, a module cache.
    /// \param module_name             The absolute module name.
    /// \param stream                  An input stream.
    ///
    /// \returns                       An interface to the loaded module or NULL
    ///                                if the module name already exists.
    ///
    /// This method works like load_module(), but reads the module content from a stream.
    /// Note that the MDL specification does not define stream modules, and they can easily
    /// add harm to your ecosystem by "stealing and hiding" valid modules. Try to avoid them.
    virtual IModule const *load_module_from_stream(
        IThread_context *context,
        IModule_cache   *cache,
        char const      *module_name,
        IInput_stream   *stream) = 0;

    /// Load a code generator.
    ///
    /// \param target_language  The name of the target language for which to generate code.<br>
    ///                         Available target languages and the returned interfaces:
    ///                          - "dag":  ICode_generator_dag
    ///                          - "jit":  ICode_generator_jit
    ///
    /// \returns                An interface to the loaded code generator or NULL if
    ///                         \p target_language is not supported.
    ///
    /// If the code generator is already loaded, no new code generator will be created,
    /// but an interface to the already loaded code generator will be returned.
    virtual ICode_generator *load_code_generator(char const *target_language) = 0;

    /// Create a printer.
    ///
    /// \param stream  an output stream the new printer will print to
    ///
    /// This method creates a new printer that will operate on the given output stream,
    /// \see IPrinter.
    ///
    /// \note Pass an IOutput_stream_colored for colored output.
    virtual IPrinter *create_printer(IOutput_stream *stream) const = 0;

    /// Access compiler options.
    ///
    /// Get access to the MDL compiler options, \see mdl_compiler_options.
    /// Note that this manipulates the global options, visible for all compilations.
    /// If the options should only be changed for a single compilation, use the options
    /// from an IThread_context.
    virtual Options &access_options() = 0;

    /// Return a builtin semantic for a given absolute intrinsic function name.
    ///
    /// \param name  the name of a builtin intrinsic function
    ///
    /// \returns  the semantic for the given intrinsic function name
    ///           \c DS_UNKNOWN otherwise
    virtual IDefinition::Semantics get_builtin_semantic(char const *name) const = 0;

    /// Evaluates an intrinsic function called on constant arguments.
    ///
    /// \param value_factory  The value factory used to create new values
    /// \param sema           The semantic of the intrinsic function
    /// \param arguments      The values of the function arguments
    /// \param n_arguments    The number of arguments
    ///
    /// \return The function result or IValue_bad if the function could
    ///         not be evaluated
    virtual IValue const *evaluate_intrinsic_function(
        IValue_factory         *value_factory,
        IDefinition::Semantics sema,
        IValue const * const   arguments[],
        size_t                 n_arguments) const = 0;

    /// Serialize a module to the given serializer.
    ///
    /// \param module                the module to serialize
    /// \param is                    the serializer data is written to
    /// \param include_dependencies  if true, the module is written including all
    ///                              modules it imports (self-contained binary)
    virtual void serialize_module(
        IModule const *module,
        ISerializer   *is,
        bool          include_dependencies) const = 0;

    /// Deserialize a module from a given deserializer.
    ///
    /// \param ds  the deserializer data is read from
    ///
    /// \return the module
    virtual IModule const *deserialize_module(IDeserializer *ds) = 0;

    /// Predefined streams kinds.
    enum Std_stream {
        OS_STDOUT,  ///< Mapped to OS specific standard out.
        OS_STDERR,  ///< Mapped to OS specific standard error.
        OS_STDDBG,  ///< Mapped to OS specific standard debug output.
    };

    /// Create an IOutput_stream standard stream.
    ///
    /// \param kind  a standard stream kind
    ///
    /// \return an IOutput_stream stream
    ///
    /// \note For an OS that supports colored output, an IOutput_stream_colored
    ///       interface is returned.
    virtual IOutput_stream *create_std_stream(Std_stream kind) const = 0;

    /// Create an IOutput_stream from a file.
    ///
    /// \param filename  the file name
    ///
    /// \return an IOutput_stream stream or NULL if the file could not be created
    virtual IOutput_stream *create_file_output_stream(char const *filename) const = 0;

    /// Create an IInput_stream from a file.
    ///
    /// \param filename  the file name
    ///
    /// \return an IInput_stream stream or NULL if the file could not be opened
    virtual IInput_stream *create_file_input_stream(char const *filename) const = 0;

    /// Serialize a code DAG to the given serializer.
    ///
    /// \param code                  the code DAG to serialize
    /// \param is                    the serializer data is written to
    virtual void serialize_code_dag(
        IGenerated_code_dag const *code,
        ISerializer               *is) const = 0;

    /// Deserialize a code DAG from a given deserializer.
    ///
    /// \param ds  the deserializer data is read from
    ///
    /// \return the code DAG
    virtual IGenerated_code_dag const *deserialize_code_dag(IDeserializer *ds) = 0;

    /// Create a new MDL lambda function.
    ///
    /// \param context  the execution context for this lambda function.
    ///
    /// \returns  a new lambda function.
    virtual ILambda_function *create_lambda_function(
        ILambda_function::Lambda_execution_context context) = 0;

    /// Create a new MDL distribution function.
    ///
    /// \returns  a new distribution function.
    virtual IDistribution_function *create_distribution_function() = 0;

    /// Serialize a lambda function to the given serializer.
    ///
    /// \param lambda                the lambda function to serialize
    /// \param is                    the serializer data is written to
    virtual void serialize_lambda(
        ILambda_function const *lambda,
        ISerializer            *is) = 0;

    /// Deserialize a lambda function from a given deserializer.
    ///
    /// \param ds  the deserializer data is read from
    ///
    /// \return the lambda function
    virtual ILambda_function *deserialize_lambda(IDeserializer *ds) = 0;

    /// Check if the given absolute module name names a builtin MDL module.
    ///
    /// \param absname  an absolute module name
    ///
    /// \return true if \p absname names a builtin module, false otherwise
    virtual bool is_builtin_module(char const *absname) const = 0;

    /// Add a new builtin module to the MDL compiler.
    ///
    /// \param abs_name    the absolute name of the module to add
    /// \param buffer      the buffer containing the source of the builtin module
    /// \param buf_len     the length of the buffer
    /// \param is_encoded  if true, buffer points to an encoded buffer
    /// \param is_native   if true, the added module is native
    ///
    /// \note Builtin modules must be added before any real MDL modules are loaded or created,
    ///       otherwise this call will fail.
    ///
    /// \return true on success, false otherwise
    virtual bool add_builtin_module(
        char const *abs_name,
        char const *buffer,
        size_t     buf_len,
        bool       is_encoded,
        bool       is_native) = 0;

    /// Get the used allocator.
    virtual mi::base::IAllocator *get_mdl_allocator() const = 0;

    /// Get the MDL version of a given Module.
    ///
    /// \param module  the module
    virtual MDL_version get_module_version(IModule const *module) const = 0;

    /// Creates a new thread context for this compiler.
    virtual IThread_context *create_thread_context() = 0;

    /// Create an MDL exporter.
    ///
    /// An IMDL_exporter interface can be used to write a valid IModule
    /// to an output stream.
    virtual IMDL_exporter *create_exporter() const = 0;

    /// Check if a given identifier is a valid MDL identifier.
    ///
    /// \param ident  the identifier to check
    ///
    /// \note The set of keywords depends on the MDL version. Hence a name that is valid for
    ///       one version of MDL might be invalid for another. This function checks against a
    ///       superset of all possible keywords, so it might deny valid names for some MDL
    ///       versions.
    virtual bool is_valid_mdl_identifier(char const *ident) const = 0;

    /// Create an MDL entity resolver.
    ///
    /// \param module_cache  If non-NULL, a module cache.
    ///
    /// This returns a new entity resolver. The resolver automatically inherits the search path
    /// from the compiler.
    virtual IEntity_resolver *create_entity_resolver(
        IModule_cache *module_cache) const = 0;

    /// Return the current MDL entity resolver.
    ///
    /// If an external resolver is installed, it is returned. Otherwise, a newly created instance
    /// of the built-in resolver using the provided module cache is returned.
    virtual IEntity_resolver *get_entity_resolver(
        IModule_cache *module_cache) const = 0;

    /// Create an MDL archive tool using this compiler.
    ///
    /// The archive tool is used to create and unpack MDL archives.
    virtual IArchive_tool *create_archive_tool() = 0;

    /// Create an MDL encapsulate tool using this compiler.
    ///
    /// The encapsulate tool stored a pre-processed module with all resources in an MDLe file.
    virtual IEncapsulate_tool *create_encapsulate_tool() = 0;

    /// Create an MDL comparator tool using this compiler.
    ///
    /// The comparator tool is used to compare MDL modules and archives for "compatibility".
    virtual IMDL_comparator *create_mdl_comparator() = 0;

    /// Create an MDL module transformer using this compiler.
    ///
    /// The module transformer operates on modules, transforming them into semantically equivalent
    /// modules.
    virtual IMDL_module_transformer *create_module_transformer() = 0;

    /// Sets a resolver interface that will be used to lookup MDL modules and resources.
    ///
    /// \param resolver  The new resolver. Pass NULL to disable a previously installed resolver and
    ///                  to continue to use the built-in resolver.
    ///
    /// \note An external resolver disables the built-in resolver.
    virtual void set_external_entity_resolver(IEntity_resolver *resolver) = 0;

    /// Check if an external entity resolver is installed.
    virtual bool uses_external_entity_resolver() const = 0;

    /// Add a foreign module translator.
    ///
    /// \param translator  the translator
    virtual void add_foreign_module_translator(
        IMDL_foreign_module_translator *translator) = 0;

    /// Remove a foreign module translator.
    ///
    /// \param translator  the translator
    ///
    /// \return true on success, false if the given translator was not found
    virtual bool remove_foreign_module_translator(
        IMDL_foreign_module_translator *translator) = 0;
};


/*!
\page mdl_compiler_options Options for the MDL compiler

You can configure the MDL compiler by setting options on the #mi::mdl::Options object
returned by #mi::mdl::IMDL::access_options().

These options are specific to the MDL compiler:

- \ref mdl_option_dump_call_graph "dump_call_graph"
- \ref mdl_option_dump_dependence_graph "dump_dependence_graph"
- \ref mdl_option_opt_level "opt_level"
- \ref mdl_option_warn "warning"

These options set render specific constants in the standard library:
 
- \ref mdl_option_limits_double_max "limits::DOUBLE_MAX"
- \ref mdl_option_limits_double_min "limits::DOUBLE_MIN"
- \ref mdl_option_limits_float_max "limits::FLOAT_MAX"
- \ref mdl_option_limits_float_min "limits::FLOAT_MIN"
- \ref mdl_option_limits_wavelength_max "limits::WAVELENGTH_MAX"
- \ref mdl_option_limits_wavelength_min "limits::WAVELENGTH_MIN"
- \ref mdl_option_state_wavelength_base_max "state::WAVELENGTH_BASE_MAX"

\section mdl_compiler_options MDL compiler options

\anchor mdl_option_dump_call_graph
- <b>dump_call_graph:</b> If set to "true", the compiler will dump a call graph for every compiled
  module. Default: "false"

\anchor mdl_option_dump_dependence_graph
- <b>dump_dependence_graph:</b> If set to "true", the compiler will dump the dependence graph for
  every analyzed function. Default: "false"

\anchor mdl_option_opt_level
- <b>opt_level:</b> Specifies the optimization level.
  Possible values are:
  - <b>0</b>: all optimizations are disabled
  - <b>1</b>: only intra procedural optimizations are enabled
  - <b>2</b>: intra and inter procedural optimizations are enabled (default)

\anchor mdl_option_warn
- <b>warning:</b> Modifies warnings, format is "value (',' value)*".
  With \<num\> representing a decimal number, possible values are:
  - <b>err</b>: all warnings are errors
  - <b>\<num\>=off</b>: deactivate warning W\<num\>
  - <b>\<num\>=on</b>:  activate warning W\<num\>
  - <b>\<num\>=err</b>: warning W\<num\> will be treated as an error

\section mdl_stdlib_options MDL standard library options

\anchor mdl_option_limits_double_max
- <b>limits::DOUBLE_MAX:</b> Sets the largest double value supported by the current platform.
  This is the value of the ::%limits::DOUBLE_MAX constant.
  Default: Set to DBL_MAX

\anchor mdl_option_limits_double_min
- <b>limits::DOUBLE_MIN:</b> Sets the smallest positive normalized double value supported by the
  current platform.
  This is the value of the ::%limits::DOUBLE_MIN constant.
  Default: Set to DBL_MIN

\anchor mdl_option_limits_float_max
- <b>limits::FLOAT_MAX:</b> Sets the largest float value supported by the current platform.
  This is the value of the ::%limits::FLOAT_MAX constant.
  Default: Set to FLT_MAX

\anchor mdl_option_limits_float_min
- <b>limits::FLOAT_MIN:</b> Sets the smallest positive normalized float value supported by the
  current platform.
  This is the value of the ::%limits::FLOAT_MIN constant.
  Default: Set to FLT_MIN

\anchor mdl_option_limits_wavelength_max
- <b>limits::WAVELENGTH_MAX:</b> Sets the largest float value that the current platform allows for
  representing wavelengths for the color type and its related functions.
  This is the value of the ::%limits::WAVELENGTH_MAX constant.
  Default: Set to 780.0

\anchor mdl_option_limits_wavelength_min
- <b>limits::WAVELENGTH_MIN:</b> Sets the smallest float value that the current platform allows for
  representing wavelengths for the color type and its related functions.
  This is the value of the ::%limits::WAVELENGTH_MIN constant.
  Default: Set to 380.0

\anchor mdl_option_state_wavelength_base_max
- <b>state::WAVELENGTH_BASE_MAX:</b> Sets the number of wavelengths returned in the result of
  ::%state::wavelength_base().
  This is the value of the ::%state::WAVELENGTH_BASE_MAX constant.
  Default: Set to 1
*/

/*!
\page mdl_compilation Compiling MDL modules

The MDL core compiler compiles MDL modules into an \c IModule.
The \c IModule basically contains the attributed abstract syntax tree of this module.
The method \c IModule::is_valid() returns true if the module is error-free and can be
used to generate code by the backends.

\section mdl_avoid_recompilation Avoid recompilation of modules

The compiler itself is stateless, i.e. it creates a new module on every invocation.
The same is true for imports, i.e. if a module \c A is compiled, followed by the
compilation of module \c B, and both import a common module \c C, then \c C is compiled
two times, once in the context of module \c A compilation, once in the context of
module \c B compilation.
However, it is ensured that a module is compiled only once during one compilation request, i.e.
if a module \c D is imported several times (for instance hidden by other imports), it is
only compiled on the first import.

This behavior allows to compile always the current view, i.e. one can modify the MDL source
on disk and recompile the current version.
It is even possible to have several versions of one module, because one compilation always
produces a new result.

For an application with a static set of modules, this behavior might compile a module more than
once, producing semantically equal results.
To avoid that, an application can implement the \c IModule_cache interface.

Before the compiler compiles a module, it also interrogates the \c IModule_cache interface.
A user implementation might cache every compiled module to minimize compilation time, or
check timestamps or similar to detect if a module needs to be recompiled.

\section mdl_compile_materials Compiling materials

After a module was successfully compiled, the next step is to compile the contained
materials.
This step is done by the DAG-Backend.
It transforms the attributed abstract syntax tree into a Directed-Acyclic-Graph
intermediate representation \c IGenerated_code_dag.
This IR is much more suited for analyzing the properties of an MDL material.
See \ref mdl_dag_ir for a deeper description of this representation.

From the DAG-IR of the materials of a module, material instances can be created.
A material instance represents a set of material arguments bound to a material.

\section mdl_material_instances_modes Instance-compilation and class-compilation

There are two different compilation modes to accommodate different needs: instance
and class compilation.
In instance compilation mode (the default) all arguments, i.e., constant expressions
and call expressions, are compiled into the body of the resulting material.
Hence, the resulting IGenerated_code_dag::IMaterial_instance is parameterless.
This mode offers most optimization possibilities.
As a downside, even the smallest argument change requires a recompilation to target code.

In class compilation mode, only call expressions, i.e., the structure of the graph
created by the call expressions, are compiled into the body of the resulting material.
The DAG constants (\c DAG_constant) remain as arguments of the material instance.
Note, that the parameters of a material instance do not in general correspond to the
parameters of the original material:

  - Unused parameters will not make it into the compiled material.
  - The order is arbitrary.
  - Function calls will become part of the material.
    Hence, if you instantiate a material parameter "int x" with a call to "math::max(1, 3)"
    (prototype "int math::max(int a, int b)"), there will be no parameter with name "x"
    but the two constants used in the call with be turned into the parameters "x.a" and "x.b".
    The constants "1" and "3" will become the default arguments of the parameters "x.a"
    and "x.b", respectively.
    The parameter names are generated by walking the path from the material parameters
    to the literals and concatenating all parameter names that are visited by ".".

\subsection mdl_material_modify_class_args Modifying arguments of a class-compiled material

If a class-compiled material still has parameters after the compilation, the generated code will
expect a pointer to a "target argument block", containing the data of the arguments in a
target-specific layout.
The layouts are available as IGenerated_code_value_layout objects returned by
IGenerated_code_executable::get_captured_arguments_layout() or ILink_unit::get_arg_block_layout().
For each class-compiled material with parameters, one layout will be added to the lists.

Using the parameter name and type information from the IGenerated_code_dag::IMaterial_instance
object corresponding to the layout, the layout can be navigated with
IGenerated_code_value_layout::get_nested_state() and the offset, kind, and size of an
argument or sub-element can be retrieved via IGenerated_code_value_layout::get_layout().

So, if you wanted to get the offset of \c "x.b" in the above example case, you would search for the
index for which IGenerated_code_dag::IMaterial_instance::get_parameter_name() returns \c "x.b" and
use this index to get the nested layout state for it.
Providing this state to IGenerated_code_value_layout::get_layout() would then result in the offset
of this argument within the target argument block data.

*/

} // mdl
} // mi

#endif // MDL_MDL_H

