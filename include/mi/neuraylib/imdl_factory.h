/***************************************************************************************************
 * Copyright (c) 2014-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      API component that gives access to some MDL functionality

#ifndef MI_NEURAYLIB_IMDL_FACTORY_H
#define MI_NEURAYLIB_IMDL_FACTORY_H

#include <mi/base/interface_declare.h>
#include <mi/neuraylib/iexpression.h>
#include <mi/neuraylib/version.h>

#if defined (MI_NEURAYLIB_DEPRECATED_12_1) || defined(MI_NEURAYLIB_DEPRECATED_14_0)
#include <mi/neuraylib/imdl_execution_context.h>
#endif

namespace mi {

class IArray;
class IString;

namespace neuraylib {

class IExpression_factory;
class IMdl_execution_context;
class IMdl_module_builder;
class IMdl_module_transformer;
class ITransaction;
class IType_factory;
class IValue_bsdf_measurement;
class IValue_factory;
class IValue_light_profile;
class IValue_texture;

/**
    \defgroup mi_neuray_mdl_misc Miscellaneous MDL-related Interfaces
    \ingroup mi_neuray

    MDL-related interfaces excluding types, scene elements, and the compiler
*/

/** \addtogroup mi_neuray_mdl_misc
@{
*/

/// Factory for various MDL interfaces and functions.
///
/// This interface gives access to the type, value, and expressions factories. It also allows to
/// create material and function variants.
class IMdl_factory : public
    mi::base::Interface_declare<0xba936279,0x4b71,0x42a4,0x95,0x37,0x98,0x69,0x97,0xb3,0x47,0x72>
{
public:
    /// Returns an MDL type factory for the given transaction.
    virtual IType_factory* create_type_factory( ITransaction* transaction) = 0;

    /// Returns an MDL value factory for the given transaction.
    virtual IValue_factory* create_value_factory( ITransaction* transaction) = 0;

    /// Returns an MDL expression factory for the given transaction.
    virtual IExpression_factory* create_expression_factory( ITransaction* transaction) = 0;

    /// Creates an execution context.
    virtual IMdl_execution_context* create_execution_context() = 0;

    /// Clones an execution context.
    ///
    /// Creates a new execution context if \p context is \c NULL (as in
    /// #create_execution_context()). There is \em no deep copy of option values of type
    /// #mi::base::IInterface, they are shared by both instances.
    ///
    /// Useful to change options temporarily.
    virtual IMdl_execution_context* clone( const IMdl_execution_context* context) = 0;

    /// Creates a value referencing a texture identified by an MDL file path.
    ///
    /// \param transaction   The transaction to be used.
    /// \param file_path     The absolute MDL file path that identifies the texture. The MDL
    ///                      search paths are used to resolve the file path. See section 2.2 in
    ///                      [\ref MDLLS] for details.
    /// \param shape         The value that is returned by
    ///                      #mi::neuraylib::IType_texture::get_shape() on the type corresponding
    ///                      to the return value.
    /// \param gamma         The value that is returned by #mi::neuraylib::ITexture::get_gamma()
    ///                      on the DB element referenced by the return value.
    /// \param selector      The selector, or \c NULL. See section 2.3.1 in [\ref MDLLS] for
    ///                      details.
    /// \param shared        Indicates whether you want to re-use the DB elements for that texture
    ///                      if it has already been loaded, or if you want to create new DB elements
    ///                      in all cases. Note that sharing is based on the location where the
    ///                      texture is finally located and includes sharing with instances that
    ///                      have not explicitly been loaded via this method, e.g., textures in
    ///                      defaults.
    /// \param context       An execution context which can be queried for detailed error messages
    ///                      after the operation has finished. Can be \c NULL. The error codes have
    ///                      the following meaning:
    ///                      -   0: Success.
    ///                      -  -1: Invalid parameters (\c NULL pointer).
    ///                      -  -2: The file path is not an absolute MDL file path.
    ///                      -  -3: No image plugin found to handle the file.
    ///                      -  -4: Failure to resolve the given filename, e.g., the file does
    ///                             not exist.
    ///                      -  -5: Failure to open the resolved file.
    ///                      -  -7: The image plugin failed to import the data.
    ///                      - -10: Failure to apply the given selector.
    /// \return              The value referencing the texture, or \c NULL in case of failure.
    ///
    /// \see #mi::neuraylib::IImage::reset_file() \if IRAY_API or
    ///      #mi::neuraylib::IVolume_data::reset_file() \endif if you are given a plain filename
    ///      instead of an MDL file path.
    virtual IValue_texture* create_texture(
        ITransaction* transaction,
        const char* file_path,
        IType_texture::Shape shape,
        Float32 gamma,
        const char* selector,
        bool shared,
        IMdl_execution_context* context) = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline IValue_texture* create_texture(
        ITransaction* transaction,
        const char* file_path,
        IType_texture::Shape shape,
        Float32 gamma,
        bool shared,
        Sint32* errors = 0)
    {
        mi::base::Handle<IMdl_execution_context> context( create_execution_context());
        IValue_texture* result = create_texture(
            transaction, file_path, shape, gamma, 0, shared, context.get());
        if( errors) {
            mi::base::Handle<const IMessage> msg( context->get_error_message( 0));
            *errors = msg ? msg->get_code() : 0;
        }
        return result;
    }
#endif // MI_NEURAYLIB_DEPRECATED_12_1

#ifdef MI_NEURAYLIB_DEPRECATED_14_0
    inline IValue_texture* create_texture(
        ITransaction* transaction,
        const char* file_path,
        IType_texture::Shape shape,
        Float32 gamma,
        const char* selector,
        bool shared,
        Sint32* errors = 0)
    {
        mi::base::Handle<IMdl_execution_context> context( create_execution_context());
        IValue_texture* result = create_texture(
            transaction, file_path, shape, gamma, selector, shared, context.get());
        if( errors) {
            mi::base::Handle<const IMessage> msg( context->get_error_message( 0));
            *errors = msg ? msg->get_code() : 0;
        }
        return result;
    }
#endif // MI_NEURAYLIB_DEPRECATED_14_0

    /// Creates a value referencing a light profile identified by an MDL file path.
    ///
    /// \param transaction   The transaction to be used.
    /// \param file_path     The absolute MDL file path that identifies the light profile. The MDL
    ///                      search paths are used to resolve the file path. See section 2.2 in
    ///                      [\ref MDLLS] for details.
    /// \param shared        Indicates whether you want to re-use the DB element for that light
    ///                      profile if it has already been loaded, or if you want to create a new
    ///                      DB element in all cases. Note that sharing is based on the location
    ///                      where the light profile is finally located and includes sharing with
    ///                      instances that have not explicitly been loaded via this method, e.g.,
    ///                      light profiles in defaults.
    /// \param context       An execution context which can be queried for detailed error messages
    ///                      after the operation has finished. Can be \c NULL. The error codes have
    ///                      the following meaning:
    ///                      -   0: Success.
    ///                      -  -1: Invalid parameters (\c NULL pointer).
    ///                      -  -2: The file path is not an absolute MDL file path.
    ///                      -  -3: Invalid filename extension (only \c .ies is supported).
    ///                      -  -4: Failure to resolve the given filename, e.g., the file does
    ///                             not exist.
    ///                      -  -5: Failure to open the resolved file.
    ///                      -  -7: File format error.
    /// \return              The value referencing the light profile, or \c NULL in case of failure.
    ///
    /// \see #mi::neuraylib::ILightprofile::reset_file() if you are given a plain filename instead
    ///      of an MDL file path.
    virtual IValue_light_profile* create_light_profile(
        ITransaction* transaction,
        const char* file_path,
        bool shared,
        IMdl_execution_context* context) = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_14_0
    inline IValue_light_profile* create_light_profile(
        ITransaction* transaction,
        const char* file_path,
        bool shared,
        Sint32* errors = 0)
    {
        mi::base::Handle<IMdl_execution_context> context( create_execution_context());
        IValue_light_profile* result = create_light_profile(
            transaction, file_path, shared, context.get());
        if( errors) {
            mi::base::Handle<const IMessage> msg( context->get_error_message( 0));
            *errors = msg ? msg->get_code() : 0;
        }
        return result;
    }
#endif // MI_NEURAYLIB_DEPRECATED_14_0

    /// Creates a value referencing a BSDF measurement identified by an MDL file path.
    ///
    /// \param transaction   The transaction to be used.
    /// \param file_path     The absolute MDL file path that identifies the BSDF measurement. The
    ///                      MDL search paths are used to resolve the file path. See section 2.2 in
    ///                      [\ref MDLLS] for details.
    /// \param shared        Indicates whether you want to re-use the DB element for that BSDF
    ///                      measurement if it has already been loaded, or if you want to create a
    ///                      new DB element in all cases. Note that sharing is based on the location
    ///                      where the BSDF measurement is finally located and includes sharing with
    ///                      instances that have not explicitly been loaded via this method, e.g.,
    ///                      BSDF measurements in defaults.
    /// \param context       An execution context which can be queried for detailed error messages
    ///                      after the operation has finished. Can be \c NULL. The error codes have
    ///                      the following meaning:
    ///                      -  0: Success.
    ///                      - -1: Invalid parameters (\c NULL pointer).
    ///                      - -2: The file path is not an absolute MDL file path.
    ///                      - -3: Invalid filename extension (only \c .mbsdf is supported).
    ///                      - -4: Failure to resolve the given filename, e.g., the file does not
    ///                            exist.
    ///                      - -5: Failure to open the resolved file.
    ///                      - -7: Invalid file format.
    /// \return              The value referencing the BSDF measurement, or \c NULL in case of
    ///                      failure.
    ///
    /// \see #mi::neuraylib::IBsdf_measurement::reset_file() if you are given a plain filename
    ///      instead of an MDL file path.
    virtual IValue_bsdf_measurement* create_bsdf_measurement(
        ITransaction* transaction,
        const char* file_path,
        bool shared,
        IMdl_execution_context* context) = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_14_0
    inline IValue_bsdf_measurement* create_bsdf_measurement(
        ITransaction* transaction,
        const char* file_path,
        bool shared,
        Sint32* errors = 0)
    {
        mi::base::Handle<IMdl_execution_context> context( create_execution_context());
        IValue_bsdf_measurement* result = create_bsdf_measurement(
            transaction, file_path, shared, context.get());
        if( errors) {
            mi::base::Handle<const IMessage> msg( context->get_error_message( 0));
            *errors = msg ? msg->get_code() : 0;
        }
        return result;
    }
#endif // MI_NEURAYLIB_DEPRECATED_14_0

    /// Creates a module builder for a given module.
    ///
    /// \param transaction          The transaction to be used.
    /// \param module_name          The DB name of the MDL module to build. If there is no such
    ///                             module, then an empty module with this name and \p
    ///                             min_module_version is created. Otherwise, the existing module
    ///                             is edited. Builtin modules or MDLE modules cannot be built or
    ///                             edited.
    /// \param min_module_version   The initial MDL version of the new module. Ignored if the
    ///                             module exists already.
    /// \param max_module_version   The maximal desired MDL version of the module. If higher than
    ///                             the current MDL version of the module, then the module builder
    ///                             will upgrade the MDL version as necessary to handle requests
    ///                             requiring newer features.
    /// \param context              An execution context which can be queried for detailed error
    ///                             messages after the operation has finished. Can be \c NULL.
    /// \return                     The module builder for the given module, or \c NULL in
    ///                             case of errors.
    virtual IMdl_module_builder* create_module_builder(
        ITransaction* transaction,
        const char* module_name,
        Mdl_version min_module_version,
        Mdl_version max_module_version,
        IMdl_execution_context* context) = 0;

    /// Creates a module transformer for a given module.
    ///
    /// \param transaction   The transaction to be used.
    /// \param module_name   The DB name of the MDL module to transform. Builtin modules cannot be
    ///                      transformed.
    /// \param context       An execution context which can be queried for detailed error messages
    ///                      after the operation has finished. Can be \c NULL.
    /// \return              The module transformer for the given module, or \c NULL in case of
    ///                      errors.
    virtual IMdl_module_transformer* create_module_transformer(
        ITransaction* transaction, const char* module_name, IMdl_execution_context* context) = 0;

    /// Returns the DB name for the MDL name of a module (or file path for MDLE modules).
    ///
    /// For example, given \c "::state", the method returns \c "mdl::state".
    ///
    /// \note This method does not check for existence of the corresponding DB element, nor does it
    ///       necessarily reject an invalid module name.
    ///
    /// \note Usage of this method is strongly recommended instead of manually prepending \c "mdl",
    ///       since (a) the mapping is more complicated than that, e.g., for MDLE modules, and (b)
    ///       the mapping might change in the future.
    ///
    /// \param mdl_name      The MDL name of the module (non-MDLE and MDLE module), or the file path
    ///                      of an MDLE module.
    /// \return              The DB name of that module, or \c NULL if \p mdl_name was detected as
    ///                      invalid.
    virtual const IString* get_db_module_name( const char* mdl_name) = 0;

    /// Returns the DB name for the MDL name of an material or function definition.
    ///
    /// For example, given \c "::state::normal()", the method returns \c "mdl::state::normal()".
    ///
    /// \note This method does not check for existence of the corresponding DB element, nor does it
    ///       necessarily reject an invalid material or function definition name.
    ///
    /// \note Usage of this method is strongly recommended instead of manually prepending \c "mdl",
    ///       since (a) the mapping is more complicated than that, e.g., for definitions from the
    ///       \c ::&lt;builtins&gt; module or MDLE modules, and (b) the mapping might change in the
    ///       future.
    ///
    /// \param mdl_name      The MDL name of the material or function definition.
    /// \return              The DB name of that material or function definition, or \c NULL if
    ///                      \p mdl_name was detected as invalid.
    virtual const IString* get_db_definition_name( const char* mdl_name) = 0;

    /// Analyzes whether an expression graph violates the uniform constraints.
    ///
    /// \note This method can be used to check already created graphs, but it can also be used to
    ///       check whether a hypothetical connection would observe the uniform constraints: First,
    ///       invoke the method with the root of the existing graph, \p root_uniform set to \c
    ///       false (at least for materials), and \p query_expr set to the graph node to be
    ///       replaced. If the call returns with \p query_result set to \c false (and no errors in
    ///       the context), then any (valid) subgraph can be connected. Otherwise, invoke the
    ///       method again with the root of the to-be-connected subgraph, \p root_uniform set to \c
    ///       true, and \p query_expr set to \c NULL. If there are no errors, then the subgraph
    ///       can be connected.
    ///
    /// \note Make sure that \p query_expr (if not \c NULL) can be reached from \p root_name,
    ///       otherwise \p query_result is always \c false. In particular, arguments passed during
    ///       call creation (or later for argument changes) are cloned, and the expression that is
    ///       part of the graph is different from the one that was used to construct the graph
    ///       (equal, but not identical).
    ///
    /// \param transaction             The transaction to be used.
    /// \param root_name               DB name of the root node of the graph (material instance or
    ///                                function call).
    /// \param root_uniform            Indicates whether the root node should be uniform.
    /// \param query_expr              A node of the call graph for which the uniform property is
    ///                                to be queried. This expression is \em only used to identify
    ///                                the corresponding node in the graph, i.e., it even makes
    ///                                sense to pass constant expressions (which by themselves are
    ///                                always uniform) to determine whether a to-be-connected call
    ///                                expression has to be uniform. Can be \c NULL.
    /// \param[out] query_result       Indicates whether \p query_expr needs to be uniform (or
    ///                                \c false if \p query_expr is \c NULL, or in case of errors).
    /// \param[out] error_path         A path to a node of the graph that violates the uniform
    ///                                constraints, or the empty string if there is no such node
    ///                                (or in case of errors). Such violations are also reported
    ///                                via \p context. Can be \c NULL.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c NULL.
    virtual void analyze_uniform(
        ITransaction* transaction,
        const char* root_name,
        bool root_uniform,
        const IExpression* query_expr,
        bool& query_result,
        IString* error_path,
        IMdl_execution_context* context) const = 0;

    /// Decodes a DB or MDL name.
    ///
    /// \param name   The encoded DB or MDL name to be decoded.
    /// \return       The decoded DB or MDL name, or \c NULL if \p name is \c NULL.
    ///
    /// \note This method should only be used for display purposes. Do \em not use the returned
    ///       name to identify functions or materials since this representation is ambiguous. For
    ///       modules, it is possible to re-encode their name without loss of information, see
    ///       #encode_module_name(). This is \em not possible for names of function or material
    ///       definitions.
    ///
    /// \note This method does not require the corresponding module to be loaded. The method does
    ///       not check whether the given name is valid, nor whether it is defined in the
    ///       corresponding module.
    virtual const IString* decode_name( const char* name) = 0;

    /// Encodes a DB or MDL module name.
    ///
    /// \param name    The decoded DB or MDL module name to be encoded.
    /// \return        The encoded DB or MDL module name, or \c NULL if \p name is \c NULL.
    ///
    /// \note This method does not require the corresponding module to be loaded. The method does
    ///       not check whether the given name is valid.
    ///
    /// \see #mi::neuraylib::IMdl_factory::encode_function_definition_name(),
    ///      #mi::neuraylib::IMdl_factory::encode_type_name()
    virtual const IString* encode_module_name( const char* name) = 0;

    /// Encodes a DB or MDL function or material definition name.
    ///
    /// \param name             The decoded DB or MDL name of a function or material definition
    ///                         \em without signature.
    /// \param parameter_types  A static or dynamic array with elements of type #mi::IString
    ///                         representing decoded positional parameter type names. The value
    ///                         \c NULL can be used for functions or materials without parameters
    ///                         (treated like an empty array).
    /// \return                 The encoded function or material definition name, or \c NULL if
    ///                         \p name or one of the array elements is \c NULL.
    ///
    /// \note This method does not require the corresponding module to be loaded. The method does
    ///       not check whether the given name is valid, nor whether it is defined in the
    ///       corresponding module.
    ///
    /// \see #mi::neuraylib::IMdl_factory::encode_module_name(),
    ///      #mi::neuraylib::IMdl_factory::encode_type_name()
    virtual const IString* encode_function_definition_name(
        const char* name, const IArray* parameter_types) const = 0;

    /// Encodes an MDL type name.
    ///
    /// \param name             The decoded MDL name of a type.
    /// \return                 The encoded MDL name of the type, or \c NULL if \p name is \c NULL.
    ///
    /// \note This method does not require the corresponding module to be loaded. The method does
    ///       not check whether the given name is valid, nor whether it is defined in the
    ///       corresponding module.
    ///
    /// \see #mi::neuraylib::IMdl_factory::encode_function_definition_name(),
    ///      #mi::neuraylib::IMdl_factory::encode_module_name()
    virtual const IString* encode_type_name( const char* name) const = 0;

    /// Indicates whether the given string is a valid MDL identifier.
    virtual bool is_valid_mdl_identifier( const char* name) const = 0;
};

/**@}*/ // end group mi_neuray_mdl_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_FACTORY_H
