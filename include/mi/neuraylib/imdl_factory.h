/***************************************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <mi/neuraylib/itype.h>

namespace mi {

class IArray;

namespace neuraylib {

class IExpression_factory;
class IMdl_execution_context;
class IMdl_module_transformer;
class ITransaction;
class IType_factory;
class IValue_bsdf_measurement;
class IValue_factory;
class IValue_light_profile;
class IValue_texture;

/** \addtogroup mi_neuray_mdl_types
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

    /// Creates a new MDL module containing variants.
    ///
    /// A variant is basically a clone of another material or function definition (the prototype)
    /// with different defaults.
    //
    /// \param transaction     The transaction to be used.
    /// \param module_name     The fully-qualified MDL name of the new module (including package
    ///                        names, starts with "::").
    /// \param variant_data    A static or dynamic array of structures of type \c Variant_data. Such
    ///                        a structure has the following members:
    ///                        - const char* \b variant_name \n
    ///                          The name of the variant (non-qualified, without module prefix). The
    ///                          DB name of the variant is created by prefixing this name with the
    ///                          DB name of the new module plus "::".
    ///                        - const char* \b prototype_name \n
    ///                          The DB name of the prototype for this variant.
    ///                        - #mi::neuraylib::IExpression_list* \b defaults \n
    ///                          The variant implicitly uses the defaults of the prototype. This
    ///                          member allows to set explicit defaults for the prototype including
    ///                          adding defaults for parameters of the prototype without default.
    ///                          The type of an argument in the expression list must match the type
    ///                          of the corresponding parameter of the prototype. \n
    ///                          Note that the expressions in \p defaults are copied. This copy
    ///                          operation is a shallow copy, e.g., DB elements referenced in call
    ///                          expressions are \em not copied. \n
    ///                          \c NULL is a valid value which is handled like an empty expression
    ///                          list.
    ///                        - #mi::neuraylib::IAnnotation_block* \b annotations \n
    ///                          The variant does not inherit any annotations from the prototype.
    ///                          This member allows to specify annotations for the variant, i.e.,
    ///                          for the material or function declaration itself (but not for its
    ///                          arguments). \n
    ///                          Note that the annotations are copied. This copy operation is a
    ///                          shallow copy. \n
    ///                          \c NULL is a valid value which is handled like an
    ///                          empty annotation block.
    /// \return
    ///                        -   1: Success (module exists already, creation was skipped).
    ///                        -   0: Success (module was actually created with the variants as its
    ///                               only material or function definitions).
    ///                        -  -1: The module name \p module_name is invalid.
    ///                        -  -2: Failed to compile the module \p module_name.
    ///                        -  -3: The DB name for an imported module is already in use but is
    ///                               not an MDL module, or the DB name for a definition in this
    ///                               module is already in use.
    ///                        -  -4: Initialization of an imported module failed.
    ///                        -  -5: \p transaction, \p module_name or \p variant_data are invalid,
    ///                               \p variant_data is empty, or a struct member for the prototype
    ///                               name, defaults, or annotations has an incorrect type.
    ///                        -  -6: A default for a non-existing parameter was provided.
    ///                        -  -7: The type of a default does not have the correct type.
    ///                        -  -8: Unspecified error.
    ///                        -  -9: One of the annotation arguments is wrong (wrong argument name,
    ///                               not a constant expression, or the argument type does not match
    ///                               the parameter type).
    ///                        - -10: One of the annotations does not exist or it has a currently
    ///                               unsupported parameter type like deferred-sized arrays.
    virtual Sint32 create_variants(
        ITransaction* transaction, const char* module_name, const IArray* variant_data) = 0;

    virtual Sint32 create_materials(
        ITransaction* transaction, const char* module_name, const IArray* material_data) = 0;

    virtual Sint32 create_materials(
        ITransaction* transaction,
        const char* module_name,
        const IArray* mdl_data,
        IMdl_execution_context* context) = 0;

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
    /// \param shared        Indicates whether you want to re-use the DB elements for that texture
    ///                      if it has already been loaded, or if you want to create new DB elements
    ///                      in all cases. Note that sharing is based on the location where the
    ///                      texture is finally located and includes sharing with instances that
    ///                      have not explicitly been loaded via this method, e.g., textures in
    ///                      defaults.
    /// \param errors        An optional pointer to an #mi::Sint32 to which an error code will be
    ///                      written. The error codes have the following meaning:
    ///                      -  0: Success.
    ///                      - -1: Invalid parameters (\c NULL pointer).
    ///                      - -2: The file path is not an absolute MDL file path.
    ///                      - -3: Failed to resolve the given file path, or no suitable image
    ///                            plugin available.
    /// \return              The value referencing the texture, or \c NULL in case of failure.
    virtual IValue_texture* create_texture(
        ITransaction* transaction,
        const char* file_path,
        IType_texture::Shape shape,
        Float32 gamma,
        bool shared,
        Sint32* errors = 0) = 0;

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
    /// \param errors        An optional pointer to an #mi::Sint32 to which an error code will be
    ///                      written. The error codes have the following meaning:
    ///                      -  0: Success.
    ///                      - -1: Invalid parameters (\c NULL pointer).
    ///                      - -2: The file path is not an absolute MDL file path.
    ///                      - -3: Failed to resolve the given file path.
    /// \return              The value referencing the light profile, or \c NULL in case of failure.
    virtual IValue_light_profile* create_light_profile(
        ITransaction* transaction, const char* file_path, bool shared, Sint32* errors = 0) = 0;

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
    /// \param errors        An optional pointer to an #mi::Sint32 to which an error code will be
    ///                      written. The error codes have the following meaning:
    ///                      -  0: Success.
    ///                      - -1: Invalid parameters (\c NULL pointer).
    ///                      - -2: The file path is not an absolute MDL file path.
    ///                      - -3: Failed to resolve the given file path.
    /// \return              The value referencing the BSDF measurement, or \c NULL in case of
    ///                      failure.
    virtual IValue_bsdf_measurement* create_bsdf_measurement(
        ITransaction* transaction, const char* file_path, bool shared, Sint32* errors = 0) = 0;
    
    /// Creates an execution context.
    virtual IMdl_execution_context* create_execution_context() = 0;

    /// Returns the DB name for the MDL name of an MDL module.
    ///
    /// For example, given \c "::state", the method returns \c "mdl::state".
    ///
    /// \note This method does not check for existence of the corresponding DB element, nor does it
    ///       check that the input is a valid MDL module name.
    ///
    /// \note Usage of this method is strongly recommended instead of manually prepending \c "mdl",
    ///       since (a) the mapping is more complicated than that, e.g., for MDLE modules, and (b)
    ///       the mapping might change in the future.
    ///
    /// \param mdl_name      The fully-qualified MDL name of the MDL module (including package
    ///                      names, starting with "::") or an MDLE filepath (absolute or relative
    ///                      to the current working directory).
    /// \return              The DB name of that module, or \c NULL if \p mdl_name is invalid.
    virtual const IString* get_db_module_name( const char* mdl_name) = 0;

    /// Returns the DB name for the MDL name of an MDL material or function definition.
    ///
    /// For example, given \c "::state::normal()", the method returns \c "mdl::state::normal()".
    ///
    /// \note This method does not check for existence of the corresponding DB element, nor does it
    ///       check that the input is a valid MDL material or definition name.
    ///
    /// \note Usage of this method is strongly recommended instead of manually prepending \c "mdl",
    ///       since (a) the mapping is more complicated than that, e.g., for MDLE modules, and (b)
    ///       the mapping might change in the future.
    ///
    /// \param mdl_name      The fully-qualified MDL name of the MDL material or function definition
    ///                      (starts with the fully-qualified MDL name of the coresponding module).
    /// \return              The DB name of that material or function definition, or \c NULL if
    ///                      \p mdl_name is invalid.
    virtual const IString* get_db_definition_name( const char* mdl_name) = 0;
    
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
};

/// Options for repairing material instances and function calls.
///
/// \see #mi::neuraylib::IMaterial_instance::repair() and  #mi::neuraylib::IFunction_call::repair().
enum Mdl_repair_options {
    MDL_REPAIR_DEFAULT = 0,           ///< Default mode, do not alter any inputs.
    MDL_REMOVE_INVALID_ARGUMENTS = 1, ///< Remove an invalid call attached to an argument.
    MDL_REPAIR_INVALID_ARGUMENTS = 2, ///< Attempt to repair invalid calls attached to an argument.
    MDL_REPAIR_OPTIONS_FORCE_32_BIT = 0xffffffffU // Undocumented, for alignment only
};

mi_static_assert(sizeof(Mdl_repair_options) == sizeof(Uint32));

/*@}*/ // end group mi_neuray_mdl_types

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_FACTORY_H
