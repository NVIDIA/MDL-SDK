/***************************************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Scene element Compiled_material

#ifndef MI_NEURAYLIB_ICOMPILED_MATERIAL_H
#define MI_NEURAYLIB_ICOMPILED_MATERIAL_H

#include <mi/neuraylib/iexpression.h>
#include <mi/neuraylib/iscene_element.h>

namespace mi {

namespace neuraylib {

class IMdl_execution_context;

/** \addtogroup mi_neuray_mdl_elements
@{
*/

/// Material slots identify parts of a compiled material.
///
/// \see #mi::neuraylib::ICompiled_material and #mi::neuraylib::ICompiled_material::get_slot_hash()
enum Material_slot {
    SLOT_THIN_WALLED,                     ///< Slot \c "thin_walled"
    SLOT_SURFACE_SCATTERING,              ///< Slot \c "surface.scattering"
    SLOT_SURFACE_EMISSION_EDF_EMISSION,   ///< Slot \c "surface.emission.emission"
    SLOT_SURFACE_EMISSION_INTENSITY,      ///< Slot \c "surface.emission.intensity"
    SLOT_SURFACE_EMISSION_MODE,           ///< Slot \c "surface.emission.mode"
    SLOT_BACKFACE_SCATTERING,             ///< Slot \c "backface.scattering"
    SLOT_BACKFACE_EMISSION_EDF_EMISSION,  ///< Slot \c "backface.emission.emission"
    SLOT_BACKFACE_EMISSION_INTENSITY,     ///< Slot \c "backface.emission.intensity"
    SLOT_BACKFACE_EMISSION_MODE,          ///< Slot \c "backface.emission.mode"
    SLOT_IOR,                             ///< Slot \c "ior"
    SLOT_VOLUME_SCATTERING,               ///< Slot \c "volume.scattering"
    SLOT_VOLUME_ABSORPTION_COEFFICIENT,   ///< Slot \c "volume.absorption_coefficient"
    SLOT_VOLUME_SCATTERING_COEFFICIENT,   ///< Slot \c "volume.scattering_coefficient"
    SLOT_VOLUME_EMISSION_INTENSITY,       ///< Slot \c "volume.emission_intensity"
    SLOT_GEOMETRY_DISPLACEMENT,           ///< Slot \c "geometry.displacement"
    SLOT_GEOMETRY_CUTOUT_OPACITY,         ///< Slot \c "geometry.cutout_opacity"
    SLOT_GEOMETRY_NORMAL,                 ///< Slot \c "geometry.normal"
    SLOT_HAIR,                            ///< Slot \c "hair"
    SLOT_FIRST = SLOT_THIN_WALLED,        ///< First slot
    SLOT_LAST  = SLOT_HAIR,               ///< Last slot
    SLOT_FORCE_32_BIT = 0xffffffffU
};

mi_static_assert( sizeof( Material_slot) == sizeof( mi::Uint32));

/// The opacity of a compiled material.
///
/// See #mi::neuraylib::ICompiled_material::get_opacity() and
/// #mi::neuraylib::ICompiled_material::get_surface_opacity().
enum Material_opacity {
    /// The material is opaque.
    OPACITY_OPAQUE,
    /// The material is transparent.
    OPACITY_TRANSPARENT,
    /// The opacity of the material is unknown, e.g., because it depends on parameters.
    OPACITY_UNKNOWN,
    OPACITY_FORCE_32_BIT = 0xffffffffU
};

mi_static_assert( sizeof( Material_opacity) == sizeof( mi::Uint32));

/// This interface represents a compiled material.
///
/// A compiled material is a canonical representation of a material instance including all its
/// arguments (constants and call expressions). In this canonical representation, all function
/// calls are (if possible) folded into one expression and common subexpressions are identified
/// (denoted as \em temporaries here).
///
/// Note that there are two modes to create compiled materials: instance compilation and class
/// compilation. In instance compilation mode all arguments of the material instance, i.e., the
/// constants and calls, are folded and the result is an expression without any references to
/// arguments anymore. In class compilation mode only the calls are folded and the result is an
/// expression where the constant arguments of the material instance are represented by symbolic
/// placeholders. The class compilation mode allows to share the compiled representation for
/// materials if they are structurally equivalent (the call structure is similar) and only the value
/// arguments differ.
///
/// The expression that represents the compiled material consists of constant values, results of
/// function calls, indices of temporaries, or indices of arguments. Constant values are represented
/// by expressions of the kind #mi::neuraylib::IExpression_constant. Function calls are represented
/// by expressions of the kind #mi::neuraylib::IExpression_direct_call. References to temporaries
/// are represented by expressions of the kind #mi::neuraylib::IExpression_temporary, whose value is
/// the index into the array of temporaries. References to arguments appear only in case of class
/// compilation. In this case they are represented by expressions of the kind
/// #mi::neuraylib::IExpression_parameter, whose value is the index into the array of arguments.
///
/// \see #mi::neuraylib::IMaterial_instance, #mi::neuraylib::IFunction_call
class ICompiled_material : public
    mi::base::Interface_declare<0x3115ab0f,0x7a91,0x4651,0xa5,0x9a,0xfd,0xb0,0x23,0x16,0xb4,0xb7,
                                neuraylib::IScene_element>
{
public:
    /// \name Material body and temporaries
    //@{

    /// Returns the body (or material root) of the compiled material.
    virtual const IExpression_direct_call* get_body() const = 0;

    /// Returns the number of temporaries.
    virtual Size get_temporary_count() const = 0;

    /// Returns a temporary.
    ///
    /// \param index   The index of the temporary.
    /// \return        The expression of the temporary, or \c NULL if \p index is out of range.
    virtual const IExpression* get_temporary( Size index) const = 0;

    /// Returns the expression of a temporary.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid &)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \tparam T      The interface type of the requested temporary.
    /// \param index   The index of the temporary.
    /// \return        The expression of the temporary, or \c NULL if \p index is out of range.
    template<class T>
    const T* get_temporary( Size index) const
    {
        const IExpression* ptr_iexpression = get_temporary( index);
        if ( !ptr_iexpression)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_iexpression->get_interface( typename T::IID()));
        ptr_iexpression->release();
        return ptr_T;
    }

    /// Looks up a sub-expression of the compiled material.
    ///
    /// \param path    The path from the material root to the expression that should be returned,
    ///                e.g., \c "surface.scattering.tint".
    /// \return        A sub-expression for \p expr according to \p path, or \c NULL in case of
    ///                errors.
    virtual const IExpression* lookup_sub_expression( const char* path) const = 0;

    /// Indicates whether the compiled material is valid.
    ///
    /// A compiled material becomes invalid, if any of the modules it uses definitions from has
    /// has been reloaded.
    ///
    /// \param context     In case of failure, the execution context can be checked for error
    ///                    messages. Can be \c NULL.
    virtual bool is_valid( IMdl_execution_context* context) const = 0;

    //@}
    /// \name Parameters and arguments (class compilation mode only)
    //@{

    /// Returns the number of parameters used by this compiled material.
    ///
    /// Parameters and arguments only exist in class compilation mode. This method always returns 0
    /// in instance compilation mode.
    virtual Size get_parameter_count() const = 0;

    /// Returns the name of a parameter.
    ///
    /// In class compilation mode, the parameters are named according to the path to the
    /// corresponding node in the open material graph that served as basis for the compiled
    /// material. For example, the path \c "a.b.x" refers to a parameter named \c "x" on a node
    /// connected to a parameter named \c "b" on a node connected to the parameter \c "a" of the
    /// material instance that has been compiled.
    ///
    /// Note that these paths here correspond to the open material graph that served as basis for
    /// the compiled material, and not to the structure of the resulting compiled material, as it
    /// is the case for #lookup_sub_expression() or #get_sub_expression_hash().
    ///
    /// \param index   The index of the parameter.
    /// \return        The name of the parameter, or \c NULL if \p index is out of range.
    virtual const char* get_parameter_name( Size index) const = 0;

    /// Returns the value of an argument.
    ///
    /// \param index   The index of the argument.
    /// \return        The value of the argument, or \c NULL if \p index is out of range.
    virtual const IValue* get_argument( Size index) const = 0;

    /// Returns the value of an argument.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid &)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template argument.
    ///
    /// \tparam T      The interface type of the requested element
    /// \param index   The index of the argument.
    /// \return        The value of the argument, or \c NULL if \p index is out of range.
    template<class T>
    const T* get_argument( Size index) const
    {
        const IValue* ptr_ivalue = get_argument( index);
        if ( !ptr_ivalue)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_ivalue->get_interface( typename T::IID()));
        ptr_ivalue->release();
        return ptr_T;
    }

    /// Looks up the DB name of a function call connected to the argument of a compiled material.
    ///
    /// \param material_instance_name   The name of the material instance this compiled material was
    ///                                 compiled from.
    /// \param parameter_index          The index of the parameter for which the DB name of the
    ///                                 connected function call is to be looked up. For example, if
    ///                                 the compiled material has a parameter named \c
    ///                                 "tint.s.texture" the function returns DB name of the
    ///                                 function connected to the tint parameter.
    /// \param errors                   An optional pointer to an #mi::Sint32 to which an error
    ///                                 code will be written. The error codes have the following
    ///                                 meaning:
    ///                                 -  0: Success.
    ///                                 - -1: \p material_instance_name is \c NULL,
    ///                                       or there is no material instance of that name.
    ///                                 - -2: \p parameter_index is out of bounds.
    ///                                 - -3: The corresponding function call could not be found in
    ///                                       the database. This might be due to the fact that the
    ///                                       given parameter is not connected to a function or the
    ///                                       material instance has been changed after the creation
    ///                                       of this compiled material.
    /// \return                         The DB name of the connected function call, or \c NULL in
    ///                                 case of errors.
    virtual const IString* get_connected_function_db_name(
        const char* material_instance_name,
        Size parameter_index,
        Sint32* errors = 0) const = 0;

    //@}
    /// \name Properties of the compiled material
    //@{

    /// Returns the conversion ration between meters and scene units.
    virtual Float32 get_mdl_meters_per_scene_unit() const = 0;

    /// Returns the smallest supported wavelength.
    virtual Float32 get_mdl_wavelength_min() const = 0;

    /// Returns the largest supported wavelength.
    virtual Float32 get_mdl_wavelength_max() const = 0;

    /// Returns the opacity of the compiled material.
    ///
    /// The method returns #OPACITY_TRANSPARENT if the cutout opacity is a constant and less than
    /// 1.0. Otherwise it checks whether a transmissive BSDF is present in the \c surface.scattering
    /// slot.
    ///
    /// See #get_surface_opacity() for a variant ignoring the cutout opacity, and
    /// #get_cutout_opacity() to retrieve the cutout opacity itself.
    virtual Material_opacity get_opacity() const = 0;

    /// Returns the surface opacity of the compiled material.
    ///
    /// The methods checks whether a transmissive BSDF is present in the \c surface.scattering slot.
    ///
    /// See #get_opacity() for a variant taking the cutout opacity into account, and
    /// #get_cutout_opacity() to retrieve the cutout opacity itself.
    virtual Material_opacity get_surface_opacity() const = 0;

    /// Returns the cutout opacity (provided it is a constant).
    ///
    /// \see #get_opacity() and #get_surface_opacity()
    ///
    /// \param[out] cutout_opacity  The cutout opacity value in case of success.
    /// \return                     \c true in case of success, \c false if the value is not a
    ///                             constant, but depends on parameters or complex user expressions.

    virtual bool get_cutout_opacity( Float32* cutout_opacity) const = 0;

    /// Returns the number of scene data attributes referenced by this compiled material.
    virtual Size get_referenced_scene_data_count() const = 0;

    /// Return the name of a scene data attribute referenced by this compiled material.
    ///
    /// \param index   The index of the scene data attribute.
    virtual const char* get_referenced_scene_data_name( Size index) const = 0;

    /// Indicates whether the compiled material depends on coordinate space transformations like
    /// \c %state::transform() and related functions.
    virtual bool depends_on_state_transform() const = 0;

    /// Indicates whether the compiled material depends on \c state::object_id().
    virtual bool depends_on_state_object_id() const = 0;

    /// Indicates whether the compiled material depends on global distribution (edf).
    virtual bool depends_on_global_distribution() const = 0;

    /// Indicates whether the compiled material depends on uniform scene data.
    virtual bool depends_on_uniform_scene_data() const = 0;

    //@}
    /// \name Hash values of the compiled material or parts thereof
    //@{

    /// Returns a hash of the body and all temporaries.
    ///
    /// The hash allows to quickly identify compiled materials that have the same body, temporaries,
    /// and parameter names. Note that the arguments themselves are not included in the hash value.
    ///
    /// \note For performance reasons, the hash for resources does not include the actual resource
    ///       data, but certain properties to identify resources: If the absolute MDL file path is
    ///       available, it is used (including the gamma value and selector for textures). If the
    ///       absolute MDL file path is not available, some internal IDs that identify the resource
    ///       in the database are used instead. \n
    ///       For the latter case, the following applies: If two otherwise identical compiled
    ///       materials share a resource (in the sense of there is one and only one DB element for
    ///       that resource), then their hash is also identical. But if the compiled materials use
    ///       distinct (but otherwise identical) copies of the same DB element, then their IDs are
    ///       different, resulting in different hashes. IDs are also different if a module is
    ///       removed from the database, and later loaded again. IDs might be different if the
    ///       module is loaded in different processes.
    ///
    /// \see #get_slot_hash() for hashes of predefined material slots, and
    ///      #get_sub_expression_hash() for hashes of arbitrary subexpressions
    virtual base::Uuid get_hash() const = 0;

    /// Returns the hash of a particular material slot.
    ///
    /// The hash allows to quickly identify compiled materials where a particular material slot
    /// is identical (corresponding parts of the body and temporaries, and all parameter names).
    /// Note that the arguments themselves are not included in the hash value. See #get_hash()
    /// for details about resources.
    ///
    /// \see #get_hash() for a hash covering all slots in one hash value, and
    ///      #get_sub_expression_hash() for hashes of arbitrary subexpressions
    virtual base::Uuid get_slot_hash( Material_slot slot) const = 0;

    /// Returns the hash of a sub-expression of the compiled material.
    ///
    /// The hash allows to quickly identify compiled materials where a particular sub-expression
    /// is identical (corresponding parts of the body and temporaries, and all parameter names).
    /// Note that the arguments themselves are not included in the hash value. See #get_hash() for
    /// details about resources.
    ///
    /// \note This hash value is computed on-demand, unless the path corresponds to one of the
    ///       predefined material slots, for which the method simply returns the precomputed hash
    ///       value.
    ///
    /// \see #get_hash() for a hash covering all slots in one hash value, and #get_slot_hash()
    ///      for hashes of predefined material slots
    ///
    /// \param path            The path from the material root to the expression that should be
    ///                        hashed, e.g., \c "surface.scattering.tint". An empty path can be
    ///                        used to identify the entire compiled material.
    /// \return                A hash for the sub-expression identified by \p path, or
    ///                        default-constructed in case invalid paths.
    virtual base::Uuid get_sub_expression_hash( const char* path) const = 0;

    //@}
};

/**@}*/ // end group mi_neuray_mdl_elements

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ICOMPILED_MATERIAL_H
