/***************************************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
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
#include <mi/neuraylib/version.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_mdl_elements
@{
*/

/// Material slots identify parts of a compiled material.
///
/// \see #mi::neuraylib::ICompiled_material and #mi::neuraylib::ICompiled_material::get_slot_hash()
enum Material_slot {
    SLOT_THIN_WALLED,                     ///< Slot thin_walled
    SLOT_SURFACE_SCATTERING,              ///< Slot surface.scattering
    SLOT_SURFACE_EMISSION_EDF_EMISSION,   ///< Slot surface.emission.emission
    SLOT_SURFACE_EMISSION_INTENSITY,      ///< Slot surface.emission.intensity
    SLOT_BACKFACE_SCATTERING,             ///< Slot backface.scattering
    SLOT_BACKFACE_EMISSION_EDF_EMISSION,  ///< Slot backface.emission.emission
    SLOT_BACKFACE_EMISSION_INTENSITY,     ///< Slot backface.emission.intensity
    SLOT_IOR,                             ///< Slot ior
    SLOT_VOLUME_SCATTERING,               ///< Slot volume.scattering
    SLOT_VOLUME_ABSORPTION_COEFFICIENT,   ///< Slot volume.absorption_coefficient
    SLOT_VOLUME_SCATTERING_COEFFICIENT,   ///< Slot volume.scattering_coefficient
    SLOT_GEOMETRY_DISPLACEMENT,           ///< Slot geometry.displacement
    SLOT_GEOMETRY_CUTOUT_OPACITY,         ///< Slot geometry.cutout_opacity
    SLOT_GEOMETRY_NORMAL,                 ///< Slot geometry.normal
    SLOT_FORCE_32_BIT = 0xffffffffU
};

mi_static_assert( sizeof( Material_slot) == sizeof( mi::Uint32));

/// This interface represents a compiled material.
///
/// A compiled material is a canonical representation of a material instance including all its
/// arguments (constants and call expressions). In this canonical representation, all function calls
/// are folded into one expression and common subexpressions are identified (denoted as \em
/// temporaries here).
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
    /// \name Common methods related to instance and class compilation
    //@{

    /// Returns the direct call expression that represents the body of the compiled material.
    virtual const IExpression_direct_call* get_body() const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_8_1
    const IExpression* get_field( const char* name) const
    {
        mi::base::Handle<const IExpression_direct_call> body( get_body());
        mi::base::Handle<const IExpression_list> arguments( body->get_arguments());
        return arguments->get_expression( name);
    }

    template<class T>
    const T* get_field( const char* name) const
    {
        mi::base::Handle<const IExpression_direct_call> body( get_body());
        mi::base::Handle<const IExpression_list> arguments( body->get_arguments());
        return arguments->get_expression<T>( name);
    }
#endif

    /// Returns the number of temporaries used by this compiled material.
    virtual Size get_temporary_count() const = 0;

    /// Returns the expression of a temporary.
    ///
    /// \param index            The index of the temporary.
    /// \return                 The expression of the temporary, or \c NULL if \p index is out of
    ///                         range.
    virtual const IExpression* get_temporary( Size index) const = 0;

    /// Returns the expression of a temporary.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid &)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \tparam T               The interface type of the requested element
    /// \param index            The index of the temporary.
    /// \return                 The expression of the temporary, or \c NULL if \p index is out of
    ///                         range.
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

    /// Returns the conversion ration between meters and scene units for this material.
    virtual Float32 get_mdl_meters_per_scene_unit() const = 0;

    /// Returns the smallest supported wavelength.
    virtual Float32 get_mdl_wavelength_min() const = 0;

    /// Returns the largest supported wavelength.
    virtual Float32 get_mdl_wavelength_max() const = 0;

    /// Indicates whether this material depends on coordinate space transformations like
    /// \c %state::transform() and related functions.
    virtual bool depends_on_state_transform() const = 0;

    /// Indicates whether this material depends on \c state::object_id().
    virtual bool depends_on_state_object_id() const = 0;

    /// Indicates whether this material depends on global distribution (edf).
    virtual bool depends_on_global_distribution() const = 0;

    //@}
    /// \name Additional methods related to class compilation
    //@{

    /// Returns the number of parameters used by this compiled material.
    virtual Size get_parameter_count() const = 0;

    /// Returns the name of a parameter.
    ///
    /// Note that the parameter name is only available if the corresponding parameter of the
    /// original material instance has a constant as argument. If that argument is a call,
    /// \c NULL is returned.
    virtual const char* get_parameter_name( Size index) const = 0;

    /// Returns the value of an argument.
    ///
    /// \param index            The index of the argument.
    /// \return                 The value of the argument, or \c NULL if \p index is out of range.
    virtual const IValue* get_argument( Size index) const = 0;

    /// Returns the value of an argument.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid &)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template argument.
    ///
    /// \tparam T               The interface type of the requested element
    /// \param index            The index of the argument.
    /// \return                 The value of the argument, or \c NULL if \p index is out of range.
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

    /// Returns a hash of the body and all temporaries.
    ///
    /// The hash allows to quickly identify compiled materials that have the same body and
    /// temporaries. Note that the arguments are not included in the hash value.
    ///
    /// \note For performance reasons, the hash for resources does not include the actual resource
    ///       data, but certain properties to identify resources: If the absolute MDL file path is
    ///       available, it is used (including the gamma value for textures). If the absolute MDL
    ///       file path is not available, some internal IDs that identify the resource in the
    ///       database are used instead. \n
    ///       For the latter case, the following applies: If two otherwise identical materials share
    ///       a resource (in the sense of there is one and only one DB element for that resource),
    ///       then their hash is also identical. But if the materials use distinct (but otherwise
    ///       identical) copies of the same DB element, then their IDs are different, resulting in
    ///       different hashes. IDs are also different if a module is removed from the database, and
    ///       later loaded again. IDs might be different if the module is loaded in different
    ///       processes.
    ///
    /// \see #get_slot_hash() for hashes for individual material slots
    virtual base::Uuid get_hash() const = 0;

    /// Returns the hash of a particular material slot.
    ///
    /// The slots hashes allow to quickly compare slots of compiled materials. Note that the
    /// arguments are not included in the hash value.
    ///
    /// \note For performance reasons, the hash for resources does not include the actual resource
    ///       data, but certain properties to identify resources: If the absolute MDL file path is
    ///       available, it is used (including the gamma value for textures). If the absolute MDL
    ///       file path is not available, some internal IDs that identify the resource in the
    ///       database are used instead. \n
    ///       For the latter case, the following applies: If two otherwise identical materials share
    ///       a resource (in the sense of there is one and only one DB element for that resource),
    ///       then their hash is also identical. But if the materials use distinct (but otherwise
    ///       identical) copies of the same DB element, then their IDs are different, resulting in
    ///       different hashes. IDs are also different if a module is removed from the database, and
    ///       later loaded again. IDs might be different if the module is loaded in different
    ///       processes.
    ///
    /// \see #get_hash() for a hash covering all slots together
    virtual base::Uuid get_slot_hash( Material_slot slot) const = 0;

    /// Looks up a sub-expression of the compiled material.
    ///
    /// \param path            The path from the material root to the expression that should be
    ///                        returned, e.g., \c "surface.scattering.tint".
    /// \return                A sub-expression for \p expr according to \p path, or \c NULL in case
    ///                        of errors.
    virtual const IExpression* lookup_sub_expression( const char* path) const = 0;

    //@}
};

/*@}*/ // end group mi_neuray_mdl_elements

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ICOMPILED_MATERIAL_H
