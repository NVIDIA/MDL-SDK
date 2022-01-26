/***************************************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Scene element Material_instance

#ifndef MI_NEURAYLIB_IMATERIAL_INSTANCE_H
#define MI_NEURAYLIB_IMATERIAL_INSTANCE_H

#include <mi/neuraylib/iexpression.h>
#include <mi/neuraylib/ifunction_call.h>
#include <mi/neuraylib/iscene_element.h>
#include <mi/neuraylib/imdl_factory.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_mdl_elements
@{
*/

class ICompiled_material;
class IMdl_execution_context;

/// This interface represents a material instance.
///
/// This interface is almost obsolete. See \ref mi_mdl_materials_are_functions and
/// #mi::neuraylib::IFunction_call. The only remaining purpose is the method
/// #create_compiled_material() and the associated enum #Compilation_options.
class IMaterial_instance : public
    mi::base::Interface_declare<0x037ec156,0x281d,0x466a,0xa1,0x56,0x3e,0xd6,0x83,0xe9,0x5a,0x00,
                                neuraylib::IScene_element>
{
public:
    /// Various options for the creation of compiled materials.
    ///
    /// \see #create_compiled_material()
    enum Compilation_options {
        DEFAULT_OPTIONS   = 0, ///< Default compilation options (e.g., instance compilation).
        CLASS_COMPILATION = 1, ///< Selects class compilation instead of instance compilation.
        COMPILATION_OPTIONS_FORCE_32_BIT = 0xffffffffU // Undocumented, for alignment only
    };

    mi_static_assert( sizeof( Compilation_options) == sizeof( mi::Uint32));

    /// Creates a compiled material.
    ///
    /// \param flags          A bitmask of flags of type #Compilation_options.
    /// \param[inout] context An optional pointer to an execution context which can be used to pass
    ///                       compilation options to the MDL compiler. The following options are
    ///                       supported for this operation:
    ///                       - #mi::Float32 "meters_per_scene_unit": The conversion ratio between
    ///                         meters and scene units for this material. Default: 1.0f.
    ///                       - #mi::Float32 "wavelength_min": The smallest supported wavelength.
    ///                         Default: 380.0f.
    ///                       - #mi::Float32 "wavelength_max": The largest supported wavelength.
    ///                         Default: 780.0f.
    ///                       .
    ///                       The following options are supported in class compilation mode:
    ///                       - \c bool "fold_ternary_on_df": Fold all ternary operators of *df
    ///                         types. Default: \c false.
    ///                       - \c bool "fold_all_bool_parameters": Fold all bool parameters.
    ///                         Default: \c false.
    ///                       - \c bool "fold_all_enum_parameters": Fold all enum parameters.
    ///                         Default: \c false.
    ///                       - #mi::base::IInterface *"fold_parameters": A static or dynamic array
    ///                         of strings of the parameters to fold. The names of the parameters
    ///                         are those that would otherwise be reported in
    ///                         #mi::neuraylib::ICompiled_material::get_parameter_name().
    ///                         Default: \c NULL
    ///                       - \c bool "fold_trivial_cutout_opacity": Fold the expression for
    ///                         geometry.cutout_opacity if it evaluates to a constant with value
    ///                         0.0f or 1.0f. Default: \c false.
    ///                       - \c bool "fold_transparent_layers": Calls to the functions
    ///                         \c df::weighted_layer(), \c df::fresnel_layer(),
    ///                         \c df::custom_curve_layer(), \c df::measured_curve_layer(), and
    ///                         their equivalents with color weights, are replaced by their
    ///                         \c base argument, if the \c weight argument evaluates to a constant
    ///                         with value 0.0f, and the \c layer argument is one of
    ///                         \c df::diffuse_transmission_bsdf(), \c df::specular_bsdf(),
    ///                         \c df::simple_glossy_bsdf(), or \c df::microfacet_*_bsdf(), and
    ///                         the \c scatter_mode argument (if present) is either
    ///                         \c df::scatter_transmit or \c df::scatter_reflect_transmit.
    ///                         In addition, the \c layer argument might be a combination of such
    ///                         BSDFs using the ternary operator.
    ///                       .
    ///                       During material compilation, messages like errors and warnings will
    ///                       be passed to the context for later evaluation by the caller. Possible
    ///                       error conditions:
    ///                       - Type mismatch, call of an unsuitable DB element, or call cycle in
    ///                         the graph of this material instance.
    ///                       - The thin-walled material instance has different transmission for
    ///                         surface and backface.
    ///                       - An argument type of the graph of this material instance is varying
    ///                         but the corresponding parameter type is uniform.
    ///                       - An element in the array for the context option
    ///                         "fold_parameters" does not have the type #mi::IString.
    /// \return               The corresponding compiled material, or \c NULL in case of failure.
    virtual ICompiled_material* create_compiled_material(
        Uint32 flags,
        IMdl_execution_context* context = 0) const = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::get_function_definition().
    virtual const char* get_material_definition() const = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::get_mdl_function_definition().
    virtual const char* get_mdl_material_definition() const = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::get_return_type().
    virtual const IType* get_return_type() const = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::get_parameter_count().
    virtual Size get_parameter_count() const = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::get_parameter_name().
    virtual const char* get_parameter_name( Size index) const = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::get_parameter_index().
    virtual Size get_parameter_index( const char* name) const = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::get_parameter_types().
    virtual const IType_list* get_parameter_types() const = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::get_arguments().
    virtual const IExpression_list* get_arguments() const = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::set_arguments().
    virtual Sint32 set_arguments( const IExpression_list* arguments) = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::set_argument(Size,const IExpression*).
    virtual Sint32 set_argument( Size index, const IExpression* argument) = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::set_argument(const char*,const IExpression*).
    virtual Sint32 set_argument( const char* name, const IExpression* argument) = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::reset_argument(Size).
    virtual Sint32 reset_argument( Size index) = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::reset_argument(const char*).
    virtual Sint32 reset_argument( const char* name) = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::is_default().
    virtual bool is_default() const = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::is_valid().
    virtual bool is_valid( IMdl_execution_context* context) const = 0;

    /// See \ref mi_mdl_materials_are_functions and
    /// #mi::neuraylib::IFunction_call::repair().
    virtual Sint32 repair(
        Uint32 flags,
        IMdl_execution_context* context) = 0;
};

/*@}*/ // end group mi_neuray_mdl_elements

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMATERIAL_INSTANCE_H
