/***************************************************************************************************
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      API component that gives access to the MDL distiller

#ifndef MI_NEURAYLIB_IMDL_DISTILLER_H
#define MI_NEURAYLIB_IMDL_DISTILLER_H

#include <mi/base/interface_declare.h>

namespace mi {

class IData;
class IMap;

namespace neuraylib {

class IBaker;
class ICanvas;
class ICompiled_material;

/** \addtogroup mi_neuray_mdl_misc
@{
*/

/// Identifies the resource(s) to be used by a baker.
///
/// \see #mi::neuraylib::IMdl_distiller_api::create_baker()
enum Baker_resource {

    /// Use only the CPU for texture baking.
    BAKE_ON_CPU,
    /// Use only the GPU for texture baking.
    BAKE_ON_GPU,
    /// Prefer using the GPU for texture baking, use the CPU as fallback.
    BAKE_ON_GPU_WITH_CPU_FALLBACK,
    //  Undocumented, for alignment only.
    BAKER_RESOURCE_FORCE_32_BIT = 0xffffffffU
};

mi_static_assert( sizeof( Baker_resource) == sizeof( Uint32));

/// Provides access to various functionality related to MDL distilling.
class IMdl_distiller_api : public
    mi::base::Interface_declare<0x074709ef,0x11b0,0x4196,0x82,0x1c,0xab,0x64,0x1a,0xa2,0x50,0xdb>
{
public:
    /// Returns the number of targets supported for distilling.
    virtual Size get_target_count() const = 0;

    /// Returns the \c index -th target name supported for distilling, or \c NULL if \p index is out
    /// of bounds.
    virtual const char* get_target_name( Size index) const = 0;

    /// Distills a material.
    ///
    /// Material distilling refers to the translation of an arbitrary input material to a
    /// predefined target model. 
    /// Supported target models are
    /// - diffuse
    /// - specular_glossy
    /// - ue4
    /// - transmissive_pbr
    ///
    /// Depending on the structure of the input material and the 
    /// complexity of the target model the resulting material can be as simple as a single bsdf
    /// or a set of bsdfs combined using layerers and mixes as illustrated in the table below
    /// using a pseudo-mdl notation.
    /// <table>
    ///   <tr>
    ///      <td>diffuse</td>
    ///      <td>surface.scattering = diffuse_reflection_bsdf<br>
    ///          geometry.normal = ()
    ///      </td>
    ///    </tr>
    ///   <tr>
    ///      <td rowspan=3>specular_glossy</td>
    ///      <td>surface.scattering = custom_curve_layer(layer: bsdf_glossy_ggx_vcavities, 
    ///          base: diffuse_reflection_bsdf)<br>
    ///          geometry.normal = ()
    ///      </td>
    ///   </tr>
    ///   <tr>
    ///      <td>
    ///          surface.scattering = bsdf_glossy_ggx_vcavities<br>
    ///          geometry.normal = ()
    ///      </td>
    ///   </tr>
    ///   <tr>
    ///      <td>
    ///          surface.scattering = diffuse_reflection_bsdf<br>
    ///          geometry.normal = ()
    ///      </td>
    ///   </tr>
    ///   <tr>
    ///      <td rowspan=2>ue4</td>
    ///      <td>surface.scattering = custom_curve_layer( // clearcoat<br>
    ///          &nbsp;&nbsp;layer: bsdf_glossy_ggx_vcavities,<br>
    ///          &nbsp;&nbsp;base: weighted_layer(<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;layer: normalized_mix(<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bsdf_glossy_ggx_vcavities,<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;custom_curve_layer(<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;layer: 
    ///          bsdf_glossy_ggx_vcavities,<br> 
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;base: 
    ///          diffuse_reflection_bsdf<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;),<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;),<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;normal: () // under-clearcoat normal<br>
    ///          &nbsp;&nbsp;),<br>
    ///          &nbsp;&nbsp;normal: () // clearcoat normal<br>
    ///          )
    ///      </td>
    ///   </tr>
    ///   <tr>
    ///      <td>
    ///          A weighted_layer of any subset of the above construct 
    ///          with an optional clearcoat on top.
    ///      </td>
    ///   </tr>
    ///   <tr>
    ///      <td rowspan=2>transmissive_pbr</td>
    ///      <td>surface.scattering = custom_curve_layer( // clearcoat<br>
    ///          &nbsp;&nbsp;layer: bsdf_glossy_ggx_vcavities,<br>
    ///          &nbsp;&nbsp;base: weighted_layer(<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;layer: normalized_mix(<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bsdf_glossy_ggx_vcavities,<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;custom_curve_layer(<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;layer: 
    ///          bsdf_glossy_ggx_vcavities,<br> 
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;base: 
    ///          normalized_mix(<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
    ///          bsdf_glossy_ggx_vcavities(scatter_transmit)<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
    ///          diffuse_reflection_bsdf<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;),<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;),<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;),<br>
    ///          &nbsp;&nbsp;&nbsp;&nbsp;normal: () // under-clearcoat normal<br>
    ///          &nbsp;&nbsp;),<br>
    ///          &nbsp;&nbsp;normal: () // clearcoat normal<br>
    ///          )
    ///      </td>
    ///   </tr>
    ///   <tr>
    ///      <td>
    ///          A weighted_layer of any subset of the above construct 
    ///          with an optional clearcoat on top.
    ///      </td>
    ///   </tr>
    /// </table>
    ///
    /// \param material           The material to be distilled.
    /// \param target             The target model. See #get_target_count() and #get_target_name().
    /// \param distiller_options  Options for the distiller. Supported options are:
    ///       - \c "top_layer_weight" of type #mi::IFloat32. This weight is given to the top
    ///         layer if a Fresnel layered BSDF is simplified to a single diffuse BSDF in the 
    ///         'diffuse' distilling target. The base layer uses one minus this weight.
    ///         Default: 0.04.
    ///       - \c "layer_normal" of type #mi::IBoolean. If \c true, it enables the aggregation
    ///         of the local normal maps of BSDF layerers to combine them with the global normal
    ///         map. Default: \c true.
    ///                           
    /// \param errors             An optional pointer to an #mi::Sint32 to which an error code will
    ///                           be written. The error codes have the following meaning:
    ///                           -  0: Success.
    ///                           - -1: Invalid parameters (\c NULL pointer).
    ///                           - -2: Invalid target model.
    ///                           - -3: Unspecified failure.
    /// \return                   The distilled material, or \c NULL in case of failure.
    virtual ICompiled_material* distill_material(
        const ICompiled_material* material,
        const char* target,
        const IMap* distiller_options = 0,
        Sint32* errors = 0) const = 0;

    /// Creates a baker for texture baking.
    ///
    /// \param material           The material of which a subexpression is to be baked.
    /// \param path               The path from the material root to the expression that should be
    ///                           baked, e.g., \c "surface.scattering.tint".
    /// \param resource           The resource to be used for baking.
    /// \param gpu_device_id      The device ID of the GPU to be used for baking (as identified by
    ///                           the CUDA runtime or driver API). Ignored if \p resource is
    ///                           #BAKE_ON_CPU.
    /// \return                   A baker for the expression of the given material identified by \p
    ///                           path, or \c NULL in case of failure. Note that returned baker
    ///                           depends on the transaction that was used to access the material.
    virtual const IBaker* create_baker(
        const ICompiled_material* material,
        const char* path,
        Baker_resource resource = BAKE_ON_CPU,
        Uint32 gpu_device_id = 0) const = 0;

    /// Returns the number of required MDL modules for the given target.
    ///
    /// If a target reports any required modules, the integrating application must query the name
    /// and MDL code for each of them using #get_required_module_name() and
    /// #get_required_module_code() and load them using
    /// #mi::neuraylib::IMdl_impexp_api::load_module_from_string() before distilling any material
    /// to that target, otherwise distilling will fail.
    ///
    /// \param target    The target material model to distill to.
    ///
    /// \return          The number of required MDL modules for that target.
    virtual Size get_required_module_count( const char* target) const = 0;

    /// Returns the MDL name of the required MDL module with the given index for the given target.
    ///
    /// \see #get_required_module_count()
    ///
    /// \param target    The target material model to distill to.
    /// \param index     The index of the required module for the given target.
    ///
    /// \return          The MDL name of the required module.
    virtual const char* get_required_module_name( const char* target, Size index) const = 0;

    /// Returns the MDL source code of the required MDL module with the given index for the given
    /// target.
    ///
    /// \see #get_required_module_count()
    ///
    /// \param target    The target material model to distill to.
    /// \param index     The index of the required module for the given target.
    ///
    /// \return          The MDL source code of the required module.
    virtual const char* get_required_module_code( const char* target, Size index) const = 0;
};

/// Allows to bake a varying or uniform expression of a compiled material into a texture or
/// constant.
class IBaker : public
    mi::base::Interface_declare<0x4dba1b1d,0x8fce,0x43d9,0x80,0xa7,0xa2,0x24,0xf3,0x1e,0xdc,0xe7>
{
public:
    /// Returns the pixel type that matches the expression to be baked best.
    virtual const char* get_pixel_type() const = 0;

    /// Indicates whether the expression to be baked is uniform or varying.
    ///
    /// Typically, varying expressions are baked into textures, and uniform expressions into
    /// constant. However, it is also possible to do it the other way round.
    virtual bool is_uniform() const = 0;

    /// Bakes the expression as texture.
    ///
    /// \param texture   The baked texture will be stored in this canvas. If the pixel type of \p
    ///                  canvas does not match the pixel type of the expression to be baked (as
    ///                  indicated by #get_pixel_type()), then the pixel data is converted as
    ///                  described in #mi::neuraylib::IImage_api::convert().
    /// \param samples   The number of samples (per pixel).
    /// \return
    ///                  -  0: Success.
    ///                  - -1: Invalid parameters (\c NULL pointer).
    ///                  - -2: The transaction that is bound to this baker is no longer open.
    ///                  - -3: The execution of the MDL code failed.
    virtual Sint32 bake_texture( ICanvas* texture, Uint32 samples = 1) const = 0;

    /// Bakes the expression as texture.
    ///
    /// \param texture   The baked texture will be stored in this canvas. If the pixel type of \p
    ///                  canvas does not match the pixel type of the expression to be baked (as
    ///                  indicated by #get_pixel_type()), then the pixel data is converted as
    ///                  described in #mi::neuraylib::IImage_api::convert().
    /// \param min_u     The lower value of the U range used for baking.
    /// \param max_u     The higher value of the U range used for baking.
    /// \param min_v     The lower value of the V range used for baking.
    /// \param max_v     The higher value of the V range used for baking.
    /// \param samples   The number of samples (per pixel).
    /// \return
    ///                  -  0: Success.
    ///                  - -1: Invalid parameters (\c NULL pointer).
    ///                  - -2: The transaction that is bound to this baker is no longer open.
    ///                  - -3: The execution of the MDL code failed.
    virtual Sint32 bake_texture(
        ICanvas* texture,
        Float32 min_u, Float32 max_u, Float32 min_v, Float32 max_v,
        Uint32 samples = 1) const = 0;

    /// Bakes the expression as constant.
    ///
    /// \param constant  An instance of #mi::IData of suitable type such that the baked constant can
    ///                  be stored in this argument. For pixel types \c "Float32" and \c
    ///                  "Float32<3>" the type name of this argument needs to match the pixel type.
    ///                  For pixel type \c "Rgb_fp" this argument needs to have the type name \c
    ///                  "Color".
    /// \param samples   The (total) number of samples.
    /// \return
    ///                  -  0: Success.
    ///                  - -1: Invalid parameters (\c NULL pointer).
    ///                  - -2: The transaction that is bound to this baker is no longer open.
    ///                  - -3: The execution of the MDL code failed.
    ///                  - -4: The type of \p constant does not match the pixel type corresponding
    ///                         to the expression to be baked.
    virtual Sint32 bake_constant( IData* constant, Uint32 samples = 1) const = 0;
};

/**@}*/ // end group mi_neuray_mdl_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_DISTILLER_H
