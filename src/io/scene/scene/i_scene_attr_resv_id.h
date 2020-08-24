/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The SCENE specific attribute id definitions.

#ifndef IO_SCENE_SCENE_I_SCENE_ATTR_RESV_ID_H
#define IO_SCENE_SCENE_I_SCENE_ATTR_RESV_ID_H

namespace MI
{
namespace SCENE
{

/// Some attributes are very common and are collected into a fixed bitmap in
/// Attribute_anchor::m_flag_*, with fixed attribute IDs in the range 0..31.
/// Update flagattrname[] and attrname[] in scene.C if changed. Most of these
/// apply to all element types. Although many (most notably, OPT_*) apply to a
/// single type only, I keep them all here in one place to simplify numbering,
/// and because the get/set functions for names and types are in io/scene/scene.
enum Attr_resv_id {
                                //------------------ any element: flags -------
    DISABLE,			///< pretend this node doesn't exist
    VISIBLE,			///< visible to primary (eye) rays
    TRANSPARENCY_CAST,		///< may cast transparency rays
    TRANSPARENCY_RECV,		///< is hit by transparency rays
    REFLECTION_CAST,		///< may cast reflection rays
    REFLECTION_RECV,		///< is hit by reflection rays
    REFRACTION_CAST,		///< may cast refraction rays
    REFRACTION_RECV,		///< is hit by refraction rays
    SHADOW_CAST,		///< may cast raytraced shadows
    SHADOW_RECV,		///< raytraced shadows fall on this object
    FINALGATHER_CAST,		///< is hit by fg rays
    FINALGATHER_RECV,		///< is illuminated by indirect light from fg
    CAUSTIC,			///< is intersected by caustic photons
    CAUSTIC_CAST,		///< emits caustic photons
    CAUSTIC_RECV,		///< is illuminated by caustic photons
    GLOBILLUM,			///< is intersected by globillum photons
    GLOBILLUM_CAST,		///< emits globillum photons
    GLOBILLUM_RECV,		///< is illuminated by globillum photons
    MOVABLE,			///< transform can be modified at runtime
    FACE_FRONT,			///< front face visible?
    FACE_BACK,			///< back face visible?
    HULL,			///< call no shaders, just enter new volume
    MATTE,                      ///< is this a matte object
    LIGHT_PORTAL,               ///< is this light a portal light
    SHADOW_TERMINATOR_OFFSET,   ///< is the shadow terminator smoothing enabled
    MATTE_CONNECT_TO_ENV,       ///< connect matte to environment or backplate instead
    MATTE_CONNECT_FROM_CAM,     ///< connect matte from camera or use real incoming ray
    BACKPLATE_MESH,             ///< is the object a 3D-environment/backplate mesh
    N_FLAGS,			///< number of reserved flags, <= 31
    N_MAXFLAGS		= 32,	///< can't have more than 32 reserved flags

// more reserved attribute IDs, that are not flags. There are no predefined
// fields in Attribute_anchor for these. All attributes not here or in
// Attr_flag_id are user-defined and must be created with Attribute::id_create,
// like shader parameters. Update attrname in scene_object_attr.C if changed.
//
// Attributes OPT_ are legal only on Options and are not inherited.
// Supported filters are box, triangle, gauss cmitchell, clanczos, or fast.
// AO-ambient occlusion, fg=final gathering, gi=globillum photons, ca=caustics,
// vs=volume scattering, no prefix: regular raytracing. When adding more OPT_*
// modes, also change Nscene_opt_attrs in TRAVERSE to read and dump them.
//
// TODO: the all-caps names reflect deprecated names, like SAMPLE_MAX instead
// of SAMPLES. This minimizes changes in this weeks' 2.1 beta. See scene.C;
// deprecated_attributes lists the IDs and reserved_attributes the new names.

                                //------------------ any element --------------
    LABEL		= 32,	///< Uint       arbitrary identifying label
    MATERIAL_ID,                ///< Uint       user-defined material label
    HANDLE_STRING,              ///< string     arbitrary identifying string label
                                //------------------ object -------------------
    MATERIAL,			///< Tag        material or array of materials
    DECALS,			///< Tag        array of decal instances
    ENABLED_DECALS,             ///< Tag        array of enabled decal instances
    DISABLED_DECALS,            ///< Tag        array of disabled decal instances
    PROJECTORS,			///< Tag        array of projector instances
    ACTIVE_PROJECTOR,           ///< Tag        the one active projector instance
    APPROX,			///< Approx     tessellation accuracies
    APPROX_CURVE,		///< Approx     curve approximation accuracies
    MATTE_SHADOW_INTENSITY,     ///< Scalar     matte_shadow_intensity for fake shadows
                                //------------------ object, no inheritance ---
    OBJ_NORMAL,			///< Vector3    vertex normal
    OBJ_MOTION,			///< Vector3[]  vertex motion path
    // note that this is for all primitives EXCEPT quads! For quad primitives
    // there is another attribute carrying the same information, called
    // OBJ_MATERIAL_INDEX_QUAD!
    OBJ_MATERIAL_INDEX,		///< Uint        per-face mtlidx in inst MATERIAL
    OBJ_MATERIAL_INDEX_QUAD,	///< Uint        per-quad-face mtl idx as above
    OBJ_DERIVS,			///< Vector3[2]  surface derivatives
    OBJ_PRIM_LABEL,		///< Uint        user-defined primitive label
                                //------------------ light --------------------
    COLOR,			///< Color       color, usually in the 0..1 range
    ENERGY,			///< Scalar      brightness: emit color * energy
    C_PHOTONS_STORE,		///< Uint        # of caustic photons to store
    C_PHOTONS_EMIT,		///< Uint        # of caustic photons to emit
    GI_PHOTONS_STORE,		///< Uint	       # of glossy/diff photons to store
    GI_PHOTONS_EMIT,		///< Uint	       # of glossy/diff photons to emit
    AREA_SAMPLES,		///< Scalar      max # of samples if area light
    SHMAP_TYPE,			///< Shadowmap_type  Woo, or bias, or detail
    SHMAP_MODE,			///< Option_mapmode  Ondemand,rebuild,freeze,off
    SHMAP_BIAS,			///< Scalar      store dist(light,occlusion)+bias
    SHMAP_FILENAME,		///< char *      load/save shadowmap here
    SHMAP_RESOLUTION_X,		///< Uint        X resolution
    SHMAP_RESOLUTION_Y,		///< Uint        Y resolution
    SHMAP_WINDOW,		///< Uint[4]     crop window: xl, yl, xh, yh
    SHMAP_SAMPLES,		///< Scalar      number of shadowmap samples
    SHMAP_LIGHT_DIAM,		///< Scalar      add distance-dependent blur
    SHMAP_SOFTNESS,		///< Scalar      add fixed blur radius
    SHMAP_FILTER,		///< Scalar      sample filter radius
    SHMAP_MOTION,		///< bool        shadowmap is motion-blurred
                                //------------------ options ------------------
    OPT_BSP_FOLDING,		///< bool        enable or disable bsp folding
    OPT_BSP_FILE_MODE,          ///< uint        MMODE: update rebuild freeze off
    OPT_BSP_FILENAME_PREFIX,    ///< string      Prefix for BSP files
    OPT_METASL_DISABLE,		///< bool        disable all MetaSL shaders
    OPT_METASL_DISABLE_LIGHT,	///< bool        disable MetaSL light shaders
    OPT_METASL_TARGET,		///< uint        0=default,1=cgfx,2=glsl,3=opengl
    OPT_ANIMATION_TIME,		///< dscalar     animation time
    OPT_OCCLUSION_THRESHOLD,	///< scalar      ignore objs showing fewer pixels
    OPT_LIGHT_MAX_COUNT,	///< uint        max n most relevnt lights, 0=inf
    OPT_SAMPLES,		///< scalar      max or fixed # samples/pixel
    OPT_SAMPLES_ADAPT,		///< scalar      factor samples / min_undersample
    OPT_SAMPLES_SHADING,	///< scalar      rasterizer: # shading/pixel
    OPT_SAMPLES_SHADING_ADAPT,	///< scalar      factor samples / min_undersample
    OPT_SAMPLES_MOTION,		///< Uint        rasterizer: # motion samples
    OPT_SAMPLE_SEQUENCE_OFFSET, ///< Uint        number of the first sample instance
    OPT_MODE,			///< uint        MMODE: update off
    OPT_PIXEL_JITTER,		///< Scalar      pixel jittering
    OPT_FILTER,			///< uint        filter type: box triangle etc
    OPT_RADIUS,			///< scalar      filter radius in pixels
    OPT_DEPTH_REFLECT,		///< uint        trace depth for reflection rays
    OPT_DEPTH_REFRACT,		///< uint        trace depth for refraction rays
    OPT_DEPTH_TRANSP,		///< uint        trace depth for transparent rays
    OPT_DEPTH,			///< uint        total trace depth
    OPT_FALLOFF_START,		///< scalar      ray length where fading begins
    OPT_FALLOFF_STOP,		///< scalar      max ray length
    OPT_IMPORTANCE_MIN,		///< scalar      kill rays with importance < min
    OPT_IMPORTANCE_MAX,		///< scalar      fade rays to black if imp. < max
    OPT_AO_MODE,		///< uint        MMODE: update rebuild freeze off
    OPT_AO_SAMPLES,		///< scalar      number of rays per map point
    OPT_AO_SAMPLES_ADAPT,	///< scalar      unused
    OPT_AO_SAMPLES_SHADING,	///< scalar      number of interpolation points
    OPT_AO_SAMPLES_SHADING_ADAPT,///<scalar      unused
    OPT_AO_SMOOTHING,		///< scalar      smoothing factor for interpolation kernel
    OPT_AO_RADIUS,		///< scalar      filter radius in pixels
    OPT_AO_FILENAME,		///< string      file name of the map
    OPT_AO_DEPTH,		///< uint        total trace depth
    OPT_AO_FALLOFF_START,	///< scalar      ray length where fading begins
    OPT_AO_FALLOFF_STOP,	///< scalar      max ray length
    OPT_AO_IMPORTANCE_MIN,	///< scalar      kill rays with importance < min
    OPT_AO_IMPORTANCE_MAX,	///< scalar      fade rays to black if imp. < max
    OPT_AO_GPU_TECHNIQUE,       ///< int         choose a certain gpu rendering technique for ao
    OPT_DL_MODE,		///< uint        MMODE: update rebuild freeze off
    OPT_DL_SAMPLES,		///< scalar      unused
    OPT_DL_SAMPLES_ADAPT,	///< scalar      unused
    OPT_DL_SAMPLES_SHADING,	///< scalar      number of interpolation points
    OPT_DL_SAMPLES_SHADING_ADAPT,///<scalar      unused
    OPT_DL_SMOOTHING,		///< scalar      unused
    OPT_DL_RADIUS,		///< scalar      unused
    OPT_DL_FILENAME,		///< string      file name of the map
    OPT_DL_DEPTH,		///< uint        unused
    OPT_DL_FALLOFF_START,	///< scalar      unused
    OPT_DL_FALLOFF_STOP,	///< scalar      unused
    OPT_DL_IMPORTANCE_MIN,	///< scalar      unused
    OPT_DL_IMPORTANCE_MAX,	///< scalar      unused
    OPT_FG_MODE,		///< uint        MMODE: update rebuild freeze off
    OPT_FG_SAMPLES,		///< scalar      number of rays per map point
    OPT_FG_SAMPLES_ADAPT,	///< scalar      unused
    OPT_FG_SAMPLES_SHADING,	///< scalar      number of interpolation points
    OPT_FG_SAMPLES_SHADING_ADAPT,///<scalar      unused
    OPT_FG_SMOOTHING,		///< scalar      smoothing factor for interpolation kernel
    OPT_FG_RADIUS,		///< scalar      filter radius in pixels
    OPT_FG_FILENAME,		///< string      file name of the map
    OPT_FG_DEPTH,		///< uint        total trace depth
    OPT_FG_FALLOFF_START,	///< scalar      ray length where fading begins
    OPT_FG_FALLOFF_STOP,	///< scalar      max ray length
    OPT_FG_IMPORTANCE_MIN,	///< scalar      kill rays with importance < min
    OPT_FG_IMPORTANCE_MAX,	///< scalar      fade rays to black if imp. < max
    OPT_FG_GPU_TECHNIQUE,       ///< uint        choose gpu technique for fg
    OPT_GLOSSY_MODE,		///< uint        MMODE: update rebuild freeze off
    OPT_GLOSSY_SAMPLES,		///< scalar      number of rays when splitting
    OPT_GLOSSY_SAMPLES_ADAPT,	///< scalar      unused
    OPT_GLOSSY_SAMPLES_SHADING,	///< scalar      number of interpolation points
    OPT_GLOSSY_SAMPLES_SHADING_ADAPT,///<scalar  unused
    OPT_GLOSSY_SMOOTHING,	///< scalar      smoothing factor for interpolation kernel
    OPT_GLOSSY_RADIUS,		///< scalar      filter radius in pixels
    OPT_GLOSSY_FILENAME,	///< string      file name of the map
    OPT_GLOSSY_DEPTH,		///< uint        glossy trace depth, before approximation
    OPT_GLOSSY_FALLOFF_START,	///< scalar      ray length where fading begins
    OPT_GLOSSY_FALLOFF_STOP,	///< scalar      max ray length
    OPT_GLOSSY_IMPORTANCE_MIN,	///< scalar      kill rays with importance < min
    OPT_GLOSSY_IMPORTANCE_MAX,	///< scalar      fade rays to black if imp. < max
    OPT_IBL_MODE,		///< uint        MMODE: update rebuild freeze off
    OPT_IBL_SAMPLES,		///< scalar      max # of lights sampled
    OPT_IBL_SAMPLES_SHADING,	///< scalar      number of interpolation points
    OPT_IBL_SAMPLES_ADAPT,	///< scalar      unused
    OPT_IBL_SAMPLES_SHADING_ADAPT,///<scalar     unused
    OPT_IBL_SMOOTHING,		///< scalar      smoothing factor for interpolation kernel
    OPT_IBL_RADIUS,		///< scalar      filter radius in pixels
    OPT_IBL_FILENAME,	        ///< string      file name of the map
    OPT_IBL_FALLOFF_START,	///< scalar      ray length where fading begins
    OPT_IBL_FALLOFF_STOP,	///< scalar      max ray length
    OPT_IBL_JITTER,		///< bool        sample jittering on/off
    OPT_IBL_IRRADIANCE_LEGACY,	///< bool        include IBL in irradiance()
    OPT_IBL_TEXTURE_RESOLUTION, ///< uint        resolution for env. shader IBL texture baking
    OPT_CA_MODE,		///< uint        MMODE: update rebuild freeze off
    OPT_CA_SAMPLES,		///< scalar      unused
    OPT_CA_SAMPLES_ADAPT,	///< scalar      unused
    OPT_CA_SAMPLES_SHADING,	///< scalar      number of interpolation points
    OPT_CA_SAMPLES_SHADING_ADAPT,///<scalar      unused
    OPT_CA_SMOOTHING,		///< scalar      unused
    OPT_CA_RADIUS,		///< scalar      unused
    OPT_CA_FILENAME,		///< string      file name of the map
    OPT_CA_DEPTH,		///< uint        unused
    OPT_CA_FALLOFF_START,	///< scalar      unused
    OPT_CA_FALLOFF_STOP,	///< scalar      unused
    OPT_CA_IMPORTANCE_MIN,	///< scalar      unused
    OPT_CA_IMPORTANCE_MAX,	///< scalar      unused
    OPT_SECTION_PLANES,         ///< struct[]    section planes
    OPT_MDL_METERS_PER_SCENE_UNIT, ///< scalar   conversion config
    OPT_MDL_DISTILLING_TARGET,  /// < uint
    OPT_FORCE_BUILTIN_BUMP_LINEAR_GAMMA, /// < bool force linear gamma for builtin bumps
                                //------------------ obsolete options ---------
    OPT_SAMPLES_MIN,		///< scalar      raytracer: min # samples/pixel
    OPT_AO_SAMPLES_MIN,		///< scalar      unused
    OPT_DL_SAMPLES_MIN,		///< scalar      unused
    OPT_FG_SAMPLES_MIN,		///< scalar      unused
    OPT_GLOSSY_SAMPLES_MIN,	///< scalar      unused
    OPT_IBL_SAMPLES_MIN,	///< scalar      unused
    OPT_CA_SAMPLES_MIN,		///< scalar      unused
    OPT_GI_MODE,		///< uint        MMODE: update rebuild freeze off
    OPT_GI_SAMPLES,		///< scalar      unused
    OPT_GI_SAMPLES_ADAPT,	///< scalar      unused
    OPT_GI_SAMPLES_SHADING,	///< scalar      number of interpolation points
    OPT_GI_SAMPLES_SHADING_ADAPT, ///< scalar    unused
    OPT_GI_SMOOTHING,		///< scalar      smoothing factor for interpolation kernel
    OPT_GI_RADIUS,		///< scalar      unused
    OPT_GI_FILENAME,		///< string      file name of the map
    OPT_GI_DEPTH,		///< uint        unused
    OPT_GI_FALLOFF_START,	///< scalar      unused
    OPT_GI_FALLOFF_STOP,	///< scalar      unused
    OPT_GI_IMPORTANCE_MIN,	///< scalar      unused
    OPT_GI_IMPORTANCE_MAX,	///< scalar      unused
                                //------------------ object, no inheritance ---
    OBJ_TEXTURE,		///< Scalar[]    first texture space
    OBJ_TEXTURE_NUM	= 256,	///<             number of texture spaces

    OBJ_USER        = OBJ_TEXTURE + OBJ_TEXTURE_NUM,  ///< Scalar[]    first user space
    OBJ_USER_NUM    = 16,                             ///<             number of user spaces
                                //------------------ deprecated ---------------
                                //		fix neuray sources, then remove
    SAMPLE_MIN		= OPT_SAMPLES_MIN,
    SAMPLE_MAX		= OPT_SAMPLES,
    SAMPLE_ADAPT	= OPT_SAMPLES_ADAPT,
    SAMPLE_SHADE	= OPT_SAMPLES_SHADING,
    SAMPLE_SHADE_ADAPT	= OPT_SAMPLES_SHADING_ADAPT,
    SAMPLE_MOTION	= OPT_SAMPLES_MOTION,
                                //------------------ summaries ----------------
    LAST_NONTEXTURE_ID	= OBJ_TEXTURE - 1,		  ///< texture is this+1
    N_IDS		= OBJ_USER + OBJ_USER_NUM,  ///< last reserved ID+1
    N_IDS_SIMPLE	= N_IDS - OBJ_TEXTURE_NUM - OBJ_USER_NUM - N_MAXFLAGS	///< non-flags & non-arrays
};

}
}

#endif
