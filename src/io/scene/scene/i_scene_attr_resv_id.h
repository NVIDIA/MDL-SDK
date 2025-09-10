/***************************************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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
    DISABLE,                    ///< pretend this node doesn't exist
    VISIBLE,                    ///< visible to primary (eye) rays
    SHADOW_CAST,                ///< may cast raytraced shadows
    SHADOW_RECV,                ///< raytraced shadows fall on this object
    MOVABLE,                    ///< transform can be modified at runtime
    MATTE,                      ///< is this a matte object
    SHADOW_TERMINATOR_OFFSET,   ///< is the shadow terminator smoothing enabled
    MATTE_CONNECT_TO_ENV,       ///< connect matte to environment or backplate instead
    MATTE_CONNECT_FROM_CAM,     ///< connect matte from camera or use real incoming ray
    BACKPLATE_MESH,             ///< is the object a 3D-environment/backplate mesh
    NOT_PICKABLE,               ///< "invisible" to pick rays
    SELECTED,                   ///< include in rendering of selection subsets
    CRACKFREE_DISPLACEMENT,     ///< enables crackfree displacement
    N_FLAGS,                    ///< number of reserved flags, <= 31
    N_MAXFLAGS          = 32,   ///< can't have more than 32 reserved flags

// more reserved attribute IDs, that are not flags. There are no predefined
// fields in Attribute_anchor for these. All attributes not here or in
// Attr_flag_id are user-defined and must be created with Attribute::id_create,
// like shader parameters. Update attrname in scene_object_attr.C if changed.
//
// Attributes OPT_ are legal only on Options and are not inherited.
// Supported filters are box, triangle, gauss cmitchell, clanczos, or fast.
// When adding more OPT_*
// modes, also change Nscene_opt_attrs in TRAVERSE to read and dump them.
//
                                //------------------ any element --------------
    LABEL  = N_MAXFLAGS,        ///< Uint       arbitrary identifying label
    MATERIAL_ID,                ///< Uint       user-defined material label
    HANDLE_STRING,              ///< string     arbitrary identifying string label
                                //------------------ object -------------------
    MATERIAL,                   ///< Tag        material or array of materials
    DECALS,                     ///< Tag        array of decal instances
    ENABLED_DECALS,             ///< Tag        array of enabled decal instances
    DISABLED_DECALS,            ///< Tag        array of disabled decal instances
    PROJECTORS,                 ///< Tag        array of projector instances
    ACTIVE_PROJECTOR,           ///< Tag        the one active projector instance
    APPROX,                     ///< Approx     tessellation accuracies
    APPROX_CURVE,               ///< Approx     curve approximation accuracies
    MATTE_SHADOW_INTENSITY,     ///< Scalar     matte_shadow_intensity for fake shadows
    VOLUME_PRIORITY,            ///< Sint8      volume stack priority    
    BACKPLATE_MESH_FUNCTION,    ///< Tag        backplate mesh function
    GHOSTLIGHT_FACTOR,          ///< Scalar     factor to lower light visibility in glossy reflections
    APPROX_TRIANGLE_LIMIT,      ///< Uint       triangle limit for tessellation
    APPROX_VERTEX_OFFSET,       ///< Scalar     factor by which vertices are offset according to the shading normal
    CRACKFREE_DISPLACEMENT_POINT_TOLERANCE,
                                ///< Scalar     tolerance when finding similar/duplicate points
                                //------------------ object, no inheritance ---
    OBJ_NORMAL,                 ///< Vector3    vertex normal
    OBJ_MOTION,                 ///< Vector3[]  vertex motion path
    // note that this is for all primitives EXCEPT quads! For quad primitives
    // there is another attribute carrying the same information, called
    // OBJ_MATERIAL_INDEX_QUAD!
    OBJ_MATERIAL_INDEX,         ///< Uint        per-face mtlidx in inst MATERIAL
    OBJ_MATERIAL_INDEX_QUAD,    ///< Uint        per-quad-face mtl idx as above
    OBJ_DERIVS,                 ///< Vector3[2]  surface derivatives
    OBJ_PRIM_LABEL,             ///< Uint        user-defined primitive label
                                //------------------ material --------------------
    EXCLUDE_FROM_WHITE_MODE,    /// < bool exclude from white mode
                                //------------------ options --------------------
    OPT_ANIMATION_TIME,         ///< dscalar     animation time
    OPT_FILTER,                 ///< uint        filter type: box triangle etc
    OPT_RADIUS,                 ///< scalar      filter radius in pixels
    OPT_SECTION_PLANES,         ///< struct[]    section planes
    OPT_MDL_METERS_PER_SCENE_UNIT, ///< scalar   conversion config
    OPT_MDL_DISTILLING_TARGET,  /// < string
    OPT_FORCE_BUILTIN_BUMP_LINEAR_GAMMA, /// < bool force linear gamma for builtin bumps
    OPT_IRAY_SPECTRAL_OBSERVER_CUSTOM_CURVE, /// < Vector3[]  photometric spectral color response curve
    OPT_IGNORE_MAX_DISPLACE,    ///< bool        ignore max displacement setting on objects
    OPT_DISPLACE_ON_GPU,        ///< bool        enables gpu displacement
                                //------------------ object, no inheritance ---------
    OBJ_TEXTURE,                ///< Scalar[]    first texture space
    OBJ_TEXTURE_NUM     = 256,  ///<             number of texture spaces

    OBJ_USER        = OBJ_TEXTURE + OBJ_TEXTURE_NUM,  ///< Scalar[]    first user space
    OBJ_USER_NUM    = 16,                             ///<             number of user spaces
                                //------------------ summaries ----------------
    LAST_NONTEXTURE_ID  = OBJ_TEXTURE - 1,                ///< texture is this+1
    N_IDS               = OBJ_USER + OBJ_USER_NUM,  ///< last reserved ID+1
    N_IDS_SIMPLE        = N_IDS - OBJ_TEXTURE_NUM - OBJ_USER_NUM - N_MAXFLAGS   ///< non-flags & non-arrays
};

}
}

#endif
