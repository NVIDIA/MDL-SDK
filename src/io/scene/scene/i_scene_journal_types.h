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
/// \brief The SCENE specific Journal_type definitions.

#ifndef IO_SCENE_SCENE_I_SCENE_JOURNAL_TYPES_H
#define IO_SCENE_SCENE_I_SCENE_JOURNAL_TYPES_H

#include <base/data/db/i_db_journal_type.h>

namespace MI
{
namespace SCENE
{

// The journal flags allow to accelerate scene preprocessing. It is important to set the correct
// journal flags, otherwise scene updates might not be taken into account by scene preprocessing.
//
// The flags are to be set on the following changes. The list of examples is meant to be
// exhaustive, but it is not authoritative and might become outdated.
//
// - JOURNAL_CHANGE_TOPOLOGY: all topology changes (changes to the DAG structure), e.g., the item
//   of an Instance, the members of a Group, the members and the function of a Select, the
//   reference and bounding box of an Assembly. Note that the flag
//   has to be set on the referencing element, not on the referenced element. The flag has to be
//   set when the reference itself changes, not when the referenced element changes. There is a
//   special flag for changing a shader.
//
// - JOURNAL_CHANGE_SHADER: all shader changes, i.e. adding, replacing, or removing a Shader
//   to/of/from Camera, Light, Material, Material_decl, Function_decl. The flag has also to be
//   set if the "material" attribute (1) changes.
//   Note that the flag has to be set on the referencing element, not on the referenced element.
//   The flag has to be set when the reference itself changes, not when the referenced element
//   changes.
//
// - JOURNAL_CHANGE_GEOMETRY: all geometry changes in objects, e.g., all mesh modifications
//   including normal and other attributes, the "approx" attribute (1), displacement shader changes,
//   the bound for the maximum displacement.
//
// - JOURNAL_CHANGE_TRANSFORM: changes of the world-to-object transformation matrix in an Instance
//
// - JOURNAL_CHANGE_SHADOW: some light changes (type, spread, distance) (2), some attributes on
//   Options (1)
//
// - JOURNAL_CHANGE_NON_SHADER_ATTRIBUTE: all changes of predefined attributes (1)
// - JOURNAL_CHANGE_VISIBLE_FLAG, JOURNAL_CHANGE_DISABLE_FLAG: needed by generate_change_lists
//   (renderapi), is always set together with JOURNAL_CHANGE_NON_SHADER_ATTRIBUTE because
//    scene-traversal only supports the latter. TODO: add support in scene-traversal. (1)
//
// - JOURNAL_CHANGE_SHADER_ATTRIBUTE: some light changes (type, spread, distance) (2), all
//   attribute changes of Function, Material, Material_decl, Function_decl (3), a connection
//   change on Function, the file referenced in an Image or in a Lightprofile,
//   the image referenced in a Texture.
//   For the last case: as for JOURNAL_CHANGE_TOPOLOGY, the flag
//   has to be set when the reference itself changes, not when the referenced element changes.
//
// - JOURNAL_CHANGE_LIGHT_FIELD: all light class members.
// - JOURNAL_CHANGE_LIGHT_GEOMETRY: special light class members which change the light geometry,
//   ie area_shader, area_size_x/y and type.
//
// - JOURNAL_CHANGE_FIELD: all changes to class members that are not inherited and have no impact
//   on the scene data structure, e.g., Camera::set_focal().
//
// - JOURNAL_CHANGE_NOTHING: No bits set, it should not be necessary to use this flag.
//
// - JOURNAL_CHANGE_UNKNOWN: All bits set, it should not be necessary to use this flag.
//
// - JOURNAL_CHANGE_DICE: all changes to DiCE DB elements
//
// - JOURNAL_CHANGE_DECAL_FIELD: all changes to the decal fields (besides shader changes)
//
// - JOURNAL_CHANGE_DECAL: all changes of decal related attributes, ie addition/removal/replacement
//
// - JOURNAL_CHANGE_PROJECTOR_FIELD: all changes to the projector fields (besides shader changes)
//
// - JOURNAL_CHANGE_PROJECTOR: all changes of projector related attributes, ie
//   addition/removal/replacement
//
// (1) The neuray API uses ATTR::Attribute_spec::get_journal_flags() to retrieve the journal flag(s)
//     to be set. It is the responsibility of io/scene/scene/scene.cpp to define these flags.
//
// (2) Only these three properties of Light are part of the shader state.
//
// (3) For simplicity the neuray API sets that flag on all attribute changes without checking
//     whether the attribute actually is a shader attribute.
//
// The neuray API implementation in api/api/neuray is supposed to adhere to these rules.
//
static const DB::Journal_type
    JOURNAL_CHANGE_UNKNOWN          = DB::Journal_type(~0),        // all bits on
    JOURNAL_CHANGE_NOTHING          = DB::Journal_type(0x0),       // no bit set
    JOURNAL_CHANGE_FIELD            = DB::Journal_type(0x01),      // bit 0 set
    JOURNAL_CHANGE_NON_SHADER_ATTRIBUTE	= DB::Journal_type(0x02),  // bit 1 set
    JOURNAL_CHANGE_SHADER_ATTRIBUTE = DB::Journal_type(0x04),      // bit 2 set
    JOURNAL_CHANGE_TRANSFORM        = DB::Journal_type(0x08),      // bit 3 set
    JOURNAL_CHANGE_SHADOW           = DB::Journal_type(0x10),      // bit 4 set
    JOURNAL_CHANGE_GEOMETRY         = DB::Journal_type(0x20),      // bit 5 set
    JOURNAL_CHANGE_TOPOLOGY         = DB::Journal_type(0x40),      // bit 6 set
    JOURNAL_CHANGE_SHADER           = DB::Journal_type(0x80),      // bit 7 set
    JOURNAL_CHANGE_LIGHT_FIELD      = DB::Journal_type(0x100),     // bit 8 set
    JOURNAL_CHANGE_LIGHT_GEOMETRY   = DB::Journal_type(0x200),     // bit 9 set
    JOURNAL_CHANGE_DISABLE_FLAG     = DB::Journal_type(0x400),     // bit 10 set
    JOURNAL_CHANGE_VISIBLE_FLAG     = DB::Journal_type(0x800),     // bit 11 set
    JOURNAL_CHANGE_DECAL_FIELD      = DB::Journal_type(0x1000),    // bit 12 set
    JOURNAL_CHANGE_DECAL            = DB::Journal_type(0x2000),    // bit 13 set
    JOURNAL_CHANGE_PROJECTOR_FIELD  = DB::Journal_type(0x4000),    // bit 14 set
    JOURNAL_CHANGE_PROJECTOR        = DB::Journal_type(0x8000),    // bit 15 set
    JOURNAL_CHANGE_DICE             = DB::Journal_type(0x8000000); // bit 31 set
}
}

#endif
