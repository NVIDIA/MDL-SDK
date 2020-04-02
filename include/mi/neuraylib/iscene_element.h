/***************************************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Base class for all scene elements

#ifndef MI_NEURAYLIB_ISCENE_ELEMENT_H
#define MI_NEURAYLIB_ISCENE_ELEMENT_H

#include <mi/neuraylib/iattribute_set.h>
#include <mi/neuraylib/version.h>

namespace mi {

namespace neuraylib {

/** \defgroup mi_neuray_scene_element Scene elements
    \ingroup mi_neuray

    All the scene elements that make up the scene graph.
*/

/** \addtogroup mi_neuray_scene_element
@{
*/

/// Distinguishes scene elements.
///
/// \see #mi::neuraylib::IScene_element::get_element_type()
enum Element_type
{
#ifndef MI_SKIP_WITH_MDL_SDK_DOXYGEN
    ELEMENT_TYPE_INSTANCE                        =  0, ///< #mi::neuraylib::IInstance
    ELEMENT_TYPE_GROUP                           =  1, ///< #mi::neuraylib::IGroup
    ELEMENT_TYPE_OPTIONS                         =  2, ///< #mi::neuraylib::IOptions
    ELEMENT_TYPE_CAMERA                          =  3, ///< #mi::neuraylib::ICamera
    ELEMENT_TYPE_LIGHT                           =  4, ///< #mi::neuraylib::ILight
#endif // MI_SKIP_WITH_MDL_SDK_DOXYGEN
    ELEMENT_TYPE_LIGHTPROFILE                    =  5, ///< #mi::neuraylib::ILightprofile
    ELEMENT_TYPE_TEXTURE                         =  7, ///< #mi::neuraylib::ITexture
    ELEMENT_TYPE_IMAGE                           =  8, ///< #mi::neuraylib::IImage
#ifndef MI_SKIP_WITH_MDL_SDK_DOXYGEN
    ELEMENT_TYPE_TRIANGLE_MESH                   = 10, ///< #mi::neuraylib::ITriangle_mesh
    ELEMENT_TYPE_ATTRIBUTE_CONTAINER             = 16, ///< #mi::neuraylib::IAttribute_container
    ELEMENT_TYPE_POLYGON_MESH                    = 18, ///< #mi::neuraylib::IPolygon_mesh
    ELEMENT_TYPE_SUBDIVISION_SURFACE             = 23, ///< #mi::neuraylib::ISubdivision_surface
    ELEMENT_TYPE_FREEFORM_SURFACE                = 24, ///< #mi::neuraylib::IFreeform_surface
    ELEMENT_TYPE_FIBERS                          = 25, ///< #mi::neuraylib::IFibers
#endif // MI_SKIP_WITH_MDL_SDK_DOXYGEN
    ELEMENT_TYPE_MODULE                          = 29, ///< #mi::neuraylib::IModule
    ELEMENT_TYPE_FUNCTION_DEFINITION             = 30, ///< #mi::neuraylib::IFunction_definition
    ELEMENT_TYPE_FUNCTION_CALL                   = 31, ///< #mi::neuraylib::IFunction_call
    ELEMENT_TYPE_MATERIAL_DEFINITION             = 32, ///< #mi::neuraylib::IMaterial_definition
    ELEMENT_TYPE_MATERIAL_INSTANCE               = 33, ///< #mi::neuraylib::IMaterial_instance
    ELEMENT_TYPE_COMPILED_MATERIAL               = 34, ///< #mi::neuraylib::ICompiled_material
    ELEMENT_TYPE_BSDF_MEASUREMENT                = 35, ///< #mi::neuraylib::IBsdf_measurement
#ifndef MI_SKIP_WITH_MDL_SDK_DOXYGEN
    ELEMENT_TYPE_IRRADIANCE_PROBES               = 36, ///< #mi::neuraylib::IIrradiance_probes
    ELEMENT_TYPE_DECAL                           = 37, ///< #mi::neuraylib::IDecal
    ELEMENT_TYPE_ON_DEMAND_MESH                  = 38, ///< #mi::neuraylib::IOn_demand_mesh
    ELEMENT_TYPE_PROJECTOR                       = 39, ///< #mi::neuraylib::IProjector
    ELEMENT_TYPE_SECTION_OBJECT                  = 40, ///< #mi::neuraylib::ISection_object
    ELEMENT_TYPE_PROXY                           = 41, ///< #mi::neuraylib::IProxy
#endif // MI_SKIP_WITH_MDL_SDK_DOXYGEN
    ELEMENT_TYPE_FORCE_32_BIT                    = 0xffffffffU
};

mi_static_assert( sizeof( Element_type)== sizeof( Uint32));

/// Common %base interface for all scene elements.
class IScene_element :
    public base::Interface_declare<0x8a2a4da9,0xe323,0x452c,0xb8,0xda,0x92,0x45,0x67,0x85,0xd7,0x78,
                                   neuraylib::IAttribute_set>
{
public:
    /// Indicates the actual scene element represented by interfaces derived from this interface.
    virtual Element_type get_element_type() const = 0;
};

/*@}*/ // end group mi_neuray_scene_element

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ISCENE_ELEMENT_H
