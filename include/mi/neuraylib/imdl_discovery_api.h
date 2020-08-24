/***************************************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief  API component to discover MDL content in archives and file systems.

#ifndef MI_NEURAYLIB_IMDL_DISCOVERY_API_H 
#define MI_NEURAYLIB_IMDL_DISCOVERY_API_H 

#include <mi/base/interface_declare.h>

namespace mi {

class IString;

namespace neuraylib {


/// Abstract interface for a discovery graph node.
/// It provides the common functionality of all different discovery graph node interfaces.
class IMdl_info : public 
    base::Interface_declare<0xd2f50312,0xe76c,0x4d64,0xa5,0x91,0xcb,0x70,0x38,0x2c,0xa9,0x9f>
{
    public:

        /// The kinds of the graph nodes.
        enum Kind {  
            DK_PACKAGE = 0,
            DK_MODULE = 1,
            DK_XLIFF = 2,
            DK_TEXTURE = 4,
            DK_LIGHTPROFILE = 8,
            DK_MEASURED_BSDF = 16,
            DK_DIRECTORY = 32,
            // next = 64
            DK_ALL = 0xffffffffU
        };

        /// Returns the kind of the graph node.
        virtual Kind                    get_kind() const = 0;

        /// Returns the qualified name of the graph node. 
        virtual const char*             get_qualified_name() const = 0;

        /// Returns the simple_name (tail of the qualified_name) of the graph node.
        virtual const char*             get_simple_name() const = 0;   
};
mi_static_assert(sizeof(IMdl_info::Kind) == sizeof(Uint32));

/// Interface for a graph node representing an MDL module.
class IMdl_module_info : public
    base::Interface_declare<0x22,0x1204,0x46,0xb1,0x5b,0xbf,0xa8,0x11,0xc7,0xe7,0xe1,IMdl_info>
        
{
    public:

        /// Returns the index of the search path where this module has been found.
        virtual Size                    get_search_path_index() const = 0;

        /// Returns the search path in the local file system that contains this module.
        virtual const char*             get_search_path() const = 0;

        /// Returns an absolute path to the module file in the local file system.
        virtual const IString*          get_resolved_path() const = 0;

        /// Returns the number of shadows of this module.
        virtual Size                    get_shadows_count() const = 0; 

        /// Returns one of the shadows this module has.
        /// \param index     Index in the shadow list of this module.
        virtual const IMdl_module_info* get_shadow(Size index) const = 0;

        /// Returns true if the module has been discovered inside of an archive, false if not.
        virtual bool                    in_archive() const = 0;
};

/// Interface for xliff files.
class IMdl_xliff_info : public
    base::Interface_declare<0x3a, 0xdf, 0x8a, 0x42, 0x94, 0x09, 0x59, 0xe6, 0xf9, 0x25, 0x67, IMdl_info>
{
public:

    /// Returns an absolute path to the xliff file in the local file system.
    virtual const char*          get_resolved_path() const = 0;

    /// Returns the index of the search path where the xliff file has been found.
    virtual Size                 get_search_path_index() const = 0;

    /// Returns the search path in the local file system that contains this xliff file.
    virtual const char*          get_search_path() const = 0;

    /// Returns true if the xliff file has been discovered inside of an archive, false if not.
    virtual bool                 in_archive() const = 0;

};

/// Interface for resources.
class IMdl_resource_info : public
    base::Interface_declare<0x28,0x54,0x6b,0xcd,0x14,0x59,0xfd,0x42,0x9d,0xfa,0xd5, IMdl_info>
{
public:

    /// Returns the index of the search path where this resource has been found.
    virtual Size                    get_search_path_index() const = 0;

    /// Returns the search path in the local file system that contains this resource.
    virtual const char*             get_search_path() const = 0;

    /// Returns the file extension of the resource file.
    virtual const char*             get_extension() const = 0;

    /// Returns an absolute path to the resource file in the local file system.
    virtual const char*             get_resolved_path() const = 0;

    /// Returns the number of shadows of this resource.
    virtual Size                    get_shadows_count() const = 0;

    /// Returns one of the shadows this resource has.
    /// \param index     Index in the shadow list of this resource.
    virtual const IMdl_resource_info* get_shadow(Size index) const = 0;

    /// Returns true if the resource has been discovered inside of an archive, false if not.
    virtual bool                    in_archive() const = 0;

};


/// Interface for texture files.
class IMdl_texture_info : public
    base::Interface_declare<0x62,0x71,0xac,0x50,0xde,0xa9,0x40,0x92,0x8b,0xf0,0x1d,IMdl_resource_info>
{

};

/// Interface for lightprofile files.
class IMdl_lightprofile_info : public
    base::Interface_declare<0x17,0x23,0x61,0xf4,0xcb,0x64,0xb3,0x40,0xb4,0x45,0x07,IMdl_resource_info>
{

};

/// Interface for measured bsdf files.
class IMdl_measured_bsdf_info : public
    base::Interface_declare<0xce,0x45,0xe6,0xef,0xdc,0x74,0x00,0x4f,0xac,0xae,0x34,IMdl_resource_info>
{

};

/// Interface for a graph node representing an MDL package.
class IMdl_package_info : public
    base::Interface_declare<0x94d,0x66,0x47a,0xb0,0xc3,0x7b,0x68,0xba,0x40,0xde,0x06,IMdl_info>
{
    public:

        /// Returns the number of modules and packages contained by this package.
        virtual Size                    get_child_count() const = 0;

        /// Returns a child of this package.
        /// \param index     Index in the child list of this package. 
        virtual const IMdl_info*        get_child(Size index) const = 0;    

        /// Returns the number of search paths of this package.
        virtual Size                    get_search_path_index_count() const = 0;

        /// Returns the search path index in the current IMdl_discovery_result.
        /// \param index     Index in the IMdl_discovery_result search path list.
        virtual Size                    get_search_path_index(Size index) const = 0;

        /// Returns a search path in the local file system where this package has been found.
        /// \param index     Index in the search path list of this package.
        virtual const char*             get_search_path(Size index) const = 0;

        /// Returns an absolute path to package in the local file system.
        /// \param index     Index in the resolved path list of this package.
        virtual const IString*          get_resolved_path(Size index) const = 0;

        /// Returns true if the package has been discovered inside of an archive, false if not.
        /// \param index     Index in the interval [0, search_path_index_count).
        virtual bool                    in_archive(Size index) const = 0;
};


/// Interface for the discovery result.
class IMdl_discovery_result : public
    base::Interface_declare<0xe3c1bc1a,0xb1db,0x4c8c,0xba,0x41,0x37,0xed,0x87,0xb7,0x86,0xb8>
{
public:

    /// Returns a pointer to the root of the discovered graph.
    virtual const IMdl_package_info*    get_graph() const = 0;

    /// Returns the number of search paths.
    virtual Size                        get_search_paths_count() const = 0;

    /// Returns the search path in the local file system referenced by index.
    /// \param index     Index in the search path list of this discovery result.
    virtual const char*                 get_search_path(Size index) const = 0;

};

/// Interface for the discovery API.
///
/// The discovery API offers functionality to find MDL content in the registered MDL search paths.
/// These search paths have to be declared with the function
/// call \if IRAY_API #mi::neuraylib::IRendering_configuration::add_mdl_path() \else
/// #mi::neuraylib::IMdl_compiler::add_module_path() \endif in advance. The ordering of the search
/// paths results from the ordering of the registration and is taken into account during the
/// discovery process.
/// 
/// The MDL modules and packages found inside a search path are assembled into a graph structure
/// consisting of package and module nodes. The discovery results of multiple search paths are 
/// merged to one result graph. If a module is found, but there exists already a 
/// module with the same qualified name in a previously discovered graph, the additional
/// module will be handled as a shadow of the existing module. Multiple packages with the same 
/// qualified name will be merged into one package. 
///
/// The discovery API works for both MDL files (.mdl) and MDL archive files (.mdr). Archives have 
/// to be installed in the top level of the search paths, but material files can be placed anywhere
/// across the sub-folders of a search path.
/// 
/// The result of a discovery process is provided as an mi::neuraylib::IMdl_discovery_result. This 
/// data structure provides information about the discovered search paths as well as access to the
/// result graph structure.
class IMdl_discovery_api : public
    base::Interface_declare<0x208aa1f2,0x08bc,0x4c81,0x8b,0x0f,0x54,0xba,0x4a,0x61,0xe9,0xd8>
{
public:

    /// Returns the file system and archive discovery result.
    /// \param filter   Bitmask, that can be used to specify which discovery kinds to include in 
    ///                 the result (see #mi::neuraylib::IMdl_info::Kind). 
    ///                 By default, all kinds are included.
    virtual const IMdl_discovery_result*  discover(
        Uint32 filter = static_cast<Uint32>(IMdl_info::DK_ALL)) const = 0;
};

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_DISCOVERY_API_H
