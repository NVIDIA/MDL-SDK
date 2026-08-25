/******************************************************************************
 * Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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
 *****************************************************************************/

// examples/mdl_sdk/entity_resolver/entity_resolver.h
//
// Simplified implementation of a custom entity resolver

#ifndef EXAMPLE_ENTITY_RESOLVER_H
#define EXAMPLE_ENTITY_RESOLVER_H

#include <map>
#include <string>
#include <vector>

#include <mi/mdl_sdk.h>

/// Minimal file system abstraction used by the entity resolver implementation below.
///
/// Does not support MDLE nor MDL archives (requires container/member name support).
/// Does not support uvtiles nor animated textures (requires directory enumeration/regex matching).
/// All names are in UTF-8.
class IFile_system : public
    mi::base::Interface_declare<0x2f783cf2,0x5dd0,0x438b,0x97,0x77,0x76,0x84,0x6f,0x16,0x64,0xf9>
{
public:
    /// Indicates whether there is a file named \p name (in UTF-8 encoding).
    virtual bool exists( const char* name) = 0;

    /// Opens the file named \p name (in UTF-8 encoding) or returns \c NULL if there is no such
    /// file.
    virtual mi::neuraylib::IReader* open( const char* name) = 0;
};

/// An implementation of IFile_system that uses the filesystem of the OS.
class Os_file_system : public mi::base::Interface_implement<IFile_system>
{
public:
    Os_file_system( mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
      : m_mdl_impexp_api( mdl_impexp_api, mi::base::DUP_INTERFACE) { }

    // interface methods
    bool exists( const char* name);
    mi::neuraylib::IReader* open( const char* name);

private:
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_mdl_impexp_api;
};

/// An implementation of IFile_system that holds file contents in memory.
class Virtual_file_system : public mi::base::Interface_implement<IFile_system>
{
public:
    Virtual_file_system( mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
      : m_mdl_impexp_api( mdl_impexp_api, mi::base::DUP_INTERFACE) { }

    /// Adds a file from the OS filesystem to the virtual filesystem.
    ///
    /// \param name     The content of this file is added to the virtual file system. The name is
    ///                 given in UTF-8.
    /// \param prefix   The virtual filesystem uses the concatenation of \p prefix and \p name as
    ///                 file name. A non-empty string is useful to ensure that all requests go
    ///                 through this abstraction, and do not access the OS filesystem directly. Note
    ///                 that the same prefix must be added to the search paths of the entity
    ///                 resolver. The prefix is given in UTF-8.
    void add_file( const char* name, const char* prefix = "");

    // interface methods
    bool exists( const char* name);
    mi::neuraylib::IReader* open( const char* name);

private:
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_mdl_impexp_api;

    /// Maps filenames to file contents.
    std::map<std::string, mi::base::Handle<mi::neuraylib::IBuffer>> m_files;
};

/// This implementation does not support string-based modules, i.e., modules without a filename.
class Mdl_resolved_module
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_module>
{
public:
    Mdl_resolved_module(
        IFile_system* file_system,
        const char* module_name,
        const char* filename)
      : m_file_system( file_system, mi::base::DUP_INTERFACE),
        m_module_name( module_name),
        m_filename( filename) { }

    const char* get_module_name() const { return m_module_name.c_str(); }

    const char* get_filename() const { return m_filename.c_str(); }

    mi::neuraylib::IReader* create_reader() const
    { return m_file_system->open( m_filename.c_str()); }

private:
    mi::base::Handle<IFile_system> m_file_system;
    std::string m_module_name;
    std::string m_filename;
};

/// This implementation does not support uv-tiles nor animated textures.
class Mdl_resolved_resource_element
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_resource_element>
{
public:
    /// Creates readers using the file system abstraction.
    Mdl_resolved_resource_element(
        IFile_system* file_system,
        const char* mdl_file_path,
        const char* filename)
      : m_file_system( file_system, mi::base::DUP_INTERFACE),
        m_mdl_file_path( mdl_file_path),
        m_filename( filename) { }

    mi::Size get_frame_number() const { return 0; }

    mi::Size get_count() const { return 1; }

    const char* get_mdl_file_path( mi::Size i) const
    { return i == 0 ? m_mdl_file_path.c_str() : nullptr; }

    // The string m_filename might not have a meaning outside of our file-system abstraction.
    // Do not return it. This also disables lazy texture loading.
    const char* get_filename( mi::Size i) const { return nullptr; }

    // Returns a reader using the file system abstraction.
    mi::neuraylib::IReader* create_reader( mi::Size i) const;

    mi::base::Uuid get_resource_hash( mi::Size i) const { return mi::base::Uuid(); }

    bool get_uvtile_uv( mi::Size i, mi::Sint32& u, mi::Sint32& v) const { return false; }

private:
    mi::base::Handle<IFile_system> m_file_system;
    std::string m_mdl_file_path;
    std::string m_filename;
};

/// This implementation does not support uv-tiles nor animated textures.
class Mdl_resolved_resource
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_resource>
{
public:
    /// Creates readers using the file system abstraction.
    Mdl_resolved_resource(
        const char* mdl_file_path_mask,
        const char* filename_mask,
        IFile_system* file_system,
        const char* mdl_file_path,
        const char* filename)
      : m_mdl_file_path_mask( mdl_file_path_mask),
        m_filename_mask( filename_mask),
        m_element( new Mdl_resolved_resource_element( file_system, mdl_file_path, filename)) { }

    bool has_sequence_marker() const { return false; }

    mi::neuraylib::Uvtile_mode get_uvtile_mode() const { return mi::neuraylib::UVTILE_MODE_NONE; }

    const char* get_mdl_file_path_mask() const { return m_mdl_file_path_mask.c_str(); }

    // The string m_filename_mask might not have a meaning outside of our file-system abstraction.
    // Do not return it. This also disables lazy texture loading.
    const char* get_filename_mask() const { return nullptr; }

    mi::Size get_count() const { return 1; }

    const mi::neuraylib::IMdl_resolved_resource_element* get_element( mi::Size i) const
    { return i == 0 ? (m_element->retain(), m_element.get()) : nullptr; }

private:
    std::string m_mdl_file_path_mask;
    std::string m_filename_mask;
    mi::base::Handle<mi::neuraylib::IMdl_resolved_resource_element> m_element;
};

/// This implementation has various limitations:
/// - No support for uv-tiles nor animated textures.
/// - No support for MDLE nor MDL archives.
class Mdl_entity_resolver
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_entity_resolver>
{
public:
    /// Constructor.
    ///
    /// \param file_system   The file system abstraction to be used by the entity resolver.
    /// \param trace         Flag to trace resolve requests and results.
    Mdl_entity_resolver(
        IFile_system* file_system,
        bool trace)
      : m_file_system( file_system, mi::base::DUP_INTERFACE),
        m_trace( trace) { }

    /// Adds a search path to the entity resolver's list of search paths.
    void add_search_path( const std::string& path) { m_search_paths.push_back( path); }

    mi::neuraylib::IMdl_resolved_module* resolve_module(
        const char* module_name,
        const char* owner_file_path,
        const char* owner_name,
        mi::Sint32 pos_line,
        mi::Sint32 pos_column,
        mi::neuraylib::IMdl_execution_context* context);

    mi::neuraylib::IMdl_resolved_resource* resolve_resource(
        const char* file_path,
        const char* owner_file_path,
        const char* owner_name,
        mi::Sint32 pos_line,
        mi::Sint32 pos_column,
        mi::neuraylib::IMdl_execution_context* context);

private:

    /// \name Modules
    //@{

    /// Checks the static rules for usage of "." and "..".
    ///
    /// The number of ".." components is checked later in #find_module_by_absolute_name().
    static bool check_module_name_for_dot_and_dot_dot( const char* module_name);

    /// Indicates whether the module name is absolute.
    static bool is_absolute_module_name( const char* module_name);

    /// Finds a module in the search paths.
    std::string find_module_by_absolute_name( const std::string& module_name);

    struct Absolute_module {
        std::string module_name;
        std::string filename;
    };

    /// Returns the absolute module name and filename for a relative module name and its owner.
    ///
    /// The method just computes those strings without checking for existence or shadowing.
    static Absolute_module get_absolute_module(
        const char* module_name, const char* owner_name, const char* owner_file_path);

    //@}
    /// \name Resources
    //@{

    /// Checks the static rules for usage of "." and "..".
    ///
    /// The number of ".." components is checked later in #find_resource_by_absolute_path().
    static bool check_resource_file_path_for_dot_and_dot_dot( const char* file_path);

    /// Indicates whether the resource file path is absolute.
    static bool is_absolute_file_path( const char* file_path);

    /// Finds a resource in the search paths.
    std::string find_resource_by_absolute_path( const std::string& file_path);

    struct Absolute_resource {
        std::string file_path;
        std::string filename;
    };

    /// Returns the absolute resource file path and filename for a relative path and its owner.
    ///
    /// The method just computes those strings without checking for existence or shadowing.
    static Absolute_resource get_absolute_resource(
        const char* file_path, const char* owner_name, const char* owner_file_path);

    //@}

    /// Finds a file in the search paths.
    ///
    /// \param filename_suffix    Name of the file to search, starting with the directory separator.
    /// \return                   The absolute filename of the file, or the empty string in case of
    ///                           failures.
    std::string find_in_search_paths( const std::string& filename_suffix);

    /// Adds an error message to the context.
    void add_error_message(
        mi::neuraylib::IMdl_execution_context* context, const char* identifier, const char* reason);

    /// File system abstraction used by the entity resolver.
    mi::base::Handle<IFile_system> m_file_system;

    /// Flag that indicates whether to trace resolve requests and results.
    bool m_trace;

    /// Search paths to be considered.
    std::vector<std::string> m_search_paths;
};

#endif // EXAMPLE_ENTITY_RESOLVER_H
