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

// examples/mdl_sdk/entity_resolver/entity_resolver.cpp
//
// Simplified implementation of a custom entity resolver

#include "entity_resolver.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <sstream>

#include "utils/sdk_strings.h"

/// Returns the OS-specific directory separator.
std::string get_directory_separator()
{
#ifdef MI_PLATFORM_WINDOWS
    return "\\";
#else // MI_PLATFORM_WINDOWS
    return "/";
#endif // MI_PLATFORM_WINDOWS
}

/// Normalizes paths.
///
/// The function collapses redundant directory separators and folds current and parent directory
/// references as possible. For example, "a//b", "a/./b", and "a/c/../b" are converted into
/// "a/b". Paths like "a/.." are converted into ".". On Windows, all slashes are converted upfront
/// to backslashes.
std::string normalize_path( std::string path)
{
#ifdef MI_PLATFORM_WINDOWS
    path = mi::examples::strings::replace( path, "/", "\\");
#endif // MI_PLATFORM_WINDOWS

    const std::string& separator = get_directory_separator();
    assert( separator.size() == 1);

    std::vector<std::string> path_components = mi::examples::strings::split( path, separator[0]);

    // resolve current and parent directory references
    std::vector<std::string> result_components;
    for( const auto& component : path_components) {

        // skip empty path components or current directory references
        if( component.empty() || component == ".")
            continue;

        // handle parent directory references
        if( component == "..") {
            if( result_components.empty() || result_components.back() == "..")
                result_components.emplace_back( "..");
            else
                result_components.pop_back();
            continue;
        }

        // handle regular path components
        result_components.push_back( component);
    }

    // convert result_components into a string
    std::string result = result_components.empty() ? "" : result_components[0];
    for( size_t i = 1, n = result_components.size(); i < n; ++i)
        result += separator + result_components[i];

    // re-add leading separator for absolute paths
    if( !path.empty() && path[0] == separator[0])
        result.insert( 0, separator);

    return result.empty() ? "." : result;
}

/// A trace helper to emit trace messages from the entity resolver.
class Trace
{
public:
    Trace( bool enabled, const std::string& message)
      : m_enabled( enabled),
        m_message( message)
    { }

    ~Trace()
    {
        if( m_enabled)
            std::cout << m_message << std::endl;
    }

    void add_failure( const std::string& reason)
    {
        if( m_enabled)
            m_message += ", failed (" + reason + ")";
    }

    void add_success( const std::string& message)
    {
        if( m_enabled)
            m_message += ", " + message;
    }

private:
    bool m_enabled;
    std::string m_message;
};

/// An implementation of IBuffer that takes its content from a reader.
class Buffer : public mi::base::Interface_implement<mi::neuraylib::IBuffer>
{
public:
    Buffer( mi::neuraylib::IReader* reader)
    {
        assert( reader->supports_absolute_access());
        mi::Sint64 size = reader->get_file_size();
        m_buffer.resize( size);
        mi::Sint64 read = reader->read( &m_buffer[0], size);
        assert( read == size);
        (void) read;
    }

    const mi::Uint8* get_data() const final
    { return reinterpret_cast<const mi::Uint8*>( m_buffer.data()); }

    mi::Size get_data_size() const final { return m_buffer.size(); }

private:
    std::vector<char> m_buffer;
};

bool Os_file_system::exists( const char* name)
{
    /// There are cheaper, but OS-dependent implementations.
    FILE* file = fopen( name, "rb");
    if( !file)
        return false;
    fclose( file);
    return true;
}

mi::neuraylib::IReader* Os_file_system::open( const char* name)
{
    return m_mdl_impexp_api->create_reader( name);
}

void Virtual_file_system::add_file( const char* name, const char* prefix)
{
    mi::base::Handle<mi::neuraylib::IReader> reader( m_mdl_impexp_api->create_reader( name));
    assert( reader);
    // Normalize name since we use plain string matches in exists() and open() instead of a regular
    // directory traversal.
    std::string new_name = normalize_path( std::string( prefix) + name);
    m_files[new_name] = new Buffer( reader.get());
}

bool Virtual_file_system::exists( const char* name)
{
    // Normalize name since we use plain string matches instead of a regular directory traversal.
    return m_files.count( normalize_path( name)) > 0;
}

mi::neuraylib::IReader* Virtual_file_system::open( const char* name)
{
    // Normalize name since we use plain string matches instead of a regular directory traversal.
    const auto& it = m_files.find( normalize_path( name));
    return it != m_files.end() ? m_mdl_impexp_api->create_reader( it->second.get()) : nullptr;
}

mi::neuraylib::IReader* Mdl_resolved_resource_element::create_reader( mi::Size i) const
{
    if( i > 0)
        return nullptr;
    return m_file_system->open( m_filename.c_str());
}

mi::neuraylib::IMdl_resolved_module* Mdl_entity_resolver::resolve_module(
    const char* module_name,
    const char* owner_file_path,
    const char* owner_name,
    mi::Sint32 /*pos_line*/,
    mi::Sint32 /*pos_column*/,
    mi::neuraylib::IMdl_execution_context* context)
{
    Trace trace(
        m_trace,
        std::string( "module_name: " ) + (module_name ? module_name : "(null)")
            + ", owner_name: " + (owner_name ? owner_name : "(null)"));

    // This check is redundant for modules as the compiler enforces these rules already. Keep it
    // here for consistency with resource, which \em do need it.
    if( !check_module_name_for_dot_and_dot_dot( module_name)) {
        trace.add_failure( "incorrect usage of \".\" or \"..\"");
        return nullptr;
    }

    std::string absolute_module_name;
    std::string result;

    if( is_absolute_module_name( module_name)) {

        absolute_module_name = module_name;
        result = find_module_by_absolute_name( absolute_module_name);
        if( result.empty()) {
            trace.add_failure( "not found");
            return nullptr;
        }

    } else {

        if( !owner_file_path || !owner_name) {
            trace.add_failure( "no owner");
            return nullptr;
        }

        Absolute_module absolute = get_absolute_module(
            module_name, owner_name, owner_file_path);
        if( absolute.module_name.empty() || absolute.filename.empty()) {
            trace.add_failure( "too many \"..\" components");
            return nullptr;
        }

        absolute_module_name = absolute.module_name;
        result = absolute.filename;

        if( !m_file_system->exists( result.c_str())) {
            trace.add_failure( "not found");
            return nullptr;
        }

        // Check 1 from section 2.2 of the MDL specification
        std::string canonical_result = find_module_by_absolute_name( absolute_module_name);
        if( canonical_result != result) {
            std::string reason = std::string( "shadowing rule: \"") + canonical_result + "\" "
                               + "shadows \"" + result + "\"";
            add_error_message( context, module_name, reason.c_str());
            trace.add_failure( reason.c_str());
            return nullptr;
        }
    }

    trace.add_success( "result: " + result);
    return new Mdl_resolved_module(
        m_file_system.get(), absolute_module_name.c_str(), result.c_str());
}

mi::neuraylib::IMdl_resolved_resource* Mdl_entity_resolver::resolve_resource(
    const char* file_path,
    const char* owner_file_path,
    const char* owner_name,
    mi::Sint32 /*pos_line*/,
    mi::Sint32 /*pos_column*/,
    mi::neuraylib::IMdl_execution_context* context)
{
    Trace trace(
        m_trace,
        std::string( "file_path: " ) + (file_path ? file_path : "(null)")
            + ", owner_name: " + (owner_name ? owner_name : "(null)"));

    if( !check_resource_file_path_for_dot_and_dot_dot( file_path)) {
        const char* reason = "incorrect usage of \".\" or \"..\"";
        add_error_message( context, file_path, reason);
        trace.add_failure( reason);
        return nullptr;
    }

    std::string absolute_file_path;
    std::string result;

    if( is_absolute_file_path( file_path)) {

        absolute_file_path = file_path;
        result = find_resource_by_absolute_path( absolute_file_path);
        if( result.empty()) {
            const char* reason = "not found";
            add_error_message( context, file_path, reason);
            trace.add_failure( reason);
            return nullptr;
        }

    } else {

        if( !owner_file_path || !owner_name) {
            const char* reason = "no owner";
            add_error_message( context, file_path, reason);
            trace.add_failure( reason);
            return nullptr;
        }

        Absolute_resource absolute = get_absolute_resource(
            file_path, owner_name, owner_file_path);
        if( absolute.file_path.empty() || absolute.filename.empty()) {
            const char* reason = "too many \"..\" components";
            add_error_message( context, file_path, reason);
            trace.add_failure( reason);
            return nullptr;
        }

        absolute_file_path = absolute.file_path;
        result = absolute.filename;

        if( !m_file_system->exists( result.c_str())) {
            const char* reason = "not found";
            add_error_message( context, file_path, reason);
            trace.add_failure( reason);
            return nullptr;
        }

        // Check 1 from section 2.2 of the MDL specification
        std::string canonical_result = find_resource_by_absolute_path( absolute_file_path);
        if( canonical_result != result) {
            std::string reason = std::string( "shadowing rule: \"") + canonical_result + "\" "
                               + "shadows \"" + result + "\"";
            add_error_message( context, file_path, "not found");
            add_error_message( context, file_path, reason.c_str());
            trace.add_failure( reason.c_str());
            return nullptr;
        }
    }

    trace.add_success( "result: " + result);
    return new Mdl_resolved_resource(
        absolute_file_path.c_str(),
        result.c_str(),
        m_file_system.get(),
        absolute_file_path.c_str(),
        result.c_str());
}

bool Mdl_entity_resolver::check_module_name_for_dot_and_dot_dot( const char* module_name)
{
    std::string s = module_name;
    if( s.substr( 0, 3) == ".::")
        // Skip single initial "." component
        s = s.substr( 3);
    else {
        // Skip initial ".." components
        while( s.substr( 0, 4) == "..::")
            s = s.substr( 4);
    }

    // Reject strings that are empty or contain only a single "." or ".." component
    if( s.empty() || s == "." || s == "..")
        return false;

    // Reject "." or ".." component in the middle.
    if( s.find( "::.::") != std::string::npos)
        return false;
    if( s.find( "::..::") != std::string::npos)
        return false;

    // Reject "." or ".." component at the end.
    size_t n = s.size();
    if( n >= 3 && s.substr( n-3, 3) == "::.")
        return false;
    if( n >= 4 && s.substr( n-4, 4) == "::..")
        return false;

    return true;
}

bool Mdl_entity_resolver::is_absolute_module_name( const char* module_name)
{
    return module_name[0] == ':' && module_name[1] == ':';
}

std::string Mdl_entity_resolver::find_module_by_absolute_name( const std::string& module_name)
{
    assert( is_absolute_module_name( module_name.c_str()));

    std::string filename_suffix = mi::examples::strings::replace(
        module_name, "::", get_directory_separator()) + ".mdl";
    return find_in_search_paths( filename_suffix);
}

Mdl_entity_resolver::Absolute_module
Mdl_entity_resolver::get_absolute_module(
    const char* module_name, const char* owner_name, const char* owner_file_path)
{
    assert( !is_absolute_module_name( module_name));
    assert( owner_name);
    assert( owner_file_path);

    const std::string separator = get_directory_separator();
    std::string name_start = owner_name;
    std::string filename_start = owner_file_path;
    std::string end = module_name;

    // Remove owner module name and owner filename from the starts.
    size_t name_pos = name_start.rfind( "::");
    size_t filename_pos = filename_start.rfind( separator);
    if( name_pos == std::string::npos || filename_pos == std::string::npos)
        return {};
    name_start = name_start.substr( 0, name_pos);
    filename_start = filename_start.substr( 0, filename_pos);

    // Strip ".::" prefix from end.
    if( end.substr( 0, 3) == ".::")
        end = end.substr( 3);

    // Strip last component from both starts for each "..::" prefix in end.
    while( end.substr( 0, 4) == "..::") {
        name_pos = name_start.rfind( "::");
        filename_pos = filename_start.rfind( separator);
        if( name_pos == std::string::npos || filename_pos == std::string::npos)
            return {};
        end = end.substr( 4);
        name_start = name_start.substr( 0, name_pos);
        filename_start = filename_start.substr( 0, filename_pos);
    }

    Absolute_module result;
    result.module_name = name_start + "::" + end;
    std::string filename_end = mi::examples::strings::replace( end, "::", separator);
    result.filename = filename_start + separator + filename_end + ".mdl";

    return result;
}

bool Mdl_entity_resolver::is_absolute_file_path( const char* file_path)
{
    return file_path[0] == '/';
}

std::string Mdl_entity_resolver::find_resource_by_absolute_path( const std::string& file_path)
{
    assert( is_absolute_file_path( file_path.c_str()));

    std::string filename_suffix = mi::examples::strings::replace(
        file_path, "/", get_directory_separator());
    return find_in_search_paths( filename_suffix);
}

bool Mdl_entity_resolver::check_resource_file_path_for_dot_and_dot_dot( const char* file_path)
{
    std::string s = file_path;
    if( s.substr( 0, 2) == "./")
        // Skip single initial "." component
        s = s.substr( 2);
    else {
        // Skip initial ".." components
        while( s.substr( 0, 3) == "../")
            s = s.substr( 3);
    }

    // Reject strings that are empty or contain only a single "." or ".." component
    if( s.empty() || s == "." || s == "..")
        return false;

    // Reject "." or ".." component in the middle.
    if( s.find( "/./") != std::string::npos)
        return false;
    if( s.find( "/../") != std::string::npos)
        return false;

    // Reject "." or ".." component at the end.
    size_t n = s.size();
    if( n >= 2 && s.substr( n-2, 2) == "/.")
        return false;
    if( n >= 3 && s.substr( n-3, 3) == "/..")
        return false;

    return true;
}

Mdl_entity_resolver::Absolute_resource
Mdl_entity_resolver::get_absolute_resource(
    const char* file_path, const char* owner_name, const char* owner_file_path)
{
    assert( !is_absolute_file_path( file_path));
    assert( owner_name);
    assert( owner_file_path);

    const std::string separator = get_directory_separator();
    std::string file_path_start = owner_name;
    std::string filename_start = owner_file_path;
    std::string end = file_path;

    // Remove owner module name and owner filename from the starts.
    size_t file_path_pos = file_path_start.rfind( "::");
    size_t filename_pos = filename_start.rfind( separator);
    if( file_path_pos == std::string::npos || filename_pos == std::string::npos)
        return {};
    file_path_start = file_path_start.substr( 0, file_path_pos);
    filename_start = filename_start.substr( 0, filename_pos);

    // Strip "./" prefix from end.
    if( end.substr( 0, 2) == "./")
        end = end.substr( 2);

    // Strip last component from both starts for each "../" prefix in end.
    while( end.substr( 0, 3) == "../") {
        file_path_pos = file_path_start.rfind( "::");
        filename_pos = filename_start.rfind( separator);
        if( file_path_pos == std::string::npos || filename_pos == std::string::npos)
            return {};
        end = end.substr( 3);
        file_path_start = file_path_start.substr( 0, file_path_pos);
        filename_start = filename_start.substr( 0, filename_pos);
    }

    Absolute_resource result;
    file_path_start = mi::examples::strings::replace( file_path_start, "::", "/");
    result.file_path = file_path_start + "/" + end;
    std::string filename_end = mi::examples::strings::replace( end, "/", separator);
    result.filename = filename_start + separator + filename_end;

    return result;
}

std::string Mdl_entity_resolver::find_in_search_paths( const std::string& filename_suffix)
{
    assert( filename_suffix.substr( 0, 1) == get_directory_separator());

    for( const auto& s: m_search_paths) {
        std::string filename = s + filename_suffix;
        if( m_file_system->exists( filename.c_str()))
            return filename;
    }

    return {};
}

void Mdl_entity_resolver::add_error_message(
    mi::neuraylib::IMdl_execution_context* context, const char* identifier, const char* reason)
{
    if( !context)
        return;

    std::string s = std::string( "Failed to resolve \"") + identifier + "\" (" + reason + ")";

    context->add_message(
        mi::neuraylib::IMessage::MSG_INTEGRATION,
        mi::base::MESSAGE_SEVERITY_ERROR,
        -1,
        s.c_str());
}
