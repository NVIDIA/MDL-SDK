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

#ifndef BASE_LIB_PATH_I_PATH_H
#define BASE_LIB_PATH_I_PATH_H

#include <string>
#include <vector>
#include <mi/base/types.h>
#include <base/system/main/i_module.h>

namespace MI {

namespace SYSTEM { class Module_registration_entry; }

namespace PATH {

/// The different kinds of search paths that can be managed by this module.
enum Kind {
    MDL      = 0, ///< For MDL modules and resources referenced by MDL modules.
    RESOURCE = 1, ///< For resources (independent of MDL modules).
    INCLUDE  = 2, ///< For include statements, e.g., in .mi files.
    N_KINDS  = 3
};

/// Public interface of the PATH module.
///
/// \note On Windows all paths are normalized in the sense that slashes are replaced by backslashes.
///       Slashes are considered as valid directory separators by the Windows API. However, in our
///       code base we expect backslashes as directory separators on Windows (and slashes look just
///       odd).
class Path_module : public SYSTEM::IModule
{
public:
    /// Returns the module registrations entry for the module.
    static SYSTEM::Module_registration_entry* get_instance();

    /// Returns the name of the module.
    static const char* get_name() { return "PATH"; }



    /// A path is a string.
    typedef std::string Path;

    /// A search path is a sequence of paths.
    typedef std::vector<std::string> Search_path;



    /// Sets the search path of the given kind.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -2: At least one of the paths is not valid.
    virtual mi::Sint32 set_search_path( Kind kind, const Search_path& paths) = 0;

    /// Returns the search path of the given kind.
    virtual const Search_path& get_search_path( Kind kind) const = 0;

    /// Clears the search path of the given kind, i.e., sets it to the empty sequence.
    virtual void clear_search_path( Kind kind) = 0;



    /// Returns the number of paths in the search path of the given kind, i.e., the length of the
    /// sequence.
    virtual size_t get_path_count( Kind kind) const = 0;

    /// Sets a path indentified by position in the search path of the given kind.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -2: \p path is not a valid path.
    ///           - -3: \p index is out of bounds.
    virtual mi::Sint32 set_path( Kind kind, size_t index, const Path& path) = 0;

    /// Returns a path indentified by position in the search path of the given kind.
    ///
    /// \return   The requested path, or the empty string if \p index is out of bounds.
    virtual const Path& get_path( Kind kind, size_t index) const = 0;



    /// Adds a path to the end of the search path of the given kind.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -2: \p path is not a valid path.
    virtual mi::Sint32 add_path( Kind kind, const Path& path) = 0;

    /// Removes a path from the search path of the given kind.
    ///
    /// In case of multiple occurrences, only the first match is removed.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -2: There is no such path in the path list.
    virtual mi::Sint32 remove_path( Kind kind, const Path& path) = 0;



    /// Searches a file in the search path of the given kind.
    ///
    /// If \p file_name is already an absolute file name and the file is accessible, the file name
    /// is returned unchanged. If it does not exist, the empty string is returned.
    ///
    /// Otherwise, for relative file names, the method loops over the search path of the given kind
    /// and combines each of its paths with the relative file name in turn. As soon as the resulting
    /// file name references an accessible file, this file name is returned. Otherwise, the empty
    /// string is returned.
    virtual std::string search( Kind kind, const std::string& file_name) const = 0;
};

} // namespace PATH

} // namespace MI

#endif // BASE_LIB_PATH_I_PATH_H
