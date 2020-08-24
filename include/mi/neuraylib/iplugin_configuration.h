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
/// \brief API component for plugin related settings.

#ifndef MI_NEURAYLIB_IPLUGIN_CONFIGURATION_H
#define MI_NEURAYLIB_IPLUGIN_CONFIGURATION_H

#include <mi/base/plugin.h>
#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_configuration
@{
*/

/// This interface is used to load plugins and to query information about loaded plugins.
class IPlugin_configuration : public
    mi::base::Interface_declare<0x11285c46,0x9791,0x498d,0xbd,0xfe,0x8f,0x51,0x84,0x81,0x98,0xd4>
{
public:
    /// Loads a plugin library.
    ///
    /// This function loads the specified shared library, enumerates all plugin classes in the
    /// specified shared library, and adds them to the system.
    ///
    /// This function can only be called before \neurayProductName has been started.
    ///
    /// \param path     The path of the shared library to be loaded. This shared library needs to be
    ///                 a valid plugin for \neurayProductName.
    /// \return         0, in case of success, -1 in case of failure.
    virtual Sint32 load_plugin_library( const char* path) = 0;

    /// Loads all plugins from a given directory.
    ///
    /// Enumerates all plugins in the given directory in alphabetic order and calls
    /// #load_plugin_library() for each of them in turn. On Windows, all files with the extension
    /// \c .dll are considered, while on Linux and MacOS X all files with the extension \c .so are
    /// considered. Additionally, on MacOS X all files with extension \c .dylib are considered.
    ///
    /// \param path     The path of the directory.
    /// \return         0, in case of success, -1 in case of failure.
    virtual Sint32 load_plugins_from_directory( const char* path) = 0;

    /// Returns the number of loaded plugins.
    ///
    /// \return         The number of loaded plugins.
    virtual Size get_plugin_length() const = 0;

    /// Returns a descriptor for the \p index -th loaded plugin.
    ///
    /// \return         A descriptor for the \p index -th loaded plugin, or \c NULL in
    ///                 case of failure.
    virtual base::IPlugin_descriptor* get_plugin_descriptor( Size index) const = 0;
};

/*@}*/ // end group mi_neuray_configuration

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IPLUGIN_CONFIGURATION_H
