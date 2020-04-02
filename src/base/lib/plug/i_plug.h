/******************************************************************************
 * Copyright (c) 2004-2020, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/

#ifndef BASE_LIB_PLUG_I_PLUG_H
#define BASE_LIB_PLUG_I_PLUG_H

#include <cstdlib>
#include <base/system/main/i_module.h>

namespace mi { namespace base { class Plugin; class IPlugin_descriptor; } }
namespace mi { namespace neuraylib { class IPlugin_api; } }

namespace MI {

namespace SYSTEM { class Module_registration_entry; }

namespace PLUG {

/// The PLUG module.
class Plug_module : public SYSTEM::IModule
{
public:
    /// \name Loading/unloading of libraries
    //@{

    /// Loads a dynamic library and invokes factories for all plugins.
    ///
    /// \note Due to reference counting, libraries without any plugin are unloaded again.
    ///
    /// \return \c true in case of success, \c false otherwise
    virtual bool load_library( const char* path) = 0;

    /// Library unloading during runtime is not supported in neuray. If needed, note that passing
    /// the same argument as for load_library() is not guaranteed to work since the path is resolved
    /// during loading. One needs to pass the path as stored in a plugin descriptor.

    //@}
    /// \name Plugin introspection
    //@{

    /// Return the number of plugins.
    virtual size_t get_plugin_count() = 0;

    /// Returns a plugin by index.
    ///
    /// \return The requested plugin, or \c NULL if \p index is out of bounds.
    virtual mi::base::IPlugin_descriptor* get_plugin( size_t index) = 0;

    /// Returns a plugin by index.
    ///
    /// \return The requested plugin, or \c NULL if there is no plugin of that name.
    virtual mi::base::IPlugin_descriptor* get_plugin( const char* name) = 0;

    //@}
    /// \name Access to Plugin API/render plugin API
    //@{

    /// Sets the pointer for the plugin API of the Iray/DiCE API.
    ///
    /// The value \c NULL can be passed to clear the pointer.
    virtual void set_plugin_api( mi::neuraylib::IPlugin_api* plugin_api) = 0;

    /// Returns the pointer to the plugin API of the Iray/DiCE API.
    ///
    /// The value \c NULL is returned if the pointer is not set (Iray/DICE API not present).
    virtual mi::neuraylib::IPlugin_api* get_plugin_api() const = 0;


    //@}

    // methods of SYSTEM::IModule

    static const char* get_name() { return "PLUG"; }

    static SYSTEM::Module_registration_entry* get_instance();
};

} // namespace PLUG

} // namespace MI

#endif // BASE_LIB_PLUG_I_PLUG_H

