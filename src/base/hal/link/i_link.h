/***************************************************************************************************
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
 **************************************************************************************************/

/// \file
/// \brief LINK module
///
/// This module provides capabilities to load and unload dynamic libraries and to retrieve symbols
/// from them.

#ifndef BASE_HAL_LINK_I_LINK_H
#define BASE_HAL_LINK_I_LINK_H

#include <mi/base/interface_declare.h>
#include <base/system/main/i_module.h>

#include <string>

namespace MI {

namespace SYSTEM { class Module_registration_entry; }

namespace LINK {

class ILibrary;

/// The LINK module.
class Link_module : public SYSTEM::IModule
{
public:
    /// Loads a dynamic library.
    ///
    /// \param path   Path of the dynamic library
    /// \return       The loaded dynamic library, or \c NULL in case of failure.
    virtual ILibrary* load_library( const char* path) = 0;

    // methods of SYSTEM::IModule

    static const char* get_name() { return "LINK"; }

    static SYSTEM::Module_registration_entry* get_instance();
};

/// Represents dynamic libraries.
///
/// The destructor unloads the dynamic library (in implementations of this interface).
class ILibrary : public
    mi::base::Interface_declare<0xa1653043,0x65d7,0x47c5,0x93,0xa7,0x3c,0xdf,0xaf,0xcf,0x3a,0x1c>
{
public:
    /// Returns the address of a symbol from the dynamic library, or \c NULL in case of failure.
    virtual void* get_symbol( const char* symbol_name) = 0;

    /// Returns the filename of the dynamic library.
    ///
    /// \param symbol_name   A symbol from this dynamic library, e.g., "mi_plugin_factory". Needed
    ///                      for technical reasons on some platforms.
    /// \return              The filename of the dynamic library, or the empty string in case of
    ///                      failure. Usually this path is absolute. It might not be absolute on
    ///                      Linux if the filename passed to dlopen() contained a slash.
    virtual std::string get_filename( const char* symbol_name) = 0;
};

} // namespace LINK

} // namespace MI

#endif // BASE_HAL_LINK_I_LINK_H
