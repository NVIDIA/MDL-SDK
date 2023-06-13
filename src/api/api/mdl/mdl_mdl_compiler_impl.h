/***************************************************************************************************
 * Copyright (c) 2012-2023, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Implementation of IMdl_compiler
 **
 ** Implements the IMdl_compiler interface
 **/

#ifndef API_API_MDL_MDL_COMPILER_IMPL_H
#define API_API_MDL_MDL_COMPILER_IMPL_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/imdl_compiler.h>

#include <base/system/main/access_module.h>
#include <boost/core/noncopyable.hpp>

namespace mi {
    class IString;
    namespace base { class ILogger; } namespace neuraylib { class IWriter; }
}

namespace MI {

namespace ATTR { class Attr_module; }
namespace MDLC { class Mdlc_module; }
namespace MEM { class Mem_module; }
namespace PATH { class Path_module; }
namespace PLUG { class Plug_module; }

namespace MDL {

class Neuray_impl;

class Mdl_compiler_impl
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_compiler>,
    public boost::noncopyable
{
public:
    /// Constructor of Mdl_compiler_impl
    ///
    /// \param neuray_impl      The neuray instance which contains this Mdl_compiler_impl
    Mdl_compiler_impl( Neuray_impl* neuray_impl);

    /// Destructor of Mdl_compiler_impl
    ~Mdl_compiler_impl();

    // public API methods

    mi::Sint32 add_builtin_module(
        const char* module_name,
        const char* module_source) override;

    // internal methods

    /// Starts this API component.
    ///
    /// The implementation of INeuray::start() calls the #start() method of each API component.
    /// This method performs the API component's specific part of the library start.
    ///
    /// \return            0, in case of success, -1 in case of failure.
    mi::Sint32 start();

    /// Shuts down this API component.
    ///
    /// The implementation of INeuray::shutdown() calls the #shutdown() method of each API
    /// component. This method performs the API component's specific part of the library shutdown.
    ///
    /// \return           0, in case of success, -1 in case of failure
    mi::Sint32 shutdown();

private:

    Neuray_impl* m_neuray_impl;                                    ///< Neuray_impl

    SYSTEM::Access_module<ATTR::Attr_module> m_attr_module;
    SYSTEM::Access_module<MDLC::Mdlc_module> m_mdlc_module;
    SYSTEM::Access_module<MEM::Mem_module> m_mem_module;
    SYSTEM::Access_module<PATH::Path_module> m_path_module;
    SYSTEM::Access_module<PLUG::Plug_module> m_plug_module;
};

} // namespace MDL

} // namespace MI

#endif // API_API_MDL_MDL_COMPILER_IMPL_H
