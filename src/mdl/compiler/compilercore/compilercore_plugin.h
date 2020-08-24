/***************************************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief MDL core plugin API

#ifndef MDL_COMPILER_COMPILERCORE_PLUGIN_H
#define MDL_COMPILER_COMPILERCORE_PLUGIN_H

#include <mi/base/interface_declare.h>
#include <mi/base/plugin.h>
#include <mi/base/types.h>

namespace mi {

namespace neuraylib { class IPlugin_api; }

namespace mdl {

class IMDL;

/// Type of MDL core plugins
#define MI_MDL_CORE_PLUGIN_TYPE "mdl_core v1"

/// Abstract interface for MDL core plugins.
class IMdl_plugin : public base::Plugin
{
public:
    /// Returns the name of the plugin.
    ///
    /// \note This method from #mi::base::Plugin is repeated here only for documentation purposes.
    virtual const char* get_name() const = 0;

    /// Initializes the plugin.
    ///
    /// \param mdl   Provides access to the IMDL instance.
    /// \return      \c true in case of success, and \c false otherwise.
    virtual bool init( neuraylib::IPlugin_api* plugin_api, IMDL* mdl) = 0;

    /// De-initializes the plugin.
    ///
    /// \param mdl   Provides access to the IMDL instance.
    /// \return      \c true in case of success, and \c false otherwise.
    virtual bool exit( neuraylib::IPlugin_api* plugin_api, IMDL* mdl) = 0;
};

} // namespace mdl

} // namespace mi

#endif // MDL_COMPILER_COMPILERCORE_PLUGIN_H
