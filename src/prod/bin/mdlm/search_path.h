/******************************************************************************
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
 *****************************************************************************/
#pragma once

#include <mi/mdl_sdk.h>
#include <string>
#include <vector>

namespace mdlm
{
    // Search_path
    //
    // Helper to:
    // - take a snapshot of the search paths and restore them
    // - clear all search paths
    // - add search path
    //
    // Usage:
    // - snapshot()
    // - mess around with search paths... Add/Remove...
    // - restore_snapshot()
    //
    class Search_path
    {
        mi::base::Handle<mi::neuraylib::IMdl_configuration> m_mdl_config;

        std::vector<std::string> m_paths;

    public:
        // Initialize
        Search_path(const mi::neuraylib::INeuray * neuray);

        ~Search_path();

        // Take a snapshot of the current search pathes
        mi::Sint32 snapshot();

        // Restore search path since last snapshot
        // If no snapshot was ever taken, nothing to restore 
        mi::Sint32 restore_snapshot() const;

        // Clear current snapshot list of paths
        mi::Sint32 clear_snapshot();

        // Add to module path
        mi::Sint32 add_module_path(const std::string & directory);

        // Is input directory in the list of searhc path
        bool find_module_path(const std::string & directory) const;

        // Remove all module paths
        mi::Sint32 clear_module_paths();

        // Get the list of MDL path
        // Require to invoke snapshot() before
        const std::vector<std::string> & paths() const { return m_paths; }

        // log_debug each path in the m_paths list
        void log_debug() const;
    };
}
