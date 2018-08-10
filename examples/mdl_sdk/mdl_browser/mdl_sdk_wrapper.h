/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief Small wrapper for the few MDL objects this application is requiring.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_SDK_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_SDK_H

#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/base/interface_implement.h>
#include <vector>

namespace mi
{
    namespace neuraylib
    {
        class INeuray;
        class IMdl_compiler;
        class IMdl_archive_api;
        class IMdl_discovery_api;
        class IDatabase;
        class IScope;
        class ITransaction;
    }
}

class Mdl_cache;


/// Custom logger 
class Mdl_browser_logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    Mdl_browser_logger(bool trace);;
    void message(mi::base::Message_severity level, const char* mc, const char* message) override;
private:
    bool m_trace;
};


class Mdl_sdk_wrapper
{
public:
    Mdl_sdk_wrapper(const std::vector<std::string>& search_paths, bool cache_rebuild);
    ~Mdl_sdk_wrapper();

    mi::neuraylib::IMdl_compiler*      get_compiler() const;
    mi::neuraylib::IMdl_archive_api*   get_archive_api() const;
    mi::neuraylib::IMdl_discovery_api* get_discovery() const;
    mi::neuraylib::ITransaction*       get_transaction() const;

    Mdl_cache* get_cache() const { return m_cache; }

private:
    bool start(const std::vector<std::string>& search_paths, bool cache_rebuild);
    bool shutdown();

    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
    mi::base::Handle<mi::neuraylib::IMdl_compiler> m_compiler;
    mi::base::Handle<mi::neuraylib::IMdl_archive_api> m_archive_api;
    mi::base::Handle<mi::neuraylib::IMdl_discovery_api> m_discovery;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<mi::base::ILogger> m_logger;
    Mdl_cache* m_cache;
};


#endif