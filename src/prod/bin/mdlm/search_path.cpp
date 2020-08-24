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
#include "search_path.h"
#include "errors.h"
#include "util.h"
using namespace mdlm;
using std::vector;
using std::string;

// Initialize
Search_path::Search_path(const mi::neuraylib::INeuray * neuray)
{
    m_mdl_config = neuray->get_api_component<mi::neuraylib::IMdl_configuration>();
}

Search_path::~Search_path()
{
}

// Take a snapshot of the current search pathes
mi::Sint32 Search_path::snapshot()
{
    if (!m_mdl_config)
    {
        return -1;
    }
    clear_snapshot();
    for (mi::Size i = 0; i < m_mdl_config->get_mdl_paths_length(); i++)
    {
        m_paths.push_back(m_mdl_config->get_mdl_path(i)->get_c_str());
    }
    return 0;
}

// Restore search path since last snapshot
// If no snapshot was ever taken, nothing to restore 
mi::Sint32 Search_path::restore_snapshot() const
{
    if (!m_mdl_config)
    {
        return -1;
    }
    mi::Sint32 rtn = 0;
    for (vector<string>::const_iterator it = m_paths.begin();
        it != m_paths.end();
        it++)
    {
        mi::Sint32 success = m_mdl_config->add_mdl_path(it->c_str());
        check_success3(success == 0, Errors::ERR_MODULE_PATH_FAILURE, it->c_str());
        rtn |= success;
    }
    return rtn;
}

mi::Sint32 Search_path::clear_snapshot()
{
    m_paths.clear();
    return 0;
}

// Add to module path
mi::Sint32 Search_path::add_module_path(const string & directory)
{
    mi::Sint32 success = m_mdl_config->add_mdl_path(directory.c_str());
    check_success3(success == 0, Errors::ERR_MODULE_PATH_FAILURE, directory.c_str());
    clear_snapshot();// Clear the current snapshot
    return success;
}

bool Search_path::find_module_path(const std::string & directory) const
{
    std::string test(directory);
    for (vector<string>::const_iterator it = m_paths.begin();
        it != m_paths.end();
        it++)
    {
        std::string current(*it);
        if (Util::equivalent(current, test))
        {
            return true;
        }
    }
    return false;
}

// Remove all module paths
mi::Sint32 Search_path::clear_module_paths()
{
    m_mdl_config->clear_mdl_paths();
    clear_snapshot();// Clear the current snapshot
    return 0;
}

void Search_path::log_debug() const
{
    Util::log_debug("+++++++++++++++++Search_path::log_debug PATH");
    for (vector<string>::const_iterator it = m_paths.begin();
        it != m_paths.end();
        it++)
    {
        Util::log_debug("Search_path::log_debug: path: " + *it);
    }
}
