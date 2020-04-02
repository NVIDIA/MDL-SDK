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
 ******************************************************************************/
#pragma once

#include <string>
#include <vector>
#include "command.h"

namespace mi
{
    namespace neuraylib
    {
        class IMdl_module_info;
        class IMdl_package_info;
    }
}
namespace i18n
{
    class Create_xliff_command : public Command
    {
        std::string m_locale;
        std::vector<std::string> m_modules;
        std::vector<std::string> m_packages;
        bool m_recursive = true; // Traverse sub-packages if set to true
        bool m_output_context = true; // Ouput group/context information
        bool m_dry_run = false; // If true, do not create XLIFF but report what would be done
        bool m_force = false; // If true, force overwriting XLIFF files
    private:
        int handle_module(const std::string & module);
        int handle_package(const std::string & package);
        int handle_modules(
            const std::vector<std::string> & modules, const std::string & filename
            , const std::string & qualified_name);

        // Log information about the current settings.
        void log_settings() const;
        bool check_file(const std::string & filename) const;
        
        // Build the filename corresponding to the given module
        // in the current locale.
        bool build_filename(
            const mi::neuraylib::IMdl_module_info * module
            , std::string & filename
        ) const;
        
        // Build the filename corresponding to the given package
        // and the given search path in the current locale.
        // Search path is necessary when a package lives in multiple search path.
        // e.g. ::nvidia is usually both in USER and SYSTEM
        bool build_filename(
              const mi::neuraylib::IMdl_package_info * module
            , const std::string & search_path
            , std::string & filename
        ) const;
    public:
        Create_xliff_command();
        void add_module(const std::string & module)
        {
            m_modules.push_back(module);
        }
        void add_package(const std::string & package)
        {
            m_packages.push_back(package);
        }
        void set_locale(const std::string & locale)
        {
            m_locale = locale;
        }
        void set_recursive(bool recursive)
        {
            m_recursive = recursive;
        }
        void set_output_context(bool output_context)
        {
            m_output_context = output_context;
        }
        void set_dry_run(bool dry_run)
        {
            m_dry_run = dry_run;
        }
        void set_force(bool force)
        {
            m_force = force;
        }
        int execute();
    public:
        typedef enum {
              UNSPECIFIED_FAILURE = -1
            , MISSING_LOCALE = -2
            , MISSING_PACKAGE_OR_MODULE = -3
            , ONLY_ONE_PACKAGE_OR_MODULE = -4
            , PACKAGE_NOT_FOUND = -5
            , ARCHIVE_NOT_SUPPORTED = -6
            , PACKAGE_IN_MANY_SEARCH_PATH = -7
            , MODULE_NOT_FOUND = -8
            , FILE_ALREADY_EXISTS = -9
            , FAILED_TO_CREATE_XLIFF_FILE = -10
            , SUCCESS = 0

        } RETURN_CODE;
    };

} // namespace i18n
