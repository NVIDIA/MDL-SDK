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

#include <mi/mdl_sdk.h>
#include <string>
#include <vector>

namespace mdlm
{
    class Version;

    /// MDL Archive helper.
    class Archive
    {
    public:
        typedef std::string NAME;
    public:
        /// Helpers
        static const std::string extension; // ".mdr"
        /// If input has the .mdr extension, return input unchanged
        /// If input does not have the .mdr extension, return input + ".mdr"
        static std::string with_extension(const std::string & input);
    protected:
        std::string m_archive_file;

    private:
        bool is_valid_archive_name() const;

        /// Get all the dependencies for the current archive
        std::vector<std::pair<std::string, Version>> dependencies() const;
    public:
        static bool Test();

    public:
        /// Initialize archive object from path to .mdr file.
        Archive(const std::string & archive_file);

        /// Destructor, cleanup, remove temp directory if needed.
        ~Archive();

        /// Archive file exists and is a valid MDL archive.
        /// Tests are:
        ///   1- Extension is .mdr
        ///   2- File is readable
        ///   3- We are able to retrieve the archive version/manifest
        bool is_valid() const;

        /// Get archive version.
        /// \return
        ///		-  0: Success
        ///		-  -1: Failure
        mi::Sint32 get_version(Version & version) const;

        /// Return archive version.
        Version get_version() const;

        /// Get archive first value given a key.
        /// \return
        ///		-  0: Success
        ///		-  -1: Failure to retrieve manifest from archive
        ///		-  -2: Key not found
        mi::Sint32 get_value(const std::string & key, std::string & value) const
        {
            std::vector<std::string> values;
            const mi::Sint32 rtn = get_values(key, values, false /*all_values*/);
            if (rtn == 0 && ! values.empty())
            {
                value = values[0];
            }
            return rtn;
        }

        /// Get all (by default) archive values given a key.
        /// If all_values is false, then retrieve only the first value
        /// \return
        ///		-  0: Success
        ///		-  -1: Failure
        ///		-  -2: Key not found
        mi::Sint32 get_values(
            const std::string & key
            , std::vector<std::string> & values
            , bool all_values = true) const;

        /// Get archive short name, e.g. archive.mdr.
        std::string base_name() const;

        /// Get archive long name with possibly directory information.
        std::string full_name() const;

        /// Get archive short name, without trailing .mdr extension
        std::string stem() const;

        /// Extract archive to the given directory.
        mi::Sint32 extract_to_directory(const std::string & directory) const;

        /// Return whether the archive might conflict with already installed archives,
        /// modules, package from the given location.
        bool conflict(const std::string & directory) const;

        /// Are all the dependencies for this archive already installed?
        bool all_dependencies_are_installed() const;
    };

} // namespace mdlm
