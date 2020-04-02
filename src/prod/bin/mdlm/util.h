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
    /// Some utility routines
    class Util
    {
    public:
        /// File or directory helper
        class File
        {
        protected:
            std::string m_path;
        public:
            File(const std::string& path);
            virtual ~File()
            {}
            bool exist() const;
            bool is_file() const;
            bool is_directory() const;
            bool is_readable() const;
            bool is_writable() const;
            // File size for file or number of elements in a directory
            bool size(mi::Sint64 & size) const;
            // Directory or file is empty
            bool is_empty() const;
            bool remove() const;
            // Returns this directory if this File is a directory
            // or the parent file directory if this File is a file
            std::string get_directory() const;
        public:
            static bool Test();
            /// Convert symbolic name directory (e.g. SYSTEM, USER) to
            /// their actual directory location on disk.
            /// Return true if the input directory was modified
            /// Return false if the input directory was not modified
            static bool convert_symbolic_directory(std::string & directory);
        public:
            static File SYSTEM;
            static File USER;
        };
    public:
        /// Utility routines for logging messages
        /// These levels are controlled by the verbosity
        static void log_fatal(const std::string & msg);
        static void log_error(const std::string & msg);
        static void log_warning(const std::string & msg);
        static void log_info(const std::string & msg);
        static void log_verbose(const std::string & msg);
        static void log_debug(const std::string & msg);
        static void log(const std::string & msg, mi::base::Message_severity severity);

        /// Utility routines for logging messages
        /// This is not controlled by verbosity
        static void log_report(const std::string & msg);

        /// string basename (Unix-like)
        static std::string basename(const std::string& path);
        
        /// string extension
        static std::string extension(const std::string& path);

        /// Get programe name from path, remove path and .exe extension if needed
        static std::string get_program_name(const std::string & path);

        /// File exists and is readable?
        static bool file_is_readable(const std::string & path);

        /// The input directory exists and is writable with current permissions
        static bool directory_is_writable(const std::string & path);

        /// Create a temporary directory
        static bool create_name_in_temp_directory(std::string & new_directory);

        /// Create an empty directory
        static bool create_directory(const std::string & new_directory);

        /// Delete a directory
        static bool delete_file_or_directory(const std::string & directory, bool recursive = false);

        /// Return true if string ends with another string
        static bool has_ending(std::string const &fullString, std::string const &ending);

        /// Get the SYSTEM directory (e.g. "C:\ProgramData\NVIDIA Corporation\mdl" on Windows)
        static std::string get_mdl_system_directory();

        /// Get the USER directory (e.g. "C:\Users\%USERNAME%\Documents\mdl" on Windows)
        static std::string get_mdl_user_directory();

        /// Normalize path, see MI::HAL::Ospath::normpath()
        static std::string normalize(const std::string & path);

        /// Remove directories wich are duplicates in the input list of directories.
        /// Implemented using boost Path and boost::filesystem::equivalent
        static bool remove_duplicate_directories(std::vector<std::string> & directories);

        /// Copy file
        static bool copy_file(std::string const & source, std::string const & destination);

        /// Aray of char * to vector of strings
        ///
        static void array_to_vector(int ac, char *av[], std::vector<std::string> & v);

        /// Valid MDL identifier
        static bool is_valid_mdl_identifier(const std::string & identifier);
        static bool is_valid_module_name(const std::string & identifier);
        static bool is_valid_archive_name(const std::string & identifier);

        static bool equivalent(const std::string & file1, const std::string & file2);
        static std::string path_appends(const std::string & path, const std::string & end);
        static std::string stem(const std::string & path);

        // Create a file name which does not already exist
        //static std::string unique_path(const std::string & model);

        // Create a file name which does not already exist in the given folder
        static std::string unique_file_in_folder(const std::string & folder);

        static std::string temp_directory_path();

        static std::vector<std::string> split(const std::string &s, char delim);
    };

    template<class T> std::string to_string(const T & value)
    {
        std::stringstream str;
        str << value;
        return str.str();
    }

    template<typename Out>
    void split(const std::string &s, char delim, Out result)
    {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            *(result++) = item;
        }
    }

} // namespace mdlm
