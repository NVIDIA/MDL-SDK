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
#include <set>
#include <map>
#include "archive.h"

namespace mdlm
{
    /// Command class
    class Command
    {
    private:
        Command(const Command &); // prevent copy ctor
    public:
        Command() {}
        virtual ~Command() {}
        virtual int execute();
    };

    /// Archive compatibilty command
    /// 
    /// Compatibity command checks that a new archive does not break compatibility
    /// with an old installed one
    class Compatibility : public Command
    {
    public:
        typedef enum
        {
              // Negative values for errors
              UNSPECIFIED_FAILURE = -3
            , INVALID_ARCHIVE = -2
            , ARCHIVE_DIFFERENT_NAME = -1
              // Positive values for success
            , COMPATIBLE = 0
            , COMPATIBLE_SAME_VERSION
            , NOT_COMPATIBLE
            , UNDEFINED

        } COMPATIBILITY;
    protected:
        std::string m_old_archive;
        std::string m_new_archive;
        COMPATIBILITY m_compatible;
        // Do not inspect inside archives, only test versions
        bool m_test_version_only;
        // Output Compatibility result
        void report_compatibility_result() const;
    public:
        Compatibility(const std::string & old_archive, const std::string & new_archive);

        /// \return
        ///		- See COMPATIBILITY
        /// COMPATIBLE_SAME_VERSION is returned same as COMPATIBLE 
        int execute() override;

        /// \return status of the compatibility 
        /// Need ot invoke execute() first.
        COMPATIBILITY compatibility() const { return m_compatible; }

        /// Will only test versions if set to trus
        void set_test_version_only(bool flag) { m_test_version_only = flag; }
    };

    /// Create archive command
    /// 
    class Create_archive : public Command
    {
    public:
        typedef enum {
              UNSPECIFIED_FAILURE = -1
            , SUCCESS = 0

        } RETURN_CODE;
    protected:
        std::string m_mdl_directory;
        std::string m_archive;
        typedef std::pair<std::string, std::string> KeyValuePair;
        std::vector<KeyValuePair> m_keys;
    public:
        Create_archive(const std::string & mdl_directory, const std::string & archive);

        /// \return
        ///		- SUCCESS: Success
        ///		- UNSPECIFIED_FAILURE: Unspecified failure.
        int execute() override;

        /// \return
        ///		- SUCCESS: Success
        ///		- UNSPECIFIED_FAILURE: Unspecified failure.
        /// key_value format should be key=value, otherwise the key/value pair is not added.
        int add_key_value(const std::string & key_value);
    };

    /// Archive install command
    /// 
    class Install : public Command
    {
    public:
        typedef enum {
              INVALID_ARCHIVE = -2
            , INVALID_MDL_PATH = -1
            , SUCCESS = 0

        } RETURN_CODE;
    protected:
        std::string m_archive;
        std::string m_mdl_directory; // Where to install
        bool m_force; // Force installation
        bool m_dryrun; // Does not install but run checks

    private:
        /// Test if the archive to be installed is already installed in the given directory
        /// If an archive exists, test the compatibility of the new archive with the 
        /// installed one.
        /// Return Compatibility::COMPATIBILITY or UNDEFINED if the archive was not found in
        /// directory
        Compatibility::COMPATIBILITY test_compatibility(const std::string & directory) const;
    public:
        Install(const std::string & archive, const std::string & mdl_path);

        /// Force installation of archive, in case of version conflict, ...
        void set_force_installation(bool force) { m_force = force; }

        /// Dry run, do not copy but run checks
        void set_dryrun(bool dryrun) { m_dryrun = dryrun; }

        /// \return
        ///		- SUCCESS: Success
        ///		- UNSPECIFIED_FAILURE: Unspecified failure.
        ///		- INVALID_ARCHIVE: Invalid input archive
        ///		- INVALID_MDL_PATH: Invalid MDL path
        int execute() override;
    };

    /// Archive test command
    /// 
    class Show_archive : public Command
    {
    public:
        typedef enum {
              INVALID_ARCHIVE = -1
            , SUCCESS = 0

        } RETURN_CODE;
    protected:
        std::string m_archive;
        std::set<std::string> m_filter_field;
        std::multimap<std::string, std::string> m_manifest;
        bool m_report = true;
    private:
        bool list_field(const std::string & field)
        {
            if (m_filter_field.empty())
            {
                return true;
            }
            return(m_filter_field.find(field) != m_filter_field.end());
        }
    public:
        Show_archive(const std::string & archive);

        /// Can filter on some fields, output will only contain the given field
        void add_filter_field(const std::string & field)
        {
            m_filter_field.insert(field);
        }

        const std::multimap<std::string, std::string> & get_manifest()const { return m_manifest; }

        /// \return
        ///		- SUCCESS: Success
        ///		- UNSPECIFIED_FAILURE: Unspecified failure.
        ///		- INVALID_ARCHIVE: Invalid input archive
        ///		- INVALID_MDL_PATH: Invalid MDL path
        int execute() override;

        /// Control wheter to report or not the result
        void set_report(bool report)
        {
            m_report = report;
        }
    };

    /// List archive dependencies command
    /// 
    class List_dependencies : public Show_archive
    {
    public:
        static const std::string dependency_keyword;

    public:
        List_dependencies(const std::string & archive);

        void get_dependencies(std::multimap<Archive::NAME, Version> & dependencies) const;
    };

    /// Extract archive command
    /// 
    class Extract : public Command
    {
    public:
        typedef enum
        {
              INVALID_ARCHIVE = -1
            , CAN_NOT_CREATE_PATH = -2
            , SUCCESS = 0

        } RETURN_CODE;
    protected:
        std::string m_archive;
        std::string m_directory; // Where to install
        bool m_force; // Force extract

    public:
        Extract(const std::string & archive, const std::string & path);

        /// Force extract of archive
        void set_force_extract(bool force)
        {
            m_force = force;
        }

        /// \return
        ///		- SUCCESS: Success
        ///		- UNSPECIFIED_FAILURE: Unspecified failure.
        ///             - PATH_EXISTS: A file or directory with same name exists
        ///		- INVALID_ARCHIVE: Invalid input archive
        ///		- INVALID_PATH: Invalid MDL path
        int execute() override;
    };

    /// Create mdle command
    /// 
    class Create_mdle : public Command
    {
    public:
        typedef enum {
              UNSPECIFIED_FAILURE = -1
            , SUCCESS = 0

        } RETURN_CODE;
    protected:
        std::string m_prototype;
        std::string m_mdle;
        std::vector<std::pair<std::string, std::string>> m_user_files;
    public:
        Create_mdle(const std::string& prototype, const std::string& mdle);

        /// \return
        ///		- SUCCESS: Success
        ///		- UNSPECIFIED_FAILURE: Unspecified failure.
        int execute() override;

        /// \return
        ///		- SUCCESS: Success
        ///		- UNSPECIFIED_FAILURE: Unspecified failure.
        /// add a user file to the MDLE 
        int add_user_file(const std::string& source_path, const std::string& target_path);
    };


    /// Check mdle command
    /// 
    class Check_mdle : public Command
    {
    public:
        typedef enum
        {
            UNSPECIFIED_FAILURE = -1
            , SUCCESS = 0

        } RETURN_CODE;
    protected:
        std::string m_mdle;
    public:
        Check_mdle(const std::string& mdle);

        /// \return
        ///		- SUCCESS: Success
        ///		- UNSPECIFIED_FAILURE: Unspecified failure.
        int execute() override;
    };
    /// Help command
    /// 
    class Help : public Command
    {
        std::string m_command;
    public:
        Help(const std::string & command)
            : m_command(command)
        {}

        /// \return
        ///		- SUCCESS: Success
        ///		- UNSPECIFIED_FAILURE: Unspecified failure.
        ///		- INVALID_ARCHIVE: Invalid input archive
        ///		- INVALID_MDL_PATH: Invalid MDL path
        int execute() override;
    };

    /// List command
    /// 
    class List_cmd : public Command
    {
    public:
        typedef enum {
            UNSPECIFIED_FAILURE = -1
            , SUCCESS = 0

        } RETURN_CODE;

    public:
        class List_result
        {
        public:
            std::vector<mdlm::Archive> m_archives;
        };

    private:
        std::string m_archive_name;
        List_result m_result;

    public:
        List_cmd(const std::string & archive);

        /// List all the locations for the given Archive
        /// \return
        ///		- SUCCESS: Success
        ///		- UNSPECIFIED_FAILURE: Unspecified failure.
        int execute() override;

        List_result get_result() const
        {
            return m_result;
        }
    };

    /// Remove command
    /// 
    class Remove_cmd : public Command
    {
    public:
        typedef enum {
            UNSPECIFIED_FAILURE = -10
            , INVALID_ARCHIVE = -3
            , ARCHIVE_NOT_FOUND = -2
            , SUCCESS = 0

        } RETURN_CODE;

    private:
        std::string m_archive_name;
    
    private:
        int find_archive(Archive & archive);

    public:
        Remove_cmd(const std::string & archive);

        /// List all the locations for the given Archive
        /// \return
        ///		- SUCCESS: Success
        ///     - ARCHIVE_NOT_FOUND: Archive does not exist
        ///     - INVALID_ARCHIVE: Archive is not valid
        ///		- UNSPECIFIED_FAILURE: Unspecified failure.
        int execute() override;
    };

    /// Command class
    class Option_set;
    class Command_factory
    {
    public:
        static Command * build_command(const Option_set & option);
    };

} // namespace mdlm
