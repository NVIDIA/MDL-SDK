/***************************************************************************************************
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
 **************************************************************************************************/
/// \file options.h

#pragma once

#include <mi/mdl_sdk.h>
#include <vector>
#include <string>
#include <set>

namespace mdlm
{
    class Option;
    typedef std::vector<Option> Option_set_type;
    class Option_set : public Option_set_type
    {
    public:
        // Find all options with that name
        Option_set find_options_from_name(const std::string & name) const;
        void output_usage(std::ostream & ostr, bool commands, unsigned int indent = 0) const;
        void output_values(std::ostream & ostr, unsigned int indent = 0) const;
    };
    class Option_parser;

    class Option
    {
    protected:
        // Different names for the option, e.g. -h, --help, ...
        std::set<std::string> m_names;
        // List of values for this option
        std::vector<std::string> m_value;
        int m_id;
        int m_number_of_parameters;
        Option_parser * m_options;
        bool m_can_appear_mulitple_times;
        bool m_is_command;
        std::string m_parameter_helper_string; //"arg" by default but can be changed
        std::vector<std::string> m_help_strings;
        std::string m_command_string;

    public:
        Option()
            : m_id(-1)
            , m_number_of_parameters(-1)
            , m_options(NULL)
            , m_can_appear_mulitple_times(false)
            , m_is_command(false)
            , m_parameter_helper_string("arg")
        {}
        Option(int id)
            : m_id(id)
            , m_options(NULL)
            , m_can_appear_mulitple_times(false)
            , m_is_command(false)
            , m_parameter_helper_string("arg")
        {}
        ~Option();
        Option(const Option & other); // Copy ctor
        Option & operator=(const Option&); // Assignement operator
        bool is_valid() const { return m_id != -1; }
        void set_can_appear_mulitple_times(bool flag) { m_can_appear_mulitple_times = flag; }
        void set_is_command(bool flag) { m_is_command = flag; }
        bool get_is_command() const { return m_is_command; }
        void add_help_string(const std::string & str) { m_help_strings.push_back(str); }
        std::string get_help_string() const
        {
            if (!m_help_strings.empty())
            {
                return m_help_strings[0];
            }
            else
            {
                return "";
            }
        }
        const std::vector<std::string> & get_help_strings() const { return m_help_strings; }
        void set_command_string(const std::string & str) { m_command_string = str; }
        std::string get_command_string() const { return m_command_string; }
        void add_name(const std::string & name)
        {
            m_names.insert(name);
        }
        bool has_name(const std::string & name) const
        {
            return (m_names.find(name) != m_names.end());
        }
        int id() const { return m_id; }
        virtual bool consume_arguments(
            const std::vector<std::string> & command_line
            , std::vector<std::string>::const_iterator & it
        );
        void set_number_of_parameters(int n)
        {
            m_number_of_parameters = n;
        }
        int get_number_of_parameters() const
        {
            return m_number_of_parameters;
        }
        void add_option(const Option & option);
        void output_usage(std::ostream & ostr) const;
        void output_values(std::ostream & ostr) const;
        /// return all names enclosed in backets separated with pipe [-h | --help]
        std::string get_names() const;
        Option_parser * get_options() const { return m_options; }
        const std::vector<std::string> & value() const { return m_value; }
        void set_parameter_helper_string(const std::string & str)
        {
            m_parameter_helper_string = str;
        }
    };

    class Option_parser
    {
    protected:
        Option_set m_known_options;
        Option_set m_options_set;

    private:
        // Return the single next option found on the command line
        // In case of ambiguity, report an error and try to resolve
        Option next_option(
            const std::vector<std::string> & command_line
            , std::vector<std::string>::const_iterator & it
        ) const;

    public:
        Option_parser() {}
        virtual ~Option_parser() {}
        bool consume_arguments(
            const std::vector<std::string> & command_line
            , std::vector<std::string>::const_iterator & it
        );
        bool is_set(const int & id, Option_set & option) const;
        bool is_command_set(Option_set & option) const;
        void add_known_option(const Option & option);
        void output_usage(std::ostream & ostr) const;
        const Option_set & get_known_options() const { return m_known_options; }
        const Option_set & get_options_set() const { return m_options_set; }
        void output_options_set(std::ostream & ostr) const;
        void cleanup() { m_options_set.clear(); }
    };

    class MDLM_option_parser : public Option_parser
    {
    public:
        static bool Test();
    public:
        typedef enum
        {
              HELP
            , HELP_CMD
            , VERBOSE
            , QUIET
            , ADD_PATH
            , NO_STD_PATH
            , FORCE
            , DRY_RUN
            , COMPATIBILITY
            , INSTALL_ARCHIVE
            , CREATE_ARCHIVE
            , SHOW_ARCHIVE
            , SET_KEY
            , LIST_ALL
            , LIST
            , REMOVE
            , DEPENDS
            , EXTRACT
            , CREATE_MDLE
            , CREATE_MDLE_ADD_USER_FILE
            , CHECK_MDLE

        } OPTIONS_AND_COMMAND;

    public:
        MDLM_option_parser();
    };

} // namespace mdlm
