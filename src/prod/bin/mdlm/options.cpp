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
///

#include "options.h"
#include "application.h"
#include "util.h"
#include <iomanip>
#include <map>
#include <iostream>
using namespace mdlm;
using std::vector;
using std::string;
using std::set;
using std::ostream;
using std::endl;
using std::cout;
using std::ostringstream;
using std::map;

bool MDLM_option_parser::Test()
{
    MDLM_option_parser optionParser;

    //cout << "=========================== TEST OPTIONS" << endl;

    Option o;
    Option o1(o); // Copy ctor
    Option o2 = o1; // Copy ctor
    Option o3 = Option(18);

    // Test cmmand line parsing
    vector<string> command_line;
    command_line.push_back("mdlm");
    command_line.push_back("-v");
    command_line.push_back("6");
    command_line.push_back("-h");
    command_line.push_back("install");
    command_line.push_back("--force");
    command_line.push_back("automotive_catalog.mdr");
    vector<string>::const_iterator it = command_line.begin();

    for (it = command_line.begin();
        !optionParser.consume_arguments(command_line, it) && it != command_line.end();
        it++)
    {
    }

    Option_set optionset;
    if (optionParser.is_set(MDLM_option_parser::HELP, optionset))
    {

    }
    if (optionParser.is_set(MDLM_option_parser::ADD_PATH, optionset))
    {

    }
    if (optionParser.is_set(MDLM_option_parser::INSTALL_ARCHIVE, optionset))
    {

    }

    //optionParser.output_usage(cout);

    return true;
}

Option::Option(const Option & rhs)
    : m_options(NULL)
{
    * this = rhs;
}

Option::~Option()
{
    if (m_options)
    {
        delete m_options;
    }
}

Option & Option::operator=(const Option& rhs)
{
    if (this != &rhs)
    {
        m_names = rhs.m_names;
        m_value = rhs.m_value;
        m_id = rhs.m_id;
        m_number_of_parameters = rhs.m_number_of_parameters;
        m_parameter_helper_string = rhs.m_parameter_helper_string;
        delete m_options;
        m_options = NULL;
        if (rhs.m_options)
        {
            m_options = new Option_parser;
            *(m_options) = *(rhs.m_options);
        }
        m_can_appear_mulitple_times = rhs.m_can_appear_mulitple_times;
        m_is_command = rhs.m_is_command;
        m_help_strings = rhs.m_help_strings;
        m_command_string = rhs.m_command_string;
    }
    return *this;
}

bool Option::consume_arguments(
    const vector<string> & command_line
    , vector<string>::const_iterator & it
)
{
    m_value.clear();

    // My options
    if (m_options)
    {
        m_options->cleanup();
        m_options->consume_arguments(command_line, it);
    }

    int number_of_parameters = m_number_of_parameters;
    while (number_of_parameters--)
    {
        if (it == command_line.end())
        {
            return false;
        }
        else
        {
            m_value.push_back(*it);
            it++;
        }
    }
    return true;
}

void Option::add_option(const Option & option)
{
    if (!m_options)
    {
        m_options = new Option_parser;
    }
    m_options->add_known_option(option);
}

void Option::output_usage(ostream & ostr) const
{
    if (!m_is_command)
    {
        Util::log_warning("Warning: usage output not implemented for non-commands");
        return;
    }

    size_t indent = 2;

    cout << endl << "Command \"" << get_command_string() << "\" help:" << endl;
    cout << string(indent, ' ') << std::left
        << "mdlm [<Global option> ...] "
        << get_command_string() << endl << endl;

    const vector<string> & hs = get_help_strings();
    vector<string>::const_iterator its;
    for (its = hs.begin(); its != hs.end(); its++)
    {
        cout << string(indent * 2, ' ') << *its << endl;
    }

    // Command options
    Option_parser * op = get_options();
    if (op)
    {
        cout << endl;

        cout << string(indent * 2, ' ') << "Global options:" << endl;

        const Option_set & options = op->get_known_options();
        options.output_usage(ostr, false/*commands*/, indent * 4);
    }
}

void Option::output_values(std::ostream & ostr) const
{
    ostr << get_names() << endl;
    std::vector<std::string>::const_iterator it;
    for (it = m_value.begin(); it != m_value.end(); it++)
    {
        ostr << *it << endl;
    }
}

std::string Option::get_names() const
{
    if (m_is_command)
    {
        return *(m_names.begin());
    }
    string rtn;
    //string rtn("[");
    //if (m_can_appear_mulitple_times)
    //{
    //    rtn += "[";
    //}
    std::set<std::string>::const_iterator it;
    size_t i = 0;
    for (i = 0, it = m_names.begin(); it != m_names.end(); it++, i++)
    {
        bool not_first = (i != 0);
        bool not_last = (i != m_names.size());

        if (not_first && not_last)
        {
            rtn += " | ";
        }
        rtn += *it;
    }
    //    rtn += "]";
    int np = m_number_of_parameters;
    while (np--)
    {
        rtn += " <" + m_parameter_helper_string + ">";
    }
    //if (m_can_appear_mulitple_times)
    //{
    //    rtn += "][...]";
    //}
    return rtn;
}

Option_set Option_set::find_options_from_name(const std::string & name) const
{
    Option_set rtn;
    Option_set_type::const_iterator it;
    for (it = begin(); it != end(); it++)
    {
        if (it->has_name(name))
        {
            rtn.push_back( *it );
        }
    }
    return rtn;
}

void Option_set::output_usage(std::ostream & ostr, bool commands, unsigned int indent) const
{
    size_t longuest = 0;
    size_t extra = 2;
    //<- indent -><--------- longuest --------------><- extra -> 
    //            [[--search - path | -p] <arg>][...]           Add path to MDL search path
    //            [--verbose | -v] <arg>                        Set verbosity level
    Option_set::const_iterator it;
    for (it = begin(); it != end(); it++)
    {
        if (it->get_is_command() == commands)
        {
            if (commands)
            {
                longuest = std::max(longuest, it->get_command_string().size());
            }
            else
            {
                longuest = std::max(longuest, it->get_names().size());
            }
        }
    }
    for (it = begin(); it != end(); it++)
    {
        if (it->get_is_command() == commands)
        {
            if (commands)
            {
                cout << string(indent, ' ') << std::left <<
                    std::setw(longuest) << it->get_command_string() <<
                    string(extra, ' ') <<
                    it->get_help_string() << endl;

            }
            else
            {
                cout << string(indent, ' ') << std::left <<
                    std::setw(longuest) << it->get_names() <<
                    string(extra, ' ') <<
                    it->get_help_string() << endl;
            }

            if (!commands)
            {
                // Only display long strings for non commands
                const vector<string> & hs = it->get_help_strings();
                if (hs.size() > 1)
                {
                    vector<string>::const_iterator its = hs.begin();
                    its++;
                    for (; its != hs.end(); its++)
                    {
                        cout << string(indent + longuest + extra, ' ') << *its << endl;
                    }
                }
            }
        }
    }
}

void Option_set::output_values(std::ostream & ostr, unsigned int indent) const
{
    Option_set::const_iterator it;
    for (it = begin(); it != end(); it++)
    {
        //        if (!it->get_is_command())
        {
            it->output_values(ostr);
            ostr << endl;
        }
    }
}

void Option_parser::add_known_option(const Option & option)
{
    m_known_options.push_back(option);
}

void Option_parser::output_usage(ostream & ostr) const
{
    // <------------------------------ maxline ------------------------------------------>
    // <---col 1--><------------------ col 2 -------------------------------------------->
    // Usage: mdlm [-h | --help][-v | --verbosity <level>][-q | --quiet][-n | --nostdpath]
    //             [[-p | --search - path <path>]][�]
    //             <command>[<args>]
    //
    // col 1 + col 2 < maxline
    int maxline = 120;
    ostringstream str;
    str << "Usage: " << Application::theApp().name() << " [<Global option> ...] <command> [<arg> ...]";
    class Line
    {
        size_t m_maxlen;// for each line
        string m_text;
    public:
        Line(size_t maxlen)
            : m_maxlen(maxlen)
        {}
        bool add_text(const string & text)
        {
            if (m_text.size() + text.size() < m_maxlen)
            {
                m_text += text;
                return true;
            }
            return false;
        }
        const string & str() const { return m_text; }
        void reset()
        {
            m_text.clear();
        }
        void indent(size_t indent)
        {
            m_text += string(indent, ' ');
        }
    };

    Line line(maxline);
    line.add_text(str.str());
    Option_set::const_iterator it;
    ostr << line.str();
    line.reset();

    ostr << line.str() << endl;
    ostr << endl;

    //Indent = 2 spaces
    // 
    //  MDL manager
    //
    size_t indent = 2;
    string application_help("MDL Manager");

    line.reset();
    line.indent(indent);
    line.add_text(application_help);
    ostr << line.str() << endl << endl;

    //Options :
    ostr << "Global options:" << endl;

    m_known_options.output_usage(ostr, false/*commands*/, indent);

    ostr << endl;

    //Local Options :
    ostr << endl;
    ostr << "Local command options:" << endl;
    line.reset();
    line.indent(indent);
    line.add_text("For descriptions of individual local command options use \"help <command>\"");
    ostr << line.str() << endl << endl;

    //Commands:
    ostr << "Commands:" << endl;

    m_known_options.output_usage(ostr, true/*commands*/, indent);
}

void Option_parser::output_options_set(std::ostream & ostr) const
{
    m_options_set.output_values(ostr);
}

MDLM_option_parser::MDLM_option_parser()
{
    Option option;
    option = Option(VERBOSE);
    option.set_number_of_parameters(1);
    option.add_name("-v");
    option.add_name("--verbose");
    option.add_help_string("Set verbosity level");
    m_known_options.push_back(option);

    option = Option(ADD_PATH);
    option.set_number_of_parameters(1);
    option.add_name("-p");
    option.add_name("--search-path");
    option.set_can_appear_mulitple_times(true);
    option.add_help_string("Add path to the MDL search paths");
    option.add_help_string(
        "Note: SYSTEM and USER path are added by default to MDL search paths");
    option.add_help_string("SYSTEM is \"" + Util::get_mdl_system_directory() + "\"");
    option.add_help_string("USER is \"" + Util::get_mdl_user_directory() + "\"");
    m_known_options.push_back(option);

    option = Option(QUIET);
    option.set_number_of_parameters(0);
    option.add_name("-q");
    option.add_name("--quiet");
    option.add_help_string("Suppress all messages");
    m_known_options.push_back(option);

    option = Option(NO_STD_PATH);
    option.set_number_of_parameters(0);
    option.add_name("-n");
    option.add_name("--nostdpath");
    option.add_help_string("Do not add SYSTEM and USER path to MDL search paths");
    m_known_options.push_back(option);

    {
        option = Option(HELP);
        option.set_number_of_parameters(0);
        option.add_name("help");
        option.set_is_command(true);
        option.add_help_string("Display help information");
        option.set_command_string("help");
        m_known_options.push_back(option);
    }
    {
        option = Option(HELP_CMD);
        option.set_number_of_parameters(1);
        option.add_name("help");
        option.set_is_command(true);
        option.add_help_string("Display help information for the given command");
        option.set_command_string("help <command>");
        m_known_options.push_back(option);
    }
    {
        Option force = Option(FORCE);
        force.set_number_of_parameters(0);
        force.add_name("-f");
        force.add_name("--force");
        force.add_help_string("Force installation even if errors or warnings");

        Option dryrun = Option(DRY_RUN);
        dryrun.set_number_of_parameters(0);
        dryrun.add_name("-d");
        dryrun.add_name("--dry-run");
        dryrun.add_help_string("Do not install but perform compatibility checks");

        option = Option(INSTALL_ARCHIVE);
        option.set_number_of_parameters(2);
        option.add_name("install");
        option.add_option(force);
        option.add_option(dryrun);
        option.set_is_command(true);
        option.add_help_string("Install archive in the given destination directory");
        option.add_help_string("Only install if it is safe to do so");
        option.add_help_string("Check possible conflicts with installed packages, modules, archives");
        option.add_help_string("");
        option.add_help_string("If an old archive is found at a higher priority path:");
        option.add_help_string("New and old archives have compatible versions: No install, error");
        option.add_help_string("New and old archives have same versions: No install, warning");
        option.add_help_string("New and old archives have incompatible versions: No install, warning");
        option.add_help_string("");
        option.add_help_string("If an old archive is found in the destination path:");
        option.add_help_string("New and old archives have compatible versions: Install");
        option.add_help_string("New and old archives have same versions: No install, warning");
        option.add_help_string("New and old archives have incompatible versions: No install, warning");
        option.add_help_string("");
        option.add_help_string("If an old archive is found at a lower priority path:");
        option.add_help_string("New and old archives have compatible versions: No install, warning");
        option.add_help_string("New and old archives have same versions: No install, warning");
        option.add_help_string("New and old archives have incompatible versions: No install, warning");
        option.set_command_string(
            "install [<Local options>] <archive> <SYSTEM | USER | directory>");
        m_known_options.push_back(option);
    }
    option = Option(COMPATIBILITY);
    option.set_number_of_parameters(2);
    option.add_name("check");
    option.set_is_command(true);
    option.add_help_string("Compare 2 archives");
    option.set_command_string("check <old archive> <new archive>");
    m_known_options.push_back(option);

    {
        Option key = Option(SET_KEY);
        key.set_number_of_parameters(1);
        key.add_name("-k");
        key.add_name("--key");
        key.set_can_appear_mulitple_times(true);
        key.set_parameter_helper_string("key=value");
        key.add_help_string("Set key/value pairs");

        option = Option(CREATE_ARCHIVE);
        option.set_number_of_parameters(2);
        option.add_name("create");
        option.set_is_command(true);
        option.add_option(key);
        option.add_help_string(
            "Create archive from the files contained in the given search path root directory");
        option.set_command_string("create [<Local options>] <search-path-root> <archive>");
        m_known_options.push_back(option);
    }

    option = Option(SHOW_ARCHIVE);
    option.set_number_of_parameters(1);
    option.add_name("show");
    option.set_is_command(true);
    option.add_help_string("Show the content of the given archive");
    option.add_help_string("Perform some validation test on the archive file");
    option.add_help_string("List the content of the archive manifest");
    option.set_command_string("show <archive>");
    m_known_options.push_back(option);

    option = Option(LIST_ALL);
    option.set_number_of_parameters(0);
    option.add_name("list");
    option.set_is_command(true);
    option.add_help_string("List all installed archives");
    option.set_command_string("list");
    m_known_options.push_back(option);

    option = Option(LIST);
    option.set_number_of_parameters(1);
    option.add_name("list");
    option.set_is_command(true);
    option.add_help_string("List locations for the given archive");
    option.set_command_string("list <archive>");
    m_known_options.push_back(option);

    option = Option(REMOVE);
    option.set_number_of_parameters(1);
    option.add_name("remove");
    option.set_is_command(true);
    option.add_help_string("Uninstall the given archive");
    option.set_command_string("remove <archive>");
    m_known_options.push_back(option);

    option = Option(DEPENDS);
    option.set_number_of_parameters(1);
    option.add_name("depends");
    option.set_is_command(true);
    option.add_help_string("List dependencies for the given archive");
    option.set_command_string("depends <archive>");
    m_known_options.push_back(option);

    {
        Option force = Option(FORCE);
        force.set_number_of_parameters(0);
        force.add_name("-f");
        force.add_name("--force");
        force.add_help_string("Force extraction even in case of conflict");

        option = Option(EXTRACT);
        option.set_number_of_parameters(2);
        option.add_name("extract");
        option.set_is_command(true);
        option.add_option(force);
        option.add_help_string("Unpack the given archive into directory");
        option.set_command_string("extract <archive> <directory>");
        m_known_options.push_back(option);
    }

    // MDLE Commands
    {
        Option user_file = Option(CREATE_MDLE_ADD_USER_FILE);
        user_file.set_number_of_parameters(2);
        user_file.add_name("--file");
        user_file.set_can_appear_mulitple_times(true);
        user_file.set_parameter_helper_string("<source file path> <target path in the mdle>");
        user_file.add_help_string("add an additional user file to the mdle");

        option = Option(CREATE_MDLE);
        option.set_number_of_parameters(2);
        option.add_name("create-mdle");
        option.set_is_command(true);
        option.add_option(user_file);
        option.add_help_string(
            "Create an MDLE from a qualified material. ");
        option.set_command_string("create-mdle [<Local options>] <qualified material name> <mdle path>");
        m_known_options.push_back(option);
    }

    {
        option = Option(CHECK_MDLE);
        option.set_number_of_parameters(1);
        option.add_name("check-mdle");
        option.set_is_command(true);
        option.add_help_string(
            "Check the integrity of an MDLE file. ");
        option.set_command_string("check-mdle <mdle path>");
        m_known_options.push_back(option);
    }
}

Option Option_parser::next_option(
    const vector<string> & command_line
    , vector<string>::const_iterator & it
) const
{
    Option found_option;
    Option_set options = m_known_options.find_options_from_name(*it);
    if (options.size() == 1)
    {
        // Single option match, just return it
        found_option = options[0];
    }
    else if (options.size() > 1)
    {
        // Found multiple options for the same argument

        // Try parsing each of the found options and only keep the one which succeed
        Option_set valid_options;
        for (auto& opt : options)
        {
            vector<string>::const_iterator init_it = it;
            init_it++;
            if (opt.consume_arguments(command_line, init_it))
            {
                // Copy parser and parse the rest of the line to see if an error occurs
                Option_parser local = *this;
                if (local.consume_arguments(command_line, init_it))
                {
                    // This option might work
                    valid_options.push_back(opt);
                }
            }
        }
        options = valid_options;
        if (options.size() == 1)
        {
            // Single option match, just return it
            found_option = options[0];
        }
        else if (options.size() > 1)
        {
            // If multiple succeed then there is ambiguity
            Util::log_error("Ambiguous command line option");

            // Only keep the option with longuest argument list
            int longuest = options[0].get_number_of_parameters();
            found_option = options[0];
            for (auto& opt : options)
            {
                if (opt.get_number_of_parameters() > longuest)
                {
                    longuest = opt.get_number_of_parameters();
                    found_option = opt;
                }
            }
            Util::log_error("Resolved ambiguous command line option to: " 
                + found_option.get_command_string());
        }
    }

    return found_option;
}

bool Option_parser::consume_arguments(
    const vector<string> & command_line
    , vector<string>::const_iterator & it
)
{
    while (it != command_line.end())
    {
        Option option = next_option(command_line, it);

        if (option.is_valid())
        {
            it++;
            if (option.consume_arguments(command_line, it))
            {
                m_options_set.push_back(option);
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    return true;
}

bool Option_parser::is_set(const int & id, Option_set & option) const
{
    option.clear();
    Option_set::const_iterator it;
    bool set = false;
    for (it = m_options_set.begin(); it != m_options_set.end(); it++)
    {
        if (it->id() == id)
        {
            option.push_back(*it);
            set = true;
        }
    }
    return set;
}

bool Option_parser::is_command_set(Option_set & option) const
{
    option.clear();
    Option_set::const_iterator it;
    bool set = false;
    for (it = m_options_set.begin(); it != m_options_set.end(); it++)
    {
        if (it->get_is_command())
        {
            option.push_back(*it);
            set = true;
        }
    }
    return set;
}
