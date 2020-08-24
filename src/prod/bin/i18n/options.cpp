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

#include "options.h"
#include "application.h"
#include "util.h"
#include <iomanip>
#include <map>
using namespace i18n;
using std::vector;
using std::string;
using std::set;
using std::ostream;
using std::endl;
using std::cout;
using std::ostringstream;
using std::map;

bool I18N_option_parser::Test()
{
    I18N_option_parser optionParser;

    //cout << "=========================== TEST OPTIONS" << endl;

    Option o;
    Option o1(o); // Copy ctor
    Option o2 = o1; // Copy ctor
    Option o3 = Option(18);

    // Test cmmand line parsing
    vector<string> command_line;
    command_line.push_back("i18n");
    command_line.push_back("-v");
    command_line.push_back("6");
    command_line.push_back("-h");
    command_line.push_back("-p");
    command_line.push_back("C:/Users/pamand/Documents/mdl");
    command_line.push_back("-p");
    command_line.push_back("C:/ProgramData/NVIDIA Corporation/mdl");
    command_line.push_back("install");
    command_line.push_back("--force");
    command_line.push_back("automotive_catalog.mdr");
    command_line.push_back("D:/MDL/Archives/automotive_catalog/trunk");
    vector<string>::const_iterator it = command_line.begin();

    for (it = command_line.begin();
        !optionParser.consume_arguments(command_line, it) && it != command_line.end();
        it++)
    {
    }

    Option_set optionset;
    if (optionParser.is_set(I18N_option_parser::HELP, optionset))
    {

    }
    if (optionParser.is_set(I18N_option_parser::ADD_PATH, optionset))
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
        << "i18n [<option> ...] "
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

        cout << string(indent * 2, ' ') << "Options:" << endl;

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
    // Usage: i18n [-h | --help][-v | --verbosity <level>][-q | --quiet][-n | --nostdpath]
    //             [[-p | --search - path <path>]][…]
    //             <command>[<args>]
    //
    // col 1 + col 2 < maxline
    int maxline = 120;
    ostringstream str;
    str << "Usage: " << Application::theApp().name() << " [<option> ...] <command> [<arg> ...]";
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
    //  I18N 
    //
    size_t indent = 2;
    string application_help("MDL Internationalization Tool");

    line.reset();
    line.indent(indent);
    line.add_text(application_help);
    ostr << line.str() << endl << endl;

    //Options :
    ostr << "Options:" << endl;

    m_known_options.output_usage(ostr, false/*commands*/, indent);

    ostr << endl;

    //Commands:
    ostr << "Commands:" << endl;

    m_known_options.output_usage(ostr, true/*commands*/, indent);
}

void Option_parser::output_options_set(std::ostream & ostr) const
{
    m_options_set.output_values(ostr);
}

I18N_option_parser::I18N_option_parser()
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
        force.add_help_string("Overwrites XLIFF file if it exists");

        Option dry_run = Option(DRY_RUN);
        dry_run.set_number_of_parameters(0);
        dry_run.add_name("-d");
        dry_run.add_name("--dry-run");
        dry_run.add_help_string("Do not create XLIFF but report what would be done");

        Option no_context = Option(NO_CONTEXT);
        no_context.set_number_of_parameters(0);
        no_context.add_name("-nc");
        no_context.add_name("--no-context");
        no_context.set_can_appear_mulitple_times(false);
        no_context.add_help_string("Do not output XLIFF groups/context");

        Option no_recursive = Option(NO_RECURSIVE);
        no_recursive.set_number_of_parameters(0);
        no_recursive.add_name("-nr");
        no_recursive.add_name("--no-recursive");
        no_recursive.set_can_appear_mulitple_times(false);
        no_recursive.add_help_string("Do not traverse sub-packages");

        Option locale = Option(LOCALE);
        locale.set_number_of_parameters(1);
        locale.add_name("-l");
        locale.add_name("--locale");
        locale.set_can_appear_mulitple_times(false);
        locale.add_help_string("Specify the XLIFF locale");

        Option module = Option(MODULE);
        module.set_number_of_parameters(1);
        module.add_name("-m");
        module.add_name("--module");
        module.set_can_appear_mulitple_times(true);
        module.add_help_string("Specify a module (qualified name)");

        Option package = Option(PACKAGE);
        package.set_number_of_parameters(1);
        package.add_name("-p");
        package.add_name("--package");
        package.set_can_appear_mulitple_times(true);
        package.add_help_string("Specify a package (qualified name)");

        option = Option(CREATE_XLIFF);
        option.set_number_of_parameters(0);
        option.add_name("create_xliff");
        option.add_option(module);
        option.add_option(package);
        option.add_option(locale);
        option.add_option(no_recursive);
        option.add_option(no_context);
        option.add_option(dry_run);
        option.add_option(force);
        option.set_is_command(true);
        option.add_help_string("Create an XLIFF file for a module or a package in \
the given locale.");
        option.add_help_string("When given a module name the output XLIFF file is named \
<module>_<locale>.xlf.");
        option.add_help_string("When given a package name the output XLIFF file is named \
<locale>.xlf and contains");
        option.add_help_string("annotations for all the modules contained in the package");
        option.set_command_string("create_xliff [<options>]");
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
            Util::log_error("Invalid option: " + *it);
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
