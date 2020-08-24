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
#include "application.h"
#include "errors.h"
#include "logger.h"
#include "command.h"
#include "util.h"
#include "neuray_factory.h"
#include "options.h"
#include "search_path.h"
#include <fstream>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <set>

using mi::base::Handle;
using mi::base::ILogger;
using mi::neuraylib::INeuray;
using mi::neuraylib::Neuray_factory;
using mi::neuraylib::IMdl_configuration;
using mi::neuraylib::IPlugin_configuration; 
using namespace mdlm;
using std::size_t;
using std::vector;
using std::string;
using std::cout;
using std::max;
using std::left;
using std::setw;
using std::endl;
using std::istringstream;
using std::set;

// Global routines which need to be provided
// This should make easy the use of the commands as API
namespace mdlm
{
    mi::neuraylib::INeuray * neuray()
    {
        return Application::theApp().neuray();
    }

    void report(const std::string & msg)
    {
        return Application::theApp().report(msg);
    }

    // Check if freeimage is available
    bool freeimage_available()
    { 
        return Application::theApp().freeimage_available(); 
    }
}

/// Configure the MDL SDK with module search paths and load necessary plugins.
void configuration(INeuray* neuray, ILogger* logger, Application::Options* options)
{
    mi::base::Handle<IMdl_configuration> mdl_config(neuray->get_api_component<IMdl_configuration>());
    mdl_config->set_logger(logger);

    // Start removing the current directory "." from the list of search paths
    mdl_config->remove_mdl_path(".");

    // Gather list of path to add to MDL search path
    vector<string> directories;
    if (!options->m_nostdpath)
    {
        // Add standard path
        string dir = Util::get_mdl_system_directory();
        if (Util::File(dir).exist())
        {
            directories.push_back(dir);
        }
        dir = Util::get_mdl_user_directory();
        if (Util::File(dir).exist())
        {
            directories.push_back(dir);
        }
    }

    // Add command line paths
    for (size_t i = 0; i < options->m_paths.size(); ++i)
    {
        directories.push_back(options->m_paths[i]);
    }
    Util::remove_duplicate_directories(directories);

    for (vector<string>::const_iterator it = directories.begin();
        it != directories.end();
        it++)
    {
        string path(* it);
        Util::File::convert_symbolic_directory(path);
        check_success3(
              mdl_config->add_mdl_path(path.c_str()) == 0
            , Errors::ERR_MODULE_PATH_FAILURE
            , path.c_str()
        );
        Util::log_verbose("Added directory to MDL path: " + path);
    }

#if defined(DEBUG)
    // Test
    {
        Search_path sp(neuray);
        sp.snapshot();
        sp.log_debug();
    }
#endif
}

Application & Application::theApp()
{
    static Application app;
    return app;
}

Application::Application()
    : m_command(NULL), m_factory(NULL), m_freeimage_loaded(false)
{
}

Application::~Application()
{
    // Shut down the MDL SDK
    if (neuray())
    {
        check_success(neuray()->shutdown() == 0);
    }
    delete m_factory;
}

template<class T> class Option_set_simple_value
{
public:
    bool get_value(const Option_set & options, T & value) const
    {
        if (!options.empty())
        {
            istringstream str(options.begin()->value()[0]);
            str >> value;
            return true;
        }
        return false;
    }
};

mi::Sint32 Application::setup_options(int ac, char *av[])
{
    vector<string> command_line;
    Util::array_to_vector(ac, av, command_line);

    MDLM_option_parser optionParser;
    vector<string>::const_iterator it;
    for (it = command_line.begin() + 1; // Do not parse command name, useless
        !optionParser.consume_arguments(command_line, it) && it != command_line.end();
        it++)
    {
    }

    Option_set optionset;
    if (optionParser.is_set(MDLM_option_parser::HELP, optionset))
    {
        optionParser.output_usage(cout);
        exit(0);
    }
    if (optionParser.is_set(MDLM_option_parser::VERBOSE, optionset))
    {
        check_success2(
            Option_set_simple_value<int>().get_value(optionset, m_options.m_verbosity)
            , Errors::ERR_PARSING_ARGUMENTS);
    }
    if (optionParser.is_set(MDLM_option_parser::QUIET, optionset))
    {
        m_options.m_quiet = true;
    }
    if (optionParser.is_set(MDLM_option_parser::ADD_PATH, optionset))
    {
        for (auto & o : optionset)
        {
            string path(o.value()[0]);
            m_options.m_paths.push_back(path);
        }
    }
    if (optionParser.is_set(MDLM_option_parser::NO_STD_PATH, optionset))
    {
        m_options.m_nostdpath = true;
    }
    if (optionParser.is_command_set(optionset))
    {
        // Try to build command
        m_command = Command_factory::build_command(optionset);
    }

    if (!m_command)
    {
        optionParser.output_usage(cout);

        Util::log_error(
            "Options: Unknown command, missing command or invalid number of parameters\n");
        exit(0);
    }

#if 0
    // Debug command line
    std::stringstream str;
    optionParser.output_options_set(str);
    Util::log_debug(str.str());
#endif
    return 0;
}

mi::Sint32 Application::initialize(int argc, char *argv[])
{
    // Command line arguments
    m_name = Util::get_program_name(argv[0]);
    check_success2(
        setup_options(argc, argv) == 0
        , Errors::ERR_PARSING_ARGUMENTS);

    // Create an instance of our logger
    if (m_options.m_quiet)
    {
        m_options.m_verbosity = 0;
    }
    m_logger = new Logger(m_options.m_verbosity);

    // Access the MDL SDK
    m_factory = new Neuray_factory(m_logger.get());
    check_success(m_factory->get_result_code() == Neuray_factory::RESULT_SUCCESS);

    // Configure the MDL SDK library
    configuration(neuray(), m_logger.get(), &m_options);

    mi::base::Handle<IPlugin_configuration> plug_config(neuray()->get_api_component<IPlugin_configuration>());

    // Load the FreeImage plugin.
    m_freeimage_loaded = plug_config->load_plugin_library("nv_freeimage" MI_BASE_DLL_FILE_EXT) == 0;

    // Start the MDL SDK
    return neuray()->start();
}

void Application::shutdown()
{
    // Shut down the MDL SDK
    if (neuray())
    {
        check_success(neuray()->shutdown() == 0);
    }
    delete m_factory;
    m_factory = NULL;
}

mi::neuraylib::INeuray * Application::neuray()
{
    if (m_factory)
    {
        return m_factory->get();
    }
    return NULL;
}

const mi::base::Handle<mi::base::ILogger> & Application::logger()
{
    return m_logger;
}

Command * Application::get_command()
{
    return m_command;
}

void Application::report(const std::string & msg) const
{
    if (!m_options.m_quiet)
    {
        Util::log_report(msg);
    }
}
