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
#include "command.h"
#include "util.h"
#include "application.h"
#include "errors.h"
#include "version.h"
#include "search_path.h"
#include "xliff.h"
#include <iostream>
#include <map>
using namespace i18n;
using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::find;
using std::map;
using mi::base::Handle;
using mi::neuraylib::IMdl_info;

int Help::execute()
{
    I18N_option_parser parser;
    parser.output_usage(cout);

    Option_set options = parser.get_known_options().find_options_from_name(m_command);
    if (options.empty())
    {
        Util::log_error("Invalid command: " + m_command);
        return -1;
    }

    Option_set::iterator it;
    for (it = options.begin(); it != options.end(); it++)
    {
        it->output_usage(cout);
    }
    return 0;
}

Command * Command_factory::build_command(const Option_set & option)
{
    Option_set_type::const_iterator it;
    for (it = option.begin(); it != option.end(); it++)
    {
        if (it->is_valid() && it->get_is_command())
        {
            // Check the number of arguments for the command
            if (it->get_number_of_parameters() != it->value().size())
            {
                break;
            }

            if (it->id() == I18N_option_parser::HELP_CMD)
            {
                return new Help(it->value()[0]);
            }
            else if (it->id() == I18N_option_parser::CREATE_XLIFF)
            {
                Create_xliff_command * cmd = new Create_xliff_command();

                Option_parser * command_options = it->get_options();
                if (command_options)
                {
                    Option_set modopt;
                    if (command_options->is_set(I18N_option_parser::MODULE, modopt))
                    {
                        for (auto& o : modopt)
                        {
                            const std::vector<std::string> & v = o.value();
                            for (auto& module : v)
                            {
                                cmd->add_module(module);
                            }
                        }
                    }
                    if (command_options->is_set(I18N_option_parser::PACKAGE, modopt))
                    {
                        for (auto& o : modopt)
                        {
                            const std::vector<std::string> & v = o.value();
                            for (auto& package : v)
                            {
                                cmd->add_package(package);
                            }
                        }
                    }
                    if (command_options->is_set(I18N_option_parser::LOCALE, modopt))
                    {
                        cmd->set_locale(modopt[0].value()[0]);
                    }
                    if (command_options->is_set(I18N_option_parser::NO_RECURSIVE, modopt))
                    {
                        cmd->set_recursive(false);
                    }
                    if (command_options->is_set(I18N_option_parser::NO_CONTEXT, modopt))
                    {
                        cmd->set_output_context(false);
                    }
                    if (command_options->is_set(I18N_option_parser::DRY_RUN, modopt))
                    {
                        cmd->set_dry_run(true);
                    }
                    if (command_options->is_set(I18N_option_parser::FORCE, modopt))
                    {
                        cmd->set_force(true);
                    }
                }
                return cmd;
            }
        }
    }
    return NULL;
}
