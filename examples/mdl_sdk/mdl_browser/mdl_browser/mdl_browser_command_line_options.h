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

 /// \file
 /// \brief command line options

#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_COMMAND_LINE_OPTIONS_H 
#define MDL_SDK_EXAMPLES_MDL_BROWSER_COMMAND_LINE_OPTIONS_H

#include "example_shared.h"
#include <iostream>

// Command line options structure.
struct Mdl_browser_command_line_options
{
    // search paths
    std::vector<std::string> search_paths;
    bool cache_rebuild;
    bool keep_open;
    bool no_qt_mode;
    std::string locale;

    // The constructor.
    Mdl_browser_command_line_options(int argc, const char* const* argv)
        : search_paths()
        , cache_rebuild(false)
        , keep_open(false)
        , no_qt_mode(false)
        , locale("")
        , prog_name(argv[0])
    {
        for (int i = 1; i < argc; ++i)
        {
            char const *opt = argv[i];
            if (opt[0] == '-')
            {
                // options without argument
                if (strcmp(opt, "-h") == 0 || strcmp(opt, "--help") == 0)
                    print_usage();

                else if (strcmp(opt, "-c") == 0 || strcmp(opt, "--cache_rebuild") == 0)
                    cache_rebuild = true;

                else if (strcmp(opt, "-k") == 0 || strcmp(opt, "--keep_open") == 0)
                    keep_open = true;

                else if (strcmp(opt, "--no_qt_mode") == 0)
                    no_qt_mode = true;

                // options with one argument
                else if (i < argc - 1)
                {
                    if (strcmp(opt, "-p") == 0 || strcmp(opt, "--mdl_path") == 0)
                        search_paths.emplace_back(argv[++i]);

                    if (strcmp(opt, "-l") == 0 || strcmp(opt, "--locale") == 0)
                        locale = argv[++i];
                }
                else
                {
                    std::cerr << "Invalid number of arguments." << std::endl;
                    print_usage();
                    exit(EXIT_FAILURE);
                }


            }
            else
            {
                std::cerr << "Invalid parameter." << std::endl;
                print_usage();
                exit(EXIT_FAILURE);
            }
        }
    }

    // print print_usage to help specifying the options correctly
    void print_usage()
    {
        std::cout
            << "Usage: " << prog_name.c_str() << " [options] [<value>]\n"
            << "Options:\n"
            << "  -h|--help                     prints these usage instructions\n"
            << "  -c|--cache_rebuild            force a rebuild of the cache file\n"
            << "  -k|--keep_open                reopens the browser until the console is closed.\n"
            << "  -p|--mdl_path <path>          mdl search path, can occur multiple times.\n"
            << "  -l|--locale <val>             localization code (see ISO 639-1 standard).\n"
            << "  --no_qt_mode                  show the standalone use-case.\n"
            << std::endl;
        exit(EXIT_FAILURE);
    }

private:
    std::string prog_name;
};

#endif
