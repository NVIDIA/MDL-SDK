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
#include "application.h"
#include "errors.h"
#include "util.h"
#include "command.h"
#if defined(DEBUG)
#include "version.h"
#include "util.h"
#include "options.h"
#endif

using namespace i18n;

// i18n tool
// ==========================
// i18n is a command line utility tool for XLIFF files and MDL annotations.
//
// The tool has options to set verbosity, select MDL search path, run in quiet mode,
// disable standard search path.
//
// Here is usage:
//
// Usage: i18n[<option> ...] <command>[<arg> ...]
//
//    MDL Internationalization Tool
//
//    Options :
//    --verbose | -v <arg>        Set verbosity level
//    --search-path | -p <arg>  Add path to the MDL search paths
//                            Note : SYSTEM and USER path are added by default to MDL search paths
//                            SYSTEM is "C:\ProgramData\NVIDIA Corporation\mdl\"
//                            USER is "C:\Users\pamand\Documents\mdl\"
//    --quiet | -q                Suppress all messages
//    --nostdpath | -n            Do not add SYSTEM and USER path to MDL search paths
//
// The command help
// --------------------------
// Help command comes in 2 variants:
//
//    help                      Display help information
//    help <command>            Display help information for the given command
//        
// The command create_xliff
// --------------------------
// This command creates XLIFF file from a single module or from a single package.
// The command has options to set the locale, set a module or a package for
// which we want XLIFF file. When setting a package we can chose to recursively traverse
// sub-packages. XLIFF files can have context embeded as XLIFF groups or we can chose not
// to have any context. A dry-run mode allows to see the result of the command without 
// creating the XLIFF file. A force option allows to overwrite existing XLIFF files.
// The created XLIFF file is stored in the same directory as the given module or
// in the package directory.
//
// Here is usage:
//
//    Command "create_xliff [<options>]" help :
//        i18n[<option> ...] create_xliff[<options>]
//
//        Create an XLIFF file for a module or a package in the given locale.
//        When given a module name the output XLIFF file is named <module>_<locale>.xlf.
//        When given a package name the output XLIFF file is named <locale>.xlf and
//        contains annotations for all the modules contained in the package
//
//        Options :
//        --module | -m <arg>   Specify a module(qualified name)
//        --package | -p <arg>  Specify a package(qualified name)
//        --locale | -l <arg>   Specify the XLIFF locale
//        --no-recursive | -nr  Do not traverse sub - packages
//        --no-context | -nc    Do not output XLIFF groups / context
//        --dry-run | -d        Do not create XLIFF but report what would be done
//        --force | -f          Overwrites XLIFF file if it exists
//
// Create XLIFF file for a module:
//
//    > i18n create_xliff -m ::nvidia::vMaterials::Design::Composites::carbon_fiber_uncoated -l fr
//
// Create XLIFF file for a package:
//
//    > i18n create_xliff -p ::nvidia::vMaterials::Design::Composites --locale fr

int run(int argc, char *argv[])
{
    check_success2(Application::theApp().initialize(argc, argv) == 0, Errors::ERR_INIT_FAILURE);
    Util::log_info("Application started");

#if defined(DEBUG)
    Util::File::Test();
    I18N_option_parser::Test();
#endif

    Command * command = Application::theApp().get_command();

    int rtn_code(0);
    if (!command)
    {
        Util::log_error("Unknown command, missing command or invalid number of parameters");
    }
    else
    {
        rtn_code = command->execute();
    }

    Application::theApp().shutdown();

    return rtn_code;
}

#ifdef MI_PLATFORM_WINDOWS
#include <string>
#include <vector>

#include <mi/base/miwindows.h>

namespace {

    template<typename T>
    class Array_holder {
    public:
        Array_holder(size_t size) : m_arr(new T[size]) {}
        ~Array_holder() { delete[] m_arr; }
        T &operator[](size_t index) { return m_arr[index]; }
        T *data() { return m_arr; }
    private:
        T *m_arr;
    };
}

int wmain(int argc, wchar_t *argvwc[])
{
    std::vector<std::string> utf8_args(argc);
    Array_holder<char *> argv_utf8(argc);

    for (int i = 0; i < argc; ++i) {
        wchar_t *warg = argvwc[i];
        DWORD size = WideCharToMultiByte(
            CP_UTF8,
            0,
            warg,
            -1,
            NULL,
            0,
            NULL,
            NULL);
        if (size > 0) {
            std::string res(size, '\0');
            DWORD result = WideCharToMultiByte(
                CP_UTF8,
                0,
                warg,
                -1,
                const_cast<char *>(res.c_str()),
                size,
                NULL,
                NULL);
            if (result == size) {
                utf8_args[i] = res;
                argv_utf8[i] = const_cast<char *>(utf8_args[i].c_str());
            }
            else {
                argv_utf8[i] = const_cast<char *>("");
            }
        }
    }
    SetConsoleOutputCP(CP_UTF8);

    char** argv = argv_utf8.data();

    return run(argc, argv);
}

#else
int main(int argc, char *argv[])
{
    return run(argc, argv);
}
#endif
