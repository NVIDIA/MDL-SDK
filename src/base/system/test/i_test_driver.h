/******************************************************************************
 * Copyright (c) 2007-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief regression test infrastructure
///
/// - main()          generic main driver for regression tests
///
/// The test library's main() jumps into the user-defined function
/// init_unit_test_suite() after initializing the system, e.g. seeding the
/// random number generators, etc.

#ifndef BASE_SYSTEM_TEST_DRIVER_H
#define BASE_SYSTEM_TEST_DRIVER_H

#include "i_test_suite.h"
#include "i_test_sigrtmin_handler.h"
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <cstring>
#ifndef WIN_NT
#  include <unistd.h>                   // define chdir() on POSIX
#endif

extern MI::TEST::Test_suite * init_unit_test_suite(int, char **);

#define MI_TEST_MAIN_CALLING_TEST_MAIN() \
    int main(int argc, char** argv) { return test_main(argc,argv); }

#ifdef MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN
int test_main(int argc, char ** argv)
#else
int main(int argc, char ** argv)
#endif
try
{
    MI::TEST::install_sigrtmin_handler();

    using namespace std;

    // Initialize the system's random number generators.

    srand(static_cast<unsigned int>(time(static_cast<time_t*>(0))));
#ifndef WIN_NT
    srandom(static_cast<unsigned int>(time(static_cast<time_t*>(0))));
#endif

    // Change current working directory to the directory that contains this
    // test binary.

    {
        string dir_name( argv[0] );
        string::size_type const pos( dir_name.find_last_of("/\\") );
        if (pos != string::npos)
        {
            assert(pos + 1 < dir_name.size());
            string const programm_name( dir_name.substr(pos + 1u) );
            dir_name.erase(pos);
#ifdef WIN_NT
            if ( !SetCurrentDirectory(dir_name.c_str()) )
#else
            if ( chdir(dir_name.c_str()) != 0 )
#endif
            {
                dir_name.insert(0u, "cannot change working directory to '");
                dir_name += "'";
                throw runtime_error(dir_name);
            }
            strcpy(argv[0], programm_name.c_str());
        }
    }

    // Run the test suite.

    size_t fails(0u);
    MI::TEST::Test_suite * suite( init_unit_test_suite(argc, argv) );
    if (suite)
    {
        suite->run(fails);
        delete suite;
    }
    cout << endl;
    if (fails)
    {
        cout << "*** " << fails << " failure" << (fails == 1u ? " " : "s ") << "detected" << endl;
        return 1;
    }
    else
    {
        cout << "All tests successful" << endl;
        return 0;
    }
}
catch(MI::TEST::Test_suite_failure const &)
{
    // The message was already printed by the contructor of the exception.
    return 1;
}
catch(MI::TEST::Test_case_failure const &)
{
    // The message was already printed by the contructor of the exception.
    return 1;
}
catch(std::exception const & e)
{
    std::cerr << "*** runtime error: " << e.what() << std::endl;
    return 1;
}
catch(...)
{
    std::cerr << "*** unknown exception" << std::endl;
    return 2;
}

#endif // BASE_SYSTEM_TEST_DRIVER_H
