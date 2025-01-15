/******************************************************************************
 * Copyright (c) 2007-2025, NVIDIA CORPORATION. All rights reserved.
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
/// - class Test_suite        a collection of test cases
/// - MI_TEST_SUITE()         define a test suite


#ifndef BASE_SYSTEM_TEST_SUITE_H
#define BASE_SYSTEM_TEST_SUITE_H

#include "i_test_case.h"
#include <set>
#include <iostream>
#ifdef WIN_NT
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN 1
#  endif
#  include <windows.h>
#  ifdef min
#    undef min
#  endif
#  ifdef max
#    undef max
#  endif
#  ifdef IGNORE
#    undef IGNORE
#  endif
#else
#  include <sys/time.h>
#endif // WIN_NT

namespace MI { namespace TEST {

class Test_suite : public Named_test_case
{
    typedef std::pair<size_t, Test_case *> Test_case_entry;

    struct less_than
    {
        bool operator() (Test_case_entry const & lhs, Test_case_entry const & rhs) const
        {
            Test_suite const * const lhs_suite( dynamic_cast<Test_suite const *>(lhs.second) );
            Test_suite const * const rhs_suite( dynamic_cast<Test_suite const *>(rhs.second) );
            if (lhs_suite)
            {
                if (rhs_suite)  return lhs.first < rhs.first;
                else            return false;
            }
            else
            {
                if (rhs_suite)  return true;
                else            return lhs.first < rhs.first;
            }
        }
    };

    typedef std::set<Test_case_entry, less_than>      Test_case_set;
    typedef Test_case_set::iterator                     Test_case_iterator;
    Test_case_set                                       _tests;
    size_t                                              _failures;
    size_t                                              _tick;

public:
    explicit Test_suite(std::string const & suite_name) :
        Named_test_case(suite_name),
        _failures(0u),
        _tick(0u)
    {
    }

    ~Test_suite()
    {
        std::for_each(_tests.begin(), _tests.end(), destroy);
    }

    void add(Test_case * ptr)
    {
        if (ptr) _tests.insert(Test_case_entry(_tick++, ptr));
    }

    void run()
    {
        std::cerr.tie( & std::cout);
        if (!name().empty())
        {
            std::cout << name() << ':' << std::endl
                        << std::string(name().size() + 1u, '=') << std::endl;
        }

        _failures = 0u;
#ifdef WIN_NT
        double frequency;
        {
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            frequency = static_cast<double>(freq.QuadPart);
        }
#else
        timeval pre, post;
#endif
        double elapsed = 0.0;

        for (Test_case_iterator i(_tests.begin()); i != _tests.end(); /**/)
        {
            Test_suite * sub_suite( dynamic_cast<Test_suite *>(i->second) );
            if (sub_suite)
            {
                std::cout << std::endl;
                sub_suite->run();
                _failures += sub_suite->_failures;
            }
            else
            {
                bool const print_progress( !i->second->name().empty() );
                try
                {
                    if (print_progress)
                    {
#ifdef WIN_NT
                        LARGE_INTEGER counter;
                        QueryPerformanceCounter(&counter);
                        elapsed = static_cast<double>(counter.QuadPart) / frequency;
#else
                        gettimeofday(&pre, 0);
#endif
                    }
                    i->second->run();
                    if (print_progress)
                    {
#ifdef WIN_NT
                        LARGE_INTEGER counter;
                        QueryPerformanceCounter(&counter);
                        elapsed = (static_cast<double>(counter.QuadPart) / frequency) - elapsed;
#else
                        gettimeofday(&post, 0);
                        elapsed = ( static_cast<double>(post.tv_sec - pre.tv_sec) * 1.0e6
                                  + static_cast<double>(post.tv_usec - pre.tv_usec)
                                  ) / 1.0e6;
#endif
                        std::cout.flush(); std::cout.clear();
                        std::cout << pad_to_length(i->second->name(), result_column())
                                    << " ok (";
                        if (elapsed < 0.001) std::cout << "<0.001";
#ifdef DISABLE_BUGZILLA_4490_WORKAROUND
                        /* Feeding a double (or float) to std::cout
                         * (or to an std::ostringstream instance)
                         * puts the test binary into an infinite loop
                         * of some kind when running on CentOS 5. See
                         * Bugzilla #4490.
                         */
                        else                 std::cout << elapsed;
#else
                        /* So feed it an integer instead... */
                        else                 std::cout << int(elapsed * 1000 + 0.5) << "m";
#endif
                        std::cout << "s)" << std::endl;
                    }
                }
                catch(Test_case_skipped&)
                {
                    if (print_progress)
                        std::cout << pad_to_length(i->second->name(), result_column())
                                    << " skipped" << std::endl;
                }
                catch(Test_case_failure const &)
                {
                    ++_failures;
                    if (print_progress)
                        std::cout << pad_to_length(i->second->name(), result_column())
                                    << " failure" << std::endl;
                }
                catch(...)
                {
                    ++_failures;
                    if (print_progress)
                        std::cout << pad_to_length(i->second->name(), result_column())
                                    << " failure" << std::endl;
                    throw;
                }
            }
            delete i->second;
            _tests.erase(i++);
        }
    }

    void run(size_t & failed_tests)
    {
        run();
        failed_tests += _failures;
    }

    void run(Test_suite & other)
    {
        other.run(_failures);
    }

private:
    static void destroy(Test_case_entry const & e) { delete e.second; }
};

#define MI_TEST_SUITE(name) new MI::TEST::Test_suite(name)

}} // MI::TEST

#endif // BASE_SYSTEM_TEST_SUITE_H

