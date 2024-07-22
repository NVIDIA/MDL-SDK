/******************************************************************************
 * Copyright (c) 2008-2024, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief
 **/


#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/base/interface_implement.h>

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/ilogging_configuration.h>
#include <mi/neuraylib/ineuray.h>

#include <cstdio>
#include <vector>

#include "test_shared.h"

class Test_logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    void message(
        mi::base::Message_severity level,
        const char* module_category,
        const mi::base::Message_details&,
        const char* message) final
    {
        const char* log_level = get_log_level( level);
        fprintf(
            stderr,
            "Log Level = '%s' : Module:Category = '%s' : Message = '%s'\n",
            log_level,
            module_category,
            message);
        m_messages.emplace_back( message);
    }

    void check_and_reset( const char* message)
    {
        for( mi::Size i = 0; i < m_messages.size(); ++i)
            if( m_messages[i].find( message) != std::string::npos) {
                m_messages.clear();
                return;
            }

        MI_CHECK( false);
    }

private:
    const char* get_log_level( mi::base::Message_severity level)
    {
        switch( level) {
            case mi::base::MESSAGE_SEVERITY_FATAL:   return "FATAL  ";
            case mi::base::MESSAGE_SEVERITY_ERROR:   return "ERROR  ";
            case mi::base::MESSAGE_SEVERITY_WARNING: return "WARNING";
            case mi::base::MESSAGE_SEVERITY_INFO:    return "INFO   ";
            case mi::base::MESSAGE_SEVERITY_VERBOSE: return "VERBOSE";
            case mi::base::MESSAGE_SEVERITY_DEBUG:   return "DEBUG  ";
            default: /* avoid compiler warning */    return "UNKNOWN";
        }
    }

    std::vector<std::string> m_messages;
};

const int levels_size = 6+1;
const int INVALID = 42;
const mi::base::Message_severity levels[levels_size] = {
    mi::base::MESSAGE_SEVERITY_FATAL,
    mi::base::MESSAGE_SEVERITY_ERROR,
    mi::base::MESSAGE_SEVERITY_WARNING,
    mi::base::MESSAGE_SEVERITY_INFO,
    mi::base::MESSAGE_SEVERITY_VERBOSE,
    mi::base::MESSAGE_SEVERITY_DEBUG,
    static_cast<mi::base::Message_severity>(INVALID)
};

const int categories_size = 14+1;
std::string categories[categories_size] = {
    "MAIN", "NETWORK", "MEMORY", "DATABASE", "DISK", "PLUGIN", "RENDER", "GEOMETRY",
    "IMAGE", "IO", "ERRTRACE", "MISC", "DISTRACE", "COMPILER", "INVALID"
};

void run_tests (
    mi::neuraylib::ILogging_configuration* logging_configuration,
    Test_logger* receiving_logger,
    bool before_start)
{
    // enable all prefix fields, delayed log messages should contain them as well

    logging_configuration->set_log_prefix( ~0u);

    // get forwarding logger

    mi::base::Handle<mi::base::ILogger> forwarding_logger(
        logging_configuration->get_forwarding_logger());
    MI_CHECK( forwarding_logger.is_valid_interface());

    forwarding_logger->message( mi::base::MESSAGE_SEVERITY_INFO, "TEST:MISC", "foo");
    forwarding_logger->printf(
        mi::base::MESSAGE_SEVERITY_INFO, "TEST:MISC", "bar %d %s %f %lf", 42, "baz", 1.0f, 1.0);
    receiving_logger->message( mi::base::MESSAGE_SEVERITY_INFO, "TEST:MISC", {}, "foo");
    receiving_logger->printf(
        mi::base::MESSAGE_SEVERITY_INFO, "TEST:MISC", "bar %d %s %f %lf", 42, "baz", 1.0f, 1.0);

    // set / get receiving logger

    logging_configuration->set_receiving_logger( receiving_logger);
    mi::base::Handle<mi::base::ILogger> logger( logging_configuration->get_receiving_logger());
    MI_CHECK( logger.get() == receiving_logger);

    logging_configuration->set_receiving_logger( nullptr);
    logger = logging_configuration->get_receiving_logger();
    MI_CHECK( logger.get() == nullptr);

    logging_configuration->set_receiving_logger( receiving_logger);
    logger = logging_configuration->get_receiving_logger();
    MI_CHECK( logger.get() == receiving_logger);

    // test with long strings (see fixed buffer size in mi::base::ILogger::printf())

    std::string s1022 = std::string( 1018, '.') + "1022";
    MI_CHECK_EQUAL( s1022.size(), 1022);
    forwarding_logger->printf(
        mi::base::MESSAGE_SEVERITY_INFO, "TEST:MISC", "%s", s1022.c_str());
    receiving_logger->check_and_reset( (std::string( ": ") + s1022).c_str());

    std::string s1023 = std::string( 1019, '.') + "1023";
    MI_CHECK_EQUAL( s1023.size(), 1023);
    forwarding_logger->printf(
        mi::base::MESSAGE_SEVERITY_INFO, "TEST:MISC", "%s", s1023.c_str());
    receiving_logger->check_and_reset( (std::string( ": ") + s1023).c_str());

    std::string s1024 = std::string( 1020, '.') + "1024";
    MI_CHECK_EQUAL( s1024.size(), 1024);
    forwarding_logger->printf(
        mi::base::MESSAGE_SEVERITY_INFO, "TEST:MISC", "%s", s1024.c_str());
    receiving_logger->check_and_reset( (std::string( ": ") + s1024.substr( 0, 1023)).c_str());

    std::string s1025 = std::string( 1021, '.') + "1025";
    MI_CHECK_EQUAL( s1025.size(), 1025);
    forwarding_logger->printf(
        mi::base::MESSAGE_SEVERITY_INFO, "TEST:MISC", "%s", s1025.c_str());
    receiving_logger->check_and_reset( (std::string( ": ") + s1025.substr( 0, 1023)).c_str());

    // set / get overall limit

    mi::Sint32 result;

    for( auto level : levels) {

        result = logging_configuration->set_log_level( level);
        MI_CHECK( (result == 0) ^ (level == INVALID));

        mi::base::Message_severity l = logging_configuration->get_log_level();
        MI_CHECK( (l == level) ^ (level == INVALID));
    }

    // set / get limits per category

    for( auto& category : categories) {
        for( auto l : levels) {

            logging_configuration->set_log_level( mi::base::MESSAGE_SEVERITY_FATAL);

            result = logging_configuration->set_log_level_by_category( category.c_str(), l);
            MI_CHECK( (result == 0) ^ ((l == INVALID) || (category == "INVALID")));

            MI_CHECK_EQUAL( logging_configuration->get_log_level(),
                mi::base::MESSAGE_SEVERITY_FATAL);

            mi::base::Message_severity level
                = logging_configuration->get_log_level_by_category( category.c_str());
            MI_CHECK( (level == l) ^ ((l == INVALID) || (category == "INVALID")));
        }
    }

    // set / get limits for all categories

    for( auto level : levels) {

        logging_configuration->set_log_level( mi::base::MESSAGE_SEVERITY_FATAL);

        result = logging_configuration->set_log_level_by_category( "ALL", level);
        MI_CHECK( (result == 0) ^ (level == INVALID));

        MI_CHECK_EQUAL( logging_configuration->get_log_level(),
            mi::base::MESSAGE_SEVERITY_FATAL);

        for( auto& category : categories) {

            mi::base::Message_severity l
                = logging_configuration->get_log_level_by_category( category.c_str());
            MI_CHECK( (l == level) ^ ((level == INVALID) || (category == "INVALID")));
        }
    }

    result = logging_configuration->set_log_level_by_category(
        "ALL", mi::base::MESSAGE_SEVERITY_INFO);
    MI_CHECK_EQUAL( 0, result);
    result = logging_configuration->set_log_level( mi::base::MESSAGE_SEVERITY_INFO);
    MI_CHECK_EQUAL( 0, result);

    // set / get logging priority

    result = logging_configuration->set_log_priority( 42);
    MI_CHECK_EQUAL( result, -1);

    MI_CHECK_EQUAL( 0, logging_configuration->get_log_priority());

    // test Log_stream

    mi::base::Log_stream s(
        forwarding_logger.get(), "TEST:MISC", mi::base::MESSAGE_SEVERITY_INFO);
    s << "An info message" << std::flush;
    s << mi::base::warning << "A warning message" << std::flush;
    s << "Another info message" << mi::base::error << "And an error message" << std::flush;
    s << "And a message that will be flushed by the destructor";
}

MI_TEST_AUTO_FUNCTION( test_ilogging_configuration )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray.is_valid_interface());

    {
        mi::base::Handle<Test_logger> receiving_logger( new Test_logger);
        MI_CHECK( receiving_logger.is_valid_interface());

        mi::base::Handle<mi::neuraylib::ILogging_configuration> logging_configuration(
            neuray->get_api_component<mi::neuraylib::ILogging_configuration>());

        run_tests( logging_configuration.get(), receiving_logger.get(), true);

        MI_CHECK_EQUAL( 0, neuray->start());
        run_tests( logging_configuration.get(), receiving_logger.get(), false);
        MI_CHECK_EQUAL( 0, neuray->shutdown());

        MI_CHECK_EQUAL( 0, neuray->start());
        run_tests( logging_configuration.get(), receiving_logger.get(), false);
        MI_CHECK_EQUAL( 0, neuray->shutdown());
    }

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

