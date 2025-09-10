/***************************************************************************************************
 * Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
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
#include "util.h"
#include "application.h"
#include "errors.h"

#include <base/hal/hal/i_hal_ospath.h>
#include <base/hal/disk/disk_utils.h>
#include <base/hal/hal/hal.h>
#include <base/util/string_utils/i_string_utils.h>

#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

using namespace i18n;
using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;

/// log
void log(const string & msg)
{
    cerr << msg << endl;
}

void log_internal(
    const mi::base::Handle<mi::base::ILogger> & logger
    , const string & msg
    , mi::base::Message_severity level
)
{
    if (logger)
    {
        logger->message(level, "MDLM", msg.c_str());
    }
    else
    {
#if ! defined(DEBUG)
        // Iray not started
        // In release build, do not log message which level is not at least warning
        if (level <= mi::base::MESSAGE_SEVERITY_WARNING)
#endif
        {
            log(msg);
        }
    }
}

void Util::log_fatal(const string & msg)
{
    log_internal(Application::theApp().logger(), msg, mi::base::MESSAGE_SEVERITY_FATAL);
}

void Util::log_error(const string & msg)
{
    log_internal(Application::theApp().logger(), msg, mi::base::MESSAGE_SEVERITY_ERROR);
}

void Util::log_warning(const string & msg)
{
    log_internal(Application::theApp().logger(), msg, mi::base::MESSAGE_SEVERITY_WARNING);
}

void Util::log_info(const string & msg)
{
    log_internal(Application::theApp().logger(), msg, mi::base::MESSAGE_SEVERITY_INFO);
}

void Util::log_verbose(const string & msg)
{
    log_internal(Application::theApp().logger(), msg, mi::base::MESSAGE_SEVERITY_VERBOSE);
}

void Util::log_debug(const string & msg)
{
    log_internal(Application::theApp().logger(), msg, mi::base::MESSAGE_SEVERITY_DEBUG);
}

void Util::log(const std::string & msg, mi::base::Message_severity severity)
{
    log_internal(Application::theApp().logger(), msg, severity);
}

void Util::log_report(const std::string & msg)
{
    cout << msg << endl;
}

string Util::basename(const string& path)
{
    return MI::HAL::Ospath::basename(path);
}

string Util::extension(const std::string& path)
{
    return MI::DISK::get_extension(path);
}

string Util::get_program_name(const string & path)
{
    string res = Util::basename(path);

#ifdef MI_PLATFORM_WINDOWS
    size_t l = res.length();
    if (l >= 4 && res.substr(l - 4) == ".exe")
    {
        res = res.substr(0, l - 4);
    }
#endif
    return res;
}

bool Util::file_is_readable(const string & path)
{
    return MI::DISK::access(path.c_str());
}

bool Util::delete_file_or_directory(const string & file_or_directory, bool recursive)
{
    std::error_code ec;
    return recursive
       ? fs::remove_all(fs::u8path(file_or_directory), ec)
       : fs::remove(fs::u8path(file_or_directory), ec);
}

bool Util::has_ending(string const &fullString, string const &ending)
{
    if (fullString.length() >= ending.length())
    {
        return (
            0 == fullString.compare(
                fullString.length() - ending.length(), ending.length(), ending));
    }
    else
    {
        return false;
    }
}

bool Util::remove_duplicate_directories(vector<string> & directories)
{
    bool modified = false;
    try
    {
        vector<string>::const_iterator it = directories.begin();
        while (it != directories.end())
        {
            vector<string>::const_iterator it2 = it;
            it2++;
            while (it2 != directories.end())
            {
                if (Util::equivalent(*it, *it2))
                {
                    Util::log_info("Duplicate MDL directory ignored: " + *it2);
                    it2 = directories.erase(it2);
                    modified = true;
                }
                else
                {
                    it2++;
                }
            }
            it++;
        }
        return modified;
    }
    catch (std::exception& e)
    {
        Util::log_error("Remove duplicate directory: " + std::string(e.what()));
    }
    catch (...)
    {
        Util::log_error("Remove duplicate directory: Exception of unknown type");
    }
    return modified;
}

/// Copy file
bool Util::copy_file(std::string const & source, std::string const & destination)
{
    std::error_code ec;
    fs::copy(fs::u8path(source), fs::u8path(destination), fs::copy_options::overwrite_existing, ec);
    return ec == std::error_code();
}

void Util::array_to_vector(int ac, char *av[], vector<string> & v)
{
    v.clear();
    for (int i = 0; i < ac; i++)
    {
        v.push_back(av[i]);
    }
}

string Util::normalize(const std::string & path)
{
    return MI::HAL::Ospath::normpath(path);
}

bool Util::equivalent(const std::string & file1, const std::string & file2)
{
    try {
       fs::path p1(fs::u8path(file1));
       p1 = fs::absolute(p1);
       p1 = p1.lexically_normal();

       fs::path p2(fs::u8path(file2));
       p2 = fs::absolute(p2);
       p2 = p2.lexically_normal();

       std::string p1norm = MI::DISK::to_string(p1);
       std::string p2norm = MI::DISK::to_string(p2);
#if WIN_NT
        // On Windows, normalize the case before testing
        MI::STRING::to_upper(p1norm);
        MI::STRING::to_upper(p2norm);
#endif

        return p1norm == p2norm;

    } catch(...) {
        return false;
    }
}

std::string Util::path_appends(const std::string & path, const std::string & end)
{
    return MI::HAL::Ospath::join(path, end);
}

std::string Util::stem(const std::string & path)
{
    std::string head;
    std::string tail;

    MI::HAL::Ospath::split(path, head, tail);

    if (!tail.empty())
    {
        std::string root;
        std::string ext;
        MI::HAL::Ospath::splitext(tail, root, ext);
        return root;
    }

    return "";
}
