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
#include "util.h"
#include "application.h"
#include "errors.h"
#include <base/hal/hal/i_hal_ospath.h>
#include <base/hal/disk/i_disk_file.h>
#include <base/hal/disk/disk.h>
#include <base/hal/hal/hal.h>
#include <base/util/string_utils/i_string_utils.h>
#include <iostream>
#include <ostream>
#include <fstream>   
#include <iterator>   
using namespace mdlm;
using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;

// Forward declarations
string get_mdl_system_directory();
string get_mdl_user_directory();

/// File
bool Util::File::Test()
{
    string new_filename;
    {
        std::string folder_name("c:\\temp");
        Util::File folder(folder_name);
        std::string newtempfile;
        if (folder.is_directory())
        {
            check_success3(true == folder.is_writable(), Errors::ERR_UNIT_TEST, "Util::File::Test()");
            check_success3(true == folder.is_readable(), Errors::ERR_UNIT_TEST, "Util::File::Test()");

            newtempfile = Util::unique_file_in_folder(folder.get_directory());
            Util::File file(newtempfile);
            check_success3(false == file.exist(), Errors::ERR_UNIT_TEST, "Util::File::Test()");

            {
                MI::DISK::File dummy;
                dummy.open(newtempfile, MI::DISK::File::Mode::M_WRITE);
                dummy.writeline("");
                dummy.close();
            }
            check_success3(true == file.exist(), Errors::ERR_UNIT_TEST, "Util::File::Test()");
            check_success3(true == file.is_readable(), Errors::ERR_UNIT_TEST, "Util::File::Test()");
            check_success3(true == file.is_writable(), Errors::ERR_UNIT_TEST, "Util::File::Test()");
            check_success3(true == file.is_file(), Errors::ERR_UNIT_TEST, "Util::File::Test()");
            check_success3(false == file.is_directory(), Errors::ERR_UNIT_TEST, "Util::File::Test()");

            bool is_empty(file.is_empty());
            check_success3(true == is_empty, Errors::ERR_UNIT_TEST, "Util::File::Test()");

            bool equivalent(Util::equivalent("c:\\temp\\..\\temp\\foobar", "c:\\temp\\foobar"));
            check_success3(true == equivalent, Errors::ERR_UNIT_TEST, "Util::File::Test()");

            std::string new_folder_name(path_appends(folder_name, "F2435087-4819-4415-911E-22BB8E6C1DC9"));
            if (Util::create_directory(new_folder_name))
            {
                Util::File new_folder(new_folder_name);
                is_empty = new_folder.is_empty();
                check_success3(true == is_empty, Errors::ERR_UNIT_TEST, "Util::File::Test()");

                bool sucess = Util::copy_file(newtempfile, new_folder_name);
                check_success3(true == sucess, Errors::ERR_UNIT_TEST, "Util::File::Test()");

                new_folder.remove();
                check_success3(true == is_empty, Errors::ERR_UNIT_TEST, "Util::File::Test()");
            }
            file.remove();
        }

        std::string stem = Util::stem("/foo/bar.txt");
        check_success3(stem == "bar", Errors::ERR_UNIT_TEST, "Util::File::Test()");
        stem = Util::stem("foo.bar.baz.tar.txt");
        check_success3(stem == "foo.bar.baz.tar", Errors::ERR_UNIT_TEST, "Util::File::Test()");
        stem = Util::stem("foo.bar.baz.tar.txt/");
        check_success3(stem == "", Errors::ERR_UNIT_TEST, "Util::File::Test()");
        string basename(Util::basename("c:/temp/foo.bar.baz.tar.txt"));
        check_success3(basename == "foo.bar.baz.tar.txt", Errors::ERR_UNIT_TEST, "Util::File::Test()");
    }
    return true;
}

Util::File Util::File::SYSTEM(get_mdl_system_directory());
Util::File Util::File::USER(get_mdl_user_directory());

Util::File::File(const string& path)
    : m_path(path)
{
}

bool Util::File::exist() const
{
    if (Util::File::is_file())
    {
        MI::DISK::File diskfile;
        return diskfile.open(m_path);
    }
    if (Util::File::is_directory())
    {
        MI::DISK::Directory dir;
        return dir.open(m_path.c_str());
    }
    return false;
}

bool Util::File::remove() const
{
    if (Util::File::is_file())
    {
        return MI::DISK::file_remove(m_path.c_str());
    }
    if(Util::File::is_directory())
    {
        // NOTE: directory need to be empty, we do not handle non empty dir
        return MI::DISK::rmdir(m_path.c_str());
    }
    return false;
}

bool Util::File::is_file() const
{   
    MI::DISK::File diskfile;
    if (diskfile.open(m_path))
    {
        return (diskfile.is_file());
    }
    return false;
}

bool Util::File::is_directory() const
{
    MI::DISK::Directory diskdir;
    return diskdir.open(m_path.c_str());
}

bool Util::File::is_readable() const
{
    bool readable = false;
    if (is_directory())
    {
        MI::DISK::Directory directory;
        readable = directory.open(m_path.c_str());
    }
    else if(is_file())
    {
        MI::DISK::File diskfile;
        readable = diskfile.open(m_path);
    }
    return readable;
}

bool Util::File::is_writable() const
{
    bool writable = false;
    if (is_directory())
    {
        // try to write in the location
        //std::string filePath = Util::path_appends(m_path, Util::unique_path("%%%%-%%%%-%%%%-%%%%"/*model*/));
        std::string filePath = Util::unique_file_in_folder(m_path);
        Util::File temp_file(filePath);

        MI::DISK::File diskfile;
        if (diskfile.open(filePath.c_str(), MI::DISK::IFile::Mode::M_WRITE))
        {
            const char* line("line");
            if (diskfile.writeline(line))
            {
                writable = true;
            }
            diskfile.close();
        }
        temp_file.remove();
        Util::log_debug(
            m_path + (writable ? " is writable" : " is not writable"));
    }
    else if (is_file())
    {
        MI::DISK::File diskfile;
        if (diskfile.open(m_path, MI::DISK::IFile::Mode::M_WRITE))
        {
            writable = true;
            diskfile.close();
        }
    }
    return writable;
}

bool Util::File::size(mi::Sint64 & rtnsize) const
{
    rtnsize = 0;
    if (Util::File::is_directory())
    {
        MI::DISK::Directory dir;
        if (dir.open(m_path.c_str()))
        {
            while (!dir.read(true/*nodot*/).empty())
            {
                rtnsize++;
            }
            return true;
        }
    }
    MI::DISK::File diskfile;
    if (diskfile.open(m_path))
    {
        if (diskfile.is_file())
        {
            rtnsize = diskfile.filesize();
            return true;
        }
    }
    return false;
}

bool Util::File::is_empty() const
{
    if (Util::File::is_directory())
    {
        MI::DISK::Directory dir;
        if (!dir.open(m_path.c_str()))
        {
            return false;
        }
        std::string fn = dir.read(true/*nodot*/);
        return fn.empty();
    }
    MI::DISK::File diskfile;
    if (diskfile.open(m_path))
    {
        if (diskfile.is_file())
        {
            return 0 == diskfile.filesize();
        }
    }
    return false;
}

std::string Util::File::get_directory() const
{
    if (Util::File::is_directory())
    {
        return m_path;
    }
    MI::DISK::File diskfile;
    if (diskfile.open(m_path))
    {
        if (diskfile.is_file())
        {
            return MI::HAL::Ospath::dirname(m_path);
        }
    }
    return "";
}

bool Util::File::convert_symbolic_directory(std::string & directory)
{
    if (directory == "SYSTEM")
    {
        directory = SYSTEM.get_directory();
        return true;
    }
    else if (directory == "USER")
    {
        directory = USER.get_directory();
        return true;
    }
    return false;
}

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
    return MI::HAL::Ospath::get_ext(path);
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
    Util::File file(path);
    return file.is_readable();
}

bool Util::directory_is_writable(const string & path)
{
    Util::File directory(path);
    return directory.is_directory() && directory.is_writable();
}

bool Util::create_directory(const string & new_directory)
{
    return MI::DISK::mkdir(new_directory.c_str());
}

bool Util::delete_file_or_directory(const string & file_or_directory, bool recursive)
{
    Util::File file(file_or_directory);
    if (file.is_directory() && recursive)
    {
        MI::DISK::Directory dir;
        if (dir.open(file_or_directory.c_str()))
        {
            while (true)
            {
                string elem(dir.read(true/*nodot*/));
                if (elem.empty())
                {
                    break;
                }
                Util::delete_file_or_directory(
                    Util::path_appends(file_or_directory,elem)
                    , recursive);
            }
        }
    }
    return file.remove();
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

std::string Util::get_mdl_system_directory()
{
    return ::get_mdl_system_directory();
}

std::string Util::get_mdl_user_directory()
{
    return ::get_mdl_user_directory();
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
    return MI::DISK::file_copy(source.c_str(), destination.c_str());
}

void Util::array_to_vector(int ac, char *av[], vector<string> & v)
{
    v.clear();
    for (int i = 0; i < ac; i++)
    {
        v.push_back(av[i]);
    }
}

/// Check if the given character is a valid MDL letter.
static bool is_mdl_letter(char c)
{
    if ('A' <= c && c <= 'Z')
        return true;
    if ('a' <= c && c <= 'z')
        return true;
    return false;
}

/// Check if the given character is a valid MDL digit.
static bool is_mdl_digit(char c)
{
    if ('0' <= c && c <= '9')
        return true;
    return false;
}

bool Util::is_valid_mdl_identifier(const std::string & identifier)
{
    // first check general identifier rules:
    // IDENT = LETTER { LETTER | DIGIT | '_' } .
    char const *p = identifier.c_str();

    if (!is_mdl_letter(*p))
        return false;

    for (++p; *p != '\0'; ++p) {
        if (*p == '_')
            continue;
        if (!is_mdl_letter(*p) && !is_mdl_digit(*p)) {
            return false;
        }
    }

    // now check for keywords
    p = identifier.c_str();

#define FORBIDDEN(name, n) if (strcmp(p + n, name + n) == 0) return false

    switch (p[0]) {
    case 'b':
        if (p[1] == 'o') {
            FORBIDDEN("bool", 2);
            FORBIDDEN("bool2", 2);
            FORBIDDEN("bool3", 2);
        }
        else if (p[1] == 's') {
            FORBIDDEN("bsdf", 2);
            FORBIDDEN("bsdf_measurement", 2); // MDL 1.1+
        }
        break;
    case 'c':
        FORBIDDEN("color", 1);
        break;
    case 'd':
        if (p[1] == 'o') {
            FORBIDDEN("double", 2);
            FORBIDDEN("double2", 2);
            FORBIDDEN("double3", 2);
            FORBIDDEN("double2x2", 2);
            FORBIDDEN("double2x3", 2);
            FORBIDDEN("double2x4", 2);
            FORBIDDEN("double3x2", 2);
            FORBIDDEN("double3x3", 2);
            FORBIDDEN("double3x4", 2);
            FORBIDDEN("double4x2", 2);
            FORBIDDEN("double4x3", 2);
            FORBIDDEN("double4x4", 2);
        }
        break;
    case 'e':
        FORBIDDEN("edf", 1);
        FORBIDDEN("export", 1);
        break;
    case 'f':
        if (p[1] == 'a') {
            FORBIDDEN("false", 2);
        }
        else if (p[1] == 'l') {
            FORBIDDEN("float", 2);
            FORBIDDEN("float2", 2);
            FORBIDDEN("float3", 2);
            FORBIDDEN("float2x2", 2);
            FORBIDDEN("float2x3", 2);
            FORBIDDEN("float2x4", 2);
            FORBIDDEN("float3x2", 2);
            FORBIDDEN("float3x3", 2);
            FORBIDDEN("float3x4", 2);
            FORBIDDEN("float4x2", 2);
            FORBIDDEN("float4x3", 2);
            FORBIDDEN("float4x4", 2);
        }
        break;
    case 'i':
        if (p[1] == 'm') {
            FORBIDDEN("import", 2);
        }
        else if (p[1] == 'n') {
            FORBIDDEN("int", 2);
            FORBIDDEN("int2", 2);
            FORBIDDEN("int3", 2);
            FORBIDDEN("intensity_mode", 2); // MDL 1.1+
            FORBIDDEN("intensity_power", 2); // MDL 1.1+
            FORBIDDEN("intensity_radiant_exitance", 2); // MDL 1.1+
        }
        break;
    case 'l':
        FORBIDDEN("light_profile", 1);
        break;
    case 'm':
        if (p[1] == 'a') {
            FORBIDDEN("material", 2);
            FORBIDDEN("material_emission", 2);
            FORBIDDEN("material_geometry", 2);
            FORBIDDEN("material_surface", 2);
            FORBIDDEN("material_volume", 2);
        }
        break;
    case 's':
        FORBIDDEN("string", 1);
        break;
    case 't':
        if (p[1] == 'e') {
            FORBIDDEN("texture_2d", 2);
            FORBIDDEN("texture_3d", 2);
            FORBIDDEN("texture_cube", 2);
            FORBIDDEN("texture_ptex", 2);
        }
        else if (p[1] == 'r') {
            FORBIDDEN("true", 2);
        }
        break;
    case 'u':
        FORBIDDEN("uniform", 1);
        FORBIDDEN("using", 1);
        break;
    case 'v':
        FORBIDDEN("varying", 1);
        FORBIDDEN("vdf", 1);
        break;
    }
#undef FORBIDDEN

    return true;
}

//-----------------------------------------------------------------------------
// Following code comes from:
// src/io/scene/mdl_elements/mdl_elements_utilities.cpp

bool is_valid_simple_package_or_module_name(const std::string& name)
{
    size_t n = name.size();
    if (n == 0)
        return false;

    for (size_t i = 0; i < n; ++i) {

        unsigned char c = name[i];
        // These characters are not permitted per MDL spec.
        if (c == '/' || c == '\\' || c < 32 || c == 127 || c == ':')
            return false;
    }

    return true;
}

bool is_valid_module_name(const std::string& name)
{
    if (name[0] != ':' || name[1] != ':')
        return false;

    size_t start = 2;
    size_t end = name.find("::", start);

    while (end != std::string::npos) {
        if (!is_valid_simple_package_or_module_name(name.substr(start, end - start)))
            return false;
        start = end + 2;
        end = name.find("::", start);
    }

    if (!is_valid_simple_package_or_module_name(name.substr(start)))
        return false;

    return true;
}

//-----------------------------------------------------------------------------

bool Util::is_valid_module_name(const std::string & identifier)
{
    return ::is_valid_module_name(identifier);
}

bool Util::is_valid_archive_name(const std::string & identifier)
{
    return ::is_valid_simple_package_or_module_name(identifier);
}

//-----------------------------------------------------------------------------
// 
// 

#ifdef _WIN32
#include <Shlobj.h>
#include <Knownfolders.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#ifdef _WIN32
//-----------------------------------------------------------------------------
// helper function to create standard mdl path inside the known folder. WINDOWS only
//
std::string get_mdl_path_in_known_folder(
    KNOWNFOLDERID id,
    const std::string &postfix)
{
    // Fetch the 'knownFolder' path.
    std::string result;
#if(_WIN32_WINNT >= 0x0600)
    wchar_t* knownFolderPath = 0;
    HRESULT hr = SHGetKnownFolderPath(id, 0, NULL, &knownFolderPath);
    if (SUCCEEDED(hr) && knownFolderPath != NULL)
    {
        // convert from wstring to string and append the postfix
        std::wstring s(knownFolderPath);
        int len;
        int slength = (int)s.length();
        len = WideCharToMultiByte(CP_ACP, 0, s.c_str(), slength, 0, 0, 0, 0);
        result = std::string(len, '\0');
        WideCharToMultiByte(CP_ACP, 0, s.c_str(), slength, &result[0], len, 0, 0);

        result.append(postfix);
        CoTaskMemFree(static_cast<void*>(knownFolderPath));
    }
#endif
    return result;
}
#endif // _WIN32

std::string get_environment(const char* env_var)
{
    std::string value;
#ifdef _WIN32
    char* buf = nullptr;
    size_t sz = 0;
    if (_dupenv_s(&buf, &sz, env_var) == 0 && buf != NULL)
    {
        value = buf;
        free(buf);
    }
#else
    const char* v = getenv(env_var);
    if (v)
        value = v;
#endif
    return value;
}

std::string get_mdl_user_directory()
{
    std::string path = get_environment("MDL_USER_PATH");
    if (!path.empty())
        return path;

#ifdef _WIN32
    return get_mdl_path_in_known_folder(FOLDERID_Documents, "/mdl");
#else
    const std::string home = getenv("HOME");
    return home + "/mdl";
#endif
}

std::string get_mdl_system_directory()
{
    std::string path = get_environment("MDL_SYSTEM_PATH");
    if (!path.empty())
        return path;

#ifdef _WIN32
    return get_mdl_path_in_known_folder(FOLDERID_ProgramData, "/NVIDIA Corporation/mdl");
#else // WIN32
#ifdef MACOSX
    return "/Library/Application Support/NVIDIA Corporation/mdl";
#else // NOT MACOSX (-> LINUX)
    return "/usr/share/NVIDIA Corporation/mdl";
#endif // MACOSX
#endif // WIN32
}

std::string get_module_name(const std::string& qualified_material_name)
{
    std::string stripped_mdl_name;
    size_t p = qualified_material_name.find('(');
    if (p == std::string::npos)
        stripped_mdl_name = qualified_material_name;
    else // strip function signature
        stripped_mdl_name = qualified_material_name.substr(0, p);

    p = stripped_mdl_name.rfind("::");
    if (p == std::string::npos)
        return qualified_material_name;

    return stripped_mdl_name.substr(0, p);
}

string Util::normalize(const std::string & path)
{
    return MI::HAL::Ospath::normpath(path);
}

bool Util::equivalent(const std::string & file1, const std::string & file2)
{
    std::string p1(file1);
    if (!MI::DISK::is_path_absolute(p1))
    {
        p1 = Util::path_appends(MI::DISK::get_cwd(), file1);
    }
    std::string p2(file2);
    if (!MI::DISK::is_path_absolute(p2))
    {
        p2 = Util::path_appends(MI::DISK::get_cwd(), file2);
    }

    std::string p1norm(Util::normalize(p1));
    std::string p2norm(Util::normalize(p2));

#if WIN_NT
    // On Windows, normalize the case before testing
    MI::STRING::to_upper(p1norm);
    MI::STRING::to_upper(p2norm);
#endif

    return p1norm == p2norm;
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

std::string Util::unique_file_in_folder(const std::string & folder)
{
    std::string prefix("BAA0A38F-3889-4A44-9C4D-852534E6AAA8");
    std::string filename(prefix);
    int i = 1;
    while (true)
    {
        std::string full_name(path_appends(folder, filename));
        Util::File file(full_name);
        if (file.exist())
        {
            std::stringstream str;
            str << prefix;
            str << i++;
            filename = str.str();
        }
        else
        {
            return full_name;
        }
    }
}

std::vector<std::string> Util::split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    mdlm::split(s, delim, std::back_inserter(elems));
    return elems;
}

