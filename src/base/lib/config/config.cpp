/***************************************************************************************************
 * Copyright (c) 2004-2025, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief The config module.

#include "pch.h"
#include "config.h"
#include "config_impl.h"

#include <any>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>

#include <base/lib/log/i_log_logger.h>
#include <base/lib/log/i_log_module.h>
#include <base/system/main/access_module.h>
#include <base/system/main/i_module_id.h>
#include <base/system/main/module_registration.h>
#include <base/util/registry/i_config_registry.h>
#include <base/util/string_utils/i_string_utils.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>

namespace MI {
namespace CONFIG {

using namespace LOG;
using std::string;


/// Module registration.
static SYSTEM::Module_registration<Config_module_impl> s_module(M_CONFIG, "CONF");

SYSTEM::Module_registration_entry* Config_module::get_instance()
{
    return s_module.init_module(s_module.get_name());
}

bool Config_module_impl::init()
{
    return true;
}

void Config_module_impl::exit()
{}



Config_module_impl::Config_module_impl()
  : m_read_commandline_already_called(false)
{
}


Config_module_impl::~Config_module_impl()
{}

namespace {

struct Bool_setter
{
    Bool_setter(bool* address, bool value=true) : m_address(address), m_value(value) {}
    ~Bool_setter() { if (m_address) *m_address = m_value; }
    bool* m_address;
    bool m_value;
};

//--------------------------------------------------------------------------------------------------

bool is_all_capital_letters_(
    const std::string& name,
    string::size_type begin,
    string::size_type end)
{
    if (name.size() < end)
        return false;
    for (string::size_type i=begin; i<end; ++i)
        if (islower(name[i]))
            return false;
    return true;
}


//--------------------------------------------------------------------------------------------------

/// Functor for the \c parse() function.
struct Stripper {
    std::string operator()(const std::string& str) const { return STRING::strip(str); }
};


//--------------------------------------------------------------------------------------------------

/// Parse a configuration line. The line comes either from the command line or from a config file.
/// The format is var=values, where values is a comma-separated list of floats, integers, or
/// strings. Strings may (should) be quoted; the quotes will be stripped off. Always store numbers
/// as floats here because we can't tell anyway, a later update may need to switch them to integers.
/// \note So far, only one value per variable is supported. Checked with an assertion.
/// \param opt the line to parse
/// \return a pair of <variable name, variable value> or an empty pair if there is a problem
std::pair<std::string, std::any> parse(
    const char* opt)
{
    std::pair<std::string, std::any> result;
    if (!opt) {
        mod_log->error(M_CONFIG, Mod_log::C_MISC, 2, "Empty configuration option not allowed");
        return result;
    }

    mod_log->debug(M_CONFIG, Mod_log::C_MISC, "Configure %s", opt);

    std::vector<std::string> token_list;
    MI::STRING::split(opt, "=", token_list);
    if (token_list.size() != 2) {
        mod_log->error(M_CONFIG, Mod_log::C_MISC, 12,
            "Illegal configuration option \"%s\", syntax is \"group_var=values\"", opt);
        return result;
    }
    // strip white space
    std::transform(token_list.begin(), token_list.end(), token_list.begin(), Stripper());

    string key = token_list[0];
    // here we do remove all leading capital letters module identifiers
    const string::size_type p = key.find_first_of('_');
    if (p != string::npos && p < key.size()-1) {
        if (is_all_capital_letters_(key, 0, p))
            key = key.substr(p+1);
    }
    const char* word = token_list[1].c_str();

    // comma-separated value list, no more whitespace allowed
    std::vector<std::string> value;                     // start of val in opt
    char        newtype = 0;                            // detect type change

    int l;
    std::string str_value;
    while (true) {
        if (*word == '"') {                             // string value:
            // this requires some caution: since the string starts with a \" it ends with a
            // \" too - before the closing "!
            // eg ""string"" should be stripped to "string", """" to "", etc...
            newtype = 's';
            word++;
            const std::string::size_type pos = token_list[1].find('\"', 1);
            for (l=0; word[l] && word[l] != '"'; l++);  // find end of string
            if (word[l] != '"') {
                mod_log->error(M_CONFIG, Mod_log::C_MISC, 14,
                    "Illegal configuration option \"%s\", missing trailing quote", opt);
                return result;
            }
            str_value = token_list[1].substr(1, pos);
            str_value = STRING::rstrip(str_value, "\"");
        }
        else if (*word && strchr("0123456789.+-", *word)) { // numerical value
            newtype = 'f';
            for (l=1; word[l] && strchr("0123456789.eE+-", word[l]); l++);
            str_value = word;
        }
        else {                                          // not string, not num
            mod_log->error(M_CONFIG, Mod_log::C_MISC, 5,
                "Illegal configuration option \"%s\", unrecognized value at '%c'", opt, *word);
            return result;
        }
        // since we currently do not support multiple values we don't check here neither
#if 0
        if (m_type && m_type != newtype) {              // all strng or all num
            mod_log->error(M_CONFIG, Mod_log::C_MISC, 15,
                "Illegal configuration option \"%s\", cannot mix strings and numbers", opt);
            return result;
        }
#endif
        // safe value
        value.push_back(str_value);

        word += l + (newtype == 's');
        if (*word <= ' ')                               // end of opt: break
            break;
        if (*word++ != ',') {                           // comma, next value
            mod_log->error(M_CONFIG, Mod_log::C_MISC, 7,
                "Illegal configuration option \"%s\", expected comma at '%c'", opt, *word);
            return result;
        }
    }

    ASSERT(M_CONFIG, !value.empty());

    // parsing succeeded, store key and values
    ASSERT(M_CONFIG, value.size() == 1);                // currently support for one argument only!
    result.first = key;

    if (newtype == 's') {                               // strings: dup
        if (!value.front().empty())
            result.second = std::any(value.front());
    }
    else {                                              // numbers: copy
        ASSERT(M_CONFIG, newtype == 'f');
        float val = float(atof(value[0].c_str()));
        result.second = std::any(val);
    }

    return result;
}

}


// for command-line overrides. Variables set with this will be ignored in
// the config file (except that the file's help text is extracted). The
// string is parsed just like a config file line. Return false on failure.
bool Config_module_impl::override(
    const char* opt)            // option to parse
{
    const char* p;
    // skip white space
    for (p=opt; *p == ' ' || *p == '\t'; p++);
    if (!*p || *p == '\n' || *p == '\r')
        // blank line
        return false;
    if (*p == '#')
        // comment
        return false;

    const std::pair<std::string, std::any> record = parse(p);
    if (record.first.empty())
        return false;

    // store now the data in the Registry - the record is not needed anymore afterwards
    m_configuration.overwrite_value(record.first, record.second);

    return true;
}

// Retrieve the value of a given config entry key as a string.
string Config_module_impl::get_config_value_as_string(
    const string& key) const
{
    std::any v = m_configuration.get_value(key);
    if (!v.has_value())
        return string();
    if (std::any_cast<string>(&v))
        return *std::any_cast<string>(&v);
    else if (std::any_cast<float>(&v))
        return STRING::lexicographic_cast_s<std::string>(*std::any_cast<float>(&v)).value();
    else if (std::any_cast<bool>(&v)) {
        if (*std::any_cast<bool>(&v))
            return "true";
        else
            return "false";
    }

    mod_log->error(M_CONFIG, Mod_log::C_MISC, 777,
        "Config value of key \"%s\" has unknown type.",
        key.c_str());
    return string();
}

// Update the host's config parameter with the current registered value.
// Return true when a value for name was already registered, false else.
bool Config_module_impl::update(
    const char* name,           // name of variable
    const char* help,           // meaning of var, for http/comments
    std::string& value) // store pointers to string here
{
    const CONFIG::Config_registry& registry = get_configuration();

    string val;
    const bool success = registry.get_value(name, val);

    // updating
    if (success)
        value = val;

    return success;
}


bool Config_module_impl::is_initialization_complete() const
{
    return m_read_commandline_already_called;
}

void Config_module_impl::set_initialization_complete(
    bool flag)
{
    m_read_commandline_already_called = flag;
}


// Retrieve the configuration.
Config_registry& Config_module_impl::get_configuration()
{
    return m_configuration;
}


// Retrieve the configuration.
const Config_registry& Config_module_impl::get_configuration() const
{
    return m_configuration;
}


// Retrieve the product version.
std::string Config_module_impl::get_product_version() const
{
    std::string result;

    const Config_registry& registry = get_configuration();
    std::any any = registry.get_value("product_version");
    if (any.has_value()) {
        std::string* res = std::any_cast<std::string>(&any);
        if (res)
            result = *res;
    }
    return result;
}


// Retrieve the product name.
std::string Config_module_impl::get_product_name() const
{
    std::string result;

    const Config_registry& registry = get_configuration();
    std::any any = registry.get_value("product_name");
    if (any.has_value()) {
        std::string* res = std::any_cast<std::string>(&any);
        if (res)
            result = *res;
    }
    return result;
}

}
}
