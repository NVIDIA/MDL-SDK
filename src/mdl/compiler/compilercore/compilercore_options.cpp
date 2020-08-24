/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include <cstdio>
#include <cstring>
#include <mi/mdl/mdl_options.h>

#include "compilercore_allocator.h"
#include "compilercore_options.h"
#include "compilercore_assert.h"

namespace mi {
namespace mdl {

Options_impl::Options_impl(IAllocator *alloc)
: Base()
, m_alloc(alloc)
, m_options(alloc)
{
}

Options_impl::~Options_impl()
{
}

// Get the number of options.
int Options_impl::get_option_count() const
{
    return m_options.size();
}

// Get the index of the option with name or -1 if this option does not exists.
int Options_impl::get_option_index(char const *name) const
{
    // yes I know there is find_if... And it is completely useless.
    for (size_t i = 0, n = m_options.size(); i < n; ++i) {
        Option const &opt = m_options[i];

        if (strcmp(opt.get_name(), name) == 0) {
            return int(i);
        }
    }
    return -1;
}

// Get the name of the option at index.
char const *Options_impl::get_option_name(int index) const
{
    Option const &opt = get_option(index);
    return opt.get_name();
}

// Get the value of the option at index.
char const *Options_impl::get_option_value(int index) const
{
    Option const &opt = get_option(index);
    if (opt.is_binary()) return NULL;
    return opt.get_value();
}

// Get the value and size of the binary option at index.
BinaryOptionData Options_impl::get_binary_option(int index) const
{
    Option const &opt = get_option(index);
    if (!opt.is_binary()) return BinaryOptionData(NULL, 0);
    return opt.get_binary_data();
}

// Get the default value of the option at index.
char const *Options_impl::get_option_default_value(int index) const
{
    Option const &opt = get_option(index);
    return opt.get_default_value();
}

// Get the description of the option at index.
char const *Options_impl::get_option_description(int index) const
{
    Option const &opt = get_option(index);
    return opt.get_description();
}

// Returns true if the requested option was modified.
bool Options_impl::is_option_modified(int index) const
{
    Option const &opt = get_option(index);
    return opt.is_modified();
}

// Set an option.
bool Options_impl::set_option(char const *name, char const *value)
{
    int index = get_option_index(name);
    if (index >= 0) {
        Option &opt = m_options[index];
        if (opt.is_binary()) return false;
        opt.set_value(value);
        return true;
    }
    return false;
}

// Set an option.
bool Options_impl::set_binary_option(char const *name, char const *data, size_t size)
{
    int index = get_option_index(name);
    if (index >= 0) {
        Option &opt = m_options[index];
        if (!opt.is_binary())
            return false;
        opt.set_binary_value(data, size);
        return true;
    }
    return false;
}

// Reset all options to their default values.
void Options_impl::reset_to_defaults()
{
    for (Option_vec::iterator it(m_options.begin()), end(m_options.end()); it != end; ++it) {
        Option &opt = *it;

        opt.reset_to_defaults();
    }
}

// Add a new option.
void Options_impl::add_option(char const *name, char const *def_value, char const *description)
{
    m_options.push_back(Option(m_alloc, name, def_value, def_value, description));
}

// Add a new binary option.
void Options_impl::add_binary_option(char const *name, char const *description)
{
    m_options.push_back(Option(m_alloc, name, NULL, size_t(0), description));
}

// Get a string option.
char const *Options_impl::get_string_option(char const *name) const
{
    int index = get_option_index(name);
    if (index >= 0) {
        if (char const *val = get_option_value(index))
            return val;
        if (char const *val = get_option_default_value(index))
            return val;
        return NULL;
    }
    MDL_ASSERT(!"string option does not exist");
    return NULL;
}

// Get a bool option.
bool Options_impl::get_bool_option(char const *name) const
{
    int index = get_option_index(name);
    if (index >= 0) {
        if (char const *val = get_option_value(index)) {
            if (strcmp(val, "1") == 0 || strcmp(val, "true") == 0)
                return true;
            if (strcmp(val, "0") == 0 || strcmp(val, "false") == 0)
                return false;
            // unknown ...
        }
        if (char const *val = get_option_default_value(index)) {
            if (strcmp(val, "1") == 0 || strcmp(val, "true") == 0)
                return true;
            if (strcmp(val, "0") == 0 || strcmp(val, "false") == 0)
                return false;
            // unknown ...
        }
        return false;
    }
    MDL_ASSERT(!"bool option does not exist");
    return false;
}

// Get an integer option.
int Options_impl::get_int_option(char const *name) const
{
    int index = get_option_index(name);
    if (index >= 0) {
        if (char const *val = get_option_value(index)) {
            int ival = 0;
            if (sscanf(val, "%d", &ival) == 1)
                return ival;
            // unknown ...
        }
        if (char const *val = get_option_default_value(index)) {
            int ival = 0;
            if (sscanf(val, "%d", &ival) == 1)
                return ival;
        }
        return 0;
    }
    MDL_ASSERT(!"integer option does not exist");
    return 0;
}

// Get an unsigned integer option.
unsigned Options_impl::get_unsigned_option(char const *name) const
{
    int index = get_option_index(name);
    if (index >= 0) {
        if (char const *val = get_option_value(index)) {
            unsigned uval = 0;
            if (sscanf(val, "%u", &uval) == 1)
                return uval;
            // unknown ...
        }
        if (char const *val = get_option_default_value(index)) {
            unsigned uval = 0;
            if (sscanf(val, "%u", &uval) == 1)
                return uval;
        }
        return 0;
    }
    MDL_ASSERT(!"unsigned integer option does not exist");
    return 0;
}

// Get a float option.
float Options_impl::get_float_option(char const *name) const
{
    int index = get_option_index(name);
    if (index >= 0) {
        if (char const *val = get_option_value(index)) {
            float fval = 0.f;
            if (sscanf(val, "%f", &fval) == 1)
                return fval;
            // unknown ...
        }
        if (char const *val = get_option_default_value(index)) {
            float fval = 0.f;
            if (sscanf(val, "%f", &fval) == 1)
                return fval;
        }
        return 0.f;
    }
    MDL_ASSERT(!"float option does not exist");
    return 0.f;
}

// Get a version option.
bool Options_impl::get_version_option(char const *name, unsigned &major, unsigned &minor) const
{
    major = minor = 0;
    int index = get_option_index(name);
    if (index >= 0) {
        char const *val = get_option_value(index);
        if (val == NULL)
            val = get_option_default_value(index);

        if (val != NULL) {
            char const *p = val;

            while ('0' <= *p && *p <= '9') {
                major *= 10;
                major += *p - '0';
                ++p;
            }

            if (p == val)
                return false;
            if (*p != '.')
                return false;
            ++p;

            while ('0' <= *p && *p <= '9') {
                minor *= 10;
                minor += *p - '0';
                ++p;
            }
            if (*p != '\0')
                return false;

            return true;
        }
        return false;
    }
    MDL_ASSERT(!"version option does not exist");
    return false;
}

// Get the data of a binary option.
BinaryOptionData Options_impl::get_binary_option(char const *name) const
{
    int index = get_option_index(name);
    if (index >= 0)
        return get_binary_option(index);

    MDL_ASSERT(!"binary option does not exist");
    return BinaryOptionData(NULL, 0);
}

// Get the option at given index.
Options_impl::Option const &Options_impl::get_option(int index) const
{
    MDL_ASSERT(0 <= index && index < m_options.size() && "Option index out of range");
    return m_options.at(index);
}

}  // mdl
}  // mi
