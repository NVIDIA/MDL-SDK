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

#ifndef MDL_COMPILERCORE_OPTIONS_H
#define MDL_COMPILERCORE_OPTIONS_H 1

#include <mi/mdl/mdl_options.h>

#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"

namespace mi {
namespace mdl {

///
/// Implementation of the Options interface.
///
class Options_impl : public Options {
    typedef Options Base;

public:
    /// An entry in the options table.
    class Option {
    public:
        /// Constructor for text options.
        Option(
            IAllocator *alloc,
            char const *name,
            char const *value,
            char const *def_value,
            char const *description)
        : m_name(name, alloc)
        , m_value(non_null(value), alloc)
        , m_def_value(non_null(def_value), alloc)
        , m_desc(non_null(description), alloc)
        , m_is_binary(false)
        , m_is_modified(false)
        {
        }

        /// Constructor for binary options.
        Option(
            IAllocator *alloc,
            char const *name,
            char const *data,
            size_t     size,
            char const *description)
        : m_name(name, alloc)
        , m_value(non_null(data), size, alloc)
        , m_def_value("", alloc)
        , m_desc(non_null(description), alloc)
        , m_is_binary(true)
        , m_is_modified(false)
        {
        }

        /// Get the name of the option.
        char const *get_name() const { return m_name.c_str(); }

        /// Get the value of the option if any.
        char const *get_value() const
        {
            return m_value.empty() ? NULL : m_value.c_str();
        }

        /// Get the binary value of the option.
        BinaryOptionData get_binary_data() const
        {
            if (m_value.empty()) return BinaryOptionData(NULL, 0);
            return BinaryOptionData(m_value.c_str(), m_value.size());
        }

        /// Get the default value of the option.
        char const *get_default_value() const
        {
            return m_def_value.empty() ? NULL : m_def_value.c_str();
        }

        /// get the description of the option if any.
        char const *get_description() const
        {
            return m_desc.empty() ? NULL : m_desc.c_str();
        }

        /// Set a new value.
        ///
        /// \param value  the new value
        void set_value(char const *value)
        {
            m_value = non_null(value);
            m_is_modified = true;
        }

        /// Set a new binary value.
        ///
        /// \param data  the new binary value
        /// \param size  the size of the data
        void set_binary_value(char const *data, size_t size)
        {
            if (data == NULL)
                m_value.clear();
            else
                m_value.assign(data, size);
            m_is_modified = true;
        }

        /// Reset to defaults.
        void reset_to_defaults()
        {
            m_value = m_def_value;
            m_is_modified = false;
        }

        /// Returns true if this is a binary option.
        bool is_binary() const { return m_is_binary; }

        /// Returns true if this option was modified.
        bool is_modified() const { return m_is_modified; }

    private:
        /// Helper, converts NULL pointer to empty string.
        static char const *non_null(char const *s) { return s != NULL ? s : ""; }

    private:
        /// The name of this option.
        string m_name;

        /// The value of this option.
        string m_value;

        /// The default value of this option.
        string m_def_value;

        /// The description of this option
        string m_desc;

        /// True, if this option is a binary option.
        bool m_is_binary;

        /// True, if this option was set by the user.
        bool m_is_modified;
    };

public:

    /// Get the number of options.
    int get_option_count() const MDL_FINAL;

    /// Get the index of the option with name or -1 if this option does not exists.
    ///
    /// \param      name    The name of the option.
    ///
    /// \returns            The index of the option.
    int get_option_index(char const *name) const MDL_FINAL;

    /// Get the name of the option at index.
    ///
    /// \param      index   The index of the option.
    ///
    /// \returns            The name of the option.
    char const *get_option_name(int index) const MDL_FINAL;

    /// Get the value of the option at index.
    ///
    /// \param      index   The index of the option.
    ///
    /// \returns            The value of the option.
    char const *get_option_value(int index) const MDL_FINAL;

    /// Get the data of the binary option at index.
    ///
    /// \param      index   The index of the binary option.
    ///
    /// \returns            The data of the binary option.
    BinaryOptionData get_binary_option(int index) const MDL_FINAL;

    /// Get the default value of the option at index.
    ///
    /// \param      index   The index of the option.
    ///
    /// \returns            The default value of the option.
    char const *get_option_default_value(int index) const MDL_FINAL;

    /// Get the description of the option at index.
    ///
    /// \param      index   The index of the option.
    ///
    /// \returns            The description of the option.
    char const *get_option_description(int index) const MDL_FINAL;

    /// Returns true if the requested option was modified.
    ///
    /// \param      index   The index of the option.
    ///
    /// \returns            The value of the option or NULL if the option is not a normal option.
    bool is_option_modified(int index) const MDL_FINAL;

    /// Set an option.
    ///
    /// It is only possible to set options that are actually present,
    /// as indicated by the above getter methods.
    /// If the name of an option was not found, this function returns false.
    /// In addition, this function may return false because the option value is not acceptable
    /// for a particular option or the option is a binary option.
    ///
    /// \param      name    The name of the option.
    /// \param      value   The value of the option.
    ///
    /// \returns            True on success and false on failure.
    bool set_option(char const *name, char const *value) MDL_FINAL;

    /// Set a binary option.
    ///
    /// It is only possible to set options that are actually present,
    /// as indicated by the above getter methods.
    /// If the name of an option was not found, this function returns false.
    /// In addition, this function may return false because the particular option is not a
    /// binary option.
    ///
    /// \param      name    The name of the binary option.
    /// \param      data    The data of the binary option.
    /// \param      size    The size of the binary data.
    ///
    /// \returns            True on success and false on failure.
    bool set_binary_option(char const *name, char const *data, size_t size) MDL_FINAL;

    /// Reset all options to their default values.
    void reset_to_defaults() MDL_FINAL;

    /// Add a new option.
    ///
    /// \param name         The name of the option.
    /// \param def_value    The default value of this option.
    /// \param description  The description of this option.
    void add_option(char const *name, char const *def_value, char const *description);

    /// Add a new binary option.
    ///
    /// \param name         The name of the option.
    /// \param description  The description of this option.
    void add_binary_option(char const *name, char const *description);

    /// Get a string option.
    ///
    /// \param name         The name of the option.
    ///
    /// \return the string option value or its default value if not set
    char const *get_string_option(char const *name) const;

    /// Get a bool option.
    ///
    /// \param name         The name of the option.
    ///
    /// \return the integer option value or its default value if not set
    bool get_bool_option(char const *name) const;

    /// Get an integer option.
    ///
    /// \param name         The name of the option.
    ///
    /// \return the integer option value or its default value if not set
    int get_int_option(char const *name) const;

    /// Get an unsigned integer option.
    ///
    /// \param name         The name of the option.
    ///
    /// \return the unsigned integer option value or its default value if not set
    unsigned get_unsigned_option(char const *name) const;

    /// Get a float option.
    ///
    /// \param name         The name of the option.
    ///
    /// \return the float option value or its default value if not set
    float get_float_option(char const *name) const;

    /// Get a version option.
    ///
    /// \param[in]  name     The name of the option.
    /// \param[out] major    The major version.
    /// \param[out] minor    The minor version.
    ///
    /// \return true if the version was correctly parsed, false otherwise
    bool get_version_option(char const *name, unsigned &major, unsigned &minor) const;

    /// Get the data of a binary option.
    ///
    /// \param  name         The name of the option.
    ///
    /// \returns             The data of the binary option or NULL if the option is not binary.
    BinaryOptionData get_binary_option(char const *name) const;

    /// Get the option at given index.
    ///
    /// \param      index   The index of the option.
    Option const &get_option(int index) const;

public:
    /// Constructor.
    ///
    /// \param alloc  the allocator to allocate from.
    explicit Options_impl(IAllocator *alloc);

    /// Destructor.
    virtual ~Options_impl();

private:
    // non copyable
    Options_impl(Options_impl const &) MDL_DELETED_FUNCTION;
    Options_impl &operator=(Options_impl const &) MDL_DELETED_FUNCTION;

private:
    // The used allocator.
    IAllocator *m_alloc;

    typedef vector<Option>::Type Option_vec;

    /// The vector of all options.
    Option_vec m_options;
};

}  // mdl
}  // mi

#endif
