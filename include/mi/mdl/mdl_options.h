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
/// \file mi/mdl/mdl_options.h
/// \brief Interfaces for options in the MDL Core API
#ifndef MDL_OPTIONS_H
#define MDL_OPTIONS_H 1

#include <cstddef>

namespace mi {
namespace mdl {

/// Helper struct for easily returning data and size of a binary option.
struct BinaryOptionData {
    /// Constructor.
    ///
    /// \param data  pointer to the binary data
    /// \param size  size of the binary data
    BinaryOptionData(char const *data, size_t size)
    : data(data), size(size)
    {}

    /// The data of the option.
    char const *data;

    /// The size of the option.
    size_t     size;
};


/// The interface to a set of options.
///
/// Several entities in the MDL Core API have options, like the MDL compiler, backends, etc.
class Options {
public:
    /// Get the number of options.
    virtual int get_option_count() const = 0;

    /// Get the index of the option with name or -1 if this option does not exists.
    ///
    /// \param name    The name of the option.
    ///
    /// \returns       The index of the option.
    virtual int get_option_index(char const *name) const = 0;

    /// Get the name of the option at index.
    ///
    /// \param index   The index of the option.
    ///
    /// \returns       The name of the option.
    virtual char const *get_option_name(int index) const = 0;

    /// Get the value of the option at index.
    ///
    /// \param index   The index of the option.
    ///
    /// \returns       The value of the option or NULL if the option is not a normal option.
    virtual char const *get_option_value(int index) const = 0;

    /// Get the data of the binary option at index.
    ///
    /// \param index   The index of the binary option.
    ///
    /// \returns       The data of the binary option or NULL if the option is not a
    ///                binary option.
    virtual BinaryOptionData get_binary_option(int index) const = 0;

    /// Get the default value of the option at index.
    ///
    /// \param index   The index of the option.
    ///
    /// \returns       The default value of the option.
    virtual char const *get_option_default_value(int index) const = 0;

    /// Get the description of the option at index.
    ///
    /// \param index   The index of the option.
    ///
    /// \returns       The description of the option.
    virtual char const *get_option_description(int index) const = 0;

    /// Returns true if the requested option was modified.
    ///
    /// \param index   The index of the option.
    ///
    /// \returns       The value of the option or NULL if the option is not a normal option.
    virtual bool is_option_modified(int index) const = 0;

    /// Set an option.
    ///
    /// It is only possible to set options that are actually present,
    /// as indicated by the above getter methods.
    /// If the name of an option was not found, this function returns false.
    /// In addition, this function may return false because the option value is not acceptable
    /// for a particular option or the option is a binary option.
    ///
    /// \param name    The name of the option.
    /// \param value   The value of the option.
    ///
    /// \returns       True on success and false on failure.
    virtual bool set_option(char const *name, char const *value) = 0;

    /// Set a binary option.
    ///
    /// It is only possible to set options that are actually present,
    /// as indicated by the above getter methods.
    /// If the name of an option was not found, this function returns false.
    /// In addition, this function may return false because the particular option is not a
    /// binary option.
    ///
    /// \param name    The name of the binary option.
    /// \param data    The data of the binary option.
    /// \param size    The size of the binary data.
    ///
    /// \returns       True on success and false on failure.
    virtual bool set_binary_option(char const *name, char const *data, size_t size) = 0;

    /// Reset all options to their default values.
    virtual void reset_to_defaults() = 0;
};

}  // mdl
}  // mi

#endif
