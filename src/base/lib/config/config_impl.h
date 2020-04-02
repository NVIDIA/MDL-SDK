/***************************************************************************************************
 * Copyright (c) 2009-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The config module implementation.

#ifndef BASE_LIB_CONFIG_CONFIG_IMPL_H
#define BASE_LIB_CONFIG_CONFIG_IMPL_H

#include "config.h"
#include <base/util/registry/config_registry_impl.h>

#include <mi/base/lock.h>

namespace MI {
namespace CONFIG {

/// Implementation of the Config_module.
class Config_module_impl : public Config_module
{
  public:
    /// Initialize the module.
    bool init();
    /// Shut down this module.
    void exit();

    /// Constructor.
    Config_module_impl();
    /// Destructor.
    ~Config_module_impl();

    /// Configure logging. Since now both the command line and the initrc are read, explicitly
    /// tell the log module to use the read values to initialize itself finally.
    virtual void configure_log_module();

    /// For command-line overrides. Variables set with this will be ignored in the config file
    /// (except that the file's help text is extracted). The string is parsed just like a
    /// config file line.
    /// \param opt option to parse
    /// \return succes, ie false on failure
    virtual bool override(
        const char* opt);
    /// Retrieve the value of a given config entry key as a string.
    virtual std::string get_config_value_as_string(
        const std::string& key) const;

    /// Update the host's config parameter with the current registered value.
    /// \param mod module ID of caller
    /// \param name name of variable
    /// \param help meaning of var, for http/comments
    /// \param [out] value module values to be updated
    /// \return true when a value for \p name was already registered, false else
    virtual bool update(
        const char* name,
        const char* help,
        std::string& value);

    /// Is now the initialization of the config module complete?
    /// \return true when initialization is complete
    virtual bool is_initialization_complete() const;

    /// Set initialization status that can be obtained by is_initialization_complete().
    /// Usually read_commandline() takes care of this setting, but not all
    /// products parse the command line, e.g., neuraylib.
    virtual void set_initialization_complete(
        bool flag);

    /// Retrieve the configuration.
    Config_registry& get_configuration();
    /// Retrieve the configuration.
    const Config_registry& get_configuration() const;

    /// Retrieve the product version.
    std::string get_product_version() const;
    /// Retrieve the product name.
    std::string get_product_name() const;

  private:
    mutable mi::base::Lock m_lock;		    ///< for thread-safe access.

    /// \todo This variable should be dealt with in a mt-safe manner.
    bool m_read_commandline_already_called;	    ///< was the member fct already called?
    Config_registry_impl m_configuration;	    ///< collected config variables - NOT THREADSAFE
};

}
}

#endif
