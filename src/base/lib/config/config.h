/***************************************************************************************************
 * Copyright (c) 2004-2020, NVIDIA CORPORATION. All rights reserved.
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
///
/// The config module maintains configuration values that are queried by modules on startup, to
/// initialize the module members. Module constructors first init their member fields to default
/// values, and then call mod_config->update to allow the config file to override the defaults.
/// When all values have been updated, the module may perform consistency checks. If you have a
/// module class for module \c X that has private config variables, do this
/// \code
/// class Mod_X: public Module {
///  private:
///   int   m_integer;
///   float m_array[2];
///   char *m_const;
///   char *m_alloc;
///   bool  m_bool;
///
///   Mod_X::Mod_X() {
///     m_integer  = 1;			// set defaults
///     m_array[0] = m_array[1] = 5;
///     m_const    = "hello, neuray";
///     m_alloc    = mod_string->dup("rayzilla");
///     m_bool     = true;
///
///     mod_config->update(M_X, "integer", "help", &m_integer);
///     mod_config->update(M_X, "array",   "help",  m_array, 2);
///     mod_config->update(M_X, "const",   "help", &m_const);
///     mod_config->update(M_X, "alloc",   "help", &m_alloc, 1, true);
///     mod_config->update(M_X, "bool",    "help", &m_bool);
/// \endcode
///
/// \c mod_config->update changes member fields m_xxx to values read from a config file;
/// m_xxx remains unchanged if the config file does not contain a value. Values may be strings,
/// integers or fixed arrays of integers, or floats or fixed arrays of floats. Note that Config
/// needs \c Disk, so \c Disk is initialized before \c Config can supply values. Note that after
/// successfully updating a string (update returned true), the string is always allocatesd even
/// if it started as a literal like m_const in the example above. update cannot assume that the
/// buffer is writable and large enough for the new string.
///
/// By convention, the variable name (second update argument) is the name of the member field
/// without the leading "m_". Config can also keep a help text for a variable (third update
/// argument). This should be a short one-line description of the meaning of the variable. Supply
/// units of measurement if applicable, like [ms] for milliseconds or [MB] for megabytes. It is
/// stored as a comment in the config file, and is also presented to the user in HTML pages.
///
/// None of the code here is mp-safe. It is assumed that all modules are initialized in a single
/// function at startup, not from various threads. The developer edition (-D DEVED) does not
/// support reading or writing an initrc file, but it does support the update mechanism.

#ifndef BASE_LIB_CONFIG_H
#define BASE_LIB_CONFIG_H

#include <base/system/main/i_module.h>

#include <string>

namespace MI {
namespace SYSTEM { class Module_registration_entry; }
namespace CONFIG {

class Config_registry;

/// The module for initializing CONFIG.
class Config_module : public SYSTEM::IModule
{
  public:
    /// Configure logging. Since now both the command line and the initrc are read, explicitly
    /// tell the log module to use the read values to initialize itself finally.
    virtual void configure_log_module() = 0;

    /// For command-line overrides. Variables set with this will be ignored in the config file
    /// (except that the file's help text is extracted). The string is parsed just like a
    /// config file line.
    /// \param opt option to parse
    /// \return success, false on failure
    virtual bool override(
        const char* opt) = 0;
    /// Retrieve the value of a given config entry key as a string.
    virtual std::string get_config_value_as_string(
        const std::string& key) const = 0;

    /// Is now the initialization of the config module complete?
    /// \return true when initialization is complete
    virtual bool is_initialization_complete() const = 0;

    /// Set initialization status that can be obtained by is_initialization_complete().
    /// Usually read_commandline() takes care of this setting, but not all
    /// products parse the command line, e.g., neuraylib.
    virtual void set_initialization_complete(
        bool flag) = 0;

    /// Retrieve the configuration.
    virtual Config_registry& get_configuration() = 0;
    /// Retrieve the configuration.
    virtual const Config_registry& get_configuration() const = 0;

    /// Retrieve the product version.
    virtual std::string get_product_version() const = 0;
    /// Retrieve the product name.
    virtual std::string get_product_name() const = 0;

    /// \name ModuleImpl
    /// Required functionality for implementing a \c SYSTEM::IModule.
    //@{
    /// Retrieve the name.
    static const char* get_name() { return "CONFIG"; }
    /// Allow link time detection.
    static SYSTEM::Module_registration_entry* get_instance();
    //@}

  private:
    /// Update the host's config parameter with the current registered value.
    /// \param mod module ID of caller
    /// \param name name of variable
    /// \param help meaning of var, for http/comments
    /// \param [out] value module values to be updated
    /// \return true when a value for \p name was already registered, false else
    virtual bool update(
        const char* name,
        const char* help,
        std::string& value) = 0;
};

}
}

#endif

