/***************************************************************************************************
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
 **************************************************************************************************/
/// \file
/// \brief API component for MDL internationalization settings.

#ifndef MI_NEURAYLIB_IMDL_I18N_CONFIGURATION_H
#define MI_NEURAYLIB_IMDL_I18N_CONFIGURATION_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_configuration
@{
*/

/// This interface is used to query and change MDL internationalization settings.
///
/// Here is a sample pseudo-code to illustrate the usage of this component:
///
/// \code
///     // Get the internationalization configuration component
///     using mi::base::Handle;
///     using mi::neuraylib::IMdl_i18n_configuration;
///     mi::base::Handle<mi::neuraylib::INeuray> neuray(...);
///     Handle<IMdl_i18n_configuration> i18n_configuration(
///             neuray->get_api_component<IMdl_i18n_configuration>());
///     const char * locale = NULL;
///
///     // Set locale to French language
///     i18n_configuration->set_locale("fr");
///
///     // Query defined locale
///     locale = i18n_configuration->get_locale();
///
///     // Query sytem defined locale
///     locale = i18n_configuration->get_system_locale();
///
///     // Use sytem defined locale
///     i18n_configuration->set_locale(i18n_configuration->get_system_keyword());
///
///     // Disable any translation
///     i18n_configuration->set_locale(NULL);
/// \endcode
///
class IMdl_i18n_configuration : public
    mi::base::Interface_declare<0xb28d4381,0x5760,0x4c1a,0xbe,0xa1,0x3f,0xa7,0x96,0x8a,0x86,0x28>
{
public:
    /// \name MDL Locale
    //@{

    /// Specifies which locale to use to translate annotations.
    ///
    /// This interface can be used to set the locale to use for translation,
    /// overwite the system locale or disable any translation. By default, the locale
    /// defined by the system is used.
    ///
    /// This function can only be called before \neurayProductName has been started.
    ///
    /// \param locale
    ///     The locale to be used.
    ///     Values:
    ///     - a string following ISO 639-1 standard (See https://en.wikipedia.org/wiki/ISO_639-1),
    ///     - the system keyword (this will fallback to use system defined locale).
    ///       See #mi::neuraylib::IMdl_i18n_configuration::get_system_keyword().
    ///     - a \c NULL value (this will disable any translation),
    ///                            
    /// \return
    ///     -  0: Success.
    ///     - -1: Failure.
    ///           This function can only be called before \neurayProductName has been started.
    ///
    /// \note
    ///     The locale "C" is ignored and will disable any translation.
    ///     By default the system locale is used.
    ///
    virtual Sint32 set_locale( const char* locale) = 0;

    /// Returns the locale used to translate annotations.
    ///
    /// \return
    ///     - the string passed to #mi::neuraylib::IMdl_i18n_configuration::set_locale() or,
    ///     - the system keyword #mi::neuraylib::IMdl_i18n_configuration::get_system_keyword() or,
    ///     - a \c NULL value if translation is disabled
    ///
    virtual const char* get_locale() const = 0;

    /// Returns the system locale.
    ///
    /// \return
    ///     The name of the locale set by the system or \c NULL if locale is not set.
    ///
    virtual const char* get_system_locale() const = 0;

    /// Returns the reserved string which can be used to fallback to system locale.
    /// See #mi::neuraylib::IMdl_i18n_configuration::set_locale() for usage.
    ///
    /// \return
    ///     The reserved string which is used to fallback to system locale.
    ///
    virtual const char* get_system_keyword() const = 0;

    //@}
};

/*@}*/ // end group mi_neuray_configuration

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_I18N_CONFIGURATION_H
