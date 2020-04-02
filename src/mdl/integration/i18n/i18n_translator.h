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
/// \file
/// \brief  Utilities for MDL i18n.
///
#pragma once

#include <string>
#include <mi/base/types.h>
#include <mi/base/interface_implement.h>
#include "i_i18n.h"

namespace MI {
namespace MDL {
namespace I18N {

class Database;

/// This class is the main interface for translation of MDL strings.
class Mdl_translator_impl : public Mdl_translator_module
{
public:
    Mdl_translator_impl();
    ~Mdl_translator_impl();

    bool init();

    void exit();

public:
    /// Does the given annotation needs translation.
    ///
    /// \param name Name of the annotation (e.g. "::anno::display_name(string)")
    ///
    /// \return true if this annotation needs to be translated.
    ///
    /// \note   The following annotations are currently translated:
    ///
    ///         "::anno::display_name(string)"
    ///         "::anno::in_group(string)"
    ///         "::anno::in_group(string,string)"
    ///         "::anno::in_group(string,string,string)"
    ///         "::anno::key_words(string[N])"
    ///         "::anno::copyright_notice(string)"
    ///         "::anno::description(string)"
    ///         "::anno::author(string)"
    ///         "::anno::contributor(string)"
    ///         "::anno::unused(string)"
    ///         "::anno::deprecated(string)"
    ///
    bool need_translation(const std::string & name) const;

    /// Translate the source string in the given context (i.e. MDL qualified name)
    ///
    /// \param sentence
    ///     Input/Output translation text.
    ///
    /// \return
    ///     -  0: Text was translated properly.
    ///     - -1: Text was not translated.
    ///     - -2: Translation is disabled.
    ///     - other: Unknown error
    ///
    mi::Sint32 translate(Translation_unit & sentence) const;

    /// Specifies which locale to use to translate annotations.
    ///
    /// \param locale
    ///     The locale to be used.
    ///     Values:
    ///     - a string following ISO 639-1 standard (See https://en.wikipedia.org/wiki/ISO_639-1),
    ///     - empty string will disable any translation
    ///
    /// \return
    ///     -  0: Success
    ///     - -1: Unknown error
    ///
    mi::Sint32 set_locale(const std::string & locale);

    /// Get the specified locale and test is translation is enabled.
    ///
    /// \param locale
    ///     The locale which is set.
    ///     This parameter is only valid if localization/translation is enabled.
    ///
    /// \return
    ///     - true if localization/translation is enabled.
    ///     - false if localization/translation is disabled.
    ///
    bool is_localization_set(std::string & locale) const;

    /// Cleanup the translation database.
    ///
    /// \return
    ///     -  0: Success
    ///     - -1: Unknown error
    ///
    mi::Sint32 cleanup_database();

private:
    Database * m_database = NULL;
    std::string m_locale;
};

} // namespace I18N
} // namespace MDL
} // namespace MI
