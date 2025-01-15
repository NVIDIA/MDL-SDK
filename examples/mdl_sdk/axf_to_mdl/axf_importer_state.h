/***************************************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Our own state adaption.

#ifndef EXAMPLE_AXF_TO_MDL_AXF_IMPORTER_STATE_H
#define EXAMPLE_AXF_TO_MDL_AXF_IMPORTER_STATE_H

#include <string>

#include "example_shared.h"

#include "axf_importer_options.h"

namespace mi {
namespace examples {
namespace impaxf {

enum Color_representation {
    COLOR_REP_RGB,
    COLOR_REP_SPECTRAL,
    COLOR_REP_ALL
};

/// Provides an adapted version of the importer state.
class Axf_impexp_state// : public ::mi::base::Interface_implement<mi::neuraylib::IImpexp_state>
{
  public: 
    /// Default Constructor
    Axf_impexp_state()
      : m_line_number(1)
    {}

    /// Constructor
    /// \param url The url of the file being imported.
    /// \param parent_state The parent state of the this state.
    Axf_impexp_state(
        const std::string& url)
      : m_line_number(1),
        m_url(url),
        m_element_list_flag(false)
    {}

    /// Destructor
    ~Axf_impexp_state() {}

    /// Initializes the state.
    /// \param i_url The url of the file being imported.
    /// \param i_parent_state The parent state of this state.
    void init( 
        const std::string& i_url);

    /// Returns the url of the file being imported.
    const char* get_uri() const;

    /// Returns MDL output filename.
    const char* get_mdl_output_filename() const {
        return m_mdl_output_filename.c_str();
    }

    /// Returns the prefix to be append to every imported element.
    const char* get_prefix() const;

    const char* get_target_color_space() const {
        return m_axf_color_space.c_str();
    }

    const std::string get_module_prefix() const {
        return m_axf_module_prefix;
    }

    Color_representation get_color_rep() const {
        return m_color_rep;
    }

    /// Returns true if all elements should be recorded.
    bool get_element_list_flag() const { return m_element_list_flag; }

    /// Returns the current line number
    mi::Uint32 get_line_number() const;

    /// Sets the current line number.
    void set_line_number(
        mi::Uint32 n);

    /// Increments the current line number.
    void incr_line_number();

    /// Checks whether \p importer_options contains the keys 'prefix' and 'list_elements' and
    /// sets the members in this class accordingly.
    void parse_importer_options(
        const mi::IMap* importer_options);

    /// Checks whether \p importer_options contains the keys 'prefix' and 'list_elements' and
    /// sets the members in this class accordingly.
    void parse_importer_options(
        const Axf_importer_options &importer_options);

    /// Creates an instance of mi::IMap suitable to pass to import calls.
    ///
    /// The keys "prefix" and "list_elements" are set according the values returned from
    /// get_prefix() and get_element_list_flag().
    mi::IMap* create_importer_options(
        mi::neuraylib::ITransaction* transaction) const;

  private:
    mi::Uint32 m_line_number;
    //const mi::neuraylib::IImpexp_state* m_parent_state;
    std::string m_url;
    bool m_element_list_flag;
    std::string m_mdl_output_filename;
    std::string m_name_space;
    std::string m_axf_color_space;
    std::string m_axf_module_prefix;
    Color_representation m_color_rep;

    // Suppress accidental copying.
    Axf_impexp_state(const Axf_impexp_state&);
    Axf_impexp_state& operator=(const Axf_impexp_state&);
};

}
}
}

#endif
