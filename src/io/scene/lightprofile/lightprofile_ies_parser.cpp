/***************************************************************************************************
 * Copyright (c) 2007-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Parser for the IES file formats.
///
/// The present parser reads lightprofile file given in the IES formats (cp. 'IESNA Standard
/// File Format for Electronic Transfer of Photometric Data' by the IESNA Computer Committee).

#include "pch.h"

#include <mi/base/handle.h>
#include <mi/math/function.h>
#include <mi/neuraylib/ilightprofile.h>
#include <mi/neuraylib/ireader.h>

#include <cctype>
#include <cmath>
#include <string>
#include <cstring>
#include <vector>

#include <base/util/string_utils/i_string_utils.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/util/string_utils/i_string_utils.h>
#include <base/hal/disk/disk.h>
#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/lib/log/log.h>
#include <base/lib/path/i_path.h>

namespace mi { namespace neuraylib { class IReader; } }

namespace MI {
namespace LIGHTPROFILE {

namespace {

// Valid IES versions strings
const static char* IESNA_1991 = "IESNA91";
const static char* IESNA_1995 = "IESNA:LM-63-1995";
const static char* IESNA_2002 = "IESNA:LM-63-2002";
const static char* IESNA_UNKNOWN = "IESNA:LM-63-";

// Specification used to give additional notes on parser errors
//
// IESNA Standard File Format for Electronic Transfer of Photometric Data
// by the IESNA Computer Committee
// http://lumen.iee.put.poznan.pl/kw/iesna.txt
const static char* ies_spec =
    "IESNA LM-63-95 - IESNA Standard File Format for Electronic Transfer of Photometric Data";

// Maximal string length not-including end of line
const static Uint MAX_LINE_LENGTH = 4096;   // though standard is 132/256

} // namespace

//
// IES file parser
//
// IESNA Standard File Format for Electronic Transfer of Photometric Data
// by the IESNA Computer Committee
// [Juri has filed a copy of the official specification by the IESNA]
//
class Lightprofile_ies_parser
{
public:
    // IES versions according to IES spec (which might be continued ...)
    enum Version {
        IESNA_LM_63_1986 = 0,
        IESNA_LM_63_1991,
        IESNA_LM_63_1995,
        IESNA_LM_63_2002
    };

    // Orientation of the lamp within the luminaire (cp. Fig.1 in specification)
    enum Lamp_orientation {
        VERTICAL_BASE_UP_DOWN               = 1,  // Vertical base up or vertical base down
        HORIZONTAL_ALONG_90_DEGREE_PLANE    = 2,  // Horizontal along the 90 degree plane
        HORIZONTAL_ALONG_0_DEGREE_PLANE     = 3   // Horizontal along the 0 degree plane
    };
    // Photometric types
    enum Photometric_type {
        TYPE_C = 1,
        TYPE_B,
        TYPE_A
    };
    // Unit types
    enum Unit_type {
        FEET  = 1,
        METER
    };

    // C'tor
    Lightprofile_ies_parser(mi::neuraylib::IReader* reader, const std::string& log_identifier);

    bool setup_lightprofile(
        Uint flags,
        Uint hermite,
        Uint& horizontal_resolution,
        Uint& vertical_resolution,
        Scalar& phi,
        Scalar& theta,
        Scalar& d_phi,
        Scalar& d_theta,
        std::vector<Scalar>& grid_values);

private:
    /// Tokenizes one line.
    ///
    /// Separators are comma, semicolon, whitespace, CR, LF. Empty tokens are skipped.
    void get_tokens(
        char* line, std::vector<std::string>& valid_tokens);

    /// Tokenizes one or several lines.
    ///
    /// Read entire lines from reader until required number of token have been found or EOF is
    /// reached. Separators are comma, semicolon, whitespace, CR, LF. Empty tokens are skipped.
    void get_tokens(
        mi::neuraylib::IReader* reader, Uint nb_tokens, std::vector<std::string>& tokens);

    bool parse_version(
        char* line);                            // Line to parse
    bool parse_label(
        char* line);                            // Line to parse
    void parse_tilt(
        char* line);                            // Line to parse
    void parse_tilt_values();
    void parse_lamp_data();
    void parse_additional_data();
    void parse_angles_data(
        std::vector<Scalar>&  angles_data,      // Targeted data array
        const std::string&    description);

    // Returns true if end of line has been reached
    char* parse_next_token(
        char*           line_pointer,           // Pointing to line position
        std::string&  token) const;             // Returned token

    // File infos
    mi::base::Handle<mi::neuraylib::IReader> m_reader;
    std::string                              m_log_identifier;
    bool                                     m_valid;
    bool                                     m_skip_line_length_warning;

    // Actual IES lightprofile version
    Version                     m_version;

    // Tilt values according to the IES standard format
    bool                        m_tilt_values_exist;          // Ignored
    Lamp_orientation            m_lamp_to_luminaire_geometry; // Ignored
    Uint                        m_nb_angles;                  // Ignored
    Uint                        m_nb_multiplying_factors;     // Ignored
    std::vector<Scalar>         m_angles;                     // Ignored
    std::vector<Scalar>         m_multiplying_factors;        // Ignored

    // Lamp data according to the IES standard format
    Uint                        m_nb_of_lamps;      // Number of lights. A single lightprofile
                                                    // file might contain several lights that
                                                    // correspond to a lightprofile. Ignored
    Scalar                      m_lumens_per_lamp;  // Ignored
    Scalar                      m_candela_multiplier;
    Uint                        m_nb_vertical_angles;
    Uint                        m_nb_horizontal_angles;
    Photometric_type            m_photometric_type;
    Unit_type                   m_units_type;       // Ignored
    Scalar                      m_width;            // Ignored
    Scalar                      m_length;           // Ignored
    Scalar                      m_height;           // Ignored

    // Additional data according to the IES standard format
    Scalar                      m_ballast_factor;
    Scalar                      m_ballast_lamp_factor;
    Scalar                      m_input_watts;      // Ignored

    // Angles in vertical and horizontal directions
    std::vector<Scalar>         m_vertical_angles;
    std::vector<Scalar>         m_horizontal_angles;

    // Candela values in horizontal-first-vertical-second-order
    std::vector<std::vector<Scalar> > m_candela_values;
};

//
// IES file parser
//
Lightprofile_ies_parser::Lightprofile_ies_parser(
    mi::neuraylib::IReader* reader, const std::string& log_identifier)
  : m_reader(reader, mi::base::DUP_INTERFACE),
    m_log_identifier(log_identifier),
    m_valid(true),
    m_skip_line_length_warning(false)
{

    char line[MAX_LINE_LENGTH];
    m_reader->readline(line, sizeof(line));

    // Parse header information
    bool has_version_string = parse_version(line);
    if(has_version_string && !m_reader->eof())
        m_reader->readline(line, sizeof(line));

    bool is_label = parse_label(line);
    while(is_label && !m_reader->eof())
    {
        m_reader->readline(line, sizeof(line));
        is_label = parse_label(line);
    }

    if(m_valid)
        parse_tilt(line);

    if(m_valid)
        parse_lamp_data();

    if(m_valid)
        parse_additional_data();

    if(m_valid)
        parse_angles_data(m_vertical_angles, "vertical angles");

    if(m_valid)
        parse_angles_data(m_horizontal_angles, "horizontal angles");

    if(m_valid)
    {
        for(Uint i=0; i<m_nb_horizontal_angles; i++)
        {
            std::string description = "candela values for horizontal angle #";
            description += std::to_string(i);
            if(m_valid)
                parse_angles_data(m_candela_values[i], description);
        }
    }
}

void Lightprofile_ies_parser::get_tokens(char* line, std::vector<std::string>& valid_tokens)
{
    std::vector<std::string> tokens;
    MI::STRING::split(line, ",; \t\r\n", tokens);

    size_t n = tokens.size();
    for (size_t i=0; i<n; ++i) {
        const std::string& token = tokens[i];
        if (!token.empty())
            valid_tokens.push_back(token);
    }
}

void Lightprofile_ies_parser::get_tokens(
    mi::neuraylib::IReader* reader, Uint nb_tokens, std::vector<std::string>& tokens)
{
    char line[MAX_LINE_LENGTH];
    while (tokens.size()<nb_tokens && !reader->eof()) {
        reader->readline(line, sizeof(line));
        if (strlen(line) >= MAX_LINE_LENGTH-1) {
            if (!m_skip_line_length_warning) {
                LOG::mod_log->error( M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                   "Line with at least %" FMT_SIZE_T " characters in %s too long.",
                   strlen(line), m_log_identifier.c_str());
                m_skip_line_length_warning = true;
            }
            tokens.clear();
            return;
        }
        get_tokens(line, tokens);
    }
}

//
// Parse version (if it exists...)
// Returns true if the version string has been found. If no version string is given
// then false will be returned so that the caller doesn't read the next line for parsing
//
bool Lightprofile_ies_parser::parse_version(
    char* version)                  // Line to parse
{
    while(*version && isspace(static_cast<unsigned char>(*version)))
        version++; // remove trailing white space

    if(!strncmp(IESNA_1991, version, strlen(IESNA_1991)))
    {
        m_version = IESNA_LM_63_1991;
        return true;
    }
    else if(!strncmp(IESNA_1995, version, strlen(IESNA_1995)))
    {
        m_version = IESNA_LM_63_1995;
        return true;
    }
    else if(!strncmp(IESNA_2002, version, strlen(IESNA_2002)))
    {
        m_version = IESNA_LM_63_2002;
        return true;
    }
    else if(!strncmp(IESNA_UNKNOWN, version, strlen(IESNA_UNKNOWN)))
    {
        // if a future version comes up, interpret it is as the highest version
        // known to the parser (instead of falling back to the old standard)
        LOG::mod_log->warning(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Unsupported IES version '%s' used in %s\n"
                "Data might not be imported correctly",
                version, m_log_identifier.c_str());

        m_version = IESNA_LM_63_2002;
        return true;
    }
    else
    {
        // no header, fall back old standard
        m_version = IESNA_LM_63_1986;
        return false;
    }
}

//
// Parse lamp data
//
void Lightprofile_ies_parser::parse_lamp_data()
{
    const Uint nb_required_lamp_values = 10;

    // Tokenize lamp data
    std::vector<std::string> tokens;
    get_tokens(m_reader.get(), nb_required_lamp_values, tokens);
    const Uint size = tokens.size();

    if(size!=nb_required_lamp_values)
    {
        LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
            "Unable to parse lamp data in %s. "
            "%u values have been found but %u have to be specified on dedicated possibly "
            "separated line(s). \n"
            "Please, see also '%s'",
            m_log_identifier.c_str(), size, nb_required_lamp_values, ies_spec);

        if(m_reader->eof())
        {
            LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Unexpected end of file in %s. ", m_log_identifier.c_str());
        }
        m_valid = false;
        return;
    }
    m_nb_of_lamps           = STRING::lexicographic_cast_s<MI::Uint, std::string>(tokens[0]);
    m_lumens_per_lamp       = STRING::lexicographic_cast_s<MI::Scalar, std::string>(tokens[1]);
    m_candela_multiplier    = STRING::lexicographic_cast_s<MI::Scalar, std::string>(tokens[2]);
    m_nb_vertical_angles    = STRING::lexicographic_cast_s<MI::Uint, std::string>(tokens[3]);
    m_vertical_angles.resize(m_nb_vertical_angles);
    m_nb_horizontal_angles  = STRING::lexicographic_cast_s<MI::Uint, std::string>(tokens[4]);
    m_horizontal_angles.resize(m_nb_horizontal_angles);
    for(Uint i=0; i<m_nb_horizontal_angles; i++)
    {
        std::vector<Scalar> vertical_candela_values(m_nb_vertical_angles);
        m_candela_values.push_back(vertical_candela_values);
    }
    Uint photometric_type   = STRING::lexicographic_cast_s<MI::Uint, std::string>(tokens[5]);
    switch(photometric_type)
    {
        case TYPE_C:    m_photometric_type = TYPE_C;    break;
        case TYPE_B:    m_photometric_type = TYPE_B;    break;
        case TYPE_A:
            LOG::mod_log->warning(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Photometric type A in %s is not supported. ", m_log_identifier.c_str());
            m_photometric_type = TYPE_A;
            break;
        default:
            LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Invalid photometric type %u in %s used. "
                "Valid photometric types are: 1 (Type C), 2 (Type B), or 3 (Type A). "
                "Type C will be assumed. \n"
                "Please, see also '%s'",
                photometric_type, m_log_identifier.c_str(), ies_spec);
            m_photometric_type = TYPE_C;
            break;
    }
    Uint units_type         = STRING::lexicographic_cast_s<MI::Uint, std::string>(tokens[6]);
    switch(units_type)
    {
        case FEET:      m_units_type = FEET;    break;
        case METER:     m_units_type = METER;   break;
        default:
            LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Invalid units type %u in %s used. "
                "Valid units types are: 1 (feet) or 2 (meters). "
                "'meters' will be assumed.\n"
                "Please, see also '%s'",
                units_type, m_log_identifier.c_str(), ies_spec);
            m_units_type = METER;
            break;
    }
    m_width                 = STRING::lexicographic_cast_s<MI::Scalar, std::string>(tokens[7]);
    m_length                = STRING::lexicographic_cast_s<MI::Scalar, std::string>(tokens[8]);
    m_height                = STRING::lexicographic_cast_s<MI::Scalar, std::string>(tokens[9]);
}

//
// Parse additional data
//
void Lightprofile_ies_parser::parse_additional_data()
{
    const Uint nb_required_data_values = 3;

    // Tokenize additional data
    std::vector<std::string> tokens;
    get_tokens(m_reader.get(), nb_required_data_values, tokens);
    const Uint size = tokens.size();

    if(size!=nb_required_data_values)
    {
        LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
            "Unable to parse additional data in %s. "
            "%u values have been found but %u have to be specified on dedicated possibly "
            "separated line(s). \n"
            "Please, see also '%s'",
            m_log_identifier.c_str(), size, nb_required_data_values, ies_spec);

        if(m_reader->eof())
        {
            LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Unexpected end of file in %s. ", m_log_identifier.c_str());
        }
        m_valid = false;
        return;
    }

    m_ballast_factor        = STRING::lexicographic_cast_s<MI::Scalar, std::string>(tokens[0]);
    m_ballast_lamp_factor   = STRING::lexicographic_cast_s<MI::Scalar, std::string>(tokens[1]);
    m_input_watts           = STRING::lexicographic_cast_s<MI::Scalar, std::string>(tokens[2]);

}

//
// Parse angle data such vertical/horizontal angles or candela per horizontal
//
void Lightprofile_ies_parser::parse_angles_data(
    std::vector<Scalar>&  angles_data,
    const std::string&    description)
{
    // Tokenize vertical data
    const Uint nb_angles = angles_data.size();
    std::vector<std::string> tokens;
    get_tokens(m_reader.get(), nb_angles, tokens);
    const Uint size = tokens.size();

    if(size!=nb_angles)
    {
        LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
            "Unable to parse %s in %s. "
            "%u values have been found but %u have to be specified on dedicated possibly "
            "separated line(s). \n"
            "Please, see also '%s'",
            description.c_str(), m_log_identifier.c_str(), size, nb_angles, ies_spec);

        if(m_reader->eof())
        {
            LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Unexpected end of file in %s. ", m_log_identifier.c_str());
        }
        m_valid = false;
        return;
    }

    for(Uint i=0; i<nb_angles; i++)
    {
        angles_data[i] = STRING::lexicographic_cast_s<MI::Scalar, std::string>(tokens[i]);
    }
}

//
// Parse keyword-label pairs
//
bool Lightprofile_ies_parser::parse_label(char* label)
{
    while(*label && isspace(static_cast<unsigned char>(*label)))
        label++; // remove trailing white space

    std::string label_tag = "";
    if(*label == '[')
    {
        label++;
        while(*label && *label!=']')
        {
            label_tag += *label;
            label++;
        }
        label++;
    }
    else if(!strncmp(label, "TILT", 4))
        return false;
    else
        label_tag = "<no keyword>";

    while(*label && isspace(static_cast<unsigned char>(*label)))
        label++; // remove trailing white space

    std::string label_contents = "";
    while(*label)
    {
        label_contents += *label;
        label++;
    }

    size_t len = label_contents.size();
    if(len>0)
    {
        while(isspace(static_cast<unsigned char>(label_contents[len-1])))
            len--;
        if(label_contents.size()!=len)
            label_contents = label_contents.substr(0, len);
    }
    else
        label_contents = "<no label>";

    return true;
}

//
// Parse tilt
//
void Lightprofile_ies_parser::parse_tilt(char* tilt)
{
    while (*tilt && isspace(static_cast<unsigned char>(*tilt)))
        tilt++;    // remove trailing white space

    if (STRING::compare_case_insensitive("TILT", tilt, 4))
        return;
    tilt += 4;     /// 4 == strlen("TILT")

    while (*tilt && isspace(static_cast<unsigned char>(*tilt)))
        tilt++;     // remove trailing white space

    if (*tilt != '=')
        return;

    tilt++;
    while (*tilt && isspace(static_cast<unsigned char>(*tilt)))
        tilt++;     // remove trailing white space

    std::string tilt_value = "";
    while (*tilt)
    {
        tilt_value += *tilt;
        tilt++;     // remove trailing white space
    }

    if (!strncmp("NONE", tilt_value.c_str(), 4))
    {
        m_tilt_values_exist = false;
    }
    else if (!strncmp("INCLUDE", tilt_value.c_str(), 7))
    {
        // Tilt values are included in the same file ... keep on parsing it ...
        parse_tilt_values();
    }
    else
    {
        // Tilt values are included in another file: open that file (if possible) and parse it ...
        size_t len = tilt_value.size();
        while (isspace(static_cast<unsigned char>(tilt_value[len-1])))
            len--;
        if (tilt_value.size()!=len)
            tilt_value = tilt_value.substr(0, len);

        DISK::File_reader_impl reader;
        if( !reader.open( tilt_value.c_str()))
        {
            LOG::mod_log->warning(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "The file \"%s\", which contains the corresponding tilt values "
                "cannot be opened. Tilt values will be omitted for %s.",
                tilt_value.c_str(), m_log_identifier.c_str());
            return;
        }

        mi::base::Handle<mi::neuraylib::IReader> orig_reader = m_reader;
        std::string orig_log_identifier = m_log_identifier;
        m_reader = make_handle_dup( &reader);
        m_log_identifier = tilt_value;

        parse_tilt_values();

        m_reader = orig_reader;
        m_log_identifier = orig_log_identifier;
    }
}

//
// Parsing tilt values
//
void Lightprofile_ies_parser::parse_tilt_values()
{
    const Uint nb_required_data_values = 3;

    // Tokenize additional data
    std::vector<std::string> tokens;
    get_tokens(m_reader.get(), nb_required_data_values, tokens);
    const Uint size = tokens.size();

    if(size!=nb_required_data_values)
    {
        LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
            "Unable to parse tilt data in %s. "
            "%u values have been found but %u have to be specified on dedicated possibly "
            "separated line(s). \n"
            "Please, see also '%s'",
            m_log_identifier.c_str(), size, nb_required_data_values, ies_spec);

        if(m_reader->eof())
        {
            LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Unexpected end of file in %s. ", m_log_identifier.c_str());
        }
        m_valid = false;
        return;
    }

    Uint lamp_geometry = STRING::lexicographic_cast_s<MI::Uint, std::string>(tokens[0]);
    switch(lamp_geometry)
    {
        case VERTICAL_BASE_UP_DOWN:
            m_lamp_to_luminaire_geometry = VERTICAL_BASE_UP_DOWN;
            break;
        case HORIZONTAL_ALONG_90_DEGREE_PLANE:
            m_lamp_to_luminaire_geometry = HORIZONTAL_ALONG_90_DEGREE_PLANE;
            break;
        case HORIZONTAL_ALONG_0_DEGREE_PLANE:
            m_lamp_to_luminaire_geometry = HORIZONTAL_ALONG_0_DEGREE_PLANE;
            break;
        default:
            LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Invalid lamp-to-luminaire-geometry %u used in %s. "
                "Valid units types are: 1 (vertical_base_up_down), 2 "
                "(horizontal_along_90_degree_plane), or 3 (horizontal_along_0_degree_plane). "
                "'vertical_base_up_down' will be assumed.\n"
                "Please, see also '%s'",
                lamp_geometry, m_log_identifier.c_str(), ies_spec);
            m_lamp_to_luminaire_geometry = VERTICAL_BASE_UP_DOWN;
            break;
    }
    m_nb_angles                 = STRING::lexicographic_cast_s<MI::Uint, std::string>(tokens[1]);
    m_angles.resize(m_nb_angles);
    m_nb_multiplying_factors    = STRING::lexicographic_cast_s<MI::Uint, std::string>(tokens[2]);
    m_multiplying_factors.resize(m_nb_multiplying_factors);

    if(m_valid)
        parse_angles_data(m_angles, "tilt angles");

    if(m_valid)
        parse_angles_data(m_multiplying_factors, "multiplying factors");

    if(m_valid)
        m_tilt_values_exist = true;
    else
        m_tilt_values_exist = false;
}

// -------------------------------------------------------------------------------------------------
// CONVERTING IES DATA TO PHOTOMETRIC GRID, WHICH THEN CAN BE USED FOR LIGHTPROFILE SAMPLING
// (cp. Lightprofile_photometric_grid)
//
namespace
{
//
// Resolve symmetries, cp. IESNA spec. Chapter 3.18
// No symmetry or no dependency on horizontal angle for photometric type C
//
static void resolve_no_symmetry_type_c(
    const std::vector<Scalar>&                  horizontal_angles_in,
    std::vector<Scalar>&                        horizontal_angles_out,
    const Uint                                  nb_vertical_angles,
    const Scalar                                ratio,
    const std::vector<std::vector<Scalar> >&    candela_values_in,
    std::vector<std::vector<Scalar> >&          candela_values_out)
{
    const Uint nb_horizontal_angles = horizontal_angles_in.size();
    horizontal_angles_out.resize(nb_horizontal_angles);

    for(Uint i=0; i<nb_horizontal_angles; i++)
        horizontal_angles_out[i] = mi::math::radians(horizontal_angles_in[i]);

    // Candela values stored in just one vector
    candela_values_out.resize(nb_horizontal_angles);
    for(Uint i=0; i<nb_horizontal_angles; i++)
    {
        const std::vector<Scalar>& candela_values = candela_values_in[i];
        for(Uint j=0; j<nb_vertical_angles; j++)
            candela_values_out[i].push_back(candela_values[nb_vertical_angles-1-j]*ratio);
    }
}
//
// 0-180 plane symmetry for photometric type C
//
static void resolve_0_180_plane_symmetry_type_c(
    const std::vector<Scalar>&                  horizontal_angles_in,
    std::vector<Scalar>&                        horizontal_angles_out,
    const Uint                                  nb_vertical_angles,
    const Scalar                                ratio,
    const std::vector<std::vector<Scalar> >&    candela_values_in,
    std::vector<std::vector<Scalar> >&          candela_values_out)
{
    const Uint nb_horizontal_angles_in = horizontal_angles_in.size();
    const Uint nb_horizontal_angles = (2*nb_horizontal_angles_in)-1;
    horizontal_angles_out.resize(nb_horizontal_angles);


    for(Uint i=0; i<nb_horizontal_angles_in; i++)
    {
        horizontal_angles_out[i] = mi::math::radians(horizontal_angles_in[i]);
    }

    for(Uint i=nb_horizontal_angles_in; i<nb_horizontal_angles; i++)    // Mirrored at 0-180 plane
    {
        const Scalar rad = mi::math::radians(horizontal_angles_in[nb_horizontal_angles-i-1]);
        horizontal_angles_out[i] = Scalar(2.0*M_PI) - rad;
    }

    // Candela values stored in just one vector
    candela_values_out.resize(nb_horizontal_angles);
    for(Uint i=0; i<nb_horizontal_angles_in; i++)
    {
        const std::vector<Scalar>& candela_values = candela_values_in[i];
        for(Uint j=0; j<nb_vertical_angles; j++)
            candela_values_out[i].push_back(candela_values[nb_vertical_angles-1-j]*ratio);
    }
    for(Uint i=nb_horizontal_angles_in; i<nb_horizontal_angles; i++)
    {
        const std::vector<Scalar>& candela_values = candela_values_in[nb_horizontal_angles-i-1];
        for(Uint j=0; j<nb_vertical_angles; j++)
            candela_values_out[i].push_back(candela_values[nb_vertical_angles-1-j]*ratio);
    }
}
//
// 90-270 plane symmetry for photometric type C
//
static void resolve_90_270_plane_symmetry_type_c(
    const std::vector<Scalar>&                  horizontal_angles_in,
    std::vector<Scalar>&                        horizontal_angles_out,
    const Uint                                  nb_vertical_angles,
    const Scalar                                ratio,
    const std::vector<std::vector<Scalar> >&    candela_values_in,
    std::vector<std::vector<Scalar> >&          candela_values_out)
{
    const Uint nb_horizontal_angles_in = horizontal_angles_in.size();
    const Uint nb_horizontal_angles = (2*nb_horizontal_angles_in)-1;
    horizontal_angles_out.resize(nb_horizontal_angles);

    for(Uint i=0; i<nb_horizontal_angles_in; i++)
        horizontal_angles_out[i] = mi::math::radians(horizontal_angles_in[i]);
    for(Uint i=nb_horizontal_angles_in; i<nb_horizontal_angles; i++)
    {
        const Scalar rad = mi::math::radians(horizontal_angles_in[nb_horizontal_angles-i-1]);
        horizontal_angles_out[i] = Scalar(3.0*M_PI) - rad;
    }

    // Candela values stored in just one vector
    candela_values_out.resize(nb_horizontal_angles);
    for(Uint i=0; i<nb_horizontal_angles_in; i++)
    {
        const std::vector<Scalar>& candela_values = candela_values_in[i];
        for(Uint j=0; j<nb_vertical_angles; j++)
            candela_values_out[i].push_back(candela_values[nb_vertical_angles-1-j]*ratio);
    }
    for(Uint i=nb_horizontal_angles_in; i<nb_horizontal_angles; i++)
    {
        const std::vector<Scalar>& candela_values = candela_values_in[nb_horizontal_angles-i-1];
        for(Uint j=0; j<nb_vertical_angles; j++)
            candela_values_out[i].push_back(candela_values[nb_vertical_angles-1-j]*ratio);
    }
}
//
// 0-90 octant symmetry for photometric type C
//
static void resolve_0_90_octant_symmetry_type_c(
    const std::vector<Scalar>&                  horizontal_angles_in,
    std::vector<Scalar>&                        horizontal_angles_out,
    const Uint                                  nb_vertical_angles,
    const Scalar                                ratio,
    const std::vector<std::vector<Scalar> >&    candela_values_in,
    std::vector<std::vector<Scalar> >&          candela_values_out)
{
    const Uint nb_horizontal_angles_0 = horizontal_angles_in.size();
    const Uint nb_horizontal_angles_2 = (2*nb_horizontal_angles_0)-1;
    const Uint nb_horizontal_angles_4 = (2*nb_horizontal_angles_2)-1;
    horizontal_angles_out.resize(nb_horizontal_angles_4);

    for(Uint i=0; i<nb_horizontal_angles_0; i++)                        // Copy original angles
    {
        horizontal_angles_out[i] = mi::math::radians(horizontal_angles_in[i]);
    }

    for(Uint i=nb_horizontal_angles_0; i<nb_horizontal_angles_2; i++)   // Apply symmetry
    {
        const Scalar rad = mi::math::radians(horizontal_angles_in[nb_horizontal_angles_2-i-1]);
        horizontal_angles_out[i] = Scalar(M_PI) - rad;
    }

    for(Uint i=nb_horizontal_angles_2; i<nb_horizontal_angles_4; i++)   // Flipping
    {
        const Scalar rad = Scalar(2.0 * M_PI) - horizontal_angles_out[nb_horizontal_angles_4-i-1];
        horizontal_angles_out[i] = rad;
    }

    // Candela values stored in just one vector
    candela_values_out.resize(nb_horizontal_angles_4);
    for(Uint i=0; i<nb_horizontal_angles_0; i++)
    {
        const std::vector<Scalar>& candela_values = candela_values_in[i];
        for(Uint j=0; j<nb_vertical_angles; j++)
        {
            candela_values_out[i].push_back(candela_values[nb_vertical_angles-1-j]*ratio);
        }
    }
    for(Uint i=nb_horizontal_angles_0; i<nb_horizontal_angles_2; i++)
    {
        const std::vector<Scalar>& candela_values =
            candela_values_in[nb_horizontal_angles_2-i-1];
        for(Uint j=0; j<nb_vertical_angles; j++)
        {
            candela_values_out[i].push_back(candela_values[nb_vertical_angles-1-j]*ratio);
        }
    }
    for(Uint i=nb_horizontal_angles_2; i<nb_horizontal_angles_4; i++)
    {
        const std::vector<Scalar>& candela_values =
            candela_values_out[nb_horizontal_angles_4-i-1];
        for(Uint j=0; j<nb_vertical_angles; j++)
        {
            candela_values_out[i].push_back(candela_values[j]);
        }
    }
}
//
// Temporary code [cp. MENTAL RAY: lprofint.cpp, lines 387..]
// Temporary code to replace vertical symmetry with 2 angles until interpolation
// cannot deal with it
//
static void resolve_special_case_type_c(
    const std::vector<Scalar>&                  horizontal_angles_in,
    std::vector<Scalar>&                        horizontal_angles_out,
    const Uint                                  nb_vertical_angles,
    const std::vector<std::vector<Scalar> >&    candela_values_in,
    std::vector<std::vector<Scalar> >&          candela_values_out)
{
    const Uint nb_horizontal_angles = 2;
    horizontal_angles_out.resize(nb_horizontal_angles);

    horizontal_angles_out[0] = horizontal_angles_in[0];
    horizontal_angles_out[1] = Scalar(2.0 * M_PI);

    // Candela values stored in just one vector
    candela_values_out.resize(nb_horizontal_angles);
    const std::vector<Scalar>& candela_values = candela_values_in[0];
    for(Uint i=0; i<nb_horizontal_angles; i++)
    {
        for(Uint j=0; j<nb_vertical_angles; j++)
            candela_values_out[i].push_back(candela_values[j]);
    }
}
//
// Resolve symmetries, cp. IESNA spec. Chapter 3.18
// No symmetry or no dependency on horizontal angle for photometric type B
//
static void resolve_no_symmetry_type_b(
    const std::vector<Scalar>&                  horizontal_angles_in,
    std::vector<Scalar>&                        horizontal_angles_out,
    const Uint                                  nb_vertical_angles,
    const Scalar                                ratio,
    const std::vector<std::vector<Scalar> >&    candela_values_in,
    std::vector<std::vector<Scalar> >&          candela_values_out)
{
    const Uint nb_horizontal_angles = horizontal_angles_in.size();
    horizontal_angles_out.resize(nb_horizontal_angles);

    for(Uint i=0; i<nb_horizontal_angles; i++)
        horizontal_angles_out[i] = mi::math::radians(horizontal_angles_in[i]);

    // Candela values stored in just one vector
    candela_values_out.resize(nb_horizontal_angles);
    for(Uint i=0; i<nb_horizontal_angles; i++)
    {
        const std::vector<Scalar>& candela_values = candela_values_in[i];
        for(Uint j=0; j<nb_vertical_angles; j++)
            candela_values_out[i].push_back(candela_values[j]*ratio);
    }
}
//
// 0-90 octant symmetry for photometric type B
//
static void resolve_0_90_symmetry_type_b(
    const std::vector<Scalar>&                  horizontal_angles_in,
    std::vector<Scalar>&                        horizontal_angles_out,
    const Uint                                  nb_vertical_angles,
    const Scalar                                ratio,
    const std::vector<std::vector<Scalar> >&    candela_values_in,
    std::vector<std::vector<Scalar> >&          candela_values_out)
{
    const Uint nb_horizontal_angles_in = horizontal_angles_in.size();
    const Uint nb_horizontal_angles = (2*nb_horizontal_angles_in)-1;
    horizontal_angles_out.resize(nb_horizontal_angles);

    for(Uint i=0; i<nb_horizontal_angles_in; i++)
    {
        horizontal_angles_out[nb_horizontal_angles_in+i-1] =
            mi::math::radians(horizontal_angles_in[i]);
        if (i > 0)
            horizontal_angles_out[nb_horizontal_angles_in-i-1] =
                -mi::math::radians(horizontal_angles_in[i]);
    }

    // Candela values stored in just one vector
    candela_values_out.resize(nb_horizontal_angles);
    for(Uint i=0; i<nb_horizontal_angles_in; i++)
    {
        const std::vector<Scalar>& candela_values = candela_values_in[i];
        for(Uint j=0; j<nb_vertical_angles; j++)
        {
            candela_values_out[nb_horizontal_angles_in+i-1].push_back(
                candela_values[j]*ratio);
            if (i > 0)
                candela_values_out[nb_horizontal_angles_in-i-1].push_back(
                    candela_values[j]*ratio);
        }
    }
}
//
// Horizontal rotation
// [inspired by MENTAL RAY: horz_rotate]
static void horizontal_rotate(
    std::vector<Scalar>& horizontal_angles,
    Dscalar rotation
)
{
    const Uint nb_horizontal_angles = horizontal_angles.size();

    while(rotation + horizontal_angles[0] < -M_PI)
        rotation += 2.0 * M_PI;

    for(Uint i=0; i<nb_horizontal_angles; i++)
        horizontal_angles[i] += Scalar(rotation);
}
//
// Flip representaion: IES normally has clockwise phi, but mi use counter -
// clockwise one. Also, flags may overwrite this.
// [inspired by MENTAL RAY: horz_flip]
static void horizontal_flip(
    Uint                                nb_vertical_angles,
    std::vector<Scalar>&                horizontal_angles,
    std::vector<std::vector<Scalar> >&  candela_values)
{
    Uint nb_horizontal_angles = horizontal_angles.size();

    Scalar max_angle = horizontal_angles[nb_horizontal_angles - 1];

    // miASSERT(-M_PI <= max_angle && max_angle <= 3 * M_PI);
    Scalar add = (max_angle>M_PI) ? Scalar(2.0*M_PI) : 0.f;   // Put reversed angle to [-PI,PI]

    Uint i = 0;
    Uint j = nb_horizontal_angles-1;
    for(; i<j; i++, j--)
    {
        const Scalar swap    =  horizontal_angles[i];
        horizontal_angles[i] = -horizontal_angles[j] + add;
        horizontal_angles[j] = -swap + add;
    }
    // Odd number, i.e., swap mid line
    if(i==j)
        horizontal_angles[i] = -horizontal_angles[i] + add;

    i=0;
    j = nb_horizontal_angles-1;
    for(; i<j; i++, j--)
    {
        std::vector<Scalar>& candela_lower = candela_values[i];
        std::vector<Scalar>& candela_upper = candela_values[j];
        for(Uint k=0; k<nb_vertical_angles; k++)
        {
            const Scalar swap = candela_lower[k];
            candela_lower[k]  = candela_upper[k];
            candela_upper[k]  = swap;
        }
    }
    // No need to swap mid line if there is any
}
//
// Convert to actual grid that includes boundary values
// [inspired by MENTAL RAY: boundary_add]
static void add_boundary(
    std::vector<Scalar>&                     vertical_angles,
    std::vector<Scalar>&                     horizontal_angles,
    std::vector<std::vector<Scalar> >&       grid
)
{
    const Uint nb_vertical_angles    = vertical_angles.size();
    const Uint nb_horizontal_angles  = horizontal_angles.size();

    // Add boundary values: boundary positions are mirror images
    if(nb_horizontal_angles>1)
    {
        const Scalar last_angle           = horizontal_angles[nb_horizontal_angles-1];
        const Scalar last_angle_minus_one = horizontal_angles[nb_horizontal_angles-2];

        // x_{-1}  = x_0 - (x_1 - x_0)
        horizontal_angles.insert(
            horizontal_angles.begin(), 2.f*horizontal_angles[0] - horizontal_angles[1]);
        // x_{n+1} = x_n + (x_n - x_{n-1})
        horizontal_angles.push_back(2.f * last_angle - last_angle_minus_one);
    }
    else if(nb_horizontal_angles==1)
    {
        const Scalar angle = horizontal_angles[0];
        horizontal_angles.insert(horizontal_angles.begin(), angle-0.125f);
        horizontal_angles.push_back(angle+0.125f);
    }
    else
    {
        // ERROR message
    }

    // Copy vertical angles and add boundary values: boundary positions are mirror images
    if(nb_vertical_angles>1)
    {
        const Scalar last_angle           = vertical_angles[nb_vertical_angles-1];
        const Scalar last_angle_minus_one = vertical_angles[nb_vertical_angles-2];

        // y_{-1}  = y_0 - (y_1 - y_0)
        vertical_angles.insert(
            vertical_angles.begin(), 2.f*vertical_angles[0] - vertical_angles[1]);
        // y_{n+1} = y_n + (y_n - y_{n-1})
        vertical_angles.push_back(2.f * last_angle - last_angle_minus_one);
    }
    else if(nb_vertical_angles==1)
    {
        const Scalar angle = vertical_angles[0];
        vertical_angles.insert(vertical_angles.begin(), angle-0.125f);
        vertical_angles.push_back(angle+0.125f);
    }
    else
    {
        // ERROR message
    }

    // Fill in the boundary values into the grid
    //
    // First: Add boundary values in vertical direction
    // No periodic boundary conditions, i.e., simply add a linear extrapolation of boundary values
    if(nb_vertical_angles>1)
    {
        for(Uint i=0; i<nb_horizontal_angles; i++)
        {
            std::vector<Scalar>& vertical_values = grid[i];
            const Scalar last_angle           = vertical_values[nb_vertical_angles-1];
            const Scalar last_value_minus_one = vertical_values[nb_vertical_angles-2];

            vertical_values.insert(
                vertical_values.begin(), 2.f*vertical_values[0] - vertical_values[1]);
            vertical_values.push_back(2.f * last_angle - last_value_minus_one);
        }
    }
    else if(nb_vertical_angles==1)
    {
        for(Uint i=0; i<nb_horizontal_angles; i++)
        {
            std::vector<Scalar>& vertical_values = grid[0];
            const Scalar last_angle           = vertical_values[nb_vertical_angles-1];
            const Scalar last_value_minus_one = vertical_values[nb_vertical_angles-2];

            vertical_values.insert(
                vertical_values.begin(), 2.f*vertical_values[0] - vertical_values[1]);
            vertical_values.push_back(2.f * last_angle - last_value_minus_one);
        }
    }
    else
    {
        // SAME ERROR message (s.above)
    }

    // Horizontal direction
    // Check for periodic boundary conditions
    Scalar delta = fabs(horizontal_angles[nb_horizontal_angles+1-1]-horizontal_angles[1]);
    while(delta>=Scalar(2.0*M_PI))
    {
        delta -= Scalar(2.0*M_PI);
    }

    // Add boundary columns in both horizontal directions
    grid.insert(grid.begin(), std::vector<Scalar>(nb_vertical_angles+2, -1));
    grid.push_back(std::vector<Scalar>(nb_vertical_angles+2, -1));

    std::vector<Scalar>& col_left             = grid[0];
    std::vector<Scalar>& col_left_plus_two    = grid[2];
    std::vector<Scalar>& col_right            = grid[nb_horizontal_angles+1];
    std::vector<Scalar>& col_right_minus_two  = grid[nb_horizontal_angles-1];
    if(fabs(delta)<0.0001)
    {
        // Periodic boundary condition
        for(Uint i=0; i<nb_vertical_angles+2; i++) // Step up in vertical direction
        {
            col_left[i]  = col_right_minus_two[i];
            col_right[i] = col_left_plus_two[i];
        }
    }
    else
    {
        std::vector<Scalar>& col_left_plus_one    = grid[1];
        std::vector<Scalar>& col_right_minus_one  = grid[nb_horizontal_angles];

        // Extent boundary linearly
        for(Uint i=0; i<nb_vertical_angles+2; i++) // Step up in vertical direction
        {
            // Left boundary
            col_left[i] = 2.f * col_left_plus_one[i] - col_left_plus_two[i];
            // Right boundary
            col_right[i] = 2.f * col_right_minus_one[i] - col_right_minus_two[i];
        }
    }
}
//
// Remap vertical angles to [0,PI] and horizontal angles to [0,2*PI]
// [inspired by MENTAL RAY: lprof_adjust_angles]
static void remap_angles(
    std::vector<Scalar>& vertical_angles,       // Vertical angles to map with boundary angles
    std::vector<Scalar>& horizontal_angles,     // Horizontal angles to map with boundary angles
    Uint                   vertical_resolution,
    Uint                   horizontal_resolution,
    Scalar&                theta,               // Starting vertical angle
    Scalar&                phi,                 // Starting horizontal angle
    Scalar&                d_theta,             // Delta angle in vertical direction
    Scalar&                d_phi)               // Delta angle in horizontal direction
{
    theta = vertical_angles[1];                 // Omitting boundary angle in vertical direction too
    vertical_angles[0] -= theta;
    vertical_angles[1] = 0.0f;
    const Uint nb_vertical_angles = vertical_angles.size();
    for(Uint i=2; i<nb_vertical_angles; i++)
    {
        vertical_angles[i] -= theta;
        ASSERT(M_LIGHTPROFILE, vertical_angles[i]>vertical_angles[i-1]);
    }

    // Map theta to [0,PI]
    ASSERT(M_LIGHTPROFILE, -0.0001f<=theta && theta<=Scalar(M_PI+0.0001));
    if(theta<0.0f)
        theta = 0.0f;
    else if(theta>Scalar(M_PI))
        theta = Scalar(M_PI);

    d_theta = vertical_angles[nb_vertical_angles-2] / (vertical_resolution-1);

    phi = horizontal_angles[1];                 // Omitting boundary angle (cp. add_boundary)
    horizontal_angles[0] -= phi;
    horizontal_angles[1] = 0.0f;
    const Uint nb_horizontal_angles = horizontal_angles.size();
    for(Uint i=2; i<nb_horizontal_angles; i++)
    {
        horizontal_angles[i] -= phi;
        ASSERT(M_LIGHTPROFILE, horizontal_angles[i]>horizontal_angles[i-1]);
    }

    // Map phi to [0,2*PI]
    while(phi>= Scalar(2.0*M_PI)) phi -= Scalar(2.0*M_PI);
    while(phi<  0.0f)             phi += Scalar(2.0*M_PI);

    d_phi = horizontal_angles[nb_horizontal_angles-2] / (horizontal_resolution-1);
}
//
// Compute linear interpolation to determine lighprofile values on a regular grid
// [inspired by MENTAL RAY: lprof_interpolate_hermite_1
static void compute_linear_interpolation(
    const std::vector<Scalar>& vertical_angles,     // Vertical angles to map with boundary angles
    const std::vector<Scalar>& horizontal_angles,   // Horizontal angles to map with boundary angles
    Uint                         vertical_resolution,
    Uint                         horizontal_resolution,
    const std::vector<std::vector<Scalar> >& candela, // Candela values incl. boundary
    Scalar                       theta,             // Starting vertical angle
    Scalar                       phi,               // Starting horizontal angle
    Scalar                       d_theta,           // Delta angle in vertical direction
    Scalar                       d_phi,             // Delta angle in horizontal direction
    const std::string&         log_identifier,
    std::vector<Scalar>&       grid_values)         // Candela values on regular grid          [out]
{
    const Uint nb_vertical_angles   = vertical_angles.size();
    const Uint nb_horizontal_angles = horizontal_angles.size();
    grid_values.resize(horizontal_resolution*vertical_resolution, -1);

    bool warning_output = false;

    Uint idx = 0;
    Uint q = 0;
    Scalar left  = 0.00001f;
    Scalar right = 0.99999f;
    for(Uint j=0; j<horizontal_resolution; j++)
    {
        const Scalar t_phi = j*d_phi;

        // Find q such that phi[q] <= t < phi[q+1]
        while(q < nb_horizontal_angles-1 && t_phi >= horizontal_angles[q+1]) // Omitting boundary
            q++;

        ASSERT(M_LIGHTPROFILE, q<nb_horizontal_angles-1);
        ASSERT(M_LIGHTPROFILE, horizontal_angles[q+1]>horizontal_angles[q]);
        ASSERT(M_LIGHTPROFILE, horizontal_angles[q]<=t_phi && horizontal_angles[q+1]>=t_phi);

        //* Compute t in [0, 1)
        const Scalar t = (t_phi-horizontal_angles[q])/(horizontal_angles[q+1]-horizontal_angles[q]);

        Uint p = 0;
        for(Uint i=0; i<vertical_resolution; i++)
        {
            const Scalar s_theta = i*d_theta;

            // Find p such that theta[p] <= t < theta[p+1]
            while(p < nb_vertical_angles-1 && s_theta >= vertical_angles[p+1]) // Omitting boundary
            {
                p++;
            }

            ASSERT(M_LIGHTPROFILE, p<nb_vertical_angles-1);
            ASSERT(M_LIGHTPROFILE, vertical_angles[p+1]>vertical_angles[p]);
            ASSERT(M_LIGHTPROFILE, vertical_angles[p]<=s_theta && vertical_angles[p+1]>=s_theta);

            //* Compute s in [0, 1)
            const Scalar s = (s_theta-vertical_angles[p])/(vertical_angles[p+1]-vertical_angles[p]);

            // Index into candela values
            const std::vector<Scalar>& candela_values_h0 = candela[q];

            // Index into candela values (next vertical column)
            const std::vector<Scalar>& candela_values_h1 = candela[q+1];

            // Warn about negative values, but only if they are actually used
            if(    (s < right && t < right && candela_values_h0[p  ] < 0.f)
                || (s > left  && t < right && candela_values_h0[p+1] < 0.f)
                || (s < right && t > left  && candela_values_h1[p  ] < 0.f)
                || (s > left  && t > left  && candela_values_h1[p+1] < 0.f))
                warning_output = true;

            // Linear interpolation along horizontal angle at smaller vertical angle (p,q), (p+1,q)
            const Scalar u = (1.f-s)*std::max(candela_values_h0[p  ],0.f)
                                 + s*std::max(candela_values_h0[p+1],0.f);

            // Linear interpolation along horizontal angle at larger vert. angle (p,q+1), (p+1,q+1)
            const Scalar v = (1.f-s)*std::max(candela_values_h1[p  ],0.f)
                                 + s*std::max(candela_values_h1[p+1],0.f);

            // Interpolate along vertical
            grid_values[idx] = (1.f-t)*u + t*v;
            idx++;
        }
    }

    if(warning_output)
        LOG::mod_log->warning(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
            "Negative values found in %s.", log_identifier.c_str());
}
//
// Compute cubic interpolation to determine lighprofile values on a regular grid
// [inspired by MENTAL RAY: lprof_interpolate_hermite_3
/*---------------------------------------------------------------------------
 * evaluate the 4 cubic Hermite basis functions for a value t: 0 <= t < 1.
 * The coefficients are determined from the following equations:
 *
 * y(t) = c[0](t) * y(0) + c[1](t) * y(1) + c[2](t) * y'(0) + c[3](t) * y'(1)
 *
 * where each c[i] is a cubic polynomial;
 *
 * That equation holds for all t with 0 <= t <= 1. Equally, for the derivative.
 * To ensure that the approximation coincides with the given values
 * y(0) and y(1) as well as with the derivatives y'(0) and y'(1) we
 * enforce the following contraints:
 *
 * c[0] (0) = 1;  c[1] (0) = 0;  c[2] (0) = 0;  c[3] (0) = 0;
 * c[0] (1) = 0;  c[1] (1) = 1;  c[2] (1) = 0;  c[3] (1) = 0;
 * c[0]'(0) = 0;  c[1]'(0) = 0;  c[2]'(0) = 1;  c[3]'(0) = 0;
 * c[0]'(1) = 0;  c[1]'(1) = 0;  c[2]'(1) = 0;  c[3]'(1) = 1;
 *
 * Using these constraints to solve for the coefficients yields the result
 *
 * c[0](t) =  2 * t^3 - 3 * t^2 + 1
 * c[1](t) = -2 * t^3 + 3 * t^2
 * c[2](t) =      t^3 - 2 * t^2 + t
 * c[3](t) =      t^3 -     t^2
 */
static inline void compute_hermite_3_coefficients(
    Scalar      t,
    Scalar&     c0,
    Scalar&     c1,
    Scalar&     c2,
    Scalar&     c3)
{
    const Scalar t2 = t*t;

    c0 = (2.f*t - 3.f)   * t2 + 1.f;
    c1 = (3.f   - 2.f*t) * t2;
    c2 = ((t-2.f)*t+1.f) * t;
    c3 = t2*(t-1.f);
}
static void compute_cubic_interpolation(
   const std::vector<Scalar>& vertical_angles,      // Vertical angles to map with boundary angles
   const std::vector<Scalar>& horizontal_angles,    // Horizontal angles to map with boundary angles
   Uint                         vertical_resolution,
   Uint                         horizontal_resolution,
   const std::vector<std::vector<Scalar> >& candela, // Candela values incl. boundary
   Scalar                       theta,              // Starting vertical angle
   Scalar                       phi,                // Starting horizontal angle
   Scalar                       d_theta,            // Delta angle in vertical direction
   Scalar                       d_phi,              // Delta angle in horizontal direction
   std::vector<Scalar>&       grid_values)          // Candela values on regular grid          [out]
{
    const Scalar eps = 0.00001f;
    const Uint nb_vertical_angles   = vertical_angles.size();
    const Uint nb_horizontal_angles = horizontal_angles.size();
    grid_values.resize(horizontal_resolution*vertical_resolution);

    Scalar h_base[4];
    Scalar v_base[4];

    Uint idx = 0;
    Uint q = 1;
    for(Uint j=0; j<horizontal_resolution; j++)     // Omitting boundary angle
    {
        const Scalar t_phi = j*d_phi;

        // Find q such that phi[q] <= t < phi[q+1]
        while(q<nb_horizontal_angles-1 && t_phi>=horizontal_angles[q+1]-eps)
        {
            q++;
        }

        ASSERT(M_LIGHTPROFILE, q<nb_horizontal_angles-1);
        ASSERT(M_LIGHTPROFILE, horizontal_angles[q]-eps<=t_phi && horizontal_angles[q+1]+eps>t_phi);

        //* Compute t in [0, 1)
        const Scalar t = (t_phi-horizontal_angles[q])/(horizontal_angles[q+1]-horizontal_angles[q]);
        compute_hermite_3_coefficients(t, h_base[0], h_base[1], h_base[2], h_base[3]);
        const Scalar delta_phi = (horizontal_angles[q+1]-horizontal_angles[q])
                                               /(horizontal_angles[q+1]-horizontal_angles[q-1]);

        Uint p = 1;
        for(Uint i=0; i<vertical_resolution; i++)  // Omitting boundary angle
        {
            const Scalar s_theta = i*d_theta;

            // Find p such that theta[p] <= t < theta[p+1]
            while(p < nb_vertical_angles-1 && s_theta >= vertical_angles[p+1]-eps)
                ++p;

            ASSERT(M_LIGHTPROFILE, p < nb_vertical_angles-1);
            ASSERT(M_LIGHTPROFILE, vertical_angles[p+1] > vertical_angles[p]);
            ASSERT(M_LIGHTPROFILE, vertical_angles[p]-eps <= s_theta
                                                        && vertical_angles[p+1]+eps > s_theta);

            //* Compute s in [0, 1)
            const Scalar s = (s_theta-vertical_angles[p])/(vertical_angles[p+1]-vertical_angles[p]);
            compute_hermite_3_coefficients(s, v_base[0], v_base[1], v_base[2], v_base[3]);
            const Scalar delta_theta = (vertical_angles[p+1]-vertical_angles[p])
                                                  / (vertical_angles[p+1]-vertical_angles[p-1]);

            // Index into candela values
            const std::vector<Scalar>& candela_values_h0 = candela[q-1];

            // d0 and d1 are difference quotients, but the division may be omitted, since we map to
            // the interval [0,1) there is also a factor of the same size, due to the chain rule.
            Scalar d0 = 0.0;
            Scalar d1 = 0.0;
            d0 = (candela_values_h0[p+1]-candela_values_h0[p-1])*delta_theta;
            if(p==nb_vertical_angles-2)
                d1 = (candela_values_h0[p  ]-candela_values_h0[p-2])*delta_theta;
            else
                d1 = (candela_values_h0[p+2]-candela_values_h0[p  ])*delta_theta;
            Scalar val_0 = v_base[0]*candela_values_h0[p] + v_base[1]*candela_values_h0[p+1]
                         + v_base[2]*d0                   + v_base[3]*d1;

            // Index into candela values
            const std::vector<Scalar>& candela_values_h1 = candela[q];

            d0 = (candela_values_h1[p+1]-candela_values_h1[p-1])*delta_theta;
            if(p==nb_vertical_angles-2)
                d1 = (candela_values_h1[p  ]-candela_values_h1[p-2])*delta_theta;
            else
                d1 = (candela_values_h1[p+2]-candela_values_h1[p  ])*delta_theta;
            Scalar val_1 = v_base[0]*candela_values_h1[p] + v_base[1]*candela_values_h1[p+1]
                         + v_base[2]*d0                   + v_base[3]*d1;

            // Index into candela values
            const std::vector<Scalar>& candela_values_h2 = candela[q+1];

            d0 = (candela_values_h2[p+1]-candela_values_h2[p-1])*delta_theta;
            if(p==nb_vertical_angles-2)
                d1 = (candela_values_h2[p  ]-candela_values_h2[p-2])*delta_theta;
            else
                d1 = (candela_values_h2[p+2]-candela_values_h2[p  ])*delta_theta;
            Scalar val_2 = v_base[0]*candela_values_h2[p] + v_base[1]*candela_values_h2[p+1]
                         + v_base[2]*d0                   + v_base[3]*d1;

            // Index into candela values
            const std::vector<Scalar>& candela_values_h3 =
                (q==nb_horizontal_angles-2) ? candela[q-2] : candela[q+2];

            d0 = (candela_values_h3[p+1]-candela_values_h3[p-1])*delta_theta;
            if(p==nb_vertical_angles-2)
                d1 = (candela_values_h3[p  ]-candela_values_h3[p-2])*delta_theta;
            else
                d1 = (candela_values_h3[p+2]-candela_values_h3[p  ])*delta_theta;
            Scalar val_3 = v_base[0]*candela_values_h3[p] + v_base[1]*candela_values_h3[p+1]
                         + v_base[2]*d0                   + v_base[3]*d1;

            d0 = (val_2-val_0)*delta_phi;
            d1 = (val_3-val_1)*delta_phi;

            Scalar w = h_base[0] * val_1 + h_base[1] * val_2
                     + h_base[2] * d0    + h_base[3] * d1;

            grid_values[idx] = (w>0.f) ? w : 0.f;
            idx++;
        }
    }
}

// Returns the greatest common divisor or a and b.
Uint gcd(Uint a, Uint b)
{
    return b == 0 ? a : gcd(b, a %b);
}

// Returns the least common multiple of a and b.
Uint lcm(Uint a, Uint b)
{
    return a / gcd(a, b) * b;
}

// Compute resolution for equidistant grid, based on possibly non-equidistant angles.
//
// If the angles are equidistant, we simply return the resolution implied by the angles, such that
// the grid is essentially unchanged. If the angles are not equidistant, we try to compute a
// resolution that keeps the given angles.
Uint compute_resolution(const std::vector<Scalar>& angles)
{
    Uint l = angles.size();
    if( l <= 2)
        return l;

    // The float values of the angles (or rather their differences) are binned into n+1 bins
    // (numbered from 0 to n), such that the following computations can be done in integers. The
    // number n is chosen such that we can represent 0.1 degrees as well as an equidistant grid of
    // resolution l.
    Dscalar range = 180.0f * (angles[l-1] - angles[0]) / M_PI;
    Uint precision = static_cast<Uint>(10 * range + 1);
    Uint n = lcm(precision-1, l-1);
    Dscalar factor = n / range * 180.0f / M_PI;

    std::vector<Uint> differences(l-1);
    for (size_t i = 0; i+1 < l; ++i)
        differences[i] = static_cast<Uint>(floor(factor * (angles[i+1]-angles[i]) + 0.5));

    Uint min = * min_element(differences.begin(), differences.end());
    Uint max = * max_element(differences.begin(), differences.end());

    // If all differences are the same (within tolerance), the grid is equidistant.
    if (min == max)
        return l;

    // Compute gcd of n and all differences.
    Uint g = n;
    for (size_t i = 0; i+1 < l; ++i)
        g = gcd(g, differences[i]);

    Uint result = n/g + 1;

    // Avoid upscaling large resolutions if l-1 is coprime to precision-1.
    Uint limit = static_cast<Uint>(2 * range + 1);
    return result <= limit ? result : limit;
}

} // anonymous namespace

//
// Set up a lightprofile from parsed data
//
bool Lightprofile_ies_parser::setup_lightprofile(
    Uint hermite,
    Uint flags,
    Uint& horizontal_resolution,
    Uint& vertical_resolution,
    Scalar& phi,
    Scalar& theta,
    Scalar& d_phi,
    Scalar& d_theta,
    std::vector<Scalar>& grid_values)
{
    if (!m_valid)
        return false;

    //
    // STEP 1: Convert IES data into legal lightprofile data
    // taking into account sampling order, IES version, special cases, ...
    //
    if(m_photometric_type == TYPE_A)
    {
        LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
            "Photometric type A in %s not supported.\n"
            "A dummy profile will be used instead.",
            m_log_identifier.c_str());
        return false;
    }

    if (m_candela_multiplier <= 0.0f)
    {
        LOG::mod_log->warning(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
            "Non-positive multiplier %f in %s will have no contribution.",
            m_candela_multiplier, m_log_identifier.c_str());
        m_candela_multiplier = 0.0f;
    }
    if (m_ballast_factor <= 0.0f)
    {
        LOG::mod_log->warning(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
            "Non-positive ballast factor %f in %s will have no contribution.",
            m_ballast_factor, m_log_identifier.c_str());
        m_ballast_factor = 0.0f;
    }
    // Resolve symmetries by creating a huge lookup table for later sampling
    Scalar ratio = m_candela_multiplier * m_ballast_factor;
    // Since 95 ballast-lamp has already been included in ballast-factor
    if(m_version == IESNA_LM_63_1986 || m_version == IESNA_LM_63_1991)
    {
        if (m_ballast_lamp_factor <= 0.0f)
        {
            m_ballast_lamp_factor = 0.0f;
            LOG::mod_log->warning(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Light profile in file \"%s\" has \"ballast lamp factor\" <= 0.0,"
                "will have no contribution.",
                m_log_identifier.c_str());
        }

        ratio *= m_ballast_lamp_factor;
    }



    std::vector<Scalar>               vertical_angles(m_nb_vertical_angles);
    std::vector<Scalar>               horizontal_angles;        // To be filled next ...
    std::vector<std::vector<Scalar> > candela_values;           // To be filled next ...

    Scalar first_angle = m_horizontal_angles[0];
    Scalar last_angle  = m_horizontal_angles[m_nb_horizontal_angles-1];

    if(m_photometric_type == TYPE_C)
    {
        // Theta in interval [0, M_PI], 0 on the north pole, i.e., vertical reverse for type C
        for(Uint i=0; i<m_nb_vertical_angles; i++)
            vertical_angles[m_nb_vertical_angles-1-i] =
                Scalar(M_PI-mi::math::radians(m_vertical_angles[i]));

        // cp. IESNA spec. 3.18
        if((first_angle == 0.f)
            && ((last_angle == 360.f) || (last_angle == 0.f)) )       // no symmetry
        {
            resolve_no_symmetry_type_c(m_horizontal_angles, horizontal_angles,
                m_nb_vertical_angles, ratio, m_candela_values, candela_values);
        }
        else if((first_angle == 0.f) && (last_angle == 180.f) )       // 0-180 plane symmetry
        {
            resolve_0_180_plane_symmetry_type_c(m_horizontal_angles, horizontal_angles,
                m_nb_vertical_angles, ratio, m_candela_values, candela_values);
        }
        else if((first_angle == 90.f) && (last_angle == 270.f) )      // 90-270 plane symmetry
        {
            // note: this mode is no longer allowed in IESNA-LM-63-02 files

            // TODO This method seems to assume that the value of the median horizontal angle is
            // 180 degrees.
            resolve_90_270_plane_symmetry_type_c(m_horizontal_angles, horizontal_angles,
                m_nb_vertical_angles, ratio, m_candela_values, candela_values);

            // MAX need to rotate type C with 90-270 degrees (according to MENTAL RAY)
            if(flags & mi::neuraylib::LIGHTPROFILE_ROTATE_TYPE_C_90_270)
                horizontal_rotate(horizontal_angles, -M_PI);    // to -pi/2, pi/2
        }
        else if((first_angle == 0.f) && (last_angle == 90.f) )        // 0-90 octant symmetry
        {
            resolve_0_90_octant_symmetry_type_c(m_horizontal_angles, horizontal_angles,
                m_nb_vertical_angles, ratio, m_candela_values, candela_values);
        }
        else if(first_angle == 0.f)
        {
            // Some IES files, such as those produced by popular EULUMDAT->IES converter, do not
            // fully commit to the IESNA standard and drop the last horizontal angle.
            // Solution: Assume no symmetry and copy 360 degree data from 0 degree.
            LOG::mod_log->warning(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Invalid symmetry given in %s. The first horizontal angle (value: %f) "
                "and the last horizontal angle (value: %f) impose no valid symmetry for a "
                "photometric type C. "
                "We assume no symmetry and add the additional horizontal angle 360. \n"
                "Please, see also Chapter 3.18 in '%s'",
                m_log_identifier.c_str(), first_angle, last_angle, ies_spec);

            // No symmetry
            resolve_no_symmetry_type_c(m_horizontal_angles, horizontal_angles,
                m_nb_vertical_angles, ratio, m_candela_values, candela_values);

            horizontal_angles.push_back(Scalar(2.0*M_PI));
            const std::vector<Scalar>& first_horizontal_candela = m_candela_values[0];
            candela_values.push_back(std::vector<Scalar>(m_nb_vertical_angles));
            std::vector<Scalar>& last_horizontal_candela =
                candela_values[m_nb_horizontal_angles];

            for(Uint j=0; j<m_nb_vertical_angles; ++j) {
                last_horizontal_candela[j] =
                    first_horizontal_candela[m_nb_vertical_angles-1-j]*ratio;
            }
        }
        else
        {
            LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Invalid symmetry given in %s. The first horizontal angle (value: %f) "
                "and the last horizontal angle (value: %f) impose no valid symmetry for a "
                "photometric type C. \n"
                "Please, see also Chapter 3.18 in '%s'\n"
                "A dummy profile will be used instead.",
                m_log_identifier.c_str(), first_angle, last_angle, ies_spec);
            return false;
        }

        // Temporary code [cp. MENTAL RAY: lprofint.cpp, lines 387..]
        // Temporary code to replace vertical symmetry with 2 angles until interpolation
        // cannot deal with it
        if((first_angle == 0.f) && (last_angle == 0.f) )
        {
            std::vector<Scalar>               new_horizontal_angles;
            std::vector<std::vector<Scalar> > new_candela_values;

            resolve_special_case_type_c(horizontal_angles, new_horizontal_angles,
                m_nb_vertical_angles, candela_values, new_candela_values);

            horizontal_angles = new_horizontal_angles;
            candela_values = new_candela_values;
        }
    }
    else // TYPE_B; Photometric type A has been rejected prior (s. above)
    {
        // Theta in interval [0, M_PI], 0 on the north pole, i.e., no vertical reverse for type B
        for(Uint i=0; i<m_nb_vertical_angles; i++)
            vertical_angles[i] = Scalar(M_PI/2.0) + mi::math::radians(m_vertical_angles[i]);

        if((first_angle == -90.f) && (last_angle == 90.f) )
        {
            resolve_no_symmetry_type_b(m_horizontal_angles, horizontal_angles,
                m_nb_vertical_angles, ratio, m_candela_values, candela_values);
        }
        else if((first_angle == 0.f) && (last_angle == 90.f) )
        {
            resolve_0_90_symmetry_type_b(m_horizontal_angles, horizontal_angles,
                m_nb_vertical_angles, ratio, m_candela_values, candela_values);
        }
        else
        {
            LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
                "Invalid symmetry given in %s. The first horizontal angle (value: %f) "
                "and the last horizontal angle (value: %f) impose no valid symmetry for a "
                "photometric type B. \n"
                "Please, see also Chapter 3.18 in '%s'\n"
                "A dummy profile will be used instead.",
                m_log_identifier.c_str(), first_angle, last_angle, ies_spec);
            return false;
        }

        // MAX need to rotate type B profiles (according to MENTAL RAY)
        if(flags & mi::neuraylib::LIGHTPROFILE_ROTATE_TYPE_B)
            horizontal_rotate(horizontal_angles, -M_PI/2.0);
    }

    if(!(flags & mi::neuraylib::LIGHTPROFILE_CLOCKWISE))
    {
        // Convert counter-clockwise ordering of sample values into clockwise ordering ...
        horizontal_flip(m_nb_vertical_angles, horizontal_angles, candela_values);
    }

    // Remember orginal resolution after unfolding symmetries
    size_t orig_nb_horizontal_angles = horizontal_angles.size();
    size_t orig_nb_vertical_angles   = vertical_angles.size();

    // Adjust requested resolution (if not set) based on IES dimensions after unfolding symmetries
    if(vertical_resolution == 0)
        vertical_resolution = compute_resolution(vertical_angles);
    if(horizontal_resolution == 0)
        horizontal_resolution = compute_resolution(horizontal_angles);

    //
    // STEP 2: Convert legal lightprofile data into a grid data that can be used for sampling.
    // Extend the profile boundary to allow for easy calculation of derivative approximations.
    //
    // Add boundary for later interpolation
    add_boundary(vertical_angles, horizontal_angles, candela_values);

    // Remap angles
    phi     = 0.0f;
    theta   = 0.0f;
    d_phi   = 0.0f;
    d_theta = 0.0f;
    remap_angles(
        vertical_angles, horizontal_angles,
        vertical_resolution, horizontal_resolution,
        theta, phi, d_theta, d_phi);

    // Compute final grid values of the photometric light
    if(hermite==mi::neuraylib::LIGHTPROFILE_HERMITE_BASE_1)
    {
        compute_linear_interpolation(
            vertical_angles, horizontal_angles,
            vertical_resolution, horizontal_resolution,
            candela_values,
            theta, phi, d_theta, d_phi, m_log_identifier, grid_values);
    }
    else if(hermite==mi::neuraylib::LIGHTPROFILE_HERMITE_BASE_3)
    {
        compute_cubic_interpolation(
            vertical_angles, horizontal_angles,
            vertical_resolution, horizontal_resolution,
            candela_values,
            theta, phi, d_theta, d_phi, grid_values);
    }
    else
    {
        // As long as no additional enums get introduced to the lightprofile interface,
        // we won't run into this error!
        ASSERT(M_LIGHTPROFILE,false);
        LOG::mod_log->error(M_LIGHTPROFILE, LOG::Mod_log::C_IO,
            "Unknown interpolation used in %s. "
            "Lightprofile sampling supports either linear (hermite=1) "
            "or cubic (hermite=3) interpolation.\n"
            "Please, see also Chapter 2.7.7 in 'Programming mental ray, Third Edition'\n"
            "A dummy profile will be used instead.",
            m_log_identifier.c_str());
        return false;
    }

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO,
        "Loading %s, "
        "original resolution %" FMT_SIZE_T "x%" FMT_SIZE_T ", "
        "interpolated resolution %ux%u.",
        m_log_identifier.c_str(),
        orig_nb_horizontal_angles, orig_nb_vertical_angles,
        horizontal_resolution, vertical_resolution);

    return true;
}

bool setup_lightprofile(
    mi::neuraylib::IReader* reader,
    const std::string& log_identifier,
    mi::neuraylib::Lightprofile_degree degree,
    mi::Uint32 flags,
    mi::Uint32& resolution_phi,
    mi::Uint32& resolution_theta,
    mi::Float32& start_phi,
    mi::Float32& start_theta,
    mi::Float32& delta_phi,
    mi::Float32& delta_theta,
    std::vector<mi::Float32>& data)
{
    Lightprofile_ies_parser parser(reader, log_identifier);
    return parser.setup_lightprofile(
        degree, flags, resolution_phi, resolution_theta,
        start_phi, start_theta, delta_phi, delta_theta, data);
}

}
}
