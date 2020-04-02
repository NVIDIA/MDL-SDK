/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Header for the Uri implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_URI_H
#define API_API_NEURAY_NEURAY_URI_H

#include <string>

#include <boost/core/noncopyable.hpp>

namespace MI {

namespace NEURAY {

/// Represents URIs as described in RFC 2396.
///
/// Note that RFC 2396 has been superseded by RFC 3986 (see Appendix D for major changes).
/// The file: scheme is also mentioned in section 3.10 of RFC 1738. Since this RFC is to be made
/// obsolete there is an unofficial draft at
/// http://tools.ietf.org/html/draft-hoffman-file-uri-03 .
///
/// There is some ambiguity how to map URI paths to Windows paths. See
/// IImport_api::convert_uri_to_filename() for the implemented logic.
class Uri : public boost::noncopyable
{
public:

    // public API methods

    // (none)

    // internal methods

    /// Constructs a URI corresponding to the passed string.
    ///
    /// If the passed string can not be parsed as a URI, then the members of this class are set to
    /// the empty string.
    ///
    /// \param uri   The string to be parsed as URI
    Uri( const char* uri);

    /// Constructs a URI corresponding to the passed base and relative URI.
    ///
    /// If the passed strings can not be parsed as a URI, then the members of this class are set to
    /// the empty string.
    ///
    /// \param base_uri       The base URI
    /// \param relative_uri   The relative URI
    Uri( const char* base_uri, const char* relative_uri);

    /// Returns a string representation of the URI.
    std::string get_str() const;

    /// Indicates whether the URI is absolute or not.
    ///
    /// Note that a URI is absolute if it starts with "${shader}" (even thought it is not absolute
    /// in a strict interpretation of RFC 2396).
    bool is_absolute() const;

    /// Returns the URI scheme.
    const std::string& get_scheme() { return m_scheme; }

    /// Returns the URI authority.
    const std::string& get_authority() { return m_authority; }

    /// Returns the URI path.
    const std::string& get_path() { return m_path; }

    /// Returns the URI query.
    const std::string& get_query() { return m_query; }

    /// Returns the URI fragment.
    const std::string& get_fragment() { return m_fragment; }

private:
    /// The URI scheme
    std::string m_scheme;

    /// The URI authority
    std::string m_authority;

    /// The URI path
    std::string m_path;

    /// The URI query
    std::string m_query;

    /// The URI fragment
    std::string m_fragment;

    /// Parses a URI into its component parts
    ///
    /// If the passed string can not be parsed as a URI, then the members of this class are left
    /// unchanged.
    void parse_uri( const std::string& uri);

    /// Parses a parent and child URI into its component parts
    ///
    /// If the passed strings can not be parsed as a URI, then the members of this class are left
    /// unchanged.
    void parse_uri( const std::string& parent, const std::string& child);
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_URI_H
