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
 ** \brief Source for the Uri implementation.
 **/

#include "pch.h"

#include "neuray_uri.h"
#include "neuray_impexp_utilities.h"

#include <regex>
#include <mi/base/config.h>        // for MI_PLATFORM_WINDOWS
#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace NEURAY {

Uri::Uri( const char* uri)
{
    parse_uri( uri ? uri : "");
}

Uri::Uri( const char* base_uri, const char* relative_uri)
{
    parse_uri( base_uri ? base_uri : "", relative_uri ? relative_uri : "");
}

std::string Uri::get_str() const
{
    // see RFC 2396, section 5.2, step 7
    std::string result;
    if( !m_scheme.empty())
        result += m_scheme + ":";
    if( !m_authority.empty())
        result += "//" + m_authority;
    if( m_authority.empty() && m_path.substr( 0, 2) == "//")
        result += "//";
    result += m_path;
    if( !m_query.empty())
        result += "?" + m_query;
    if( !m_fragment.empty())
        result += "#" + m_fragment;
    return result;
}

bool Uri::is_absolute() const
{
    std::string uri = get_str();
    if( Impexp_utilities::is_shader_path( uri))
        return true;
    return !m_scheme.empty() || ( !m_path.empty() && (m_path[0] == '/'));
}

// Regular expression of RFC 2396, appendix B. Parses a HTTP URI into the following parts
//
//     scheme    = $2
//     authority = $4
//     path      = $5
//     query     = $7
//     fragment  = $9
//
// where $n indicates the n-th parentheses of the expression.
static const std::regex regex(
    "^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\\?([^#]*))?(#(.*))?", std::regex::extended);

void Uri::parse_uri( const std::string& uri)
{
    ASSERT( M_NEURAY_API, regex.mark_count() == 9);

    std::smatch matches;
    if( regex_match( uri, matches, regex)) {
        ASSERT( M_NEURAY_API, matches.size() == 10);
        m_scheme    = matches[2].str();
        m_authority = matches[4].str();
        m_path      = matches[5].str();
        m_query     = matches[7].str();
        m_fragment  = matches[9].str();
    }
}

void Uri::parse_uri( const std::string &parent, const std::string &child)
{
    // Create URI to hold base URI
    Uri base( parent.c_str());

    // Create URI to hold absolute URI, modified below
    parse_uri( !parent.empty() ? parent : std::string());

    // For the origin of this algorithm see RFC 2396 Section 5.2

    //    This section describes an example algorithm for resolving URI
    //    references that might be relative to a given base URI.

    //    The base URI is established according to the rules of Section 5.1 and
    //    parsed into the four main components as described in Section 3.  Note
    //    that only the scheme component is required to be present in the base
    //    URI; the other components may be empty or undefined.  A component is
    //    undefined if its preceding separator does not appear in the URI
    //    reference; the path component is never undefined, though it may be
    //    empty.  The base URI's query component is not used by the resolution
    //    algorithm and may be discarded.
    m_query.clear();

    // 1) The URI reference is parsed into the potential four components and
    //    fragment identifier, as described in Section 4.3.
    Uri reference( child.c_str());

    // 2) If the path component is empty and the scheme, authority, and
    //    query components are undefined, then it is a reference to the
    //    current document and we are done.  Otherwise, the reference URI's
    //    query and fragment components are defined as found (or not found)
    //    within the URI reference and not inherited from the base URI.
    bool path_empty = reference.m_path.empty();
    bool query_empty = reference.m_query.empty();
    bool scheme_empty = reference.m_scheme.empty();
    bool authority_empty = reference.m_authority.empty();
    if( path_empty && scheme_empty && authority_empty && query_empty) {
        m_fragment = reference.m_fragment;
        return;
    }

    m_query = reference.m_query;
    m_fragment = reference.m_fragment;

    // 3) If the scheme component is defined, indicating that the reference
    //    starts with a scheme name, then the reference is interpreted as an
    //    absolute URI and we are done.  Otherwise, the reference URI's
    //    scheme is inherited from the base URI's scheme component.
    if( !scheme_empty) {
        parse_uri( child);
        return;
    }

    m_scheme = base.m_scheme;

    //    Due to a loophole in prior specifications [RFC1630], some parsers
    //    allow the scheme name to be present in a relative URI if it is the
    //    same as the base URI scheme.  Unfortunately, this can conflict
    //    with the correct parsing of non-hierarchical URI.  For backwards
    //    compatibility, an implementation may work around such references
    //    by removing the scheme if it matches that of the base URI and the
    //    scheme is known to always use the <hier_part> syntax.  The parser
    //    can then continue with the steps below for the remainder of the
    //    reference components.  Validating parsers should mark such a
    //    misformed relative reference as an error.

    // 4) If the authority component is defined, then the reference is a
    //    network-path and we skip to step 7.  Otherwise, the reference
    //    URI's authority is inherited from the base URI's authority
    //    component, which will also be undefined if the URI scheme does not
    //    use an authority component.
    if( !authority_empty)
      return; // Step 7 simply returns

    m_authority = base.m_authority;

    // 5) If the path component begins with a slash character ("/"), then
    //    the reference is an absolute-path and we skip to step 7.
    if( !path_empty && (reference.m_path[0] == '/'))
      return; // Step 7 simply returns

    // 6) If this step is reached, then we are resolving a relative-path
    //    reference.  The relative path needs to be merged with the base
    //    URI's path.  Although there are many ways to do this, we will
    //    describe a simple method using a separate string buffer.
    //
    //    a) All but the last segment of the base URI's path component is
    //       copied to the buffer.  In other words, any characters after the
    //       last (right-most) slash character, if any, are excluded.
    std::string buffer = base.m_path;
    std::string::size_type slash_position = buffer.rfind( '/');
    if( std::string::npos != slash_position)
        buffer = buffer.substr( 0, slash_position + 1);

    //    b) The reference's path component is appended to the buffer
    //       string.
    buffer += reference.m_path;

    //    c) All occurrences of "./", where "." is a complete path segment,
    //       are removed from the buffer string.
    std::string::size_type dot_slash_position = buffer.find( "./");
    while( std::string::npos != dot_slash_position) {
        buffer.replace( dot_slash_position, 2, "");
        dot_slash_position = buffer.find( "./");
    }

    //    d) If the buffer string ends with "." as a complete path segment,
    //       that "." is removed.
    if(( buffer.size() - 1) == buffer.rfind( '.'))
        buffer = buffer.substr( 0, buffer.size() - 1);

    //    e) All occurrences of "<segment>/../", where <segment> is a
    //       complete path segment not equal to "..", are removed from the
    //       buffer string.  Removal of these path segments is performed
    //       iteratively, removing the leftmost matching pattern on each
    //       iteration, until no matching pattern remains.
    std::string::size_type dot_dot_slash_position = buffer.find( "../");
    while( std::string::npos != dot_dot_slash_position) {
        buffer.replace(dot_dot_slash_position, 3, "");
        dot_dot_slash_position = buffer.find( "../");
    }

    //    f) If the buffer string ends with "<segment>/..", where <segment>
    //       is a complete path segment not equal to "..", that
    //       "<segment>/.." is removed.
    if(( buffer.size() - 2) == buffer.rfind( ".."))
        buffer = buffer.substr( 0, buffer.size() - 2);

    //    g) If the resulting buffer string still begins with one or more
    //       complete path segments of "..", then the reference is
    //       considered to be in error.  Implementations may handle this
    //       error by retaining these components in the resolved path (i.e.,
    //       treating them as part of the final URI), by removing them from
    //       the resolved path (i.e., discarding relative levels above the
    //       root), or by avoiding traversal of the reference.

    //    h) The remaining buffer string is the reference URI's new path
    //       component.
    m_path = buffer;

    //   7) The resulting URI components, including any inherited from the
    //    base URI, are recombined to give the absolute form of the URI
    //    reference.  Using pseudocode, this would be
    //
    //       result = ""
    //
    //       if scheme is defined then
    //         append scheme to result
    //         append ":" to result
    //
    //       if authority is defined then
    //         append "//" to result
    //         append authority to result
    //
    //       append path to result
    //
    //       if query is defined then
    //         append "?" to result
    //         append query to result
    //
    //       if fragment is defined then
    //         append "#" to result
    //         append fragment to result
    //
    //       return result

    // Step 7 is done in Uri::get_str()
}

} // namespace NEURAY

} // namespace MI
