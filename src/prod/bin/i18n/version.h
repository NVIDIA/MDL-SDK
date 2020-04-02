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
#pragma once

#include <mi/mdl_sdk.h>
#include <string>

// In the GNU C Library, "minor" and "major" are defined
#ifdef minor
    #undef minor
#endif

#ifdef major
    #undef major
#endif

namespace i18n
{
/// Helper for version strings.
///
/// The format of the value must follow Semantic Versioning 2.0.0[semver.org].
/// Pre - release labels are permitted, e.g., "1.0.0-alpha.1"
class Version
{
    mi::Uint32 m_major;
    mi::Uint32 m_minor;
    mi::Uint32 m_patch;
    std::string m_label;
public:
    static bool Test();
public:
    /// Default constructor
    Version()
        :m_major(0), m_minor(0), m_patch(0)
    {}

    /// Build from major.minor.patch[label].
    Version(
        const mi::Uint32 & major
        , const mi::Uint32 & minor
        , const mi::Uint32 & patch
        , const std::string & label)
        :m_major(major), m_minor(minor), m_patch(patch), m_label(label)
    {}

    /// Build from string, e.g., "1.0.0-alpha.1"
    Version(const std::string & version);

    /// Parse string version from string.
    ///
    /// Input: "1.0.0-alpha.1" will output:
    ///     major = 1
    ///     minor = 0
    ///     patch = 0
    ///     label = "alpha.1"
    ///
    /// \param  version     The version string (e.g. "1.0.0-alpha.1").
    /// \param  major       Output "major" part of the version string
    /// \param  minor       Output "minor" part of the version string
    /// \param  patch       Output "patch" part of the version string
    /// \param  label       Output "label" part of the version string
    /// \return
    ///		-  0: Success
    mi::Sint32 parse_version(
        const std::string & version
        , mi::Uint32 & major
        , mi::Uint32 & minor
        , mi::Uint32 & patch
        , std::string & label
    ) const
    {
        char * lbl = new char[version.size()];
        lbl[0] = '\0';
        sscanf(version.c_str(), "%d.%d.%d-%s", &major, &minor, &patch, lbl);
        label = lbl;
        return 0;
    }

    /// Return major
    mi::Uint32 major() const
    {
        return m_major;
    }

    /// Return minor
    mi::Uint32 minor() const
    {
        return m_minor;
    }

    /// Return patch
    mi::Uint32 patch() const
    {
        return m_patch;
    }

    /// Return label
    std::string label() const
    {
        return m_label;
    }

    /// Return major, minor, patch depending on the index.
    ///
    /// index == 0, return major
    /// index == 1, return minor
    /// index == 2, return label
    mi::Uint32 operator[](int i) const
    {
        switch (i)
        {
        case 0: return m_major;
        case 1: return m_minor;
        case 2: return m_patch;
        };
        return 0;
    }

    /// Pretty print Version to string
    friend std::ostream& operator<<(std::ostream& os, const Version& dt);
};

/// Pretty print Version to string.
std::ostream& operator<<(std::ostream& os, const Version& dt);
/// Version comparison operators.
bool operator< (const Version & lhs, const Version & rhs);
bool operator> (const Version & lhs, const Version & rhs);
bool operator<=(const Version & lhs, const Version & rhs);
bool operator>=(const Version & lhs, const Version & rhs);
bool operator==(const Version & lhs, const Version & rhs);
bool operator!=(const Version & lhs, const Version & rhs);

} // namespace i18n
