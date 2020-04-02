/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief State passed to recursive calls of importers and exporters.

#ifndef MI_NEURAYLIB_IIMPEXP_STATE_H
#define MI_NEURAYLIB_IIMPEXP_STATE_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

/** 
\ifnot MDL_SOURCE_RELEASE
\addtogroup mi_neuray_impexp
@{
\endif
*/

/// This interface represents states that are passed to recursive calls of importers and exporters.
///
/// It carries all state information that a recursive importer call may need to access from its
/// calling importer, such as the URI and the current line number. It is, for example, used to
/// provide proper diagnostics. The class implementing the state needs to be derived from this
/// #mi::neuraylib::IImpexp_state class.
///
/// Importer and exporter classes have a factory function to create the state object. This allows
/// the calling function to create a state, initialize it and querying the number of lines imported
/// after running the actual importer.
///
/// Importers and exporters can enrich their implementation of the state with additional information
/// they may need to import or export specific formats recursively.
class IImpexp_state :
    public base::Interface_declare<0x8646a2cb,0x609f,0x453d,0xbd,0xd6,0xc7,0xbf,0xea,0xdd,0x82,0x1d>
{
public:
    /// Returns the URI for this file.
    ///
    /// \return   The URI of the associated file, or \c NULL if there is no associated file, e.g.,
    ///           for string-based import/export operations.
    virtual const char* get_uri() const = 0;

    /// Returns the line number after the last read or write operation.
    ///
    /// Line number counting starts with 1 for the first line. Only applicable for line oriented
    /// file formats.
    virtual Uint32 get_line_number() const = 0;

    /// Sets the line number to \p n.
    ///
    /// Line number counting starts with 1 for the first line.
    virtual void set_line_number( Uint32 n) = 0;

    /// Convenience function that increments the line number by one.
    ///
    /// Line number counting starts with 1 for the first line.
    virtual void incr_line_number() = 0;

    /// Returns the state of the parent importer or exporter.
    ///
    /// The parent importer or exporter is the one that called the current importer or exporter.
    /// Returns \c NULL if there is no parent importer or exporter.
    virtual const IImpexp_state* get_parent_state() const = 0;
};

/*
\ifnot MDL_SOURCE_RELEASE
@}
\endif
*/ // end group mi_neuray_impexp

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IIMPEXP_STATE_H

