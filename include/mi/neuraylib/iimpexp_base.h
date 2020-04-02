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
/// \brief Abstract %base interface common for importers and exporters.

#ifndef MI_NEURAYLIB_IIMPEXP_BASE_H
#define MI_NEURAYLIB_IIMPEXP_BASE_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

class IImpexp_state;

/** 
\ifnot MDL_SOURCE_RELEASE
\addtogroup mi_neuray_impexp
@{
\endif
*/

/// Confidence in capabilities of an importer or exporter.
///
/// If there is more than one importer or exporter that accepts a certain file format, the importer
/// or exporter with the highest confidence is taken.
///
/// \see #mi::neuraylib::IImpexp_base::get_priority()
enum Impexp_priority
{
    /// The highest confidence, reserved for importer and exporter from user plugins. An importer
    /// or exporter of this priority will always take precedence over all internal importers and
    /// exporters.
    IMPEXP_PRIORITY_OVERRIDE        = 4,
    /// The highest possible priority for internal importer or exporter. Use only for importer or
    /// exporter whose test function cannot fail and whose support for the format is complete.
    IMPEXP_PRIORITY_WELL_DEFINED    = 3,
    /// The test function might fail, or the format might not be fully supported.
    IMPEXP_PRIORITY_AMBIGUOUS       = 2,
    /// The importer or exporter does not have a good way of detecting the format or the support
    /// is very preliminary or incomplete.
    IMPEXP_PRIORITY_GUESS           = 1,
    //  Undocumented, for alignment only
    IMPEXP_PRIORITY_FORCE_32_BIT    = 0xffffffffU
};

mi_static_assert( sizeof( Impexp_priority) == sizeof( Uint32));

/// Abstract %base interface common for importers and exporters.
///
/// The common functionality of importers and exporters comprises a factory function for the
/// corresponding state, a test function to indicate support of a particular file extension, and
/// additional methods to provide information about the importer or exporter, like author, version
/// number, UUID, and so on.
class IImpexp_base :
    public base::Interface_declare<0xf14bab60,0x91d8,0x4a78,0xaa,0xc4,0x6d,0x14,0x02,0xb1,0x97,0x47>
{
public:
    /// Returns a state suitable for passing it to a recursive import or export call.
    ///
    /// The parameters should be used to initialize the corresponding properties of the state. The
    /// initial line number should be set to 1.
    ///
    /// \param uri            The URI of the associated file, or \c NULL if there is no associated
    ///                       file, e.g., for string-based import/export operations.
    /// \param parent_state   The state of the parent importer or exporter. The parent importer or
    ///                       exporter is the one that called the current importer or exporter.
    virtual IImpexp_state* create_impexp_state(
        const char* uri,
        const IImpexp_state* parent_state = 0) const = 0;

    /// Indicates whether a file name extension is supported.
    ///
    /// Returns \c true if the importer or exporter can handle the file type determined by
    /// the file name extension. The extension is defined as the part of the file name after and
    /// including the first dot, for example, \c ".mi".
    /// \ifnot MDL_SOURCE_RELEASE
    /// \see #mi::neuraylib::IImporter::test_file_type(),
    ///      #mi::neuraylib::IExporter::test_file_type() \n
    /// \endif
    ///      These more specific versions also pass an #mi::neuraylib::IReader or
    ///      #mi::neuraylib::IWriter as argument, which can be used to look at lookahead data for
    ///      magic file headers (readers only) or to decide if the reader or writer capabilities are
    ///      sufficient to do the import or export (for example, random access capability).
    virtual bool test_file_type( const char* extension) const = 0;

    /// Returns the priority of the importer or exporter.
    ///
    /// The priority expresses the confidence of the importer or exporter that its #test_file_type()
    /// method can identify the file and that the file format is fully supported.
    virtual Impexp_priority get_priority() const = 0;

    /// Returns the \p i -th supported file name extension.
    ///
    /// \return   The file name extension including the separating dot, or \c NULL if \p i is out of
    ///           range.
    virtual const char* get_supported_extensions( Uint32 i) const = 0;

    /// Returns a concise single-line clear text description of the importer or exporter.
    ///
    /// The description should name, besides maybe a product or brand name, the supported file
    /// format and, if applicable, the major file type version or versions that are supported.
    /// If a file format differs sufficiently from version to version, you may as well register
    /// different importers or exporters for each version.
    ///
    /// This description may be used to support user interaction, such as in command-line help or
    /// selection boxes of graphical user interfaces. This method must return a valid string.
    /// This description is also used in the importer or exporter selection algorithm, which selects
    /// among importers and exporters of the same name and same author the one that has the highest
    /// version number.
    virtual const char* get_name() const = 0;

    /// Returns a concise single-line clear text description of the author of this importer
    /// or exporter.
    ///
    /// This description may be used to support user interaction, such as in command-line help or
    /// selection boxes of graphical user interfaces. This method must return a valid string.
    ///
    /// This description is also used in the importer or exporter selection algorithm, which selects
    /// among importers and exporters of the same name and same author the one that has the highest
    /// version number.
    virtual const char* get_author() const = 0;

    /// Returns the unique identifier for the importer or exporter.
    ///
    /// You can register only one importer or exporter for a particular UUID. If you wish to support
    /// installations with multiple versions of your importer or exporter, you have to change the
    /// UUID for each minor and major version number change.
    virtual base::Uuid get_uuid() const = 0;

    /// Returns the major version number of the importer or exporter.
    ///
    /// If you register multiple importers or exporters with equal name and author, the importer or
    /// exporter with the higher version number will be taken. A version number is higher if the
    /// version number is higher, or if it is equal and the minor version number is higher
    /// major (lexicographic order).
    virtual Uint32 get_major_version() const = 0;

    /// Returns the minor version number of the importer or exporter.
    ///
    /// If you register multiple importers or exporters with equal name and author, the importer or
    /// exporter with the higher version number will be taken. A version number is higher if the
    /// version number is higher, or if it is equal and the minor version number is higher
    /// major (lexicographic order).
    virtual Uint32 get_minor_version() const = 0;
};

/*
\ifnot MDL_SOURCE_RELEASE
@}
\endif
*/ // end group mi_neuray_impexp

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IIMPEXP_BASE_H

