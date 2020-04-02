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
/// \brief Abstract interface for exporters.

#ifndef MI_NEURAYLIB_IEXPORTER_H
#define MI_NEURAYLIB_IEXPORTER_H

#include <mi/base/interface_declare.h>
#include <mi/neuraylib/iimpexp_base.h>

namespace mi {

class IArray;
class IMap;

namespace neuraylib {

class IExport_result;
class IImpexp_state;
class ITransaction;
class IWriter;

/** \addtogroup mi_neuray_impexp
@{
*/

/// Abstract interface for exporters.
class IExporter :
    public base::Interface_declare<0x543b5252,0x7c50,0x4998,0xa9,0xf3,0x97,0xa9,0x4e,0x11,0xfd,0x3a,
                                   neuraylib::IImpexp_base>
{
public:
    /// Indicates whether the exporter can handle the file type.
    ///
    /// Returns \c true if the exporter can handle the file type determined by the file name
    /// extension and if the writer has sufficient capabilities for export (such as random access
    /// or availability of a file handle if that is required for this exporter). The extension is
    /// defined as the part of the filename after and including the first dot, for example, ".mi".
    /// In the case of missing capabilities the exporter shall return \c false immediately.
    virtual bool test_file_type(const char* extension, const IWriter* writer) const = 0;

    using IImpexp_base::test_file_type;

    /// Exports a scene via a writer.
    ///
    /// Writes the scene identified by the root group, the camera instance, and the options to
    /// the writer in a format determined by the exporter and the given extension.
    ///
    /// Any elements referenced by the three argument elements are written as well, recursively,
    /// in an order that allows re-importing (e.g., in \c .%mi files, a referenced element must
    /// precede the reference). The scene graph cannot contain cyclic references and an exporter
    /// does not need to take any precautions to detect cycles.
    ///
    /// In addition to exporter-specific options, every exporter has to implement the following
    /// standard option:
    /// - \c "strip_prefix" of type #mi::IString: This prefix is to be stripped from the names
    ///   of all exported elements if they have the same prefix. Default: the empty string.
    ///
    /// It is strongly recommended that names for exporter-specific options use a prefix related to
    /// the exporter to avoid name conflicts, e.g., the file name extension.
    ///
    /// In the case of the \c .%mi file format, the exported file will contain a render statement.
    ///
    /// \param transaction        The transaction to be used.
    /// \param extension          The file name extension (which might influence the file format).
    /// \param writer             The writer to write the byte stream to.
    /// \param state              The current exporter state.
    /// \param rootgroup          The root group of the scene to be exported.
    /// \param caminst            The camera instance of the scene to be exported (optional).
    /// \param options            The options of the scene to be exported (optional).
    /// \param exporter_options   The options that control the way the exporter works (optional).
    /// \return                   An instance of #mi::neuraylib::IExport_result indicating success
    ///                           or failure.
    ///
    /// \ifnot MDL_SOURCE_RELEASE
    /// \see #mi::neuraylib::IExport_api::get_export_dependencies()
    /// \endif
    virtual IExport_result* export_scene(
        ITransaction* transaction,
        const char* extension,
        IWriter* writer,
        const char* rootgroup,
        const char* caminst,
        const char* options,
        const IMap* exporter_options,
        IImpexp_state* state) const = 0;

    /// Exports a set of named elements via a writer.
    ///
    /// Writes the named elements to the writer in a format determined by the exporter and the given
    /// extension. Note that in contrast to #export_scene() references are not followed recursively.
    ///
    /// The exporter can expect that the elements array contains no duplicates and that elements
    /// that are referenced come before elements that reference them. It is possible that these two
    /// conditions on the elements array are violated, but the exporter may then have undefined
    /// behavior, for example, produce invalid files.
    ///
    /// In addition to exporter-specific options, every exporter has to implement the following
    /// standard option:
    /// - \c "strip_prefix" of type #mi::IString: If present, this prefix is to be stripped from
    ///   all names of database elements if they have the same prefix. Default: the empty string.
    ///
    /// It is strongly recommended that names for exporter-specific options use a prefix related to
    /// the exporter to avoid name conflicts, e.g., the file name extension.
    ///
    /// In the case of the \c .%mi file format, the exported file will contain no render statement.
    ///
    /// \param transaction        The transaction to be used.
    /// \param extension          The file name extension (which might influence the file format).
    /// \param writer             The writer to write the byte stream to.
    /// \param state              The current exporter state.
    /// \param elements           The array of elements to be exported.
    /// \param exporter_options   The options that control the way the exporter works (optional).
    /// \return                   An instance of #mi::neuraylib::IExport_result indicating success
    ///                           or failure.
    virtual IExport_result* export_elements(
        ITransaction* transaction,
        const char* extension,
        IWriter* writer,
        const IArray* elements,
        const IMap* exporter_options,
        IImpexp_state* state) const = 0;
};

/*@}*/ // end group mi_neuray_impexp

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IEXPORTER_H

