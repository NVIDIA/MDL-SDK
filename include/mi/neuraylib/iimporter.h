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
/// \brief Abstract interface for importers.

#ifndef MI_NEURAYLIB_IIMPORTER_H
#define MI_NEURAYLIB_IIMPORTER_H

#include <mi/base/interface_declare.h>
#include <mi/neuraylib/iimpexp_base.h>

namespace mi {

class IMap;

namespace neuraylib {

class IImpexp_state;
class IImport_result;
class IReader;
class ITransaction;

/** \addtogroup mi_neuray_impexp
@{
*/

/// Abstract interface for importers.
class IImporter :
    public base::Interface_declare<0x1288a8b7,0x13ba,0x4010,0xb9,0x0c,0xbc,0xf5,0xca,0x49,0x70,0xdd,
                                   neuraylib::IImpexp_base>
{
public:
    /// Indicates whether the importer can handle the file type.
    ///
    /// Returns \c true if the importer can handle the file type determined by the file name
    /// extension or by looking at the lookahead. The extension is defined as the part of the file
    /// name after and including the first dot, for example, ".mi". This method also checks if the
    /// reader has sufficient capabilities to import successfully. If not, the importer shall return
    /// \c false immediately.
    ///
    /// For formats that have a mandatory and sufficiently distinguishing magic header, the importer
    /// shall use an available lookahead to determine the answer.
    /// Otherwise, the importer shall use the file name extension. In addition, if the lookahead is
    /// available the importer may check if the file header is plausible. If the file header is not
    /// understood by the importer, it should return \c false.
    virtual bool test_file_type(const char* extension, const IReader* reader) const = 0;

    using IImpexp_base::test_file_type;

    /// Imports a scene via a reader.
    ///
    /// Imports all elements from the reader in a format determined by the file extension and
    /// (optionally) the lookahead of the reader.
    ///
    /// In addition to importer-specific options, every importer has to implement the following
    /// standard options:
    /// - \c "prefix" of type #mi::IString: This prefix is to be prepended to the names of all
    ///   imported elements. Default: the empty string.
    /// - \c "list_elements" of type #mi::IBoolean: If \c true, the name of each imported element
    ///   has to be stored in the returned instance of #mi::neuraylib::IImport_result (e.g. via
    ///   #mi::neuraylib::IImport_result_ext::element_push_back()). Default: \c false.
    ///
    /// It is strongly recommended that names for importer-specific options use a prefix related to
    /// the importer to avoid name conflicts, e.g., the file name extension.
    ///
    /// \param transaction        The transaction to be used.
    /// \param extension          The file name extension (which might influence the file format).
    /// \param reader             The reader to read the byte stream from.
    /// \param importer_options   The options that control the way the importer works (optional).
    /// \param state              The current importer state.
    /// \return                   An instance of #mi::neuraylib::IImport_result indicating success
    ///                           or failure.
    virtual IImport_result* import_elements(
        ITransaction* transaction,
        const char* extension,
        IReader* reader,
        const IMap* importer_options,
        IImpexp_state* state) const = 0;
};

/*@}*/ // end group mi_neuray_impexp

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IIMPORTER_H
