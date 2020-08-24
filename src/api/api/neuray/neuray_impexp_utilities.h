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
 ** \brief Header for the Impexp_utilities implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_IMPEXP_UTILITIES_H
#define API_API_NEURAY_NEURAY_IMPEXP_UTILITIES_H

#include <mi/base/enums.h>

#include <set>
#include <string>
#include <vector>

#include <boost/core/noncopyable.hpp>
#include <base/data/db/i_db_tag.h>

namespace mi {

class IArray;
class IString;

namespace neuraylib {

class IExport_result;
class IExport_result_ext;
class IImport_result;
class IImport_result_ext;
class IImpexp_base;
class IImpexp_state;
class IReader;
class IWriter;
class ITransaction;

}

}

namespace MI {

namespace DB { class Transaction; }

namespace NEURAY {

class Recording_transaction;
class Transaction_impl;

class Impexp_utilities : public boost::noncopyable
{
public:

    // public API methods

    // (none)

    // internal methods

    // Convenience methods to create instances of various interfaces
    // =============================================================

    /// Creates an instance of #mi::neuraylib::IImport_result.
    ///
    /// Convenience function used by the importer wrappers.
    ///
    /// \param transaction        The transaction that is used to create the instance.
    /// \param message_number     The message number to set in the import result.
    /// \param message_severity   The message severity to set in the import result.
    /// \param message            The message message to set in the import result.
    /// \param rootgroup          The rootgroup to set in the import result.
    /// \param camera_inst        The camera_inst to set in the import result.
    /// \param options            The options to set in the import result.
    /// \param elements           The elements to set in the import result.
    /// \return                   The created instance of #mi::neuraylib::IImport_result.
    static mi::neuraylib::IImport_result* create_import_result(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 message_number,
        mi::base::Message_severity message_severity,
        const char* message,
        DB::Tag rootgroup = DB::Tag(),
        DB::Tag camera_inst = DB::Tag(),
        DB::Tag options = DB::Tag(),
        const std::vector<std::string>& elements = std::vector<std::string>());

    /// Creates an instance of #mi::neuraylib::IExport_result.
    ///
    /// Convenience function used by the exporter wrappers.
    ///
    /// \param transaction        The transaction that is used to create the instance.
    /// \param message_number     The message number to set in the export result.
    /// \param message_severity   The message severity to set in the export result.
    /// \param message            The message message to set in the export result.
    /// \return                   The created instance of #mi::neuraylib::IExport_result.
    static mi::neuraylib::IExport_result* create_export_result(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 message_number,
        mi::base::Message_severity message_severity,
        const char* message);

    /// Creates an instance of #mi::neuraylib::IExport_result.
    ///
    /// As above, but with exactly one %s format specifier in \p message which is replaced by
    /// \p argument.
    static mi::neuraylib::IExport_result* create_export_result(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 message_number,
        mi::base::Message_severity message_severity,
        const char* message,
        const char* argument);

    /// Creates an instance of #mi::neuraylib::IReader for the given URI.
    ///
    /// Convenience function used by Impexp_utilities and Import_api_impl.
    ///
    /// \param uri             The URI to be handled by the reader.
    /// \param path            The method returns the path that was considered for the URI \p uri.
    ///                        The empty string is returned for invalid URIs.
    /// \return                A reader for the URI, or \c NULL in case of errors, e.g., the URI is
    ///                        invalid, or the URI is valid, but the file denoted by the URI could
    ///                        not be opened.
    static mi::neuraylib::IReader* create_reader(
        const std::string& uri, std::string& path);

    /// Creates an instance of #mi::neuraylib::IReader for the given data.
    ///
    /// Convenience function used by Import_api_impl.
    ///
    /// \param data            Pointer to the start of the data block.
    /// \param length          Length of the data block.
    /// \return                A reader for the data block. The reader wraps the data block, it does
    ///                        not copy the data.
    static mi::neuraylib::IReader* create_reader( const char* data, mi::Size length);

    /// Creates an instance of #mi::neuraylib::IWriter for the given URI.
    ///
    /// Convenience function used by Impexp_utilities and Export_api_impl.
    ///
    /// \param uri             The URI to be handled by the writer.
    /// \param path            The method returns the path that was considered for the URI \p uri.
    ///                        The empty string is returned for invalid URIs.
    /// \return                A writer for the URI, or \c NULL in case of errors, e.g., the URI is
    ///                        invalid, or the URI is valid, but the file denoted by the URI could
    ///                        not be opened.
    static mi::neuraylib::IWriter* create_writer(
        const std::string& uri, std::string& path);

    /// Creates an instance of #mi::neuraylib::IImport_result_ext from an instance of
    /// #mi::neuraylib::IImport_result.
    ///
    /// \param transaction     The transaction that is used to create the instance.
    /// \param import_result   The instance to convert to #mi::neuraylib::IImport_result_ext.
    /// \return                If \p import_result implements the interface
    ///                        #mi::neuraylib::IImport_result_ext, this interface is simply
    ///                        returned. Otherwise, a new instance of
    ///                        #mi::neuraylib::IImport_result_ext is created and the data copied.
    static mi::neuraylib::IImport_result_ext* create_import_result_ext(
        mi::neuraylib::ITransaction* transaction,
        mi::neuraylib::IImport_result* import_result);

    /// Returns the elements names recorded by the recording transaction.
    static std::vector<std::string> get_recorded_elements(
        const Transaction_impl* transaction, Recording_transaction* recording_transaction);

    /// Return the file name extension of given URI
    ///
    /// The current implementation assumes that the URI scheme is "file:" (or missing) and ends
    /// with a filename. The file name extension is defined as the last dot (".") and any subsequent
    /// characters after(!) the last path/directory delimiter.
    ///
    /// \param uri             The URI.
    /// \return                The file name extension of the URI including the dot (or "" if there
    ///                        is none).
    static std::string get_extension( const std::string& uri);

    // Handling of ${shader}
    // =====================

    /// Returns \c true if \p path is "${shader}" or starts with "${shader}" and is followed by
    /// a path delimiter.
    static bool is_shader_path( const std::string& path);

    /// Returns the given path with "${shader}" replaced by the given shader path.
    ///
    /// If #is_shader_path() returns \c false, \p path is returned unchanged.
    static std::string resolve_shader_path(
        const std::string& path, const std::string& shader_path);

    // URI handling
    // ============

    /// Resolves the given URI against the URI of the parent state.
    ///
    /// If the URI of \p parent_state is absolute and the URI \p child is relative, then the
    /// returned URI is constructed as set forth in RFC 2396. Otherwise, (incl. \p parent_state
    /// being \c NULL), \p child is returned.
    ///
    /// Note that in this context \p child is consider as relative URI if it starts with
    /// "${shader}" (even though it would be considered as relative URI in a strict
    /// interpretation of RFC 2396).
    static std::string resolve_uri_against_state(
        const char* child, const mi::neuraylib::IImpexp_state* parent_state);

    /// Converts a filename into a URI.
    ///
    /// Returns the empty string if \p filename is the empty string. Otherwise returns an URI
    /// without URI scheme and URI authority. The URI path is constructed from the filename
    /// according to the following rules.
    ///
    /// On Linux and MacOS X, the URI path equals the filename.
    /// On Windows, backslashes in relative filenames are converted to slashes to obtain the URI
    /// path. Absolute filenames are mapped to URI paths according to the following table.
    ///
    /// <table>
    ///   <tr>
    ///     <th>Filename</th>
    ///     <th>URI path</th>
    ///     <th>Comment</th>
    ///   </tr>
    ///   <tr>
    ///     <td>C:\\dir1\\dir2\\file</td>
    ///     <td>/C:/dir1/dir2/file</td>
    ///     <td>-</td>
    ///   </tr>
    ///   <tr>
    ///     <td>\\dir1\\dir2\\file</td>
    ///     <td>/dir1/dir2/file</td>
    ///     <td>-</td>
    ///   </tr>
    ///   <tr>
    ///     <td>\\\\share\\dir1\\dir2\\file</td>
    ///     <td>//share/dir1/dir2/file</td>
    ///     <td>Note that an empty URI authority (\c //) is prepended since otherwise the
    ///         the share name is interpreted as URI authority.</td>
    ///   </tr>
    /// </table>
    ///
    /// \note There are no checks whether \p filename identifies an existing file, or whether that
    /// file is readable. The filename is simply converted to an URI according to some fixed rules.
    ///
    /// \note This method does not understand the special variable \c "${shader}".
    ///
    /// \see \ref mi_neuray_impexp for general information about URIs.
    static std::string convert_filename_to_uri( const std::string& filename);

    /// Converts a URI into a filename.
    ///
    /// Returns the empty string if
    /// - \p uri is the empty string,
    /// - the URI scheme is non-empty and different from \c "file",
    /// - the URI authority is non-empty, or
    /// - the URI path is empty.
    ///
    /// In all other cases the URI path is converted into a filename according to the following
    /// rules.
    ///
    /// On Linux and MacOS X, the filename equals the URI path.
    /// On Windows, slashes in relative URI paths are replaced by backslashes to obtain the
    /// filename. Absolute URI paths are mapped to file system paths according to the following
    /// table.
    ///
    /// <table>
    ///   <tr>
    ///     <th>URI path</th>
    ///     <th>Filename</th>
    ///     <th>Comment</th>
    ///   </tr>
    ///   <tr>
    ///     <td>/C:/dir1/dir2/file</td>
    ///     <td>C:\\dir1\\dir2\\file</td>
    ///     <td>-</td>
    ///   </tr>
    ///  <tr>
    ///     <td>/C/dir1/dir2/file</td>
    ///     <td>C:\\dir1\\dir2\\file</td>
    ///     <td>This mapping is supported in addition to the first one since a colon is a reserved
    ///        character in URIs.</td>
    ///   </tr>
    ///   <tr>
    ///     <td>/dir1/dir2/file</td>
    ///     <td>\\dir1\\dir2\\file</td>
    ///     <td>This mapping is only supported for top-level directory names not consisting of a
    ///         single letter.</td>
    ///   </tr>
    ///   <tr>
    ///     <td>//share/dir1/dir2/file</td>
    ///     <td>\\\\share\\dir1\\dir2\\file</td>
    ///     <td>This mapping requires an (otherwise optional) empty URI authority (\c //) since
    ///        otherwise the share name is interpreted as URI authority.</td>
    ///   </tr>
    /// </table>
    ///
    /// \note There are no checks whether \p uri identifies an existing file, or whether that
    /// file is readable. The URI is simply converted to a filename according to some fixed rules.
    ///
    /// \note This method does not understand the special variable \c "${shader}".
    ///
    /// \see \ref mi_neuray_impexp for general information about URIs.
    static std::string convert_uri_to_filename( const std::string& uri);

    // Name-to-Tag and String conversions
    // ==================================

    /// Converts element names to tags.
    ///
    /// The method returns an empty vector if an array element is not of type #mi::IString
    /// or there is no DB element of that name.
    static std::vector<DB::Tag> convert_names_to_tags(
        mi::neuraylib::ITransaction* transaction, const mi::IArray* names);

    /// Converts tags to element names.
    static mi::IArray* convert_tags_to_names(
        mi::neuraylib::ITransaction* transaction, const std::vector<DB::Tag>& tags);

    // Im-/exporter selection
    // ======================

    /// The less-than functor for im-/exporter selection.
    class Impexp_less
    {
    public:
        /// The less-than functor for im-/exporter selection.
        ///
        /// First, the priorities are considered. If different, they decide the result of the
        /// call. If equal, the author/name pairs are considered. If equal, the versions are
        /// considered. If the versions are different, they decide the result of the comparison.
        /// In all other cases, the result is \c false.
        bool operator()(
            const mi::neuraylib::IImpexp_base* lhs, const mi::neuraylib::IImpexp_base* rhs);
    };

    // Export dependencies
    // ===================

    /// Computes the elements that need to be exported by an exporter.
    ///
    /// \param db_transaction   The DB transaction to be used.
    /// \param tags             The elements to be exported.
    /// \param recurse          Indicates whether dependecies are to be included.
    /// \param time_stamp       Only elements that have been changed since the time stamp are
    ///                         included (or \c NULL to include all elements).
    /// \param shortcuts_mdl    If \c true, some MDL dependencies are skipped/simplified, e.g.,
    ///                         dependencies of MDL modules are skipped, and definitions are
    ///                         replaced by their module.
    /// \return                 A sorted list of elements to be exported.
    static std::vector<DB::Tag> get_export_elements(
        DB::Transaction* db_transaction,
        const std::vector<DB::Tag>& tags,
        bool recurse,
        DB::Tag_version* time_stamp,
        bool shortcuts_mdl);

private:

    /// The constant "${shader}".
    static std::string s_shader;

    /// Computes the elements that need to be exported by an exporter (recursive helper function).
    ///
    /// \param db_transaction   The DB transaction to be used.
    /// \param tag              The element to be exported.
    /// \param recurse          Indicates whether dependecies are to be included.
    /// \param time_stamp       Only elements that have been changed since the time stamp are
    ///                         included (or \c NULL to include all elements).
    /// \param shortcuts_mdl    If \c true, some MDL dependencies are skipped/simplified, e.g.,
    ///                         dependencies of MDL modules are skipped, and definitions are
    ///                         replaced by their module.
    /// \param result           A sorted list of elements to be exported (intermediate result).
    /// \param tags_seen        Set of elements already handled.
    static void get_export_elements_internal(
        DB::Transaction* db_transaction,
        DB::Tag tag,
        bool recurse,
        DB::Tag_version* time_stamp,
        bool shortcuts_mdl,
        std::vector<DB::Tag>& result,
        std::set<DB::Tag>& tags_seen);

    /// Checks whether a character is a drive letter
    ///
    /// \param ch  The character to check
    /// \return    \c true if \p character is in [A-Za-z], otherwise \c false
    static bool is_drive_letter( char c);
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_IMPEXP_UTILITIES_H
