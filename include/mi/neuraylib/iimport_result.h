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
/// \brief Result of an import operation.

#ifndef MI_NEURAYLIB_IIMPORT_RESULT_H
#define MI_NEURAYLIB_IIMPORT_RESULT_H

#include <mi/base/enums.h>
#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {
    
/** 
\ifnot MDL_SOURCE_RELEASE
\addtogroup mi_neuray_impexp
@{
\endif
*/

/// This interface represents the result of an import operation.
///
/// \ifnot MDL_SOURCE_RELEASE
/// Such an import operation is triggered by #mi::neuraylib::IImport_api::import_elements() or
/// #mi::neuraylib::IImport_api::import_elements_from_string(). It gives access
/// to messages, message numbers, and message severities, and to important scene elements like root
/// group, camera instance, and options. Furthermore you can query the imported element names, if
/// requested during import.
///
/// Importers should use the message severities according to the following guidelines:
/// - #mi::base::details::MESSAGE_SEVERITY_FATAL \n
///   The importer is no longer usable.
/// - #mi::base::details::MESSAGE_SEVERITY_ERROR \n
///   The file contains errors and not all elements could be imported.
/// - #mi::base::details::MESSAGE_SEVERITY_WARNING \n
///   The file contains minor errors.
/// - #mi::base::details::MESSAGE_SEVERITY_INFO \n
///   A normal operational message.
/// - #mi::base::details::MESSAGE_SEVERITY_VERBOSE and
///   #mi::base::details::MESSAGE_SEVERITY_DEBUG \n
///   These message severities should be avoided by importers.
/// \endif
class IImport_result :
    public base::Interface_declare<0xa47741d4,0x49c5,0x418d,0xa5,0x4b,0xa6,0xfb,0xf4,0xa0,0x91,0x44>
{
public:
    /// Returns the name of the root group.
    ///
    /// If the scene file has a scene root (like a \c render or \c root statement in a \c .%mi
    /// file), the method returns the name of the root element. For most other files, this is the
    /// name of a group that was created by the importer and that contains the top-level scene
    /// elements. Returns \c NULL for \c .mdl files.
    virtual const char* get_rootgroup() const = 0;

    /// Returns the name of the camera instance.
    ///
    /// If the scene file has a scene root (like a \c render statement in a \c .%mi file) containing
    /// a camera instance, the method returns the name of the camera instance element.
    virtual const char* get_camera_inst() const = 0;

    /// Returns the name of the options element.
    /// If the scene file has a scene root (like a \c render statement in a \c .%mi file) containing
    /// an options element, the method returns the name of the options element.
    virtual const char* get_options() const = 0;

    /// Returns the length of the element array.
    virtual Size get_elements_length() const = 0;

    /// Returns the name of the element indicated by \p index.
    virtual const char* get_element( Size index) const = 0;

    /// Returns the number of the first error.
    ///
    /// The error number indicates the status of the import operation: 0 means success, all other
    /// values indicate failures, in which case #get_error_message() provides a diagnostic message.
    /// Numbers in the range 4000-5999 are reserved for custom importers.
    /// All other numbers are reserved for other purposes.
    ///
    /// It is possible to query the message numbers of all messages, see
    /// #get_messages_length() and #get_message_number(mi::Size)const. This method just reports the
    /// number of the first message of severity #mi::base::details::MESSAGE_SEVERITY_ERROR or above,
    /// or 0 if there is no such message.
    virtual Uint32 get_error_number() const = 0;

    /// Returns the message of the first error.
    ///
    /// A message describing the error condition corresponding to the error reported from
    /// #get_error_number().
    ///
    /// It is possible to query the all messages, see #get_messages_length() and
    /// #get_message(mi::Size)const. This method just reports the the first message of severity
    /// #mi::base::details::MESSAGE_SEVERITY_ERROR or above, or \c NULL if there is no such
    /// message.
    virtual const char* get_error_message() const = 0;

    /// Returns the number of messages.
    virtual Size get_messages_length() const = 0;

    /// Returns the message number for a given message from the array of messages.
    /// \see #get_messages_length()
    virtual Uint32 get_message_number( Size index) const = 0;

    /// Returns a message from the array of messages.
    /// \see #get_messages_length()
    virtual const char* get_message( Size index) const = 0;

    /// Returns the severity for a given message from the array of messages.
    /// \see #get_messages_length()
    virtual base::Message_severity get_message_severity( Size index) const = 0;
};

/// This interface represents the result of an import operation.
///
/// It is derived from the #mi::neuraylib::IImport_result interface and is intended to be used by
/// importer writers. In addition to the #mi::neuraylib::IImport_result interface it provides
/// methods to set all values.
///
/// \ifnot MDL_SOURCE_RELEASE
/// See #mi::neuraylib::IImport_api::import_elements() for common message numbers. Numbers in the
/// range 4000-5999 are reserved for importer-specific messages. All other numbers are reserved for
/// other purposes.
/// \endif
///
/// \note In case of a successful Import operation the array of messages should not contain an
///       explicit message with message number 0. If there are no other importer-specific messages,
///       messages, the message array should then just be empty, such that
///       #mi::neuraylib::IImport_result::get_error_number() returns 0 and
///       #mi::neuraylib::IImport_result::get_error_message() returns \c NULL.
class IImport_result_ext :
    public base::Interface_declare<0xe43ae7a3,0x7816,0x4915,0xb1,0x98,0x42,0x12,0x1d,0x1b,0xe2,0x09,
                                   neuraylib::IImport_result>
{
public:
    /// Sets the name of the root group.
    ///
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    virtual Sint32 set_rootgroup( const char* group) = 0;

    /// Sets the name of the camera instance.
    ///
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    virtual Sint32 set_camera_inst( const char* camera) = 0;

    /// Sets the name of the options.
    ///
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    virtual Sint32 set_options( const char* options) = 0;

    /// Appends an element to the array of recorded elements.
    ///
    /// \param element  The name of the element to append.
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    virtual Sint32 element_push_back( const char* element) = 0;

    /// Replaces an element in the array of recorded elements.
    ///
    /// The operation is skipped if \p index is out of bounds.
    ///
    /// \param index    The index of the element to be replaced.
    /// \param element  The name of the element element to be replaced.
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    ///                 - -2: \p index is out of bounds.
    virtual Sint32 set_element( Size index, const char* element) = 0;

    /// Removes all elements.
    virtual void clear_elements() = 0;

    /// Replaces all messages by the given message number, severity, and message.
    ///
    /// Equivalent to clear_messages(), followed by message_push_back().
    ///
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    ///
    /// \see #mi::neuraylib::IImport_result_ext for valid message numbers
    virtual Sint32 set_message(
        Uint32 number, base::Message_severity severity, const char* message) = 0;

    /// Appends a message number, severity, and message to the array of recorded message numbers,
    /// severities, and messages.
    ///
    /// \param number   The message number to append.
    /// \param severity The message severity to append.
    /// \param message  The message to append.
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    ///
    /// \see #mi::neuraylib::IImport_result_ext for valid message numbers
    virtual Sint32 message_push_back(
        Uint32 number, base::Message_severity severity, const char* message) = 0;

    /// Replaces a message number, severity, and message in the array of recorded message numbers,
    /// severities, and messages.
    ///
    /// \param index    The index of the message to be replaced.
    /// \param number   The message number to append.
    /// \param severity The message severity to append.
    /// \param message  The message to append.
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    ///                 - -2: \p index is out of bounds.
    ///
    /// \see #mi::neuraylib::IImport_result_ext for valid message numbers
    virtual Sint32 set_message(
        Size index, Uint32 number, base::Message_severity severity, const char* message) = 0;

    /// Removes all messages.
    virtual void clear_messages() = 0;

    /// Appends all elements in \p import_result to this instance.
    ///
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    virtual Sint32 append_elements( const IImport_result* import_result) = 0;

    /// Appends all messages in \p import_result to this instance.
    ///
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    virtual Sint32 append_messages( const IImport_result* import_result) = 0;
};


/*
\ifnot MDL_SOURCE_RELEASE
@}
\endif
*/ // end group mi_neuray_impexp

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IIMPORT_RESULT_H

