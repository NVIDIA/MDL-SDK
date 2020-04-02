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
/// \brief Result of an export operation.

#ifndef MI_NEURAYLIB_IEXPORT_RESULT_H
#define MI_NEURAYLIB_IEXPORT_RESULT_H

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

/// This interface represents the result of an export operation.
///
/// \ifnot MDL_SOURCE_RELEASE
/// Such an export operation is triggered by #mi::neuraylib::IExport_api::export_elements() or
/// #mi::neuraylib::IExport_api::export_scene(). It gives access to messages, message numbers, and
/// and message severities.
/// \endif
///
/// Exporters should use the message severities according to the following guidelines:
/// - #mi::base::details::MESSAGE_SEVERITY_FATAL \n
///   The exporter is no longer usable.
/// - #mi::base::details::MESSAGE_SEVERITY_ERROR \n
///   The exporter was not able to export all elements successfully.
/// - #mi::base::details::MESSAGE_SEVERITY_WARNING \n
///   A minor problem occurred during export.
/// - #mi::base::details::MESSAGE_SEVERITY_INFO \n
///   A normal operational message.
/// - #mi::base::details::MESSAGE_SEVERITY_VERBOSE and
///   #mi::base::details::MESSAGE_SEVERITY_DEBUG \n
///   These message severities should be avoided by exporters.
class IExport_result :
    public base::Interface_declare<0xb900251e,0x34e9,0x4a56,0x83,0x77,0x69,0x97,0x69,0x6b,0x82,0x84>
{
public:
    /// Returns the number of the first error.
    ///
    /// The error number indicates the status of the import operation: 0 means success, all other
    /// values indicate failures, in which case #get_error_message() provides a diagnostic message.
    /// Numbers in the range 6000-7999 are reserved for custom exporters.
    /// All other numbers are reserved for other purposes.
    ///
    /// It is possible to query the message numbers of all messages, see #get_messages_length() and
    /// #get_message_number(mi::Size)const. This method just reports the number of the first
    /// message of severity #mi::base::details::MESSAGE_SEVERITY_ERROR or above, or 0 if there is
    /// no such message.
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
    /// see #get_messages_length()
    virtual Uint32 get_message_number( Size index) const = 0;

    /// Returns a message from the array of messages.
    /// see #get_messages_length()
    virtual const char* get_message( Size index) const = 0;

    /// Returns the severity for a given message from the array of messages.
    /// see #get_messages_length()
    virtual base::Message_severity get_message_severity( Size index) const = 0;
};

/// This interface represents the result of an export operation.
///
/// It is derived from the #mi::neuraylib::IExport_result interface and is intended to be used by
/// exporter writers. In addition to the #mi::neuraylib::IExport_result interface it provides
/// methods to set all values.
///
/// \ifnot MDL_SOURCE_RELEASE
/// See #mi::neuraylib::IExport_api::export_scene() and
/// #mi::neuraylib::IExport_api::export_elements() for common message numbers. Numbers in the range
/// 6000-7999 are reserved for exporter-specific messages. All other numbers are reserved for other
/// purposes.
/// \endif
///
/// \note In case of a successful export operation the array of messages should not contain an
///       explicit message with message number 0. If there are no other exporter-specific messages,
///       messages, the message array should then just be empty, such that
///       #mi::neuraylib::IExport_result::get_error_number() returns 0 and
///       #mi::neuraylib::IExport_result::get_error_message() returns \c NULL.
class IExport_result_ext :
    public base::Interface_declare<0xfbf13ba1,0x7310,0x4e1a,0x80,0x0a,0x88,0xc4,0x20,0x3e,0xad,0x96,
                                   neuraylib::IExport_result>
{
public:
    /// Replaces all messages by the given message number, severity, and message.
    ///
    /// Equivalent to clear_messages(), followed by message_push_back().
    ///
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    ///
    /// \see #mi::neuraylib::IExport_result_ext for valid message numbers
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
    /// \see #mi::neuraylib::IExport_result_ext for valid message numbers
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
    /// \see #mi::neuraylib::IExport_result_ext for valid message numbers
    virtual Sint32 set_message(
        Size index, Uint32 number, base::Message_severity severity, const char* message) = 0;

    /// Removes all messages.
    virtual void clear_messages() = 0;

    /// Appends all messages in \p export_result to this instance.
    /// \return
    ///                 -  0: Success.
    ///                 - -1: Invalid parameters (\c NULL pointer).
    virtual Sint32 append_messages( const IExport_result* export_result) = 0;
};

/*
\ifnot MDL_SOURCE_RELEASE
@}
\endif
*/ // end group mi_neuray_impexp

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IEXPORT_RESULT_H

