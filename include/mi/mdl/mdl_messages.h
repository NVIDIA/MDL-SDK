/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/
/// \file mi/mdl/mdl_messages.h
/// \brief Interfaces for compiler generated (error and warning) messages.
#ifndef MDL_MESSAGES_H
#define MDL_MESSAGES_H 1

#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_positions.h>

namespace mi {
namespace mdl {

/// A compiler generated message.
class IMessage : public Interface_owned
{
public:
    /// The possible severities of messages.
    enum Severity {
        MS_ERROR,       ///< An error message.
        MS_WARNING,     ///< A warning message.
        MS_INFO         ///< An informational message.
    };

    /// Get the message severity.
    virtual Severity get_severity() const = 0;

    /// Get the message code.
    ///
    /// The code is just a unique ID for the message that allows the user application
    /// to react on several error or warning messages without parsing the message
    /// text (which might be translated into a local language).
    virtual int get_code() const = 0;

    /// Get the message string.
    ///
    /// This is a human readable string that might be localized. An application
    /// should not try to parse it.
    virtual char const *get_string() const = 0;

    /// Get the file name associated with this massage if any.
    virtual char const *get_file() const = 0;

    /// Get the position to which the message is associated.
    virtual Position const *get_position() const = 0;

    /// Get the number of notes attached to this message.
    ///
    /// Notes can be used to describe an error message further or add additional details,
    /// like the set of possible overloads or the location of a previous definition.
    /// An application can suppress the output of notes, but typically they deliver
    /// viable information for the user.
    virtual size_t get_note_count() const = 0;

    /// Get the note attached at index.
    ///
    /// \param index  The index of the attached note.
    ///
    /// \returns      The attached note or NULL if the index does not exists.
    virtual IMessage const *get_note(size_t index) const = 0;

    /// Get the message class.
    ///
    /// The message class gives a hint which part of the MDL Core  (compiler, backend,
    /// archive tool) issued this message.
    virtual char get_class() const = 0;
};

/// The interface describes a list of messages.
///
/// If all messages have a source code position they are ordered in ascending position
/// order.
class Messages : public Interface_owned {
public:
    /// Get the number of messages.
    virtual size_t get_message_count() const = 0;

    /// Get the message at index.
    virtual IMessage const *get_message(size_t index) const = 0;

    /// Get number of error messages.
    virtual size_t get_error_message_count() const = 0;

    /// Get the error message at index.
    virtual IMessage const *get_error_message(size_t index) const = 0;

    /// Add a message.
    ///
    /// \param severity        The severity of the message.
    /// \param code            The code of the message.
    /// \param msg_class       The class of the message.
    /// \param str             The message string.
    /// \param file            The file from which the message originates.
    /// \param start_line      The line on which the message starts.
    /// \param start_column    The column on which the message starts.
    /// \param end_line        The line on which the message ends.
    /// \param end_column      The column on which the message ends.
    ///
    /// \returns               The current index of the message.
    virtual size_t add_message(
        IMessage::Severity severity,
        int                code,
        char               msg_class,
        char const         *str,
        char const         *file,
        int                start_line,
        int                start_column,
        int                end_line,
        int                end_column) = 0;

    /// Add a note to a message.
    ///
    /// \param index           The index of the message to which to add the note.
    /// \param severity        The severity of the message.
    /// \param code            The code of the message.
    /// \param msg_class       The class of the message.
    /// \param str             The message string.
    /// \param file            The file from which the message originates.
    /// \param start_line      The line on which the message starts.
    /// \param start_column    The column on which the message starts.
    /// \param end_line        The line on which the message ends.
    /// \param end_column      The column on which the message ends.
    ///
    /// \returns               The index of the note.
    virtual size_t add_note(
        size_t             index,
        IMessage::Severity severity,
        int                code,
        char               msg_class,
        char const         *str,
        char const         *file,
        int                start_line,
        int                start_column,
        int                end_line,
        int                end_column) = 0;
};

} // mdl
} // mi

#endif
