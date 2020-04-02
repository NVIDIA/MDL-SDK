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
/// \brief Readers, used by importers.

#ifndef MI_NEURAYLIB_IREADER_H
#define MI_NEURAYLIB_IREADER_H

#include <mi/neuraylib/ireader_writer_base.h>

namespace mi {

namespace neuraylib {

/** \if MDL_SDK_API \addtogroup mi_neuray_mdl_sdk_misc
    \else \addtogroup mi_neuray_impexp
    \endif
@{
*/

/// A reader supports binary block reads and string-oriented line reads that zero-terminate the
/// result.
class IReader :
    public base::Interface_declare<0xc03de0cf,0x5a03,0x4e8f,0xa1,0x59,0x6c,0xad,0xd6,0xf8,0xae,0x58,
                                   neuraylib::IReader_writer_base>
{
public:
    /// Reads a number of bytes from the stream.
    ///
    /// \param buffer   The buffer to store the data in.
    /// \param size     The number of bytes to be read.
    /// \return         The number of bytes read, which may be less than \p size,
    ///                 or -1 in case of errors.
    virtual Sint64 read( char* buffer, Sint64 size) = 0;

    /// Reads a line from the stream.
    ///
    /// Reads at most \p size - 1 characters from the stream and stores them in \p buffer.
    /// Reading stops after a newline character or an end-of-file.
    /// If a newline is read, it is stored in the buffer.
    /// The buffer contents will be zero-terminated.
    ///
    /// \param buffer   The buffer to store the data in.
    /// \param size     The maximum number of bytes to be read.
    /// \return         \c true in case of success, or \c false in case of errors.
    virtual bool readline( char* buffer, Sint32 size) = 0;

    /// \name Lookahead capability
    //@{

    /// Indicates whether lookahead is (in principle) supported.
    ///
    /// An actual lookahead request might still report zero bytes of available lookahead data.
    virtual bool supports_lookahead() const = 0;

    /// Gives access to the lookahead data.
    ///
    /// The first \p size bytes of binary data are made available from the reader without changing
    /// its read position.
    /// The method sets the buffer pointer to point to the available lookahead data. It may be set
    /// to 0 if no lookahead is available, in which case the return value must be either 0 or -1.
    ///
    /// \param size          The size of the desired lookahead.
    /// \param[out] buffer   The address of a pointer to the buffer with the lookahead data. The
    ///                      address might be \c NULL if no lookahead is available, in which case
    ///                      the return value is either 0 or -1. The buffer is owned by the reader
    ///                      and must not be changed or deleted.
    /// \return              The number of bytes made available in the buffer, which may larger
    ///                      than, equal to, or less than the requested size, or 0 if no lookahead
    ///                      is available, or -1 in case of errors.
    virtual Sint64 lookahead( Sint64 size, const char** buffer) const = 0;

    //@}

};

/*@}*/ // end group mi_neuray_impexp

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IREADER_H
