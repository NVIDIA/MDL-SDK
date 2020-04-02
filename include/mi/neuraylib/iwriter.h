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
/// \brief Writers, used by exporters.

#ifndef MI_NEURAYLIB_IWRITER_H
#define MI_NEURAYLIB_IWRITER_H

#include <mi/neuraylib/ireader_writer_base.h>

namespace mi {

namespace neuraylib {

/** \if MDL_SDK_API \addtogroup mi_neuray_mdl_sdk_misc
    \else \addtogroup mi_neuray_impexp
    \endif
@{
*/

/// A writer supports binary block writes and string-oriented line writes that accept a
/// zero-terminated string as argument.
class IWriter :
    public base::Interface_declare<0x0e6ecfbc,0x78c3,0x4082,0xba,0x51,0xa3,0x60,0xbb,0x1d,0x6f,0xc0,
                                   neuraylib::IReader_writer_base>
{
public:
    /// Writes a number of bytes to the stream.
    ///
    /// \param buffer  The buffer from where to read the data.
    /// \param size    The number of bytes to write.
    /// \return        The number of bytes written, or -1 in case of errors.
    virtual Sint64 write( const char* buffer, Sint64 size) = 0;

    /// Writes a zero-terminated string to the stream.
    ///
    /// \param str     The string to be written. Note that the writer does not add an extra
    ///                newline character at the end.
    /// \return        \c true in case of success, or \c false in case of errors.
    virtual bool writeline( const char* str) = 0;

    /// Flushes all buffered output to the stream.
    /// \return        \c true in case of success, or \c false in case of errors.
    virtual bool flush() = 0;
};

/*@}*/ // end group mi_neuray_impexp

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IWRITER_H
