/******************************************************************************
 * Copyright (c) 2009-2020, NVIDIA CORPORATION. All rights reserved.
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
 ******************************************************************************/

/// \file
/// \brief A buffered reader for \c DISK::IFile to avoid repeated
/// system calls for reading small amounts of data.

#ifndef BASE_HAL_I_DISK_BUFFERED_READER_H
#define BASE_HAL_I_DISK_BUFFERED_READER_H

#include "disk.h"

#include <cstring>
#include <base/lib/log/i_log_assert.h>
#include <base/system/stlext/i_stlext_scoped_array.h>

namespace MI {
namespace DISK {

/// A buffered reader for \c DISK::IFile to avoid repeated
/// system calls for reading small amounts of data.
/// When this class is used to read from a file, read
/// access should not be mixed with the access methods
/// of the underlying file itself. However the file can
/// be used directly again after destroying this instance.
class Buffered_reader
{
  public:
    /// Create a buffered reader.
    /// \param file The file, which must be opened in read mode.
    /// The file must stay valid as long as this instance exists.
    /// \param buffer_size The buffer size, i.e. the maximum size
    /// of chunks read from the file and kept in memory.
    /// Defaults to 32 MB.
    Buffered_reader(
        IFile* file,
        const size_t buffer_size = 32 << 20) :
        m_file(file)
    {
        ASSERT(M_DISK, m_file->is_open());

        // compute remaining file size
        const Sint64 cur_pos = m_file->tell();
        m_file->seek(0, 2); // jump to EOF
        const Sint64 end_pos = m_file->tell();
        m_file->seek(cur_pos, 0); // jump back
        m_max_read = size_t(end_pos - cur_pos);
        // don't allocate more than we can read
        m_buffer_size = min(m_max_read, buffer_size);
        m_buffer.reset(new char[m_buffer_size]);
        m_remaining = 0; // buffer empty
        m_pos = 0;
    }

    /// Destructor.
    ~Buffered_reader()
    {
        // seek the file pointer to the place
        // before the unconsumed data.
        m_file->seek(-Sint64(m_remaining), 1);
    }

    /// Read raw bytes, see \c DISK::IFile::read().
    /// \param buf Copy data here.
    /// \param num Copy this many bytes.
    /// \return True on success, false on error.
    bool read(
        char* buf,
        size_t num)
    {
        while (num)
        {
            if (m_remaining) // data in buffer left
            {
                const size_t copy_size = min(m_remaining, num);
                memcpy(buf, m_pos, copy_size);
                num -= copy_size;
                m_remaining -= copy_size;
                m_pos += copy_size;
                buf += copy_size;
            }
            else if (num >= m_buffer_size) // optimization for large requests
            {
                const Sint64 n = m_file->read(buf, num);
                if (n <= 0) // error
                    return false;
                m_max_read -= size_t(n);
                num -= size_t(n);
                buf += n;
            }
            else // need to refill buffer
            {
                m_pos = m_buffer.get();
                const size_t read_size = min(m_max_read, m_buffer_size);
                const Sint64 n = m_file->read(m_pos, read_size);
                if (n <= 0) // error
                    return false;
                m_max_read -= size_t(n);
                m_remaining = size_t(n);
            }
        }

        return true;
    }

    /// Read a simple POD type.
    /// \param result Store data here.
    /// \return True on success, false on error.
    template <typename T>
    bool read(
        T* result)
    {
        ASSERT(M_DISK, result);

        if (m_remaining >= sizeof(T))
        {
            *result = *reinterpret_cast<const T*>(m_pos);
            m_remaining -= sizeof(T);
            m_pos += sizeof(T);
            return true;
        }

        // not enough data left in the buffer
        return read(reinterpret_cast<char*>(result), sizeof(T));
    }
        
  private:
    static size_t min(const size_t a, const size_t b) { return a < b ? a : b; }

    IFile* m_file; ///< the file to read from
    size_t m_max_read; ///< remaining file size
    size_t m_buffer_size; ///< data buffer size
    STLEXT::Scoped_array<char> m_buffer; ///< data buffer
    size_t m_remaining; ///< remaining number of bytes in the buffer
    char* m_pos; ///< current position in the buffer
};

} // namespace MI
} // namespace DISK

#endif // BASE_HAL_I_DISK_BUFFERED_READER_H
