/******************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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
/** \file
 ** \brief
 **/

#ifndef BASE_HAL_CUDA_DEVICE_ID_H
#define BASE_HAL_CUDA_DEVICE_ID_H


namespace MI {
namespace CUDA {


/** \brief A Physical CUDA device ID.

 The purpose of this class is to avoid confusion between device indices
 and physical IDs.
 */
class Device_id
{
public:
    static const int INVALID = -42;
    static const int CPU = -1;

    Device_id() = default;
    explicit Device_id(int id) : m_id(id) {}
    int get_id() const { return m_id; }
    bool operator==(const Device_id& o) const { return m_id == o.m_id; }
    bool operator!=(const Device_id& o) const { return m_id != o.m_id; }
    bool operator<(const Device_id& o) const { return m_id < o.m_id; }
    bool is_valid() const { return INVALID != m_id; }
    bool is_cpu() const { return CPU == m_id; }
    bool is_cuda_device() const { return is_valid() && !is_cpu(); }

    static Device_id cpu() { return Device_id{CPU}; }
private:
    int m_id = INVALID;
};


}}


#endif //BASE_HAL_CUDA_DEVICE_ID_H
