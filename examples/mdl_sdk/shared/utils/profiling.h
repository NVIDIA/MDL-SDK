/******************************************************************************
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/shared/utils/profiling.h
//
// Code shared by all examples

#ifndef MDL_D3D12_PROFILING_H
#define MDL_D3D12_PROFILING_H

#include <chrono>
#include <string>


namespace mi { namespace examples { namespace profiling
{
    struct Timing
    {
        explicit Timing(std::string operation)
            : m_operation(operation)
        {
            m_start = std::chrono::steady_clock::now();
        }

        ~Timing()
        {
            auto stop = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = stop - m_start;
            printf("Finished '%s' after %f seconds.\n",
                m_operation.c_str(), elapsed_seconds.count());
        }

    private:
        std::string m_operation;
        std::chrono::steady_clock::time_point m_start;
    };
} } } // mi::examples::profiling

#endif  // MDL_D3D12_PROFILING_H
