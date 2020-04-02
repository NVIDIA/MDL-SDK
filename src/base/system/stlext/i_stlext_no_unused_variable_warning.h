/******************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file
/// \brief Helper function to avoid "unused variable" warnings.

#ifndef BASE_SYSTEM_NO_UNUSED_VARIABLE_WARNING_H
#define BASE_SYSTEM_NO_UNUSED_VARIABLE_WARNING_H

namespace MI { namespace STLEXT {

/**
 * \brief Access a given variable in order to avoid "unused variable" warnings.
 *
 * This function helps to avoid "unused variable" warnings from the compiler in
 * a portable and overhead-free fashion, which is mainly useful in code that
 * has to rely on conditional compilation, i.e. debug vs. release builds. The
 * function is used as follows:
 *
 * <pre>
 *   int i;
 *   no_unused_variable_warning_please(i);
 * </pre>
 */

template <class T>
inline void no_unused_variable_warning_please(T & /* ignored */)
{
}

}} // namespace MI::STLEXT

#endif  // BASE_SYSTEM_NO_UNUSED_VARIABLE_WARNING_H
