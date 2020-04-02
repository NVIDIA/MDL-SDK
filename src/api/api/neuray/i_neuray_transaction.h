/***************************************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the NEURAY::ITransaction declaration.
 **/

#ifndef API_API_NEURAY_I_NEURAY_TRANSACTION_H
#define API_API_NEURAY_I_NEURAY_TRANSACTION_H

#include <mi/neuraylib/itransaction.h>

namespace MI {

namespace DB { class Transaction; }

namespace NEURAY {

/// Internal extension of the public mi::neuraylib::ITransaction interface.
///
/// This interface adds one additional method that allows to retrieve the wrapped DB::Transaction
/// pointer without the need to know the implementation class. This interface allows to pass a
/// DB::Transaction pointer around where a mi::neuraylib::ITransaction pointer is expected.
class ITransaction : public
    mi::base::Interface_declare<0xb440b146,0x64a9,0x4d7f,0xbd,0x3c,0x6a,0x68,0x79,0x01,0xd5,0xb5,
                                mi::neuraylib::ITransaction>
{
public:
    /// Returns the wrapped DB::Transaction pointer.
    virtual DB::Transaction* get_db_transaction() const = 0;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_I_NEURAY_TRANSACTION_H
