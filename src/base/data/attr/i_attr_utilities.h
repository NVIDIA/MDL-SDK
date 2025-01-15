/***************************************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Collection of useful tools.

#ifndef BASE_DATA_ATTR_I_ATTR_UTILITIES_H
#define BASE_DATA_ATTR_I_ATTR_UTILITIES_H

#include "i_attr_type.h"
#include "i_attr_types.h"

#include <string>
#include <cstring>

#include <base/system/stlext/i_stlext_no_unused_variable_warning.h>
#include <base/data/db/i_db_tag.h>

namespace MI {
namespace ATTR {

class Type;

/// Evaluate the given typecode and find out how many fields of which type are available
/// and what is the overall size of the data described by the type. It's important to identify
/// tags (type 't') separately so the serializer stores them as type Tag in .mib files, because
/// impmib for instance remaps tag values.
/// \param the typecode to be evaluated
/// \param[out] type store the type of the primitives here
/// \param[out] count store the count of the primitives here
/// \param[out] size store the overall size here
void eval_typecode(
    Type_code typecode,
    int* type,
    int* count,
    int* size);

//--------------------------------------------------------------------------------------------------

}
}

#endif

