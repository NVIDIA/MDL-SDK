/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "i_attr_utilities.h"
#include "i_attr_type.h"
#include "attr.h"

#include <base/lib/cont/i_cont_rle_array.h>
#include <base/lib/log/i_log_logger.h>

#include <mi/math.h>

#include <sstream>

namespace MI {
namespace ATTR {

using namespace std;
using namespace CONT;
using namespace LOG;

using namespace mi::math;



// evaluate the given typecode and find out how many fields of which type are
// available and what is the overall size of the data described by the type.
// It's important to identify tags (type 't') separately so the serializer
// stores them as type Tag in .mib files, because impmib remaps tag values.
void eval_typecode(
    Type_code typecode,                                 // the typecode to be evaluated
    int* type,                                          // store the type of the primitives here
    int* count,                                         // store the count of the primitives here
    int* size)                                          // store the overall size here
{
    *type = *count = *size = 0;
    switch(typecode) {
      case TYPE_BOOLEAN:                *type = 'c'; *size = 1; *count = 1;     break;
      case TYPE_INT8:                   *type = 'c'; *size = 1; *count = 1;     break;
      case TYPE_INT16:                  *type = 's'; *size = 2; *count = 1;     break;
      case TYPE_INT32:                  *type = 'i'; *size = 4; *count = 1;     break;
      case TYPE_INT64:                  *type = 'q'; *size = 8; *count = 1;     break;
      case TYPE_SCALAR:                 *type = 'f'; *size = 4; *count = 1;     break;
      case TYPE_VECTOR2:                *type = 'f'; *size = 4; *count = 2;     break;
      case TYPE_VECTOR3:                *type = 'f'; *size = 4; *count = 3;     break;
      case TYPE_VECTOR4:                *type = 'f'; *size = 4; *count = 4;     break;
      case TYPE_DSCALAR:                *type = 'd'; *size = 8; *count = 1;     break;
      case TYPE_DVECTOR2:               *type = 'd'; *size = 8; *count = 2;     break;
      case TYPE_DVECTOR3:               *type = 'd'; *size = 8; *count = 3;     break;
      case TYPE_DVECTOR4:               *type = 'd'; *size = 8; *count = 4;     break;
          // case TYPE_MATRIX4X4:       // identical to type matrix
      case TYPE_MATRIX:                 *type = 'f'; *size = 4; *count = 16;    break;
      case TYPE_DMATRIX:                *type = 'd'; *size = 8; *count = 16;    break;
      case TYPE_QUATERNION:             *type = 'f'; *size = 4; *count = 4;     break;
      case TYPE_STRING:                 *type = '*'; *size = sizeof(void*); *count = 1; break;
          // case TYPE_RGBA_FP: // identical to type color
      case TYPE_COLOR:                  *type = 'f'; *size = 4; *count = 4;     break;
      case TYPE_RGB:                    *type = 'c'; *size = 1; *count = 3;     break;
      case TYPE_RGBA:                   *type = 'c'; *size = 1; *count = 4;     break;
      case TYPE_RGBE:                   *type = 'c'; *size = 1; *count = 4;     break;
      case TYPE_RGBEA:                  *type = 'c'; *size = 1; *count = 5;     break;
      case TYPE_RGB_16:                 *type = 's'; *size = 2; *count = 3;     break;
      case TYPE_RGBA_16:                *type = 's'; *size = 2; *count = 4;     break;
      case TYPE_RGB_FP:                 *type = 'f'; *size = 4; *count = 3;     break;
      case TYPE_TAG:                    *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_TEXTURE:                *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_TEXTURE1D:              *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_TEXTURE2D:              *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_TEXTURE3D:              *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_TEXTURE_CUBE:           *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_LIGHTPROFILE:           *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_BRDF:                   *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_LIGHT:                  *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_BSDF_MEASUREMENT:       *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_ENUM:                   *type = 'i'; *size = 4; *count = 1;     break;
      case TYPE_VECTOR2I:               *type = 'i'; *size = 4; *count = 2;     break;
      case TYPE_VECTOR3I:               *type = 'i'; *size = 4; *count = 3;     break;
      case TYPE_VECTOR4I:               *type = 'i'; *size = 4; *count = 4;     break;
      case TYPE_VECTOR2B:               *type = 'c'; *size = 1; *count = 2;     break;
      case TYPE_VECTOR3B:               *type = 'c'; *size = 1; *count = 3;     break;
      case TYPE_VECTOR4B:               *type = 'c'; *size = 1; *count = 4;     break;
      case TYPE_MATRIX2X2:              *type = 'f'; *size = 4; *count = 4;     break;
      case TYPE_MATRIX2X3:              *type = 'f'; *size = 4; *count = 6;     break;
      case TYPE_MATRIX3X2:              *type = 'f'; *size = 4; *count = 6;     break;
      case TYPE_MATRIX3X3:              *type = 'f'; *size = 4; *count = 9;     break;
      case TYPE_MATRIX4X3:              *type = 'f'; *size = 4; *count = 12;    break;
      case TYPE_MATRIX3X4:              *type = 'f'; *size = 4; *count = 12;    break;
      case TYPE_MATRIX4X2:              *type = 'f'; *size = 4; *count = 8;     break;
      case TYPE_MATRIX2X4:              *type = 'f'; *size = 4; *count = 8;     break;
      case TYPE_SHADER:                 *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_PARTICLE_MAP:           *type = 't'; *size = 4; *count = 1;     break;
      case TYPE_SPECTRUM:               *type = 'f'; *size = 4; *count = 3;     break;
      case TYPE_ID:                     *type = 'i'; *size = 4; *count = 1;     break;
      case TYPE_PARAMETER:              *type = 'i'; *size = 4; *count = 1;     break;
      case TYPE_TEMPORARY:              *type = 'i'; *size = 4; *count = 1;     break;
      case TYPE_DMATRIX2X2:             *type = 'd'; *size = 8; *count = 4;     break;
      case TYPE_DMATRIX2X3:             *type = 'd'; *size = 8; *count = 6;     break;
      case TYPE_DMATRIX3X2:             *type = 'd'; *size = 8; *count = 6;     break;
      case TYPE_DMATRIX3X3:             *type = 'd'; *size = 8; *count = 9;     break;
      case TYPE_DMATRIX4X3:             *type = 'd'; *size = 8; *count = 12;    break;
      case TYPE_DMATRIX3X4:             *type = 'd'; *size = 8; *count = 12;    break;
      case TYPE_DMATRIX4X2:             *type = 'd'; *size = 8; *count = 8;     break;
      case TYPE_DMATRIX2X4:             *type = 'd'; *size = 8; *count = 8;     break;
      case TYPE_CALL:                   *type = '*'; *size = sizeof(void*); *count = 2; break;

      case TYPE_UNDEF:
      case TYPE_STRUCT:
      case TYPE_ARRAY:
      case TYPE_RLE_UINT_PTR:
      case TYPE_ATTACHABLE:
      case TYPE_NUM:
          ASSERT(M_ATTR, 0);
    }
}


}
}
