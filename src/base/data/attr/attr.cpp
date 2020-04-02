/***************************************************************************************************
 * Copyright (c) 2004-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief non-inlined functions
/// 
/// An attribute is an extra piece of data that can be attached to a database element. It is named
/// (although names are converted to perfect hashes early on, for faster lookup). All inheritance is
/// based on attributes. Attributes are collected in attribute sets, one of which is in every
/// Element. Sets are also the traversal state during scene preprocessing and inheritance.

#include "pch.h"

#include "attr.h"
#include "i_attr_utilities.h"

#include <base/data/serial/i_serializer.h>
#include <base/lib/cont/i_cont_rle_array.h>
#include <base/lib/log/i_log_logger.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace MI {
namespace ATTR {

using namespace LOG;
using namespace CONT;
using namespace std;

// iterate over a type using the given type iterator and serialize all data
// to the given serializer. This is able to handle structs as well as static
// and dynamic arrays
static void do_serialize_data(
    SERIAL::Serializer	*serializer,	// useful functions for byte streams
    Type_iterator	*it)		// parameter iterator
{
    // iterate over all elements of the type
    for (; !it->at_end(); it->to_next()) {
        char *value = it->get_value();
        int arraysize = it->get_arraysize();
        if (!arraysize && it->get_typecode() != TYPE_ARRAY) {
            // this is a dynamic array. We need to transmit the size of the
            // array. In a static array this is part of the type and will not
            // be transmitted with the data.
            const Dynamic_array *array = (const Dynamic_array*)value;

            // adapt arraysize so that we process the correct number of members
            arraysize = array->m_count;
            serializer->write(arraysize);
            // let the value point to the memory dynamically allocated for the
            // array contents
            value = array->m_value;
        }

        if (it->get_typecode() == TYPE_RLE_UINT_PTR) {
            // runlength encoded arrays of unsigned integers are handled
            // specially
            Rle_array<Uint> *array = *(Rle_array<Uint>**)value;
            Uint size = (Uint)array->get_index_size();
            serializer->write(size);
            Rle_chunk_iterator<Uint> iterator = array->begin_chunk();
            for (Uint i = 0; i < size; i++) {
                serializer->write((Uint)iterator.count());
                serializer->write(iterator.data());
                iterator++;
            }
        } else if (it->get_typecode() == TYPE_STRUCT || it->get_typecode() == TYPE_ATTACHABLE) {
            size_t size = it->sizeof_elem();
            for (int a=0; a < arraysize; a++) {
                Type_iterator sub(it, value);
                do_serialize_data(serializer, &sub);
                value += size;
            }
        } else if (it->get_typecode() == TYPE_CALL) {
            size_t size = it->sizeof_elem() - 2*Type::sizeof_one(TYPE_STRING);
            for (int a=0; a < arraysize; a++) {
                // first serialize two const char* - actually one DB::Tag and one const char*
                serializer->write(*(DB::Tag*)value);
                value += Type::sizeof_one(TYPE_STRING);
                serializer->write(*(char**)value);
                value += Type::sizeof_one(TYPE_STRING);
                // now dive into "structure"
                Type_iterator sub(it, value);
                do_serialize_data(serializer, &sub);
                value += size;
            }
        } else if (it->get_typecode() != TYPE_ARRAY) {
            // this is just an array of primitive types
            int type, count, size;
            eval_typecode(it->get_typecode(), &type, &count, &size);

            for (int a=0; a < arraysize; a++) {
                for (int i=0; i < count; i++) {
                    switch(type) {
                      case '*': serializer->write(*(char   **)value); break;
                      case 'c': serializer->write(*(Sint8   *)value); break;
                      case 's': serializer->write(*(Sint16  *)value); break;
                      case 'i': serializer->write(*(Sint32  *)value); break;
                      case 'q': serializer->write(*(Sint64  *)value); break;
                      case 'f': serializer->write(*(Scalar  *)value); break;
                      case 'd': serializer->write(*(Dscalar *)value); break;
                      case 't': serializer->write(*(DB::Tag *)value); break;
                      default:  ASSERT(M_ATTR, 0);
                    }
                    value += size;
                }
            }
        }
        else {
            ASSERT(M_ATTR, it->get_typecode() == TYPE_ARRAY);
            // since with the old attribute system even the array element types hold the
            // arraysize those array element types will be serialized arraysize times. Hence
            // the current array type should simply be skipped.
            Type_iterator sub(it, value);
            size_t offset = sub.sizeof_elem() * sub.get_arraysize();
            do_serialize_data(serializer, &sub);
            // increase offset by the array element type's size since an array has no size itself
            value += offset;
        }
    }
}


// serialize the given data assuming that the data is described by this
// type.
void Type::serialize_data(
    SERIAL::Serializer	*serializer,	// useful functions for byte streams
    char		*values) const	// the actual data to be serialized
{
    Type_iterator it(this, values);
    do_serialize_data(serializer, &it);
}


// iterate over a type using the given type iterator and deserialize all data
// to the given deserializer. This is able to handle structs as well as static
// and dynamic arrays
static void do_deserialize_data(
    SERIAL::Deserializer *deser,	// useful functions for byte streams
    Type_iterator	 *it)		// parameter iterator
{
    // iterate over all elements of the type
    for (; !it->at_end(); it->to_next()) {
        char *value = it->get_value();
        int arraysize = it->get_arraysize();
        if (!arraysize && it->get_typecode() != TYPE_ARRAY) {
            // this is a dynamic array. We will receive the number of elements
            // in the serialized data. In a static array this is part of the
            // type.
            Dynamic_array *array = (Dynamic_array*)value;
            deser->read(&arraysize);
            array->m_count = arraysize;
            array->m_value = arraysize > 0 ? new char[it->sizeof_elem() * arraysize] : 0;
            value = array->m_value;
        }

        if (it->get_typecode() == TYPE_RLE_UINT_PTR) {
            Rle_array<Uint> *array = new Rle_array<Uint>;
            *(Rle_array<Uint>**)value = array;
            Uint size;
            deser->read(&size);
            for (Uint i = 0; i < size; i++) {
                Uint count, data;
                deser->read(&count);
                deser->read(&data);
                array->push_back(data, count);
            }
        }
        else if (it->get_typecode() == TYPE_STRUCT ||it->get_typecode() == TYPE_ATTACHABLE) {
            size_t size = it->sizeof_elem();
            for (int a=0; a < arraysize; a++) {
                Type_iterator sub(it, value);
                do_deserialize_data(deser, &sub);
                value += size;
            }
        }
        else if (it->get_typecode() == TYPE_CALL) {
            size_t size = it->sizeof_elem() - 2*Type::sizeof_one(TYPE_STRING);
            for (int a=0; a < arraysize; a++) {
                // first serialize two const char* - actually one DB::Tag and one const char*
                deser->read((DB::Tag*)value);
                value += Type::sizeof_one(TYPE_STRING);
                deser->read((char   **)value);
                value += Type::sizeof_one(TYPE_STRING);
                // now dive into "structure"
                Type_iterator sub(it, value);
                do_deserialize_data(deser, &sub);
                value += size;
            }
        }
        else if (it->get_typecode() != TYPE_ARRAY) {
            int type, count, size;
            eval_typecode(it->get_typecode(), &type, &count, &size);

            for (int a=0; a < arraysize; a++) {
                for (int i=0; i < count; i++) {
                    switch(type) {
                      case '*': deser->read((char   **)value); break;
                      case 'c': deser->read((Sint8   *)value); break;
                      case 's': deser->read((Sint16  *)value); break;
                      case 'i': deser->read((Sint32  *)value); break;
                      case 'q': deser->read((Sint64  *)value); break;
                      case 'f': deser->read((Scalar  *)value); break;
                      case 'd': deser->read((Dscalar *)value); break;
                      case 't': deser->read((DB::Tag *)value); break;
                      default:  ASSERT(M_ATTR, 0);
                    }
                    value += size;
                }
            }
        }
        else {
            ASSERT(M_ATTR, it->get_typecode() == TYPE_ARRAY);
            // please refer to the corresponding documentation in do_serialize_data()
            Type_iterator sub(it, value);
            size_t offset = sub.sizeof_elem() * sub.get_arraysize();
            do_deserialize_data(deser, &sub);
            // increase offset by the array element type's size since an array has no size itself
            value += offset;
        }
    }
}

// serialize the given data assuming that the data is described by this
// type.
void Type::deserialize_data(
    SERIAL::Deserializer *deser,	// useful functions for byte streams
    char		 *values)		// deserialize to here
{
    Type_iterator it(this, values);
    do_deserialize_data(deser, &it);
}


//--------------------------------------------------------------------------------------------------

#ifdef MI_PLATFORM_MACOSX
// This constructor should not be used - MacOS compiler somehow insists to do so nevertheless.
// Hence we feed him the implementation of the Type_iterator(const Type*, char*) constructor.
// Maybe a newer compiler on MacOS might solve it?
Type_iterator::Type_iterator(const Type_iterator& iter)
{
    set(iter.operator->(), iter.get_value());
}
#endif

}
}
