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
/// \brief The Scene_element inlined implementation.

namespace MI
{
namespace SCENE
{
//
// construct an instance of this class
//

template <class T, SERIAL::Class_id C, class P>
SERIAL::Serializable *Scene_element<T, C, P>::factory()
{
    return new T;
}


//
// default constructor needed because of the copy constructor
//

template <class T, SERIAL::Class_id C, class P>
inline Scene_element<T, C, P>::Scene_element()
{
}


//
// copy constructor which will call the copy constructor of the base class
//

template <class T, SERIAL::Class_id C, class P>
inline Scene_element<T, C, P>::Scene_element(
    const Scene_element	&source)		// the source to copy from
    : P(source)
{
}


//
// get the class id of an instance of our class
//

template <class T, SERIAL::Class_id C, class P>
inline SERIAL::Class_id Scene_element<T, C, P>::get_class_id() const
{
    return id;
}


//
// make a copy of this instance
//

template <class T, SERIAL::Class_id C, class P>
inline DB::Element_base *Scene_element<T, C, P>::copy() const
{
    T *element = new T(*(T*)this);
    return element;
}


//
// Return the approximate size in bytes of the element including all its
// substructures. This is used to make decisions about garbage collection.
//

template <class T, SERIAL::Class_id C, class P>
inline size_t Scene_element<T, C, P>::get_size() const
{
    return sizeof(*this)
        + P::get_size() - sizeof(P);
}


// Check, if this object is of the given type. This is true, if either the
// class id of this object equals the given class id, or the class is
// derived from another class which has the given class id.
template <class T, SERIAL::Class_id C, class P>
inline bool Scene_element<T, C, P>::is_type_of(
    SERIAL::Class_id id) const		// the class id to check
{
    return id == C ? true : P::is_type_of(id);
}

}
}
