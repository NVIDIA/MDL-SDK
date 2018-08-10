/***************************************************************************************************
 * Copyright (c) 2012-2018, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Type that holds a reference to a database element or a value.

#ifndef MI_NEURAYLIB_IATTACHABLE_H
#define MI_NEURAYLIB_IATTACHABLE_H

#include <mi/neuraylib/idata.h>

namespace mi {

class IAttachable :
    public base::Interface_declare<0x5e7a28b7,0x38aa,0x45ac,0x98,0x65,0x97,0xac,0xcd,0x81,0xc4,0x0e,
                                   IData_simple>
{
public:
    virtual Sint32 set_reference( const base::IInterface* db_element) = 0;

    virtual Sint32 set_reference( const char* name) = 0;

    virtual const base::IInterface* get_reference() const = 0;

    template <class T>
    const T* get_reference() const
    {
        const base::IInterface* ptr_iinterface = get_reference();
        if ( !ptr_iinterface)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    virtual base::IInterface* get_reference() = 0;

    template <class T>
    T* get_reference()
    {
        base::IInterface* ptr_iinterface = get_reference();
        if ( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    virtual const char* get_reference_name() const = 0;

    virtual const IData* get_value() const = 0;

    template<class T>
    const T* get_value() const
    {
        const IData* ptr_idata = get_value();
        if ( !ptr_idata)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_idata->get_interface( typename T::IID()));
        ptr_idata->release();
        return ptr_T;
    }

    virtual IData* get_value() = 0;

    template<class T>
    T* get_value()
    {
        IData* ptr_idata = get_value();
        if ( !ptr_idata)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_idata->get_interface( typename T::IID()));
        ptr_idata->release();
        return ptr_T;
    }
};

/*@}*/ // end group mi_neuray_simple_types

} // namespace mi

#endif // MI_NEURAYLIB_IATTACHABLE_H
