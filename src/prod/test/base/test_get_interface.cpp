/******************************************************************************
 * Copyright (c) 2008-2024, NVIDIA CORPORATION. All rights reserved.
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

/**
 ** \file
 **/

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/interface_declare.h>
#include <mi/base/interface_implement.h>

class IMy_interface : public
    mi::base::Interface_declare<0xf3451c54,0x039b,0x4f08,0xb7,0x91,0xac,0x9d,0x04,0xda,0xea,0x42,
                                mi::base::IInterface>
{
};

class IBase : public
    mi::base::Interface_declare<0xe66b0688,0x8dd2,0x4015,0x91,0x7e,0x58,0x1e,0x54,0x29,0x8b,0xc5,
                                IMy_interface>
{
};

class IDerived : public
    mi::base::Interface_declare<0x5b9a53d8,0xd901,0x40d9,0xa8,0xde,0x56,0x0c,0x1a,0xf4,0x36,0x8f,
                                IBase>
{
};

class Base : public mi::base::Interface_implement<IBase>
{
};

class Derived : public mi::base::Interface_implement<IDerived>
{
};

MI_TEST_AUTO_FUNCTION( test_get_interface )
{
    Base base;
    Derived derived;

    auto* ibase = static_cast<IBase*> (&base);
    auto* iderived = static_cast<IDerived*> (&derived);

    const IBase* c_ibase = ibase;
    const IDerived* c_iderived = iderived;

    mi::base::IInterface* i;
    const mi::base::IInterface* ci;

    // Base*

    i = ibase->get_interface( mi::base::IInterface::IID());
    MI_CHECK( i);
    i->release();

    i = ibase->get_interface( IMy_interface::IID());
    MI_CHECK( i);
    i->release();

    i = ibase->get_interface( IBase::IID());
    MI_CHECK( i);
    i->release();

    i = ibase->get_interface( IDerived::IID());
    MI_CHECK( ! i); // Base does not implement IDerived

    // Derived*

    i = iderived->get_interface( mi::base::IInterface::IID());
    MI_CHECK( i);
    i->release();

    i = iderived->get_interface( IMy_interface::IID());
    MI_CHECK( i);
    i->release();

    i = iderived->get_interface( IBase::IID());
    MI_CHECK( i);
    i->release();

    i = iderived->get_interface( IDerived::IID());
    MI_CHECK( i);
    i->release();

    // const Base*

    ci = c_ibase->get_interface( mi::base::IInterface::IID());
    MI_CHECK( ci);
    ci->release();

    ci = c_ibase->get_interface( IMy_interface::IID());
    MI_CHECK( ci);
    ci->release();

    ci = c_ibase->get_interface( IBase::IID());
    MI_CHECK( ci);
    ci->release();

    ci = c_ibase->get_interface( IDerived::IID());
    MI_CHECK( ! ci); // Base does not implement IDerived

    // const Derived*

    ci = c_iderived->get_interface( mi::base::IInterface::IID());
    MI_CHECK( ci);
    ci->release();

    ci = c_iderived->get_interface( IMy_interface::IID());
    MI_CHECK( ci);
    ci->release();

    ci = c_iderived->get_interface( IBase::IID());
    MI_CHECK( ci);
    ci->release();

    ci = c_iderived->get_interface( IDerived::IID());
    MI_CHECK( ci);
    ci->release();
}

MI_TEST_AUTO_FUNCTION( test_get_interface_template )
{
    Base base;
    Derived derived;

    auto* ibase = static_cast<IBase*> (&base);
    auto* iderived = static_cast<IDerived*> (&derived);

    const IBase* c_ibase = ibase;
    const IDerived* c_iderived = iderived;

    mi::base::IInterface* i;
    const mi::base::IInterface* ci;
    IMy_interface* m;
    const IMy_interface* cm;
    IBase* b;
    const IBase* cb;
    IDerived* d;
    const IDerived* cd;

    // Base*

    i = ibase->get_interface<mi::base::IInterface>();
    MI_CHECK( i);
    i->release();

    m = ibase->get_interface<IMy_interface>();
    MI_CHECK( m);
    m->release();

    b = ibase->get_interface<IBase>();
    MI_CHECK( b);
    b->release();

    d = ibase->get_interface<IDerived>();
    MI_CHECK( ! d); // Base does not implement IDerived

    // Derived*

    i = iderived->get_interface<mi::base::IInterface>();
    MI_CHECK( i);
    i->release();

    m = iderived->get_interface<IMy_interface>();
    MI_CHECK( m);
    m->release();

    b = iderived->get_interface<IBase>();
    MI_CHECK( b);
    b->release();

    d = iderived->get_interface<IDerived>();
    MI_CHECK( d);
    d->release();

    // const Base*

    ci = c_ibase->get_interface<mi::base::IInterface>();
    MI_CHECK( ci);
    ci->release();

    cm = c_ibase->get_interface<IMy_interface>();
    MI_CHECK( cm);
    cm->release();

    cb = c_ibase->get_interface<IBase>();
    MI_CHECK( cb);
    cb->release();

    cd = c_ibase->get_interface<IDerived>();
    MI_CHECK( ! cd); // Base does not implement IDerived

    // const Derived*

    ci = c_iderived->get_interface<mi::base::IInterface>();
    MI_CHECK( ci);
    ci->release();

    cm = c_iderived->get_interface<IMy_interface>();
    MI_CHECK( cm);
    cm->release();

    cb = c_iderived->get_interface<IBase>();
    MI_CHECK( cb);
    cb->release();

    cd = c_iderived->get_interface<IDerived>();
    MI_CHECK( cd);
    cd->release();
}

MI_TEST_AUTO_FUNCTION( test_get_iid )
{
    Base base;
    Derived derived;

    auto* ibase = static_cast<IBase*> (&base);
    auto* iderived = static_cast<IDerived*> (&derived);

    mi::base::Uuid u;

    mi::base::IInterface* i;
    IMy_interface* m;
    IBase* b;
    IDerived* d;

    // Base*

    i = ibase->get_interface<mi::base::IInterface>();
    MI_CHECK( i);
    u = i->get_iid();
    MI_CHECK( u == Base::IID());
    i->release();

    m = ibase->get_interface<IMy_interface>();
    MI_CHECK( m);
    u = m->get_iid();
    MI_CHECK( u == Base::IID());
    m->release();

    b = ibase->get_interface<IBase>();
    MI_CHECK( b);
    u = b->get_iid();
    MI_CHECK( u == Base::IID());
    b->release();

    d = ibase->get_interface<IDerived>();
    MI_CHECK( ! d); // Base does not implement IDerived

    // Derived*

    i = iderived->get_interface<mi::base::IInterface>();
    MI_CHECK( i);
    u = i->get_iid();
    MI_CHECK( u == Derived::IID());
    i->release();

    m = iderived->get_interface<IMy_interface>();
    MI_CHECK( m);
    u = m->get_iid();
    MI_CHECK( u == Derived::IID());
    m->release();

    b = iderived->get_interface<IBase>();
    MI_CHECK( b);
    u = b->get_iid();
    MI_CHECK( u == Derived::IID());
    b->release();

    d = iderived->get_interface<IDerived>();
    MI_CHECK( d);
    u = d->get_iid();
    MI_CHECK( u == Derived::IID());
    d->release();
}
