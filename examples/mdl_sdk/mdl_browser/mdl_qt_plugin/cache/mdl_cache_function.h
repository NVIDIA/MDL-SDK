/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Implementation of the IMdl_cache_function interface.


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_CACHE_ITEM_FUNCTION
#define MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_CACHE_ITEM_FUNCTION

#include "imdl_cache.h"
#include "mdl_cache_impl.h"

namespace mi
{
    namespace base
    {
        class IInterface;
    }

    namespace neuraylib
    {
        class INeuray;
        class ITransaction;
    }
}

class Mdl_cache_function : public Mdl_cache_element<IMdl_cache_function>
{
public:
    explicit Mdl_cache_function() = default;
    virtual ~Mdl_cache_function() = default;

    IMdl_cache_item::Kind get_kind() const override { return CK_FUNCTION; }

    bool update(mi::neuraylib::INeuray* neuray, 
                mi::neuraylib::ITransaction* transaction, 
                const mi::base::IInterface* module) override;

protected:
    typedef Mdl_cache_element<IMdl_cache_function> Base;
};

#endif