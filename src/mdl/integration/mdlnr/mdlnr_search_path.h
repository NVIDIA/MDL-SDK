/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_INTERGRATION_MDLNR_SEARCH_PATH_H
#define MDL_INTERGRATION_MDLNR_SEARCH_PATH_H 1

#include <string>

#include <mi/base/interface_implement.h>
#include <mi/mdl/mdl_mdl.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>

namespace mi { namespace mdl { class IMDL; } }

namespace MI {
namespace MDLC {

/// A search path helper for neuray.
class MDL_search_path : public mi::mdl::Allocator_interface_implement<mi::mdl::IMDL_search_path>
{
    typedef mi::mdl::Allocator_interface_implement<mi::mdl::IMDL_search_path> Base;
    friend class mi::mdl::Allocator_builder;
public:
    /// Get the number of search paths.
    ///
    /// \param set  the path set
    virtual size_t get_search_path_count(Path_set set) const;

    /// Get the i'th search path.
    ///
    /// \param set  the path set
    /// \param i    index of the path
    virtual char const *get_search_path(Path_set set, size_t i) const;

public:
    /// Factory.
    ///
    /// \param alloc  the allocator.
    static MDL_search_path *create(mi::mdl::IAllocator *alloc);

private:
    /// Constructor.
    ///
    /// \param alloc  the allocator.
    explicit MDL_search_path(mi::mdl::IAllocator *alloc);

private:
    mutable std::string tmp;
};

}  // MDLC
}  // MI

#endif // MDL_INTERGRATION_MDLNR_SEARCH_PATH_H
