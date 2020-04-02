/******************************************************************************
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
 *****************************************************************************/

 /// \file
 /// \brief Implement tree links.
 ///
 ///        This class implements the shared aspects of tree-shaped
 ///        containers (Set_intr and Pqueue_intr).

#ifndef BASE_LIB_CONT_TREE_H
#define BASE_LIB_CONT_TREE_H

#include <base/lib/log/i_log_assert.h>
#include <base/lib/mem/i_mem_allocatable.h>

namespace MI {

namespace CONT {

template<
    class T>        // Type of tree elements.
class Tree_link : public MEM::Allocatable {

public:

    // Link pointers should be private, and the classes using it should be
    // friends. Unfortunately, this is not possible because these classes
    // are templates that requires a (subclass of) Tree_link as a parameter,
    // and therefore can not be declared as friends before the definition
    // of Tree_link is complete.

    // Pointer to parent.
    T *m_parent;

    // Pointer to left child.
    T *m_left;

    // Pointer to right child.
    T *m_right;

#ifdef ENABLE_ASSERT
    // Pointer to container.
    void *m_cont;
#endif

    // Constructor.
    Tree_link()
    {
        m_parent = 0;
        m_left = 0;
        m_right = 0;
#ifdef ENABLE_ASSERT
        m_cont = 0;
#endif
    }

    // Destructor.
    ~Tree_link()
    {
#if 0	// Re-enable when the DEBUG version of Set_intr::clear is implemented
	ASSERT(M_CONT,m_cont == 0);
#endif
    }

private:

    // Copy constructor.
    Tree_link(
                const Tree_link&);	// The tree link to copy.
public:

    // Copy assignment.
    //
    // Dummy implementation to satisfy requirements of std::vector. It was not needed with
    // STLport, let's hope it is only required to exist, but not used with the native STL.
    Tree_link &operator=(
                const Tree_link&)	// The tree link to assign.
    { ASSERT(M_CONT, false); return *this; }

};

}

}

#endif
