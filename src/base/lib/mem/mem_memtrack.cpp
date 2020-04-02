/***************************************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

/*
Copyright (c) 2002, 2008 Curtis Bartley
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the
distribution.

- Neither the name of Curtis Bartley nor the names of any other
contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* ---------------------------------------- includes */

#include "pch.h"

#ifdef MI_MEM_TRACKER

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include <cxxabi.h>

#include <mi/base/lock.h>

#undef new // IMPORTANT!

namespace MI {

namespace MEM {

/// Demangles a type name.
///
/// \param mangled       The mangled type name.
/// \param initial_len   Hint for the length of the unmangled type name.
/// \return              The demangled type name. The calles has to call free() on the return value.
char* demangle (const char* mangled, size_t initial_len)
{
    size_t len = initial_len;
    char* demangled = static_cast<char *> (malloc (len));
    int status;
    char* ret = abi::__cxa_demangle (mangled, demangled, &len, &status);
    if (ret) {
        demangled = ret;
    } else {
        ::strncpy (demangled, mangled, len);
        ::strncat (demangled, "()", len);
        demangled[len-1] = '\0';
    }
    return demangled;
}

/* ------------------------------------------------------------ */
/* --------------------- class Block_header ------------------- */
/* ------------------------------------------------------------ */

class Block_list;

class Block_header
{
public:
    Block_header(size_t requested_size);
    ~Block_header();

    size_t get_requested_size() const { return m_requested_size; }
    const char* get_filename() const { return m_filename; }
    int get_line_num() const { return m_line_number; }
    const char* get_type_name() const { return m_type_name; }

    void stamp(char const* filename, int line_number, char const* type_name);
    
    friend class Block_list;

private:
    size_t m_requested_size;
    const char* m_filename;
    int m_line_number;
    char* m_type_name;

    Block_header* m_prev_node;
    Block_header* m_next_node;
};

Block_header::Block_header(size_t requested_size)
{
    m_requested_size = requested_size;
    m_filename = "[unknown]";
    m_line_number = 0;
    m_type_name = strdup ("[unknown]");

    m_prev_node = NULL;
    m_next_node = NULL;
}

Block_header::~Block_header()
{
    free (m_type_name);
}
    
void Block_header::stamp(const char* filename, int line_number, const char* type_name)
{
    m_filename = filename;
    m_line_number = line_number;
    free (m_type_name);
    m_type_name = strdup (type_name);
}

/* ------------------------------------------------------------ */
/* --------------------- class Block_list --------------------- */
/* ------------------------------------------------------------ */

class Block_list
{
public:
    /// Adds node \p node to the list.
    static void add_node(Block_header* node);
    /// Removes node \p note from the list.
    static void remove_node(Block_header* node);
    /// Returns the size of the list.
    static size_t count_blocks();
    /// Copies the pointer to all list blocks to \p pointers.
    static void get_blocks(Block_header** pointers);
    /// Comparison operator.
    static bool type_greater_than(Block_header* header1, Block_header* header2);

private:
    static Block_header* m_first_node;

};

Block_header* Block_list::m_first_node = NULL;

void Block_list::add_node(Block_header* node)
{
    assert(node);
    assert(!node->m_prev_node);
    assert(!node->m_next_node);
    assert(!m_first_node || !m_first_node->m_prev_node);

    if (m_first_node)
        m_first_node->m_prev_node = node;

    node->m_next_node = m_first_node;
    m_first_node = node;
}

void Block_list::remove_node(Block_header* node)
{
    assert(node);
    assert(m_first_node);
    assert(!m_first_node->m_prev_node);

    if (m_first_node == node)
        m_first_node = node->m_next_node;
    
    if (node->m_prev_node)
        node->m_prev_node->m_next_node = node->m_next_node;
    
    if (node->m_next_node)
        node->m_next_node->m_prev_node = node->m_prev_node;

    node->m_prev_node = NULL;
    node->m_next_node = NULL;
}

size_t Block_list::count_blocks()
{
    size_t count = 0;
    Block_header* current_node = m_first_node;
    while (current_node) {
        count++;
        current_node = current_node->m_next_node;
    }
    return count;
}

void Block_list::get_blocks(Block_header** pointers)
{
    Block_header* current_node = m_first_node;
    while (current_node) {
        *pointers = current_node;
        pointers++;
        current_node = current_node->m_next_node;
    }
}

bool Block_list::type_greater_than(Block_header* header1, Block_header* header2)
{
    return strcmp(header1->m_type_name, header2->m_type_name) > 0;
}

/* ------------------------------------------------------------ */
/* ---------------------- class Signature --------------------- */
/* ------------------------------------------------------------ */

class Signature
{
public:
    Signature() : m_signature1(s_SIGNATURE1), m_signature2(s_SIGNATURE2) {};
    ~Signature() { m_signature1 = 0; m_signature2 = 0; }
    
    static bool is_valid(const Signature* prospective_signature)
    {
        try {
            if (prospective_signature->m_signature1 != s_SIGNATURE1) return false;
            if (prospective_signature->m_signature2 != s_SIGNATURE2) return false;
            return true;
        }
        catch (...) {
            return false;
        }
    }

private:
    static const unsigned int s_SIGNATURE1 = 0xCAFEBABE;
    static const unsigned int s_SIGNATURE2 = 0xFACEFACE;

    unsigned int m_signature1;
    unsigned int m_signature2;
};

/* ------------------------------------------------------------ */
/* -------------------- address conversion -------------------- */
/* ------------------------------------------------------------ */

/* We divide the memory blocks we allocate into two "chunks", the
 * "prolog chunk" where we store information about the allocation,
 * and the "user chunk" which we return to the caller to use.
 */

/* ---------------------------------------- alignment */

const size_t ALIGNMENT = 4;

/* If "value" (a memory size or offset) falls on an alignment boundary,
 * then just return it.  Otherwise return the smallest number larger
 * than "value" that falls on an alignment boundary.
 */    

#define PAD_TO_ALIGNMENT_BOUNDARY(value) \
    ((value) + ((ALIGNMENT - ((value) % ALIGNMENT)) % ALIGNMENT))

/* ---------------------------------------- chunk structs */

/* We declare incomplete structures for each chunk, just to 
 * provide type safety.
 */

struct Prolog_chunk;
struct User_chunk;

/* ---------------------------------------- chunk sizes and offsets */

const size_t SIZE_Block_header = PAD_TO_ALIGNMENT_BOUNDARY(sizeof(Block_header));
const size_t SIZE_Signature = PAD_TO_ALIGNMENT_BOUNDARY(sizeof(Signature));

const size_t OFFSET_Block_header = 0;
const size_t OFFSET_Signature = OFFSET_Block_header + SIZE_Block_header;
const size_t OFFSET_User_chunk = OFFSET_Signature + SIZE_Signature;

const size_t SIZE_Prolog_chunk = OFFSET_User_chunk;

/* ---------------------------------------- get_user_address */

static User_chunk* get_user_address(Prolog_chunk* prolog)
{
    char* pch_prolog = reinterpret_cast<char *>(prolog);
    char* pch_user = pch_prolog + OFFSET_User_chunk;
    User_chunk* user = reinterpret_cast<User_chunk *>(pch_user);
    return user;
}

/* ---------------------------------------- get_prolog_address */

static Prolog_chunk* get_prolog_address(User_chunk* user)
{
    char* pch_user = reinterpret_cast<char *>(user);
    char* pch_prolog = pch_user - OFFSET_User_chunk;
    Prolog_chunk* prolog = reinterpret_cast<Prolog_chunk *>(pch_prolog);
    return prolog;
}

/* ---------------------------------------- get_header_address */

static Block_header* get_header_address(Prolog_chunk* prolog)
{
    char* pch_prolog = reinterpret_cast<char *>(prolog);
    char* pch_header = pch_prolog + OFFSET_Block_header;
    Block_header* pHeader = reinterpret_cast<Block_header *>(pch_header);
    return pHeader;
}

/* ---------------------------------------- get_signature_address */

static Signature* get_signature_address(Prolog_chunk* prolog)
{
    char* pch_prolog = reinterpret_cast<char *>(prolog);
    char* pch_signature = pch_prolog + OFFSET_Signature;
    Signature* pSignature = reinterpret_cast<Signature *>(pch_signature);
    return pSignature;
}

/* ------------------------------------------------------------ */
/* -------------- memory allocation and stamping -------------- */
/* ------------------------------------------------------------ */

// Lock to make public functions thread-safe
mi::base::Lock s_lock;

void* track_allocation(size_t size)
{
    mi::base::Lock::Block block (&s_lock);

    // Allocate the memory, including space for the prolog.
    Prolog_chunk* prolog = (Prolog_chunk *)malloc(SIZE_Prolog_chunk + size);
    
    // If the allocation failed, then return NULL.
    if (!prolog) return NULL;
    
    // Use placement new to construct the block header in place.
    Block_header* block_header = new (prolog) Block_header(size);
    
    // Link the block header into the list of existant block headers.
    Block_list::add_node(block_header);
    
    // Use placement new to construct the signature in place.
    Signature* pSignature = new (get_signature_address(prolog)) Signature;
    (void) pSignature;
    
    // Get the offset to the user chunk and return it.
    User_chunk* user = get_user_address(prolog);
    
    return user;
}

void track_deallocation(void* p)
{
    mi::base::Lock::Block block (&s_lock);

    // It's perfectly valid for "p" to be null; return if it is.
    if (!p) return;

    // Get the prolog address for this memory block.
    User_chunk* user = reinterpret_cast<User_chunk *>(p);    
    Prolog_chunk* prolog = get_prolog_address(user);
   
    // Check the signature, and if it's invalid, return immediately.
    Signature* signature = get_signature_address(prolog);
    if (!Signature::is_valid(signature)) return;
    
    // Destroy the signature.
    signature->~Signature();
    signature = NULL;

    // Unlink the block header from the list and destroy it.
    Block_header* block_header = get_header_address(prolog);
    Block_list::remove_node(block_header);
    block_header->~Block_header();
    block_header = NULL;

    // Free the memory block.    
    free(prolog);
}

void stamp(void* p, const Context& context, const char* type_name)
{
    mi::base::Lock::Block block (&s_lock);

    // Get the header and signature address for this pointer.
    User_chunk* user = reinterpret_cast<User_chunk *>(p);
    Prolog_chunk* prolog = get_prolog_address(user);
    Block_header* header = get_header_address(prolog);
    Signature* signature = get_signature_address(prolog);

    // If the signature is not valid, then return immediately.
    if (!Signature::is_valid(signature)) return;

    // Demangle the type name.
    char* demangled_type_name = demangle(type_name, 128);
    
    // Modify the type name according some fixed rules.
    char* recorded_type_name = demangled_type_name;
    
    // 1. Strip off Boost shared pointers.
    const char* s1 = "boost::detail::sp_counted_impl_p<";
    if (strncmp(recorded_type_name, s1, strlen(s1)) == 0)
        recorded_type_name += strlen (s1);

    // 2. Strip off MI::PAGER containers.
    const char* s2 = "MI::PAGER::Vector<";
    if (strncmp(recorded_type_name, s2, strlen(s2)) == 0)
        recorded_type_name += strlen (s2);
    
    // 3. Aggregate top-level classes in std::priv.
    const char* s3 = "std::priv::";
    if (strncmp(recorded_type_name, s3, strlen(s3)) == 0) {
        char* pos = index(recorded_type_name + strlen(s3), '<');
        *pos = '\0';
    }
    
    // 4. Aggregate per namespace in MI::.
    const char* s4 = "MI::";
    if (strncmp(recorded_type_name, s4, strlen(s4)) == 0) {
        char* pos = index(recorded_type_name + strlen(s4), ':');
        *pos = '\0';
    }

    // "stamp" the information onto the header.
    header->stamp(context.m_file_name, context.m_line_number, recorded_type_name);

    // Clean up.
    free(demangled_type_name);
}

/* ------------------------------------------------------------ */
/* -------------- memory usage -------------------------------- */
/* ------------------------------------------------------------ */

void dump_blocks()
{
    mi::base::Lock::Block block (&s_lock);

    // Get an array of pointers to all existant blocks.
    size_t num_blocks = Block_list::count_blocks();
    Block_header** pointers = (Block_header**) calloc(num_blocks, sizeof(Block_header*));
    Block_list::get_blocks(pointers);

    // Dump information about the memory blocks.
    printf("\n");
    printf("=====================\n");
    printf("Current Memory Blocks\n");
    printf("=====================\n");
    printf("\n");

    for (size_t i = 0; i < num_blocks; i++) {
        Block_header* block_header = pointers[i];
        const char* type_name = block_header->get_type_name();
        size_t size = block_header->get_requested_size();
        const char* file_name = block_header->get_filename();
        int line_number = block_header->get_line_num();
        printf("#%-6zd %5zd bytes %-50s %s:%d\n", i, size, type_name, file_name, line_number);
    }

    // Clean up.
    free(pointers);
}

struct Mem_digest
{
    const char* type_name;
    int m_block_count;
    size_t m_total_size;

    static bool total_size_greater_than(const Mem_digest &md1, const Mem_digest &md2)
    { return md1.m_total_size > md2.m_total_size; }
};


static void summarize_memory_usage_for_type(
    Mem_digest* digest,
    Block_header** block_headers,
    size_t start_index,
    size_t end_index)
{
    digest->type_name = block_headers[start_index]->get_type_name();
    digest->m_block_count = 0;
    digest->m_total_size = 0;

    for (size_t i = start_index; i < end_index; i++) {
        digest->m_block_count++;
        digest->m_total_size += block_headers[i]->get_requested_size();
        assert(strcmp(block_headers[i]->get_type_name(), digest->type_name) == 0);
    }
}

void list_memory_usage()
{
    mi::base::Lock::Block block (&s_lock);

    // If there are no allocated blocks, then return now.
    size_t num_blocks = Block_list::count_blocks();
    if (num_blocks == 0) return;

    // Get an array of pointers to all existant blocks.
    Block_header** pointers = (Block_header**) calloc(num_blocks, sizeof(Block_header*));
    Block_list::get_blocks(pointers);

    // Sort the blocks by type name.
    std::sort(pointers, pointers + num_blocks, Block_list::type_greater_than);

    // Find out how many unique types we have.
    size_t num_unique_types = 1;
    for (size_t i = 1; i < num_blocks; i++) {
        const char* prev_type_name = pointers[i-1]->get_type_name();
        const char* curr_type_name = pointers[i  ]->get_type_name();
        if (strcmp(prev_type_name, curr_type_name) != 0)
            num_unique_types++;
    }

    // Create an array of "digests" summarizing memory usage by type.
    size_t start_index = 0;
    size_t current_digest = 0;
    Mem_digest* digests = (Mem_digest *)calloc(num_unique_types, sizeof(Mem_digest));
    for (size_t i = 1; i <= num_blocks; i++) {   // yes, less than or *equal* to
        const char* prevTypeName = pointers[i-1]->get_type_name();
        const char* currTypeName = (i < num_blocks) ? pointers[i]->get_type_name() : "";
        if (strcmp(prevTypeName, currTypeName) != 0) {
            size_t end_index = i;
            summarize_memory_usage_for_type(
                digests + current_digest, pointers, start_index, end_index);
            start_index = end_index;
            current_digest++;
        }
    }
    assert(current_digest == num_unique_types);

    // Sort the digests by total memory usage.
    std::sort(digests, digests + num_unique_types, Mem_digest::total_size_greater_than);

    // Compute the grand total memory usage.
    size_t grand_total_num_blocks = 0;
    size_t grand_total_size = 0;
    for (size_t i = 0; i < num_unique_types; i++) {
        grand_total_num_blocks += digests[i].m_block_count;
        grand_total_size += digests[i].m_total_size;
    }

    // Dump the memory usage statistics.
    printf("\n");
    printf("-----------------------\n");
    printf("Memory Usage Statistics\n");
    printf("-----------------------\n");
    printf("\n");
    printf("%-50s%6s  %5s %7s %s \n", "allocated type", " blocks", "", "kbytes", "");
    printf("%-50s%6s  %5s %7s %s \n", "--------------", "-------", "", "------", "");

    for (size_t i = 0; i < num_unique_types; i++) {
        const Mem_digest& digest = digests[i];
        size_t block_count = digest.m_block_count;
        double block_count_pct = 100.0 * block_count / grand_total_num_blocks;
        size_t total_size = digest.m_total_size;
        double total_size_pct = 100.0 * total_size / grand_total_size;

        printf( "%-50s %6zd %5.1f%% %7zd %5.1f%%\n",
            digest.type_name, block_count, block_count_pct, total_size/1024, total_size_pct
        );
    }

    printf("%-50s %6s %5s  %7s %s \n", "--------", "------", "", "-------", "");
    printf("%-50s %6zd %5s  %7zd %s \n", "[totals]",
        grand_total_num_blocks, "", grand_total_size/1024, "");

    // Clean up.
    free(pointers);
    free(digests);
}

} // namespace MEM

} // namespace MI

/* ------------------------------------------------------------ */
/* -------------- global operators----------------------------- */
/* ------------------------------------------------------------ */

/// redefinition of global operator new
///
/// Note: we really need std::bad_alloc here, not std::bad_alloc.
void* operator new(size_t size) throw (st""d::bad_alloc)
{
    void* p = MI::MEM::track_allocation(size);
    // simulate the (wrong) behaviour of the MEM module by *not* throwing if p == NULL
    return p;
}

/// redefinition of global operator delete
void operator delete (void* p) throw()
{
    MI::MEM::track_deallocation(p);
}

/// redefinition of global operator new[]
///
/// Note: we really need std::bad_alloc here, not std::bad_alloc.
void* operator new[](size_t size) throw (st""d::bad_alloc)
{
    void* p = MI::MEM::track_allocation(size);
    // simulate the (wrong) behaviour of the MEM module by *not* throwing if p == NULL
    return p;
}

/// redefinition of global operator delete[]
void operator delete[](void* p) throw()
{
    MI::MEM::track_deallocation(p);
}

#endif // MI_MEM_TRACKER
