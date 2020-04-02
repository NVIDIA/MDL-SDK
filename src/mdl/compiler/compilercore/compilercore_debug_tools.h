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

#ifndef MDL_COMPILERCORE_DEBUG_TOOLS_H
#define MDL_COMPILERCORE_DEBUG_TOOLS_H 1

#include <mi/base/lock.h>
#include <mi/base/interface_implement.h>
#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"

namespace mi {
namespace mdl {
namespace dbg {

///
/// A simple memory allocator using malloc/free that supports tracking.
///
/// Can be controlled by the environment variable NV_MDL_DEBUG=cmds
///
/// cmds : cmd { ';' cmd }.
/// cmd  : 'capture' (NUMBER | 'all' | 'ref')
///      | 'depth' NUMBER
///      | 'skip' NUMBER
///      | 'bpa' NUMBER
///      | 'bpf' NUMBER
///      | 'bpi' NUMBER
///      | 'bpd' NUMBER
///      | 'abort'
///      | 'noabort'
///      | 'expensive'
///      | 'noexpensive'
///      | 'size'
///      | 'nosize'
///      | 'help'.
///
/// Commands:
///   capture id:  Capture stack frame for all all operations
///                (allocation, reference count +/-) on Block id.
///   capture all: Capture stack frame for all blocks.
///   capture ref: Capture stack frame for reference counted objects.
///   depth num:   Sets the depth of the captured frame, default 5.
///   skip num:    Skip the first num frames when capturing, default 3.
///   bpa num:     Break into debugger if block num is allocated.
///   bpf num:     Break into debugger if block num is freed.
///   bpi num:     Break into debugger if object id's reference count is increased.
///   bpd num:     Break into debugger if object id's reference count is decreased.
///   abort:       Abort after memory errors were reported (default in DEBUG mode).
///   noabort:     Do not abort after memory errors were reported (default in RELEASE mode).
///   expensive:   Enable expensive checks
///   noexpensive: Disable expensive checks (default).
///   size:        Regularly dump allocated size.
///   nosize:      Do not dump allocated size.
///   
///  
/// At program end reports all leaked reference counted objects and memory blocks, and
/// wrong free calls.
/// If capture was enabled for a block, prints the captured stack frames for all
/// operation locations locations.
/// For wrong free calls, capture is always enabled.
/// If memory errors were found, aborts if abort-on-error is enabled.
///
class DebugMallocAllocator : public mi::base::Interface_implement<IDebugAllocator>
{
    /// The reason why a frame was captured.
    enum Capture_reason {
        CR_ALLOC,          ///< Allocation frame.
        CR_FREE,           ///< Destroy frame.
        CR_REF_COUNT_INC,  ///< Reference count increased.
        CR_REF_COUNT_DEC   ///< Reference count decreased.
    };

    typedef void *Address;

    /// A captured Frame.
    struct Captured_frame {
        Captured_frame *next;     ///< Points to the next captured frame.
        Capture_reason reason;    ///< Capture frame reason.
        unsigned       length;    ///< Number of captures frames.
        Address        frames[1]; ///< The frames.
    };

    /// A free error.
    struct Free_error {
        Free_error     *next;     ///< Points to the next free error entry.
        Address        adr;       ///< The wrong address.
        Captured_frame *frame;    ///< The captures frame of this error.
    };

    /// The block header.
    ///
    /// \note The size of this header must be a multiple of 16 for x86.
    struct Header {
        unsigned       magic1;    ///< First magic.
        unsigned       magic2;    ///< Second magic.
        size_t         size;      ///< The size of this block's payload.
        Header         *next;     ///< Points to the next used block.
        Header         *prev;     ///< Points to the previous used block.
        size_t         num;       ///< The unique number (ID) of this block.
        size_t         ref_count; ///< If this block contains a reference counted object, its count.
        size_t         flags;     ///< Flags.
        Captured_frame *frames;   ///< Captures Frames.
        char const     *cls_name; ///< Class name of the allocated object if any.
        size_t         pad;       ///< Padding to 16 byte

        enum Magics {
            HEADER_MAGIC = 0x55AA55AA,
        };
    };

    enum Flags {
        BH_REF_COUNTED = 1 << 0 ///< This block contains a reference counted object.
    };

    struct Breakpoint {
        unsigned      token;    ///< The breakpoint type token.
        size_t        id;       ///< The breakpoint block id.
    };

public:
    void *malloc(mi::Size size) MDL_FINAL;

    void free(void *memory) MDL_FINAL;

    /// Allocates a memory block for a class instance.
    void *objalloc(char const *cls_name, Size size) MDL_FINAL;

    /// Marks the given object as reference counted and set the initial count.
    void mark_ref_counted(void const *obj, Uint32 initial) MDL_FINAL;

    /// Increments the reference count of an reference counted object.
    void inc_ref_count(void const *obj) MDL_FINAL;

    /// Decrements the reference count of an reference counted object.
    void dec_ref_count(void const *obj) MDL_FINAL;

    /// This object is not reference counted.
    Uint32 retain() const MDL_FINAL;

    /// This object is not reference counted.
    Uint32 release() const MDL_FINAL;

public:
    /// Constructor.
    ///
    /// \param alloc  if non-NULL, use this allocator to retrieve raw memory blocks
    DebugMallocAllocator(mi::base::IAllocator *alloc = NULL);

    /// Destructor, dump memory leaks.
    ~DebugMallocAllocator();

private:
    /// Really allocate memory.
    void *internal_malloc(size_t size);

    /// Really free memory.
    void internal_free(void *memory);

    /// Check memory leaks and dump them.
    void check_memory_leaks();

    /// Parse commands.
    ///
    /// \param cmd  the command string
    void parse_commands(char const *cmd);

    /// Check if obj is a known object and return its header by doing
    /// an O(n) search in the lint on allocated blocks.
    ///
    /// \param obj  the object for which the header is searched
    ///
    /// \return the header of the object or NULL if this is not an allocated
    ///         object
    Header *find_header(void const *obj) const;

    /// Check if obj is a known object and return its header, much faster
    /// but unsafe variant of find_header().
    ///
    /// \param obj  the object for which the header is searched
    ///
    /// \return the header of the object or NULL if this is not an allocated
    ///         object
    Header *find_header_fast(void const *obj) const;

    /// Get an empty frame for capturing.
    ///
    /// \return an empty capture frame
    Captured_frame *get_empty_frame();

    /// Free an captured frame.
    ///
    /// \param frame  the frame to be freed
    void free_frame(Captured_frame *frame);

    /// Check if the given object should be traced.
    ///
    /// \param h         the current header
    /// \param ref_mode  if true, check for REF only
    bool is_traced(Header const *h, bool ref_mode = false) const;

    /// Capture the stack frame.
    ///
    /// \param h       the header for which the frame is captured
    /// \param reason  the reason for the capture
    void capture_stack_frame(Header *h, Capture_reason reason);

    /// Dump captures frames.
    ///
    /// \param frames  the list of captures frames
    void dump_frames(Captured_frame const *frames);

    /// Set a breakpoint.
    ///
    /// \param token   the breakpoint type token
    /// \param id      the block id
    void set_breakpoint(unsigned token, size_t id);

    /// Check if a breakpoint must be executed.
    ///
    /// \param token  the breakpoint type token
    /// \param id     the current block ID
    void handle_breakpoint(unsigned token, size_t id) const;

    /// Called when the internal data structures are damaged.
    void memory_corruption_detected() const;

    /// Report current size.
    void report_size() const;

private:

    enum Capture_mode {
        CAPTURE_NONE,     //< No capture at all
        CAPTURE_REF,      //< capture reference counted only
        CAPTURE_ALL       //< capture all
    };

    /// The lock for operations on this allocator.
    mutable mi::base::Lock m_lock;

    /// The internal allocator if set.
    mi::base::Handle<mi::base::IAllocator> m_internal_alloc;

    /// Points to the first used block.
    Header *m_first;
    
    /// Points to the last used block.
    Header *m_last;

    /// The list of free errors.
    Free_error *m_errors;

    /// Next block number.
    size_t m_next_block_num;

    /// The number of frames to capture.
    unsigned m_num_captures_frames;

    /// The number of frames to skip in capture.
    unsigned m_num_skip_frames;

    /// Temporary capture store.
    Address *m_temp_frames;

    /// Captures frames free list;
    Captured_frame *m_free_frames;

    /// The capture mode for allocations.
    Capture_mode m_capture_mode;

    /// If set, a memory error was detected.
    bool m_had_memory_error;

    /// If set, do abort on errors
    bool m_abort_on_error;

    /// If set, enable expensive free checks.
    bool m_expensive_checks;

    /// If set, regularly dump allocated sizes.
    bool m_dump_size;

    /// Currently allocated size.
    size_t m_allocated_size;

    /// Maximum allocated size.
    size_t m_max_allocated_size;

    /// Number of allocated blocks.
    size_t m_allocated_blocks;

    /// Maximum number of allocated blocks.
    size_t m_max_allocated_blocks;

    /// Last reported size.
    mutable size_t m_last_reported_size;

    /// The number of tracked IDs.
    size_t m_num_tracked;

    /// The number of set breakpoints.
    size_t m_num_breakpoints;

    /// The tracked IDs.
    size_t m_tracked_ids[16];

    /// The breakpoints.
    Breakpoint m_breakpoints[16];
};

}  // dbg
}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_DEBUG_TOOLS_H
