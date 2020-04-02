/******************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

/// \file
/// \brief The debug memory allocation implementation for neuray, a stripped down version of the
///        MDL debug allocator.

#ifdef DEBUG

#ifdef _WIN32

#include <new>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tchar.h>

#include <dbghelp.h>

#elif defined(__GNUC__)

#include <execinfo.h>
#include <cxxabi.h>
#include <signal.h>

#endif

#include <cstring>
#include <cstdlib>
#include <cstdarg>
#include <cctype>
#include <cstdio>

#include <mi/base/lock.h>

#include "mem_debug_alloc.h"

namespace MI {
namespace MEM {
namespace DBG {

/// Returns the dimension of an array.
template<typename T, size_t n>
inline size_t dimension_of(T (&c)[n]) { return n; }

/// Interface to write symbols.
class ISymbol_lookup {
public:
    virtual void write_symbol(void *Address) = 0;
};

/// OS-specific frame capture.
///
/// \param skip    number of frames to skip
/// \param n       number of frames to capture
/// \param frames  destination
/// \param tmp     a temporary array of size skip + n
///
/// \return number of captured frames
static unsigned os_capture_frames(size_t skip, size_t n, void *frames[], void *tmp[]);

/// Return the OS-specific Symbol lookup interface.
static ISymbol_lookup *os_get_symbol_lookup();

/// OS-specific break into debugger.
static void os_break_point();

/// OS-specific printer
int os_printf(char const *fmt, ...);

#ifdef _MSC_VER
// MS runtime does not support %z type modifier
#ifdef _WIN64
typedef unsigned __int64 SIZE;
#define PRT_SIZE "%I64u"
#else
#define PRT_SIZE "%lu"
typedef unsigned long SIZE;
#endif
#else
typedef size_t SIZE;
#define PRT_SIZE "%zu"
#endif

namespace {

static const char *reserved[] = {
    "capture",
    "depth",
    "skip",
    "bpa",
    "bpf",
    "bpi",
    "bpd",
    "abort",
    "noabort",
    "expensive",
    "noexpensive",
    "size",
    "nosize",
    "all",
    "ref",
    "help"
};

static void show_commands() {
    os_printf("MEURAY memory allocator internal debugger commands:\n"
        "capture id   capture allocations for block id\n"
        "capture all  capture all allocations (Danger: slow, huge memory consumption)\n"
        "depth num    sets the depth of the captured frame, default 5\n"
        "skip num     skip the first num frames when capturing, default 3\n"
        "bpa id       break into debugger if block id is allocated\n"
        "bpf id       break into debugger if block id is freed\n"
        "bpi id       break into debugger if object id's reference count is increased\n"
        "bpd id       break into debugger if object id's reference count is decreased\n"
        "abort        abort after memory errors were reported (default in DEBUG mode)\n"
        "noabort      do not abort after memory errors were reported (default in RELEASE mode)\n"
        "expensive    enable expensive checks (default in DEBUG mode)\n"
        "noexpensive  disable expensive check (default in RELEASE mode)\n"
        "size         regularly dump allocated size\n"
        "nosize       do not dump allocated size\n"
        "help         this help text\n");
}

/// Simple lexer for debug commands.
class Debug_cmd_lexer {
public:
    enum Tokens {
        tok_capture = 256,
        tok_depth,
        tok_skip,
        tok_bpa,
        tok_bpf,
        tok_bpi,
        tok_bpd,
        tok_abort,
        tok_noabort,
        tok_expensive,
        tok_noexpensive,
        tok_size,
        tok_nosize,
        tok_all,
        tok_ref,
        tok_help,
        tok_num,
        tok_error,
        tok_eof
    };

    /// Constructor for a new debug command lexer.
    Debug_cmd_lexer(const char *input)
    : m_has_token(0)
    , m_cur_token(tok_error)
    , m_s(NULL)
    , m_len(0)
    , m_num(0)
    , m_curr_pos(input)
    , m_end_pos(input + strlen(input))
    , m_tok_start(NULL)
    {
    }

    /// The lexer.
    unsigned get_token() {
        char c;

        do {
            c = next_char();
        } while (c != '\0' && isspace(c));

        m_tok_start = m_curr_pos - 1;
        if (isalpha(c)) {
            int len = 0;

            do {
                c = next_char();
                ++len;
            } while (isalpha(c));
            unput();

            for (int i = dimension_of(reserved) - 1; i >= 0; --i) {
                if (strncmp(reserved[i], m_tok_start, len) == 0)
                    return 256 + i;
            }

            // else an identifier
            m_s   = m_tok_start;
            m_len = len;
            return tok_error;
        } else if (isdigit(c)) {
            size_t value = c - '0';

            for (;;) {
                c = next_char();
                if (isdigit(c))
                    value = value * 10 + c - '0';
                else
                    break;
            }
            unput();
            m_num = value;
            return tok_num;
        } else if (c == '\0')
            return tok_eof;
        return c;
    }

    /// Return the length of the current identifier.
    size_t get_ident_length() const { return m_len; }

    /// Return the current identifier's text.
    char const *get_ident_text() const { return m_s; }

    /// Return the current token's start for error messages.
    char const *tok_start() const { return m_tok_start; }

    /// Return the current tokens numerical value.
    size_t get_num_value() const { return m_num; }

private:
    /// Get the next char from input.
    char next_char() {
        if (m_curr_pos >= m_end_pos)
            return '\0';
        return *m_curr_pos++;
    }

    /// Put one character back into the buffer.
    void unput() {
        if (m_curr_pos < m_end_pos)
            --m_curr_pos;
    }

private:
    int        m_has_token;
    unsigned   m_cur_token;
    char const *m_s;
    size_t     m_len;
    size_t     m_num;

    char const *m_curr_pos;
    char const *m_end_pos;
    char const *m_tok_start;
};

}  // anonymous

void *DebugMallocAllocator::malloc(size_t size)
{
    return DebugMallocAllocator::objalloc(NULL, size);
}

void DebugMallocAllocator::free(void *memory)
{
    if (memory == NULL)
        return;

    mi::base::Lock::Block block(&m_lock);

    Header *h = m_expensive_checks ? find_header(memory) : find_header_fast(memory);

    if (h == NULL) {
        // create an error
        Free_error *err = (Free_error *)internal_malloc(sizeof(*err));
        if (err != NULL) {
            err->next  = m_errors;
            err->adr   = memory;
            err->frame = NULL;

            m_errors = err;

            if (Captured_frame *frame = get_empty_frame()) {
                unsigned count = os_capture_frames(
                    m_num_skip_frames,
                    frame->length,
                    frame->frames,
                    m_temp_frames);

                if (count > 0) {
                    frame->reason = CR_FREE;
                    frame->length = count;

                    err->frame = frame;
                }
            }
        }
        return;
    }

    handle_breakpoint(Debug_cmd_lexer::tok_bpf, h->num);

    // free all captures frames if any
    for (Captured_frame *n, *f = h->frames; f != NULL; f = n) {
        n = f->next;
        free_frame(f);
    }

    if (h->prev != NULL) {
        h->prev->next = h->next;
    } else {
        m_first = h->next;
    }

    if (h->next != NULL) {
        h->next->prev = h->prev;
    } else {
        m_last = h->prev;
    }

    m_allocated_size -= h->size;
    --m_allocated_blocks;

    if (m_dump_size)
        report_size();

    internal_free(h);
}

// Allocates a memory block for a class instance.
void *DebugMallocAllocator::objalloc(char const *cls_name, size_t size)
{
    mi::base::Lock::Block block(&m_lock);

    handle_breakpoint(Debug_cmd_lexer::tok_bpa, m_next_block_num);

    Header *h = (Header *)internal_malloc(size + sizeof(Header));

    if (h == NULL && size != 0) {
        os_printf("*** Memory exhausted (after " PRT_SIZE "Mb).\n",
            m_allocated_size >> size_t(20));
        abort();
    }

    m_allocated_size += size;
    if (m_allocated_size > m_max_allocated_size)
        m_max_allocated_size = m_allocated_size;
    ++m_allocated_blocks;
    if (m_allocated_blocks > m_max_allocated_blocks)
        m_max_allocated_blocks = m_allocated_blocks;

    memset(h, 0xDC, sizeof(*h));
    h->magic1    =
    h->magic2    = Header::HEADER_MAGIC;
    h->flags     = 0;
    h->ref_count = 0;
    h->frames    = NULL;
    h->cls_name  = cls_name;

    if (m_first == NULL) {
        m_first = h;
        h->prev = NULL;
    } else {
        h->prev = m_last;
    }
    if (m_last != NULL) {
        m_last->next = h;
    }

    h->next = NULL;
    h->size = size;
    h->num  = m_next_block_num++;
    m_last = h;

    if (is_traced(h))
        capture_stack_frame(h, CR_ALLOC);

    if (m_dump_size)
        report_size();

    return (void *)(h + 1);
}

// Marks the given object as reference counted and set the initial count.
void DebugMallocAllocator::mark_ref_counted(
    void const *obj,
    unsigned   initial)
{
    mi::base::Lock::Block block(&m_lock);

    if (Header *h = find_header_fast(obj)) {
        h->flags     |= BH_REF_COUNTED;
        h->ref_count  = initial;

        if (is_traced(h, /*ref_mode=*/true))
            capture_stack_frame(h, CR_ALLOC);
    }
}

// Increments the reference count of an reference counted object.
void DebugMallocAllocator::inc_ref_count(void const *obj)
{
    mi::base::Lock::Block block(&m_lock);

    if (Header *h = find_header_fast(obj)) {
        ++h->ref_count;

        if (is_traced(h))
            capture_stack_frame(h, CR_REF_COUNT_INC);

        handle_breakpoint(Debug_cmd_lexer::tok_bpi, h->num);
    }
}

// Decrements the reference count of an reference counted object.
void DebugMallocAllocator::dec_ref_count(void const *obj)
{
    mi::base::Lock::Block block(&m_lock);

    if (Header *h = find_header_fast(obj)) {
        --h->ref_count;

        if (is_traced(h))
            capture_stack_frame(h, CR_REF_COUNT_DEC);

        handle_breakpoint(Debug_cmd_lexer::tok_bpd, h->num);
    }
}

// Constructor.
DebugMallocAllocator::DebugMallocAllocator()
: m_lock()
, m_first(NULL)
, m_last(NULL)
, m_errors(NULL) 
, m_next_block_num(0)
, m_num_captures_frames(10)
, m_num_skip_frames(4)
, m_temp_frames(NULL)
, m_free_frames(NULL)
, m_capture_mode(CAPTURE_ALL)
, m_had_memory_error(false)
, m_abort_on_error(false)
, m_expensive_checks(false)
, m_dump_size(false)
, m_allocated_size(0)
, m_max_allocated_size(0)
, m_allocated_blocks(0)
, m_max_allocated_blocks(0)
, m_last_reported_size(0)
, m_num_tracked(0)
, m_num_breakpoints(0)
{
#ifdef DEBUG
    // in debug mode abort by default, so unit tests will fail without extra command
    m_abort_on_error = true;
#endif

    if (char const *cmd = ::getenv("NV_NEURAY_DEBUG"))
        parse_commands(cmd);

    size_t tmp_size = m_num_captures_frames + m_num_skip_frames;
    if (tmp_size > 0) {
        m_temp_frames = (Address *)internal_malloc(sizeof(m_temp_frames[0]) * tmp_size);
    }
}

DebugMallocAllocator::~DebugMallocAllocator()
{
    check_memory_leaks();

    if (m_dump_size) {
        size_t max_size = (m_max_allocated_size + size_t(1 << 20) - 1) >> size_t(20);
        os_printf("*** Maximum NEURAY memory allocated: " PRT_SIZE "Mb, " PRT_SIZE " blocks\n",
            max_size, m_max_allocated_blocks);
    }

    if (m_temp_frames != NULL)
        internal_free(m_temp_frames);

    for (Header *n, *h = m_first; h != NULL; h = n) {
        n = h->next;

        for (Captured_frame *nf, *f = h->frames; f != NULL; f = nf) {
            nf = f->next;

            free_frame(f);
        }
        h->frames = NULL;
        internal_free(h);
    }

    for (Free_error *nf, *f = m_errors; f != NULL; f = nf) {
        nf = f->next;

        if (f->frame != NULL)
            free_frame(f->frame);

        internal_free(f);
    }

    for (Captured_frame *nf, *f = m_free_frames; f != NULL; f = nf) {
        nf = f->next;

        internal_free(f);
    }

    m_free_frames = NULL;
    m_first = m_last = NULL;

    if (m_abort_on_error && m_had_memory_error) {
        // abort in debug mode, so unit test will fail if they have memory errors
        os_printf("*** NEURAY Memory errors found, aborting ...\n");
        fflush(stderr);
        abort();
    }
}

// Really allocate memory.
void *DebugMallocAllocator::internal_malloc(size_t size)
{
    return ::malloc(size);
}

// Really free memory.
void DebugMallocAllocator::internal_free(void *memory)
{
    ::free(memory);
}

static const size_t max_iter = 100000;

void DebugMallocAllocator::check_memory_leaks()
{
    mi::base::Lock::Block block(&m_lock);

    if (m_first != NULL) {
        os_printf("*** Lost memory in NEURAY detected:\n");
        size_t num_obj = 0;
        for (Header const *h = m_first; h != NULL; h = h->next) {
            if (h->flags & BH_REF_COUNTED) {
                // reference counted object leaked
                fprintf(
                    stderr, "%s ", h->cls_name != NULL ? h->cls_name : "Object");
                fprintf(
                    stderr, "%p (" PRT_SIZE "), refcount %u, size " PRT_SIZE "\n", h+1,
                    SIZE(h->num),
                    (unsigned)h->ref_count,
                    SIZE(h->size));
                dump_frames(h->frames);
                ++num_obj;
            }
        }
        if (num_obj > 0)
            os_printf(PRT_SIZE " object(s) still referenced\n\n", SIZE(num_obj));

        size_t num_blocks = 0;
        for (Header const *h = m_first; h != NULL; h = h->next) {
            if ((h->flags & BH_REF_COUNTED) == 0) {
                if (num_blocks < max_iter) {
                    fprintf(
                        stderr, "%s ", h->cls_name != NULL ? h->cls_name : "Block");
                    fprintf(
                        stderr, "%p (" PRT_SIZE "), size " PRT_SIZE "\n", h+1,
                        SIZE(h->num), SIZE(h->size));
                    dump_frames(h->frames);
                }
                ++num_blocks;
            }
        }
        if (num_blocks >= max_iter)
            os_printf("...\n" PRT_SIZE " more block(s) lost.\n", SIZE(num_blocks - max_iter));

        os_printf("\n");
    }
    if (m_errors != NULL) {
        os_printf("*** Memory errors in NEURAY detected:\n");

        for (Free_error const *f = m_errors; f != NULL; f = f->next) {
            os_printf("Wrong free %p\n", f->adr);
            dump_frames(f->frame);
        }
        os_printf("\n");
    }
    m_had_memory_error |= m_first != NULL;
    m_had_memory_error |= m_errors != NULL;
}

// Parse commands.
void DebugMallocAllocator::parse_commands(char const *cmd)
{
    Debug_cmd_lexer lexer(cmd);
    for (;;) {
        unsigned token = lexer.get_token();

        switch (token) {
        case Debug_cmd_lexer::tok_eof:
            goto leave;
        case Debug_cmd_lexer::tok_capture:
            token = lexer.get_token();
            if (token == Debug_cmd_lexer::tok_all) {
                m_capture_mode = CAPTURE_ALL;
            } else if (token == Debug_cmd_lexer::tok_ref) {
                m_capture_mode = CAPTURE_REF;
            } else if (token == Debug_cmd_lexer::tok_num) {
                if (m_num_tracked < dimension_of(m_tracked_ids)) {
                    m_tracked_ids[m_num_tracked++] = lexer.get_num_value();
                } else {
                    os_printf("DebugAllocator: cannot track more individual blocks\n");
                }
            } else
                goto error;
            break;
        case Debug_cmd_lexer::tok_depth:
            token = lexer.get_token();
            if (token != Debug_cmd_lexer::tok_num)
                goto error;
            m_num_captures_frames = lexer.get_num_value();
            break;
        case Debug_cmd_lexer::tok_skip:
            token = lexer.get_token();
            if (token != Debug_cmd_lexer::tok_num)
                goto error;
            m_num_skip_frames = lexer.get_num_value();
            break;
        case Debug_cmd_lexer::tok_bpa:
        case Debug_cmd_lexer::tok_bpf:
        case Debug_cmd_lexer::tok_bpi:
        case Debug_cmd_lexer::tok_bpd:
            {
                unsigned ntoken = lexer.get_token();
                if (ntoken != Debug_cmd_lexer::tok_num)
                    goto error;
                set_breakpoint(token, lexer.get_num_value());
            }
            break;
        case Debug_cmd_lexer::tok_abort:
            m_abort_on_error = true;
            break;
        case Debug_cmd_lexer::tok_noabort:
            m_abort_on_error = false;
            break;
        case Debug_cmd_lexer::tok_expensive:
            m_expensive_checks = true;
            break;
        case Debug_cmd_lexer::tok_noexpensive:
            m_expensive_checks = false;
            break;
        case Debug_cmd_lexer::tok_size:
            m_dump_size = true;
            break;
        case Debug_cmd_lexer::tok_nosize:
            m_dump_size = false;
            break;
        case Debug_cmd_lexer::tok_help:
            show_commands();
            break;
        default:
error:
            printf("DebugAllocator: command parse error before '%s'\n", lexer.tok_start());
            goto leave;
        }
        token = lexer.get_token();
        if (token == Debug_cmd_lexer::tok_eof)
            break;
        if (token != ';')
            goto error;
    }
leave:
    return;
}

// Check if obj is a known object and return its header ...
DebugMallocAllocator::Header *DebugMallocAllocator::find_header(void const *obj) const
{
    Header *h = (Header *)obj;
    h -= 1;

    if (h->magic1 != Header::HEADER_MAGIC || h->magic2 != Header::HEADER_MAGIC)
        return NULL;

    // check if this is our block
    Header *p = m_first;
    for (; p != NULL; p = p->next) {
        if (p->magic1 != Header::HEADER_MAGIC || p->magic2 != Header::HEADER_MAGIC) {
            memory_corruption_detected();
            return NULL;
        }
        if (p == h)
            return p;
    }
    return NULL;
}

// Check if obj is a known object and return its header ...
DebugMallocAllocator::Header *DebugMallocAllocator::find_header_fast(void const *obj) const
{
    // do some fast checks
    size_t adr = size_t(obj);
    if (adr & 7) {
        // lower 3 bits should always be 0 in a valid address
        return NULL;
    }

    Header *h = (Header *)obj;
    h -= 1;

    if (h->magic1 != Header::HEADER_MAGIC || h->magic2 != Header::HEADER_MAGIC)
        return NULL;

    if (Header const *p = h->prev) {
        if (p->magic1 != Header::HEADER_MAGIC || p->magic2 != Header::HEADER_MAGIC) {
            memory_corruption_detected();
            return NULL;
        }
    } else if (m_first != h) {
        memory_corruption_detected();
        return NULL;
    }

    if (Header const *n = h->next) {
        if (n->magic1 != Header::HEADER_MAGIC || n->magic2 != Header::HEADER_MAGIC) {
            memory_corruption_detected();
            return NULL;
        }
    } else if (m_last != h) {
        memory_corruption_detected();
        return NULL;
    }

    return h;
}

// Get an empty frame for capturing.
DebugMallocAllocator::Captured_frame *DebugMallocAllocator::get_empty_frame()
{
    Captured_frame *frame = m_free_frames;

    if (frame != NULL) {
        m_free_frames = frame->next;
    } else {
        frame = (Captured_frame *)internal_malloc(
            sizeof(Captured_frame)
            + (m_num_captures_frames - 1) * sizeof(frame->frames[0]));
        if (frame == NULL)
            return NULL;

    }

    frame->length = m_num_captures_frames;
    frame->next   = NULL;
    return frame;
}

// Free an captured frame.
void DebugMallocAllocator::free_frame(Captured_frame *frame)
{
    frame->next = m_free_frames;
    m_free_frames = frame;
}

/// Check if the given object should be traced.
bool DebugMallocAllocator::is_traced(Header const *h, bool ref_mode) const
{
    if (ref_mode) {
        return (m_capture_mode == CAPTURE_REF && (h->flags & BH_REF_COUNTED));
    }

    if (m_capture_mode == CAPTURE_ALL)
        return true;
    if (m_capture_mode == CAPTURE_REF && (h->flags & BH_REF_COUNTED))
        return true;
    for (size_t i = 0; i < m_num_tracked; ++i) {
        if (h->num == m_tracked_ids[i])
            return true;
    }
    return false;
}

// Capture the stack frame.
void DebugMallocAllocator::capture_stack_frame(
    Header         *h,
    Capture_reason reason)
{
    Captured_frame *frame = get_empty_frame();

    if (!frame)
        return;

    unsigned count = os_capture_frames(
        m_num_skip_frames, frame->length, frame->frames, m_temp_frames);

    if (count == 0) {
        free_frame(frame);
        return;
    }

    frame->reason = reason;
    frame->length = count;
    frame->next   = h->frames;
    h->frames     = frame;
}

// Dump captures frames.
void DebugMallocAllocator::dump_frames(Captured_frame const *frames)
{
    ISymbol_lookup *lookup = os_get_symbol_lookup();

    for (Captured_frame const *f = frames; f != NULL; f = f->next) {
        os_printf("Captured Frame ------------\n");
        switch (f->reason) {
        case CR_ALLOC:
            os_printf("Allocation:\n");
            break;
        case CR_FREE:
            os_printf("Freeing:\n");
            break;
        case CR_REF_COUNT_INC:
            os_printf("retain():\n");
            break;
        case CR_REF_COUNT_DEC:
            os_printf("release():\n");
            break;
        }
        for (unsigned i = 0; i < f->length; ++i) {
            os_printf("%p: ", f->frames[i]);

            if (lookup != NULL) {
                lookup->write_symbol(f->frames[i]);
            }
            os_printf("\n");
        }
    }
    if (frames != NULL)
        os_printf("\n");
}

// Set a breakpoint.
void DebugMallocAllocator::set_breakpoint(unsigned token, size_t id)
{
    if (m_num_breakpoints < dimension_of(m_breakpoints)) {
        m_breakpoints[m_num_breakpoints].token = token;
        m_breakpoints[m_num_breakpoints].id    = id;
        ++m_num_breakpoints;
    } else {
        os_printf("Cannot set more then %u breakpoints\n",
            unsigned(dimension_of(m_breakpoints)));
    }
}

// Check if a breakpoint must be executed.
void DebugMallocAllocator::handle_breakpoint(unsigned token, size_t id) const
{
    for (size_t i = 0; i < m_num_breakpoints; ++i) {
        if (id == m_breakpoints[i].id && token == m_breakpoints[i].token) {
            os_printf("*** Breakpoint on block " PRT_SIZE " reached\n", SIZE(id));
            os_break_point();
        }
    }
}

// Called when the internal data structures are damaged.
void DebugMallocAllocator::memory_corruption_detected() const
{
    os_printf("***Memory Corruption detected\n");
}

// report current size.
void DebugMallocAllocator::report_size() const
{
    size_t curr = (m_allocated_size + size_t(1 << 20) - 1) >> size_t(20);

    if (curr != m_last_reported_size) {
        m_last_reported_size = curr;
        os_printf("*** Current NEURAY memory usage: " PRT_SIZE "Mb in " PRT_SIZE " blocks\n",
            curr, m_allocated_blocks);
    }
}

// ------------------------ OS specific stuff ------------------------

#ifdef WIN32

class Win32_symbol_lookup : public ISymbol_lookup {
public:
    Win32_symbol_lookup();
    ~Win32_symbol_lookup();

    virtual void write_symbol(void *Address);

private:
    /// Prints the GetLastError() result.
    void report_error(TCHAR const *text);

private:
    /// If set to true, the dbghelp.dll is loaded and all addresses are resolved.
    bool      m_initialized;

    /// The current process handle.
    HANDLE    m_hProcess;

    /// The handle of the dbghelp.dll
    HINSTANCE m_hDbgHelp;

    typedef BOOL (WINAPI *SymInitialize_t)(
        __in HANDLE hProcess,
        __in_opt PCSTR UserSearchPath,
        __in BOOL fInvadeProcess);

    typedef BOOL (WINAPI *SymFromAddr_t)(
        __in HANDLE hProcess,
        __in DWORD64 Address,
        __out_opt PDWORD64 Displacement,
        __inout SYMBOL_INFO *Symbol);

    typedef BOOL (WINAPI *SymGetLineFromAddr64_t)(
        __in   HANDLE hProcess,
        __in   DWORD64 dwAddr,
        __out  PDWORD pdwDisplacement,
        __out  IMAGEHLP_LINE64 *Line);

    SymInitialize_t        SymInitialize;
    SymFromAddr_t          SymFromAddr;
    SymGetLineFromAddr64_t SymGetLineFromAddr64;
};

#define _STR(x) #x
#define STR(x) _STR(x)

#define GetFunction(x) (x##_t)GetProcAddress(m_hDbgHelp, _T(STR(x)))

// Constructor
Win32_symbol_lookup::Win32_symbol_lookup()
: m_hProcess(GetCurrentProcess())
, m_initialized(false)
, m_hDbgHelp(NULL)
, SymInitialize(NULL)
, SymFromAddr(NULL)
, SymGetLineFromAddr64(NULL)
{
    m_hDbgHelp = LoadLibrary(_T("dbghelp.dll"));
    if (m_hDbgHelp != NULL) {
        // succeeded
        SymInitialize        = GetFunction(SymInitialize);
        SymFromAddr          = GetFunction(SymFromAddr);
        SymGetLineFromAddr64 = GetFunction(SymGetLineFromAddr64);
    } else {
        report_error(_T("Could not load dbghelp.dll"));
    }

    if (SymInitialize != NULL && SymFromAddr != NULL && SymGetLineFromAddr64 != NULL) {
        m_initialized = true;

        BOOL res = SymInitialize(
            m_hProcess,
            /*UserSearchPath=*/NULL,
            /*fInvadeProcess=*/TRUE);
        if (!res) {
            report_error(_T("SymInitialize"));
        }
    }
}

// Destructor
Win32_symbol_lookup::~Win32_symbol_lookup()
{
    if (m_hDbgHelp != NULL) {
        FreeLibrary(m_hDbgHelp);
        m_hDbgHelp = NULL;
        m_initialized = false;
    }
}

void Win32_symbol_lookup::write_symbol(void *Address)
{
    if (!m_initialized)
        return;

    DWORD64 displacement;

    struct {
        SYMBOL_INFO sym_info;
        TCHAR       buf[1024];
    } s;

    memset(&s, 0, sizeof(s));
    s.sym_info.SizeOfStruct = sizeof(s.sym_info);
    s.sym_info.MaxNameLen   = dimension_of(s.buf);

    BOOL res = SymFromAddr(
        m_hProcess,
        DWORD64(Address),
        &displacement,
        &s.sym_info);

    if (!res) {
        report_error(_T("SymFromAddr"));
        return;
    }
    os_printf(_T("%s"), s.sym_info.Name);

    IMAGEHLP_LINE64 line;
    memset(&line, 0, sizeof(line));
    line.SizeOfStruct = sizeof(line);

    DWORD line_displacement;

    res = SymGetLineFromAddr64(
        m_hProcess,
        DWORD64(Address),
        &line_displacement,
        &line);

    if (res) {
        os_printf(_T(" %s(%lu)"),
            line.FileName,
            (unsigned long)line.LineNumber);
    }
}

void Win32_symbol_lookup::report_error(TCHAR const *text)
{
    DWORD err = GetLastError();
    TCHAR buffer[2048];

    FormatMessage(
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        /*lpSource=*/NULL,
        err,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        buffer,
        dimension_of(buffer),
        /*Arguments =*/NULL);
    _ftprintf(stderr, _T("%s: %s\n"), text, buffer);
}

static unsigned os_capture_frames(
    size_t skip,
    size_t n,
    void *frames[],
    void *tmp[])
{
    if (n == 0)
        return 0;
    return RtlCaptureStackBackTrace(
        /*FramesToSkip=*/ULONG(skip),
        /*FrameToCapture=*/ULONG(n),
        frames,
        /*BackTraceHash=*/NULL);
}


static ISymbol_lookup *os_get_symbol_lookup()
{
    static Win32_symbol_lookup *win32_symbol_lookup = NULL;

    if (win32_symbol_lookup == NULL) {
        win32_symbol_lookup = (Win32_symbol_lookup *)::malloc(sizeof(*win32_symbol_lookup));

        new (win32_symbol_lookup) Win32_symbol_lookup;
    }
    return win32_symbol_lookup;
}

static void os_break_point()
{
    DebugBreak();
}

int os_printf(char const *fmt, ...)
{
    char buf[32*1024];
    va_list va;

    va_start(va, fmt);
    int res = vfprintf(stderr, fmt, va);

    vsnprintf(buf, sizeof(buf), fmt, va);
    buf[sizeof(buf) - 1] = '\0';

    OutputDebugStringA(buf);

    va_end(va);
    return res;
}

/// os_printf for WCHAR data.
int os_printf(WCHAR const *fmt, ...)
{
    WCHAR buf[32*1024];
    va_list va;

    va_start(va, fmt);
    int res = vfwprintf(stderr, fmt, va);

    _vsnwprintf(buf, sizeof(buf)/sizeof(buf[0]), fmt, va);
    buf[sizeof(buf)/sizeof(buf[0]) - 1] = L'\0';

    OutputDebugStringW(buf);

    va_end(va);
    return res;
}

#elif defined(__GNUC__)


class Linux_symbol_lookup : public ISymbol_lookup {
public:
    virtual void write_symbol(void *Address);

private:
    char *demangle(char const *mangled, size_t len);
};

// Linux address decoding based of backtrace_symbols().
void Linux_symbol_lookup::write_symbol(void *Address)
{
    if (char **symbols = backtrace_symbols(&Address, 1)) {
        char *m = *symbols;

        char *begin = NULL, *end = NULL, *adr = NULL;

        for (char *p = *symbols; *p != '\0'; ++p) {
            if (*p == '(') {
                begin = p;
            } else if (*p == '+') {
                end = p;
            } else if (*p == '[') {
                adr = p;
            }
        }

        if (begin != NULL && end != NULL) {
            ++begin;
            *end = '\0';
            char *demangled = demangle(begin, 256);

            os_printf("%s", demangled);
            free(demangled);
        } else {
            if (adr != NULL)
                *adr = '\0';
            os_printf("%s", m);
        }
        free(symbols);
    }
}

char *Linux_symbol_lookup::demangle(char const *mangled, size_t len)
{
    char *demangled = (char *)::malloc(len);
    int status;
    size_t n_len = len;
    char *ret = abi::__cxa_demangle(mangled, demangled, &n_len, &status);
    if (ret) {
        demangled = ret;
    } else {
        ::strncpy(demangled, mangled, len);
        ::strncat(demangled, "()", len);
        demangled[len-1] = '\0';
    }
    return demangled;
}

static unsigned os_capture_frames(
    size_t skip,
    size_t n,
    void *frames[],
    void *tmp[])
{
    int res;

    if (n == 0)
        return 0;
    if (skip > 0) {
        res = backtrace(tmp, int(skip + n));
        if (res > 0) {
            memcpy(frames, tmp + skip, n * sizeof(frames[0]));
            return unsigned(res) - unsigned(skip);
        }
    } else {
        res = backtrace(frames, int(n));
        if (res > 0)
            return unsigned(res);
    }
    return 0;
}

static Linux_symbol_lookup linux_symbol_lookup;

static ISymbol_lookup *os_get_symbol_lookup()
{
    return &linux_symbol_lookup;
}

static void os_break_point()
{
#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64))
    __asm__ __volatile__("int3");
#else
    // poor unix way
    raise(SIGINT);
#endif
}

int os_printf(char const *fmt, ...)
{
    va_list va;
    
    va_start(va, fmt);
    int res = vfprintf(stderr, fmt, va);
    va_end(va);
    return res;
}

#endif  // WIN32

} // DBG
} // MEM
} // MI

#endif  // DEBUG
