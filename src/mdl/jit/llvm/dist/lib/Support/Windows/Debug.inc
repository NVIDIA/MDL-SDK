//===- llvm/Support/Win32/Debug.inc ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 extra Debug support.
//
//===----------------------------------------------------------------------===//

#include "Windows.h"
#include "llvm/Support/raw_ostream.h"

#if defined(__CYGWIN__)
#include <io.h>
#endif

#if defined(_MSC_VER)
#include <io.h>
#ifndef STDERR_FILENO
# define STDERR_FILENO 2
#endif
#endif

namespace llvm {
namespace {

/// raw_dbg_ostream - A raw_ostream that writes to a file descriptor and Windows Debug output.
///
class raw_dbg_ostream : public raw_ostream {
  /// TheStream - The real stream we output to. We set it to be
  /// unbuffered, since we're already doing our own buffering.
  ///
  raw_ostream *TheStream;

  /// OwnsStream - Are we responsible for managing the underlying
  /// stream?
  ///
  bool OwnsStream;

  uint64_t pos;

  static size_t const BufSize = 256;

  /// write_impl - See raw_ostream::write_impl.
  virtual void write_impl(const char *Ptr, size_t Size) override
  {
    pos += Size;

    TheStream->write(Ptr, Size);

    char Buf[BufSize + 1];
    while (Size > 0) {
      if (Size >= BufSize) {
        memcpy(Buf, Ptr, BufSize);
        Buf[BufSize] = '\0';
        Ptr  += BufSize;
        Size -= BufSize;
      } else {
        memcpy(Buf, Ptr, Size);
        Buf[Size] = '\0';
        Size = 0;
      }
      OutputDebugStringA(Buf);
    }
  }

  /// current_pos - Return the current position within the stream, not
  /// counting the bytes currently in the buffer.
  virtual uint64_t current_pos() const override { return 0; }

  /// preferred_buffer_size - Determine an efficient buffer size.
  virtual size_t preferred_buffer_size() const override {
    // BUFSIZ is intended to be a reasonable default.
    return BUFSIZ;
  }

public:
  raw_dbg_ostream(raw_ostream &Stream, bool Owns = false)
    : raw_ostream(/*unbuffered*/true),
      TheStream(&Stream),
      OwnsStream(Owns),
      pos(0)
  {
  }

  ~raw_dbg_ostream()
  {
    flush();
    if (OwnsStream)
      delete TheStream;
  }
};

}  // anonymous

  raw_ostream &extra_dbgs() {
     static raw_fd_ostream S(STDERR_FILENO, false, true);
     static raw_dbg_ostream DS(S, false);
     return DS;
  }
}