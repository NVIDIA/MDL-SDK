//===-- tools/bugpoint/ToolRunner.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes an abstraction around a platform C compiler, used to
// compile C and assembly code.  It also exposes an "AbstractIntepreter"
// interface, which is used to execute code using one of the LLVM execution
// engines.
//
//===----------------------------------------------------------------------===//

#ifndef BUGPOINT_TOOLRUNNER_H
#define BUGPOINT_TOOLRUNNER_H

#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SystemUtils.h"
#include <exception>
#include <vector>

namespace llvm {

extern cl::opt<bool> SaveTemps;
extern Triple TargetTriple;

class LLC;

//===---------------------------------------------------------------------===//
// GCC abstraction
//
class GCC {
  MISTD::string GCCPath;                // The path to the gcc executable.
  MISTD::string RemoteClientPath;       // The path to the rsh / ssh executable.
  MISTD::vector<MISTD::string> gccArgs; // GCC-specific arguments.
  GCC(StringRef gccPath, StringRef RemotePath,
      const MISTD::vector<MISTD::string> *GCCArgs)
    : GCCPath(gccPath), RemoteClientPath(RemotePath) {
    if (GCCArgs) gccArgs = *GCCArgs;
  }
public:
  enum FileType { AsmFile, ObjectFile, CFile };

  static GCC *create(MISTD::string &Message,
                     const MISTD::string &GCCBinary,
                     const MISTD::vector<MISTD::string> *Args);

  /// ExecuteProgram - Execute the program specified by "ProgramFile" (which is
  /// either a .s file, or a .c file, specified by FileType), with the specified
  /// arguments.  Standard input is specified with InputFile, and standard
  /// Output is captured to the specified OutputFile location.  The SharedLibs
  /// option specifies optional native shared objects that can be loaded into
  /// the program for execution.
  ///
  int ExecuteProgram(const MISTD::string &ProgramFile,
                     const MISTD::vector<MISTD::string> &Args,
                     FileType fileType,
                     const MISTD::string &InputFile,
                     const MISTD::string &OutputFile,
                     MISTD::string *Error = 0,
                     const MISTD::vector<MISTD::string> &GCCArgs =
                         MISTD::vector<MISTD::string>(),
                     unsigned Timeout = 0,
                     unsigned MemoryLimit = 0);

  /// MakeSharedObject - This compiles the specified file (which is either a .c
  /// file or a .s file) into a shared object.
  ///
  int MakeSharedObject(const MISTD::string &InputFile, FileType fileType,
                       MISTD::string &OutputFile,
                       const MISTD::vector<MISTD::string> &ArgsForGCC,
                       MISTD::string &Error);
};


//===---------------------------------------------------------------------===//
/// AbstractInterpreter Class - Subclasses of this class are used to execute
/// LLVM bitcode in a variety of ways.  This abstract interface hides this
/// complexity behind a simple interface.
///
class AbstractInterpreter {
  virtual void anchor();
public:
  static LLC *createLLC(const char *Argv0, MISTD::string &Message,
                        const MISTD::string              &GCCBinary,
                        const MISTD::vector<MISTD::string> *Args = 0,
                        const MISTD::vector<MISTD::string> *GCCArgs = 0,
                        bool UseIntegratedAssembler = false);

  static AbstractInterpreter* createLLI(const char *Argv0, MISTD::string &Message,
                                        const MISTD::vector<MISTD::string> *Args=0);

  static AbstractInterpreter* createJIT(const char *Argv0, MISTD::string &Message,
                                        const MISTD::vector<MISTD::string> *Args=0);

  static AbstractInterpreter*
  createCustomCompiler(MISTD::string &Message,
                       const MISTD::string &CompileCommandLine);

  static AbstractInterpreter*
  createCustomExecutor(MISTD::string &Message,
                       const MISTD::string &ExecCommandLine);


  virtual ~AbstractInterpreter() {}

  /// compileProgram - Compile the specified program from bitcode to executable
  /// code.  This does not produce any output, it is only used when debugging
  /// the code generator.  It returns false if the code generator fails.
  virtual void compileProgram(const MISTD::string &Bitcode, MISTD::string *Error,
                              unsigned Timeout = 0, unsigned MemoryLimit = 0) {}

  /// OutputCode - Compile the specified program from bitcode to code
  /// understood by the GCC driver (either C or asm).  If the code generator
  /// fails, it sets Error, otherwise, this function returns the type of code
  /// emitted.
  virtual GCC::FileType OutputCode(const MISTD::string &Bitcode,
                                   MISTD::string &OutFile, MISTD::string &Error,
                                   unsigned Timeout = 0,
                                   unsigned MemoryLimit = 0) {
    Error = "OutputCode not supported by this AbstractInterpreter!";
    return GCC::AsmFile;
  }

  /// ExecuteProgram - Run the specified bitcode file, emitting output to the
  /// specified filename.  This sets RetVal to the exit code of the program or
  /// returns false if a problem was encountered that prevented execution of
  /// the program.
  ///
  virtual int ExecuteProgram(const MISTD::string &Bitcode,
                             const MISTD::vector<MISTD::string> &Args,
                             const MISTD::string &InputFile,
                             const MISTD::string &OutputFile,
                             MISTD::string *Error,
                             const MISTD::vector<MISTD::string> &GCCArgs =
                               MISTD::vector<MISTD::string>(),
                             const MISTD::vector<MISTD::string> &SharedLibs =
                               MISTD::vector<MISTD::string>(),
                             unsigned Timeout = 0,
                             unsigned MemoryLimit = 0) = 0;
};

//===---------------------------------------------------------------------===//
// LLC Implementation of AbstractIntepreter interface
//
class LLC : public AbstractInterpreter {
  MISTD::string LLCPath;               // The path to the LLC executable.
  MISTD::vector<MISTD::string> ToolArgs; // Extra args to pass to LLC.
  GCC *gcc;
  bool UseIntegratedAssembler;
public:
  LLC(const MISTD::string &llcPath, GCC *Gcc,
      const MISTD::vector<MISTD::string> *Args,
      bool useIntegratedAssembler)
    : LLCPath(llcPath), gcc(Gcc),
      UseIntegratedAssembler(useIntegratedAssembler) {
    ToolArgs.clear();
    if (Args) ToolArgs = *Args;
  }
  ~LLC() { delete gcc; }

  /// compileProgram - Compile the specified program from bitcode to executable
  /// code.  This does not produce any output, it is only used when debugging
  /// the code generator.  Returns false if the code generator fails.
  virtual void compileProgram(const MISTD::string &Bitcode, MISTD::string *Error,
                              unsigned Timeout = 0, unsigned MemoryLimit = 0);

  virtual int ExecuteProgram(const MISTD::string &Bitcode,
                             const MISTD::vector<MISTD::string> &Args,
                             const MISTD::string &InputFile,
                             const MISTD::string &OutputFile,
                             MISTD::string *Error,
                             const MISTD::vector<MISTD::string> &GCCArgs =
                               MISTD::vector<MISTD::string>(),
                             const MISTD::vector<MISTD::string> &SharedLibs =
                                MISTD::vector<MISTD::string>(),
                             unsigned Timeout = 0,
                             unsigned MemoryLimit = 0);

  /// OutputCode - Compile the specified program from bitcode to code
  /// understood by the GCC driver (either C or asm).  If the code generator
  /// fails, it sets Error, otherwise, this function returns the type of code
  /// emitted.
  virtual GCC::FileType OutputCode(const MISTD::string &Bitcode,
                                   MISTD::string &OutFile, MISTD::string &Error,
                                   unsigned Timeout = 0,
                                   unsigned MemoryLimit = 0);
};

} // End llvm namespace

#endif
