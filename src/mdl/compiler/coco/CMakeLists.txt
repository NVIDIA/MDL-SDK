#*****************************************************************************
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*****************************************************************************

# name of the target and the resulting library
set(PROJECT_NAME mdl-compiler-coco)

# collect sources
set(PROJECT_HEADERS 
    "Action.h"
    "ArrayList.h"
    "BitArray.h"
    "CharClass.h"
    "CharSet.h"
    "Comment.h"
    "DFA.h"
    "Generator.h"
    "Graph.h"
    "HashTable.h"
    "Melted.h"
    "Node.h"
    "Parser.h"
    "ParserGen.h"
    "Position.h"
    "Scanner.h"
    "Sets.h"
    "SortedList.h"
    "State.h"
    "StringBuilder.h"
    "Symbol.h"
    "Tab.h"
    "Target.h"
    )

set(PROJECT_SOURCES
    "Action.cpp"
    "ArrayList.cpp"
    "BitArray.cpp"
    "CharClass.cpp"
    "CharSet.cpp"
    "Coco.cpp"
    "Comment.cpp"
    "DFA.cpp"
    "Generator.cpp"
    "HashTable.cpp"
    "Melted.cpp"
    "Node.cpp"
    "Parser.cpp"
    "ParserGen.cpp"
    "Position.cpp"
    "Scanner.cpp"
    "SortedList.cpp"
    "State.cpp"
    "StringBuilder.cpp"
    "Symbol.cpp"
    "Tab.cpp"
    "Target.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    SOURCES ${PROJECT_SOURCES}
)
