#!/bin/bash
#*****************************************************************************
# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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

# Wrapper script for dot. Modifies the input file for dot on-the-fly for improved visual appearance.

FULL_FILENAME="$1"

# Remove all explicit line breaks and IIDs.
sed @SED_I@ -E 's/\\l//g;s/0x[0-9a-f]{1,8}(, 0x[0-9a-f]{1,4}){2}(, 0x[0-9a-f]{1,2}){8}/\.\.\./g' "$FULL_FILENAME"
rm "$FULL_FILENAME.bak"

# Use alternate direction for some inheritance diagrams.
FILENAME_ONLY="${FULL_FILENAME##*/}"
if [   $FILENAME_ONLY = classmi_1_1base_1_1IInterface__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1base_1_1Interface__declare__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1neuraylib_1_1IAttribute__set__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1neuraylib_1_1IScene__element__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1neuraylib_1_1IFunctor__base__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1IData__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1IData__simple__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1INumber__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1IData__collection__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1ICompound__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1neuraylib_1_1IType__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1neuraylib_1_1IValue__inherit__graph.dot \
    -o $FILENAME_ONLY = classmi_1_1neuraylib_1_1IExpression__inherit__graph.dot ]; then
    DIRECTION=-Grankdir=LR
    # Add line break after "mi::base::Interface_declare< ...,", i.e., with explicit last argument.
    sed @SED_I@ 's/mi::base::Interface_declare\\< ...,/mi::base::Interface_declare\\<\\l ...,/g' "$FULL_FILENAME"
    rm "$FULL_FILENAME.bak"
else
    DIRECTION=-Grankdir=TB
fi

@REAL_DOT@ "$@" $DIRECTION
