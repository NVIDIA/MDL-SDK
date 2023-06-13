#!/bin/bash
#*****************************************************************************
# Copyright 2023 NVIDIA Corporation. All rights reserved.
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
