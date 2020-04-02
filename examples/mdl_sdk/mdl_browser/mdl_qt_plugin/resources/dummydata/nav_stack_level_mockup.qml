/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Controls.Universal 2.3
import QtQuick.Layouts 1.3

ListModel {

    property var currentPackage: nav_node_mockup

    ListElement {
        packageName: "Adipiscing"
        packageIsModule: false
        packageRating: 0.9
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Aliqua"
        packageIsModule: false
        packageRating: 4.8
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Amet"
        packageIsModule: false
        packageRating: 1.6
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Consectetur"
        packageIsModule: false
        packageRating: 4.9
        moduleIsShadowing: true
        //moduleShadows: ["AAA", "BBB"]
    }

    ListElement {
        packageName: "Do"
        packageIsModule: false
        packageRating: 1.5
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Dolor"
        packageIsModule: false
        packageRating: 0.9
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Dolore"
        packageIsModule: false
        packageRating: 1.5
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Eabore"
        packageIsModule: false
        packageRating: 2.8
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Eiusmod"
        packageIsModule: false
        packageRating: 0.3
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Elit"
        packageIsModule: false
        packageRating: 2.1
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Incididunt"
        packageIsModule: true
        packageRating: 4.8
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Ipsom"
        packageIsModule: true
        packageRating: 4.4
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Labore"
        packageIsModule: true
        packageRating: 3.7
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Lorem"
        packageIsModule: true
        packageRating: 2.3
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Magna"
        packageIsModule: true
        packageRating: 2.0
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Sed"
        packageIsModule: true
        packageRating: 3.6
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Sit"
        packageIsModule: true
        packageRating: 1.1
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Tempor"
        packageIsModule: true
        packageRating: 2.1
        packageIsShadowing: false
    }

    ListElement {
        packageName: "Ut"
        packageIsModule: true
        packageRating: 4.4
        packageIsShadowing: false
    }
} 
