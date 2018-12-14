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
