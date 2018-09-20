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
