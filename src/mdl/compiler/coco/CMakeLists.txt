# name of the target and the resulting library
set(PROJECT_NAME mdl-compiler-coco)

# collect sources
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
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    SOURCES ${PROJECT_SOURCES}
)
