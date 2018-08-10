Coco/R
======

This directory contains a slightly changed version of the original
[__Coco/R__](http://www.ssw.uni-linz.ac.at/Coco/) for C++ ported and
maintained by Markus L&ouml;berbauer and Csaba Balazs.

The original version can be downloaded from [__here__](http://www.ssw.uni-linz.ac.at/Coco/CPP/CocoSourcesCPP.zip).

Changes by NVIDIA
=================

- 2017/01/10  
  Let Coco/R generate the Start set as const, no functional change, but saves some code and cycles.

- 2016/11/07  
  Added a pragma "$tokenPrefix" and an option "-token_prefix" to the Coco/R parser generator.

  By default, Coco/R uses "_" as prefix for token names, i.e., a declared
  token T is used in the generated C++ code as "_T". This is problematic
  as it might (and in the case of OS/X does) clash with builtin definitions.
  Using this option allows to change the prefix.

- 2015/05/12  
  Changed the format of the Coco/R messages to MS format, so Visual Studio understands them,
  i.e., report errors in file(line,col) format.

- 2015/05/12  
  Fixed Coco/R -lines option output under Windows.
  Ensure that backslashes are escaped in the #line directive.

- 2015/04/07  
  Made the extra token "kinds" maxT and noSym enum constants instead of member variables,
  which simplifies hand-written scanners and is useful in general.

- 2015/04/01  
  Removed unused Buffer::GetString() from the default Coco/R Scanner.frame.

- 2015/02/23  
  Made Coco/R more C++ like, no functional change.

  Remove some "Java-ishm" code style to simplify later changes:
  - use enum types where appropriate instead of int
  - use switch instead of if cascade

- 2013/12/18  
  Suppress the "Misplaced resolver: no LL(1) conflict." warning.

  We use the resolvers to compute a context sensitive rule so the
  warning is ok, but regulary confuses the build log inspectors.
  Add a -no_misplaced_resolver option to Coco/R to suppress it.
