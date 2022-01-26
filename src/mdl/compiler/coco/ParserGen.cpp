/*-------------------------------------------------------------------------
ParserGen -- Generation of the Recursive Descent Parser
Compiler Generator Coco/R,
Copyright (c) 1990, 2004 Hanspeter Moessenboeck, University of Linz
ported to C++ by Csaba Balazs, University of Szeged
extended by M. Loeberbauer & A. Woess, Univ. of Linz
with improvements by Pat Terry, Rhodes University

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 2, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

As an exception, it is allowed to write an extension of Coco/R that is
used as a plugin in non-free software.

If not otherwise stated, any source code generated by Coco/R (other than
Coco/R itself) does not fall under the GNU General Public License.
-------------------------------------------------------------------------*/

#include <ctype.h>
#include "ArrayList.h"
#include "ParserGen.h"
#include "Parser.h"
#include "BitArray.h"
#include "Scanner.h"
#include "Generator.h"

namespace Coco {

void ParserGen::Indent (int n) {
	for (int i = 1; i <= n; i++) fwprintf(gen, L"\t");
}

// use a switch if more than 5 alternatives and none starts with a resolver, and no LL1 warning
bool ParserGen::UseSwitch (Node *p) {
	BitArray *s1, *s2;
	if (p->typ != Node::alt) return false;
	int nAlts = 0;
	s1 = new BitArray(tab->terminals->Count);	
	while (p != NULL) {
		s2 = tab->Expected0(p->sub, curSy);
		// must not optimize with switch statement, if there are ll1 warnings
		if (s1->Overlaps(s2)) { return false; }
		s1->Or(s2);
		++nAlts;
		// must not optimize with switch-statement, if alt uses a resolver expression
		if (p->sub->typ == Node::rslv) return false;
		p = p->down;
	}
	return nAlts > 5;
}
    
int ParserGen::GenNamespaceOpen(const wchar_t *nsName) {
	if (nsName == NULL || coco_string_length(nsName) == 0) {
		return 0;
	}
	const int len = coco_string_length(nsName);
	int startPos = 0;
	int nrOfNs = 0;
	do {
		int curLen = coco_string_indexof(nsName + startPos, COCO_CPP_NAMESPACE_SEPARATOR);
		if (curLen == -1) { curLen = len - startPos; }
		wchar_t *curNs = coco_string_create(nsName, startPos, curLen);
		fwprintf(gen, L"namespace %ls {\n", curNs);
		coco_string_delete(curNs);
		startPos = startPos + curLen + 1;
		if (startPos < len && nsName[startPos] == COCO_CPP_NAMESPACE_SEPARATOR) {
			++startPos;
		}
		++nrOfNs;
	} while (startPos < len);
	return nrOfNs;
}

void ParserGen::GenNamespaceClose(int nrOfNs) {
	for (int i = 0; i < nrOfNs; ++i) {
		fwprintf(gen, L"} // namespace\n");
	}
}

// Escape backslashes.
static void write_escaped(FILE *gen, wchar_t const *s)
{
	for (wchar_t c = *s++; c != L'\0'; c = *s++) {
		if (c == L'\\')
			fwprintf(gen, L"\\\\");
		else
			fwprintf(gen, L"%lc", c);
	}
}

void ParserGen::CopySourcePart (Position *pos, int indent) {
	// Copy text described by pos from atg to gen
	int ch, i;
	if (pos != NULL) {
		buffer->SetPos(pos->beg); ch = buffer->Read();
		if (tab->emitLines && pos->line) {
			fwprintf(gen, L"\n#line %d \"", pos->line);
			write_escaped(gen, tab->srcName);
			fwprintf(gen, L"\"\n");
		}
		Indent(indent);
		while (buffer->GetPos() <= pos->end) {
			while (ch == CR || ch == LF) {  // eol is either CR or CRLF or LF
				fwprintf(gen, L"\n"); Indent(indent);
				if (ch == CR) { ch = buffer->Read(); } // skip CR
				if (ch == LF) { ch = buffer->Read(); } // skip LF
				for (i = 1; i <= pos->col && (ch == ' ' || ch == '\t'); i++) {
					// skip blanks at beginning of line
					ch = buffer->Read();
				}
				if (buffer->GetPos() > pos->end) goto done;
			}
			fwprintf(gen, L"%lc", ch);
			ch = buffer->Read();
		}
		done:
		if (indent > 0) fwprintf(gen, L"\n");
	}
}

void ParserGen::GenErrorMsg (ErrorType errTyp, Symbol *sym) {
	errorNr++;
	const int formatLen = 1000;
	wchar_t format[formatLen];
	coco_swprintf(format, formatLen, L"\t\t\tcase %d: s = L\"", errorNr);
	coco_string_merge(err, format);
	switch (errTyp) {
	case tErr:
		if (sym->tokenKind == Symbol::litToken || sym->tokenKind == Symbol::fixedToken) {
                        wchar_t const *name = sym->name;
                        if (name[0] != '"') {
                            Iterator *iter = tab->literals->GetIterator();
                            while (iter->HasNext()) {
                                DictionaryEntry *e = iter->Next();
                                if (e->val == sym) {
                                    name = e->key;
                                    break;
                                }
                            }
                        }
			coco_swprintf(format, formatLen, L"%ls expected", tab->Escape(name));
			coco_string_merge(err, format);
		} else {
			coco_swprintf(format, formatLen, L"%ls expected", sym->name);
			coco_string_merge(err, format);
		}
		break;
	case altErr:
		coco_swprintf(format, formatLen, L"invalid %ls", sym->name);
		coco_string_merge(err, format);
		break;
	case syncErr:
		coco_swprintf(format, formatLen, L"this symbol not expected in %ls", sym->name);
		coco_string_merge(err, format);
		break;
	}
	coco_swprintf(format, formatLen, L"\"; break;\n");
	coco_string_merge(err, format);
}

int ParserGen::NewCondSet (const BitArray *s) {
	for (int i = 1; i < symSet->Count; i++) // skip symSet[0] (reserved for union of SYNC sets)
		if (Sets::Equals(s, (BitArray*)(*symSet)[i])) return i;
	symSet->Add(s->Clone());
	return symSet->Count - 1;
}

void ParserGen::GenCond (const BitArray *s, Node *p) {
	if (p->typ == Node::rslv) CopySourcePart(p->pos, 0);
	else {
		int n = Sets::Elements(s);
		if (n == 0) fwprintf(gen, L"false"); // happens if an ANY set matches no symbol
		else if (n <= maxTerm) {
			Symbol *sym;
			for (int i=0; i<tab->terminals->Count; i++) {
				sym = (Symbol*)((*(tab->terminals))[i]);
				if ((*s)[sym->n]) {
					fwprintf(gen, L"la->kind == ");
					WriteSymbolOrCode(gen, sym);
					--n;
					if (n > 0) fwprintf(gen, L" || ");
				}
			}
		} else
			fwprintf(gen, L"StartOf(%d)", NewCondSet(s));
	}
}

void ParserGen::PutCaseLabels (const BitArray *s) {
	Symbol *sym;
	for (int i=0; i<tab->terminals->Count; i++) {
		sym = (Symbol*)((*(tab->terminals))[i]);
		if ((*s)[sym->n]) {
			fwprintf(gen, L"case ");
			WriteSymbolOrCode(gen, sym);
			fwprintf(gen, L": ");
		}
	}
}

void ParserGen::GenCode (Node *p, int indent, BitArray *isChecked) {
	Node *p2;
	BitArray *s1, *s2;
	while (p != NULL) {
		switch (p->typ) {
		case Node::nt:
			Indent(indent);
			fwprintf(gen, L"%ls(", p->sym->name);
			CopySourcePart(p->pos, 0);
			fwprintf(gen, L");\n");
			break;
		case Node::t:
			Indent(indent);
			// assert: if isChecked[p->sym->n] is true, then isChecked contains only p->sym->n
			if ((*isChecked)[p->sym->n])
			    fwprintf(gen, L"Get();\n");
			else {
				fwprintf(gen, L"Expect(");
				WriteSymbolOrCode(gen, p->sym);
				fwprintf(gen, L");\n");
			}
			break;
		case Node::wt:
			Indent(indent);
			s1 = tab->Expected(p->next, curSy);
			s1->Or(tab->allSyncSets);
			fwprintf(gen, L"ExpectWeak(");
			WriteSymbolOrCode(gen, p->sym);
			fwprintf(gen, L", %d);\n", NewCondSet(s1));
			break;
		case Node::any:
		{
			Indent(indent);
			int acc = Sets::Elements(p->set);
			if (tab->terminals->Count == (acc + 1) || (acc > 0 && Sets::Equals(p->set, isChecked))) {
				// either this ANY accepts any terminal (the + 1 = end of file), or exactly what's allowed here
				fwprintf(gen, L"Get();\n");
			} else {
				GenErrorMsg(altErr, curSy);
				if (acc > 0) {
					fwprintf(gen, L"if ("); GenCond(p->set, p); fwprintf(gen, L") Get(); else SynErr(%d);\n", errorNr);
				} else fwprintf(gen, L"SynErr(%d); // ANY node that matches no symbol\n", errorNr);
			}
			break;
		}
		case Node::eps:		// nothing
			break;
		case Node::rslv:	// nothing
			break;
		case Node::sem:
			CopySourcePart(p->pos, indent);
			break;
		case Node::sync:
			Indent(indent);
			GenErrorMsg(syncErr, curSy);
			fwprintf(gen, L"while (!("); GenCond(p->set, p); fwprintf(gen, L")) {");
			fwprintf(gen, L"SynErr(%d); Get();", errorNr); fwprintf(gen, L"}\n");
			break;
		case Node::alt:
		{
			s1 = tab->First(p);
			bool equal = Sets::Equals(s1, isChecked);
			bool useSwitch = UseSwitch(p);
			if (useSwitch) { Indent(indent); fwprintf(gen, L"switch (la->kind) {\n"); }
			p2 = p;
			while (p2 != NULL) {
				s1 = tab->Expected(p2->sub, curSy);
				Indent(indent);
				if (useSwitch) {
					PutCaseLabels(s1); fwprintf(gen, L"{\n");
				} else if (p2 == p) {
					fwprintf(gen, L"if ("); GenCond(s1, p2->sub); fwprintf(gen, L") {\n");
				} else if (p2->down == NULL && equal) { fwprintf(gen, L"} else {\n");
				} else {
					fwprintf(gen, L"} else if (");  GenCond(s1, p2->sub); fwprintf(gen, L") {\n");
				}
				GenCode(p2->sub, indent + 1, s1);
				if (useSwitch) {
					Indent(indent + 1); fwprintf(gen, L"break;\n");
					Indent(indent); fwprintf(gen, L"}\n");
				}
				p2 = p2->down;
			}
			Indent(indent);
			if (equal) {
				fwprintf(gen, L"}\n");
			} else {
				GenErrorMsg(altErr, curSy);
				if (useSwitch) {
					fwprintf(gen, L"default: SynErr(%d); break;\n", errorNr);
					Indent(indent); fwprintf(gen, L"}\n");
				} else {
					fwprintf(gen, L"} else {\n");
					Indent(indent + 1); fwprintf(gen, L"SynErr(%d);\n", errorNr);
					Indent(indent); fwprintf(gen, L"}\n");
				}
			}
			break;
		}
		case Node::iter:
			Indent(indent);
			p2 = p->sub;
			fwprintf(gen, L"while (");
			if (p2->typ == Node::wt) {
				s1 = tab->Expected(p2->next, curSy);
				s2 = tab->Expected(p->next, curSy);
				fwprintf(gen, L"WeakSeparator(");
				WriteSymbolOrCode(gen, p2->sym);
				fwprintf(gen, L",%d,%d) ", NewCondSet(s1), NewCondSet(s2));
				s1 = new BitArray(tab->terminals->Count);  // for inner structure
				if (p2->up || p2->next == NULL) p2 = NULL; else p2 = p2->next;
			} else {
				s1 = tab->First(p2);
				GenCond(s1, p2);
			}
			fwprintf(gen, L") {\n");
			GenCode(p2, indent + 1, s1);
			Indent(indent); fwprintf(gen, L"}\n");
			break;
		case Node::opt:
			s1 = tab->First(p->sub);
			Indent(indent);
			fwprintf(gen, L"if ("); GenCond(s1, p->sub); fwprintf(gen, L") {\n");
			GenCode(p->sub, indent + 1, s1);
			Indent(indent); fwprintf(gen, L"}\n");
			break;
		default:
			break;
		}
		if (p->typ != Node::eps && p->typ != Node::sem && p->typ != Node::sync)
			isChecked->SetAll(false);  // = new BitArray(Symbol.terminals.Count);
		if (p->up) break;
		p = p->next;
	}
}


void ParserGen::GenTokensHeader() {
	Symbol *sym;
	int i;
	bool isFirst = true;

	fwprintf(gen, L"\tenum TokenKind {\n");

	// tokens
	for (i=0; i<tab->terminals->Count; i++) {
		sym = (Symbol*)((*(tab->terminals))[i]);
		if (!isalpha(sym->name[0])) { continue; }

		if (isFirst) { isFirst = false; }
		else { fwprintf(gen , L",\n"); }

		fwprintf(gen , L"\t\t%ls%ls=%d", tab->tokenPrefix, sym->name, sym->n);
	}
	// generate helper values
	if (!isFirst)
		fwprintf(gen , L",\n");
	fwprintf(gen, L"\t\tmaxT=%d,\n", tab->terminals->Count - 1);
	fwprintf(gen, L"\t\tnoSym = %d", tab->noSym->n);

	// pragmas
	for (i=0; i<tab->pragmas->Count; i++) {
		if (isFirst) { isFirst = false; }
		else { fwprintf(gen , L",\n"); }

		sym = (Symbol*)((*(tab->pragmas))[i]);
		fwprintf(gen , L"\t\t_%ls=%d", sym->name, sym->n);
	}

	fwprintf(gen, L"\n\t};\n");
}

void ParserGen::GenCodePragmas() {
	Symbol *sym;
	for (int i=0; i<tab->pragmas->Count; i++) {
		sym = (Symbol*)((*(tab->pragmas))[i]);
		fwprintf(gen, L"\t\tif (la->kind == ");
		WriteSymbolOrCode(gen, sym);
		fwprintf(gen, L") {\n");
		CopySourcePart(sym->semPos, 4);
		fwprintf(gen, L"\t\t}\n");
	}
}

void ParserGen::WriteSymbolOrCode(FILE *gen, const Symbol *sym) {
	if (!isalpha(sym->name[0])) {
		fwprintf(gen, L"%d /* %ls */", sym->n, sym->name);
	} else {
		fwprintf(gen, L"%ls%ls", tab->tokenPrefix, sym->name);
	}
}

void ParserGen::GenProductionsHeader() {
	Symbol *sym;
	for (int i=0; i<tab->nonterminals->Count; i++) {
		sym = (Symbol*)((*(tab->nonterminals))[i]);
		curSy = sym;
		fwprintf(gen, L"\tvoid %ls(", sym->name);
		CopySourcePart(sym->attrPos, 0);
		fwprintf(gen, L");\n");
	}
}

void ParserGen::GenProductions() {
	Symbol *sym;
	for (int i=0; i<tab->nonterminals->Count; i++) {
		sym = (Symbol*)((*(tab->nonterminals))[i]);
		curSy = sym;
		fwprintf(gen, L"void Parser::%ls(", sym->name);
		CopySourcePart(sym->attrPos, 0);
		fwprintf(gen, L") {\n");
		CopySourcePart(sym->semPos, 2);
		GenCode(sym->graph, 1, new BitArray(tab->terminals->Count));
		fwprintf(gen, L"}\n"); fwprintf(gen, L"\n");
	}
}

void ParserGen::InitSets() {
	fwprintf(gen, L"\tstatic bool const set[%d][%d] = {\n", symSet->Count, tab->terminals->Count+1);

	for (int i = 0; i < symSet->Count; i++) {
		BitArray *s = (BitArray*)(*symSet)[i];
		fwprintf(gen, L"\t\t{");
		int j = 0;
		Symbol *sym;
		for (int k=0; k<tab->terminals->Count; k++) {
			sym = (Symbol*)((*(tab->terminals))[k]);
			if ((*s)[sym->n]) fwprintf(gen, L"T,"); else fwprintf(gen, L"x,");
			++j;
			if (j%4 == 0) fwprintf(gen, L" ");
		}
		if (i == symSet->Count-1) fwprintf(gen, L"x}\n"); else fwprintf(gen, L"x},\n");
	}
	fwprintf(gen, L"\t};\n\n");
}

void ParserGen::WriteParser () {
	Generator g = Generator(tab, errors);
	int oldPos = buffer->GetPos();  // Pos is modified by CopySourcePart
	symSet->Add(tab->allSyncSets);

	fram = g.OpenFrame(L"Parser.frame");
	gen = g.OpenGen(L"Parser.h");

	Symbol *sym;
	for (int i=0; i<tab->terminals->Count; i++) {
		sym = (Symbol*)((*(tab->terminals))[i]);
		GenErrorMsg(tErr, sym);
	}

	g.GenCopyright();
	g.SkipFramePart(L"-->begin");

	g.CopyFramePart(L"-->prefix");
	g.GenPrefixFromNamespace();

	g.CopyFramePart(L"-->prefix");
	g.GenPrefixFromNamespace();

	g.CopyFramePart(L"-->headerdef");

	if (usingPos != NULL) {CopySourcePart(usingPos, 0); fwprintf(gen, L"\n");}
	g.CopyFramePart(L"-->namespace_open");
	int nrOfNs = GenNamespaceOpen(tab->nsName);

	g.CopyFramePart(L"-->constantsheader");
	GenTokensHeader();  /* ML 2002/09/07 write the token kinds */
	g.CopyFramePart(L"-->declarations"); CopySourcePart(tab->semDeclPos, 0);
	g.CopyFramePart(L"-->productionsheader"); GenProductionsHeader();
	g.CopyFramePart(L"-->namespace_close");
	GenNamespaceClose(nrOfNs);

	g.CopyFramePart(L"-->implementation");
	fclose(gen);

	// Source
	gen = g.OpenGen(L"Parser.cpp");

	g.GenCopyright();
	g.SkipFramePart(L"-->begin");
	g.CopyFramePart(L"-->namespace_open");
	nrOfNs = GenNamespaceOpen(tab->nsName);

	g.CopyFramePart(L"-->pragmas"); GenCodePragmas();
	g.CopyFramePart(L"-->productions"); GenProductions();
	g.CopyFramePart(L"-->parseRoot"); fwprintf(gen, L"\t%ls();\n", tab->gramSy->name); if (tab->checkEOF) fwprintf(gen, L"\tExpect(0);");
	g.CopyFramePart(L"-->constants");
	g.CopyFramePart(L"-->initialization"); InitSets();
	g.CopyFramePart(L"-->errors"); fwprintf(gen, L"%ls", err);
	g.CopyFramePart(L"-->namespace_close");
	GenNamespaceClose(nrOfNs);
	g.CopyFramePart(NULL);
	fclose(gen);
	buffer->SetPos(oldPos);
}


void ParserGen::WriteStatistics () {
	fwprintf(trace, L"\n");
	fwprintf(trace, L"%d terminals\n", tab->terminals->Count);
	fwprintf(trace, L"%d symbols\n", tab->terminals->Count + tab->pragmas->Count +
	                               tab->nonterminals->Count);
	fwprintf(trace, L"%d nodes\n", tab->nodes->Count);
	fwprintf(trace, L"%d sets\n", symSet->Count);
}


ParserGen::ParserGen (Parser *parser) {
	maxTerm = 3;
	CR = '\r';
	LF = '\n';
	tab = parser->tab;
	errors = parser->errors;
	trace = parser->trace;
	buffer = parser->scanner->buffer;
	errorNr = -1;
	usingPos = NULL;

	symSet = new ArrayList();
	err = NULL;
}

}; // namespace
