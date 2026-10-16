MAIN := paper
OUT := out

.PHONY: all clean

all: $(OUT)/$(MAIN).pdf
	@test -s $(OUT)/$(MAIN).synctex.gz || { echo 'Missing SyncTeX metadata; run ./compile.sh to rebuild.' >&2; exit 1; }

$(OUT)/$(MAIN).pdf: $(MAIN).tex refs.bib acl.sty acl_natbib.bst $(wildcard ../figures/*.pdf figures/*.tex)
	mkdir -p $(OUT)
	texfot --quiet pdflatex -halt-on-error -synctex=1 -output-directory $(OUT) $(MAIN).tex
	bibtex $(OUT)/$(MAIN)
	texfot --quiet pdflatex -halt-on-error -synctex=1 -output-directory $(OUT) $(MAIN).tex
	texfot --quiet pdflatex -halt-on-error -synctex=1 -output-directory $(OUT) $(MAIN).tex

clean:
	rm -f $(OUT)/$(MAIN).aux $(OUT)/$(MAIN).bbl $(OUT)/$(MAIN).blg
	rm -f $(OUT)/$(MAIN).log $(OUT)/$(MAIN).out $(OUT)/$(MAIN).pdf
	rm -f $(OUT)/$(MAIN).synctex.gz
