# Rebuild dummy-paper.pdf from dummy-paper.tex with `make fixtures` here, or from the repository root
# with `make -C tests/fixtures/dummy-paper fixtures`. The PDF is committed, so the tests need no TeX;
# a fixed SOURCE_DATE_EPOCH, an empty trailer id and no pdfTeX banner keep a rebuild byte-for-byte
# reproducible with the same TeX installation.
SOURCE_DATE_EPOCH ?= 1700000000

.PHONY: fixtures clean
fixtures: dummy-paper.pdf

dummy-paper.pdf: dummy-paper.tex
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1 pdflatex -interaction=nonstopmode -halt-on-error dummy-paper.tex
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1 pdflatex -interaction=nonstopmode -halt-on-error dummy-paper.tex
	rm -f dummy-paper.aux dummy-paper.log dummy-paper.out dummy-paperNotes.bib

clean:
	rm -f dummy-paper.aux dummy-paper.log dummy-paper.out dummy-paperNotes.bib
