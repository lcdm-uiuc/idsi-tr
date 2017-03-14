REPORTS_DIR=reports
REPORTS_TEX=$(wildcard $(REPORTS_DIR)/**/*.tex)
REPORTS=$(basename $(REPORTS_TEX))
REPORTS_PDF=$(addsuffix .pdf,$(REPORTS))
TEXTMP_DIR=.textmp

BIB = bibtex

all: reports

.PHONY: all

reports: $(REPORTS_PDF)

$(REPORTS_PDF): %.pdf : %.tex
	-mkdir $(dir $@)/$(TEXTMP_DIR)
	-pdflatex -synctex=1 -interaction=nonstopmode \
		-output-directory=$(dir $@)$(TEXTMP_DIR) \
		$<
	mv $(dir $@)$(TEXTMP_DIR)/$(notdir $@) $@

clean:
	-rm $(REPORTS_PDF)
.PHONY: clean
.SILENT:clean
