REPORTS_DIR=reports
REPORTS_TEX=reports/sample/template.tex #$(wildcard $(REPORTS_DIR)/**/*.tex)
REPORTS=$(basename $(REPORTS_TEX))
DIRECTORIES=$(dir $(REPORTS))
CONFIGS=$(addsuffix config.cls,$(DIRECTORIES))
REPORTS_PDF=$(addsuffix .pdf,$(REPORTS))
TEXTMP_DIR=.textmp
$(info $(CONFIGS))
all: reports

.PHONY: all

reports: $(REPORTS_PDF)

.SECONDEXPANSION:
$(REPORTS_PDF): %.pdf : %.tex $$(dir %)config.cls
	-echo $*
	-mkdir $(dir $@)/$(TEXTMP_DIR)
	-TEXINPUTS=.:$(dir $@): pdflatex -synctex=1 -interaction=nonstopmode \
		-output-directory=$(dir $@)$(TEXTMP_DIR) \
		$<
	mv $(dir $@)$(TEXTMP_DIR)/$(notdir $@) $@

$(CONFIGS): %.cls: %.yaml
	script/generate_config.py $<

clean:
	-rm $(REPORTS_PDF)
.PHONY: clean
.SILENT:clean
