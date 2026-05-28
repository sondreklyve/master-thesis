.PHONY: pdf clean
pdf:
	cd thesis && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	cd thesis && latexmk -C main.tex
