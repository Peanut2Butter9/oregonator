# Original slides (June 2024)

The Beamer source and figures as submitted, with one change: the student ID
numbers (NIM) on the group slide were removed. `pic/` keeps the files the
slides include, under their original names.

`slide.pdf` was built with `latexmk -xelatex`. On macOS the `ctex` package
looks for system Chinese fonts and fails, so the PDF was built from a copy
with `\usepackage{ctex, hyperref}` replaced by
`\usepackage[fontset=fandol]{ctex}\usepackage{hyperref}`. The corrected slides
in `../slides/` drop `ctex` altogether; the deck has no Chinese text.
