.PHONY: all
all: docs/epipolar_geometry.pdf docs/non_ml_focal_length_estimation.pdf

docs/%.pdf: notes/%.org
	emacs -u "$(id -un)" --batch $^ -f org-latex-export-to-pdf
	cp $^ $@
