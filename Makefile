.PHONY: all
all: docs/epipolar_geometry.html docs/non_ml_focal_length_estimation.html docs/final.html

docs/%.html: notes/%.org
	scripts/fix_svg.py $^ docs/
	pandoc $^ > $@
