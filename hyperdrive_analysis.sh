#!/bin/bash

for i in {0..3}; do
	/home/nate/.local/bin/hyperdrive deploy python3 analogical_comparison.py $i
done
python3 analogical_comparison.py 4 5 6
