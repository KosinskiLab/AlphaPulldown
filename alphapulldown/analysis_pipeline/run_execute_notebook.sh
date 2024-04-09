#!/bin/bash
source activate programme_notebook
jupyter nbconvert --to notebook --inplace --execute $@  