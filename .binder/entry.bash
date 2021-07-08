#!/usr/bin/env bash

export DISPLAY=:99
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
export PYVISTA_PLOT_THEME=document
which Xvfb
Xvfb :99 -screen 0 1024x768x24  &
sleep 3
exec "$@"
