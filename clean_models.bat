echo off
if exist lightning_logs rmdir /Q /S lightning_logs
if exist models_sc_1 rmdir /Q /S models_sc_1
if exist models_sc_2 rmdir /Q /S models_sc_2
if exist interm_models rmdir /Q /S interm_models
if exist sc.pkl del sc.pkl