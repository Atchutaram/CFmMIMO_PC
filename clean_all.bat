echo off
if exist data_logs_training rmdir /Q /S data_logs_training
if exist data_logs_testing rmdir /Q /S data_logs_testing
if exist lightning_logs rmdir /Q /S lightning_logs
if exist models_sc_1 rmdir /Q /S models_sc_1
if exist models_sc_2 rmdir /Q /S models_sc_2
if exist interm_models rmdir /Q /S interm_models
if exist sc.pkl del sc.pkl