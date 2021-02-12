#!/bin/bash

pip install -r requirements.txt
[[ -d data ]] || ((gdown --id 1OUd7dQybiioKMu7NXtWxIITdun8SaaUX) && unzip data.zip)