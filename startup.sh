#!/bin/bash
cd /home/site/wwwroot
streamlit run app.py --server.port=8000 --server.enableCORS=false