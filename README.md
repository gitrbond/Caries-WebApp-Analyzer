# Caries-WebApp-Analyzer
A simple Flask web app that analyzes the uploaded picture and recognizes caries with tensorflow object detect

Запуск:
pip install -r packages_to_be_intalled.txt

cd cav_detection_tf_obj_det_api/models/research

python -m pip install .

add cav_detection_tf_obj_det_api/models/research to PYTHONPATH

cd ../../..
python app.py

go to http://localhost:5000
