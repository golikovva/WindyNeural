# NN for wind speed forecast using station data and GFS forecast 
By Viktor Golikov for Windy.app
##Data
We use station data with 5 parameters which are wind speed module, sinus and cosinus of wind direction (angle between north and the direction wind is blowing from), hour of day and day of year 
## Training
To train the model run:
'''
python main.py
'''
## Evaluation and inference
To evaluate model run:
'''
python eval/evaluate.py
'''
To inference model use function 'inference_model()' from 'eval/inference_model.py' or run:
'''
python eval/inference_model.py
'''
