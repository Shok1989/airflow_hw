import os
import dill
import pandas as pd
import json
import glob
from datetime import datetime


def predict():
	path = os.environ.get('PROJECT_PATH', '/Users/igor/airflow_hw')
	# определяем путь до моделей
	mdl_path = "{0}/data/models/".format(path)
	# получим последнею обученную модель
	# сначала определим какой файл с моделью последний
	only_files = [f for f in os.listdir(mdl_path) if os.path.isfile(os.path.join(mdl_path, f))]
	last_model_path = mdl_path + str(max(only_files))
	with open(last_model_path, 'rb') as file:
		model = dill.load(file)

	def pred(work_model, test_json):
		with open(test_json, 'r') as file:
			js = json.load(file)
		test_df = pd.DataFrame.from_dict([js])
		y = work_model.predict(test_df)
		rs = {'car_id': test_df['id'].values[0], 'predict': y[0]}
		return rs

	# определяем путь до тестовых данных и куда складывать предсказания
	tst_path = "{0}/data/test/".format(path)
	pred_path = "{0}/data/predictions/".format(path)

	# прогоняем все тестовые файлы через модель
	file_type = tst_path + '*.json'
	fnl_dt = pd.DataFrame(columns=['car_id', 'predict'])
	for filename in glob.glob(file_type):
		x = pred(model, filename)
		work_df = pd.DataFrame.from_dict([x])
		fnl_dt = pd.concat([fnl_dt, work_df], axis=0)
	fnl_dt.to_csv(f'{pred_path}preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
	predict()
